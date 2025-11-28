// Copyright 2025 The tumix Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package model

import (
	json "encoding/json/v2"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/respjson"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestOpenAIEnsureUserContent(t *testing.T) {
	t.Run("adds_default_when_empty", func(t *testing.T) {
		req := &model.LLMRequest{}
		m := &openAIModel{}

		m.ensureUserContent(req)

		if len(req.Contents) != 1 {
			t.Fatalf("ensureUserContent added %d contents, want 1", len(req.Contents))
		}
		if req.Contents[0].Role != genai.RoleUser {
			t.Fatalf("role = %q, want %q", req.Contents[0].Role, genai.RoleUser)
		}
		if got, want := req.Contents[0].Parts[0].Text, "Handle the requests as specified in the System Instruction."; got != want {
			t.Fatalf("default text = %q, want %q", got, want)
		}
	})

	t.Run("appends_when_last_not_user", func(t *testing.T) {
		req := &model.LLMRequest{
			Contents: []*genai.Content{genai.NewContentFromText("system guidance", "system")},
		}
		m := &openAIModel{}

		m.ensureUserContent(req)

		if got, want := len(req.Contents), 2; got != want {
			t.Fatalf("len(contents) = %d, want %d", got, want)
		}
		if req.Contents[1].Role != genai.RoleUser {
			t.Fatalf("role = %q, want %q", req.Contents[1].Role, genai.RoleUser)
		}
	})

	t.Run("no_change_when_last_user", func(t *testing.T) {
		req := &model.LLMRequest{
			Contents: []*genai.Content{genai.NewContentFromText("hello", genai.RoleUser)},
		}
		m := &openAIModel{}

		m.ensureUserContent(req)

		if got, want := len(req.Contents), 1; got != want {
			t.Fatalf("len(contents) = %d, want %d", got, want)
		}
	})
}

func TestGenaiToOpenAIMessages(t *testing.T) {
	contents := []*genai.Content{
		genai.NewContentFromText("hello", genai.RoleUser),
		{
			Role: string(genai.RoleModel),
			Parts: []*genai.Part{
				genai.NewPartFromText("thinking"),
				{
					FunctionCall: &genai.FunctionCall{
						Name: "lookup",
						Args: map[string]any{"q": "foo"},
					},
				},
				{
					FunctionResponse: &genai.FunctionResponse{
						Name:     "lookup",
						Response: map[string]any{"result": "bar"},
					},
				},
			},
		},
	}

	msgs, err := genaiToOpenAIMessages(contents)
	if err != nil {
		t.Fatalf("genaiToOpenAIMessages err = %v", err)
	}

	if got, want := len(msgs), 3; got != want {
		t.Fatalf("len(msgs) = %d, want %d", got, want)
	}

	if user := msgs[0].OfUser; user == nil || user.Content.OfString.Value != "hello" {
		t.Fatalf("user message = %+v", user)
	}

	tool := msgs[1].OfTool
	if tool == nil {
		t.Fatalf("tool response message missing")
	}
	if got, want := tool.ToolCallID, "tool_1_2"; got != want {
		t.Fatalf("tool call id = %q, want %q", got, want)
	}
	if got := tool.Content.OfString.Value; !strings.Contains(got, `"result":"bar"`) {
		t.Fatalf("tool response content = %q", got)
	}

	asst := msgs[2].OfAssistant
	if asst == nil {
		t.Fatalf("assistant message missing")
	}
	if got, want := asst.Content.OfString.Value, "thinking"; got != want {
		t.Fatalf("assistant content = %q, want %q", got, want)
	}
	if len(asst.ToolCalls) != 1 {
		t.Fatalf("assistant tool calls = %d, want 1", len(asst.ToolCalls))
	}
	fn := asst.ToolCalls[0].OfFunction
	if fn == nil {
		t.Fatalf("assistant tool call missing function payload")
	}
	if got, want := fn.ID, "tool_1_1"; got != want {
		t.Fatalf("function id = %q, want %q", got, want)
	}
	if got, want := fn.Function.Name, "lookup"; got != want {
		t.Fatalf("function name = %q, want %q", got, want)
	}
	if got, want := fn.Function.Arguments, `{"q":"foo"}`; got != want {
		t.Fatalf("function args = %q, want %q", got, want)
	}
}

func TestOpenAIResponseToLLM(t *testing.T) {
	const raw = `{
		"id": "chatcmpl-1",
		"object": "chat.completion",
		"created": 1,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"finish_reason": "stop",
			"logprobs": {"content": [], "refusal": []},
			"message": {
				"role": "assistant",
				"content": "hello",
				"tool_calls": [{
					"id": "call-1",
					"type": "function",
					"function": {"name": "lookup_city", "arguments": "{\"city\":\"Paris\"}"}
				}]
			}
		}],
		"usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
	}`

	var resp openai.ChatCompletion
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("unmarshal chat completion: %v", err)
	}
	got, err := openAIResponseToLLM(&resp)
	if err != nil {
		t.Fatalf("openAIResponseToLLM err = %v", err)
	}

	want := &model.LLMResponse{
		Content: &genai.Content{
			Role: genai.RoleModel,
			Parts: []*genai.Part{
				genai.NewPartFromText("hello"),
				{
					FunctionCall: &genai.FunctionCall{
						ID:   "call-1",
						Name: "lookup_city",
						Args: map[string]any{"city": "Paris"},
					},
				},
			},
		},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     3,
			CandidatesTokenCount: 5,
			TotalTokenCount:      8,
		},
		FinishReason: genai.FinishReasonStop,
	}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("openAIResponseToLLM diff (-want +got):\n%s", diff)
	}
}

func TestOpenAIResponseToLLM_LegacyFunctionCall(t *testing.T) {
	const raw = `{
		"id": "chatcmpl-legacy",
		"object": "chat.completion",
		"created": 1,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"finish_reason": "function_call",
			"logprobs": {"content": [], "refusal": []},
			"message": {
				"role": "assistant",
				"content": "",
				"function_call": {"name": "legacy", "arguments": "{\"foo\":123}"}
			}
		}]
	}`
	var resp openai.ChatCompletion
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("unmarshal legacy chat completion: %v", err)
	}

	got, err := openAIResponseToLLM(&resp)
	if err != nil {
		t.Fatalf("openAIResponseToLLM err = %v", err)
	}

	if len(got.Content.Parts) != 1 || got.Content.Parts[0].FunctionCall == nil {
		t.Fatalf("function call not parsed: %+v", got.Content.Parts)
	}
	fn := got.Content.Parts[0].FunctionCall
	if fn.Name != "legacy" || fn.Args["foo"] != float64(123) {
		t.Fatalf("function call = %+v", fn)
	}
	if got.FinishReason != genai.FinishReasonOther {
		t.Fatalf("finish reason = %q, want %q", got.FinishReason, genai.FinishReasonOther)
	}
}

func TestOpenAIStreamAggregator(t *testing.T) {
	agg := newOpenAIStreamAggregator()

	chunk1 := openai.ChatCompletionChunk{
		Choices: []openai.ChatCompletionChunkChoice{{
			Delta: openai.ChatCompletionChunkChoiceDelta{
				Content: "Hel",
				ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{{
					Index: 0,
					Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
						Name:      "lookup_city",
						Arguments: `{"city":"Par`,
					},
				}},
			},
		}},
	}
	chunk2 := openai.ChatCompletionChunk{
		Choices: []openai.ChatCompletionChunkChoice{{
			Delta: openai.ChatCompletionChunkChoiceDelta{
				Content: "lo",
				ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{{
					Index: 0,
					ID:    "call-1",
					Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
						Arguments: `is"}`,
					},
				}},
			},
		}},
	}
	chunk3 := openai.ChatCompletionChunk{
		Choices: []openai.ChatCompletionChunkChoice{
			{
				FinishReason: "stop",
			},
		},
		Usage: openai.CompletionUsage{
			PromptTokens:     4,
			CompletionTokens: 2,
			TotalTokens:      6,
		},
	}
	chunk3.JSON.Usage = respjson.NewField("{}") // mark usage as present

	out := agg.Process(&chunk1)
	if len(out) != 1 || !out[0].Partial || out[0].Content.Parts[0].Text != "Hel" {
		t.Fatalf("chunk1 partial = %+v", out)
	}

	out = agg.Process(&chunk2)
	if len(out) != 1 || !out[0].Partial || out[0].Content.Parts[0].Text != "lo" {
		t.Fatalf("chunk2 partial = %+v", out)
	}

	if fin := agg.Process(&chunk3); len(fin) != 0 {
		t.Fatalf("chunk3 should not yield partials, got %d", len(fin))
	}

	final := agg.Final()
	if final == nil {
		t.Fatal("Final() returned nil")
	}
	if final.Partial {
		t.Fatal("Final response should not be partial")
	}
	if final.Content == nil || len(final.Content.Parts) != 2 {
		t.Fatalf("final content parts = %+v", final.Content)
	}
	if got, want := final.Content.Parts[0].Text, "Hello"; got != want {
		t.Fatalf("aggregated text = %q, want %q", got, want)
	}

	fn := final.Content.Parts[1].FunctionCall
	if fn == nil || fn.Name != "lookup_city" {
		t.Fatalf("function call = %+v", fn)
	}
	if got, want := fn.ID, "call-1"; got != want {
		t.Fatalf("function call id = %q, want %q", got, want)
	}
	if city, ok := fn.Args["city"]; !ok || city != "Paris" {
		t.Fatalf("function call args = %+v", fn.Args)
	}

	if final.UsageMetadata == nil || final.UsageMetadata.TotalTokenCount != 6 {
		t.Fatalf("usage metadata = %+v", final.UsageMetadata)
	}
	if !final.TurnComplete {
		t.Fatalf("TurnComplete = false, want true")
	}
	if got, want := final.FinishReason, genai.FinishReasonStop; got != want {
		t.Fatalf("FinishReason = %q, want %q", got, want)
	}
}
