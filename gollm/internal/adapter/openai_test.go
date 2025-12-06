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

package adapter_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/zchee/tumix/gollm/internal/adapter"
)

func TestGenAIToResponsesInput(t *testing.T) {
	contents := []*genai.Content{
		genai.NewContentFromText("hello", genai.RoleUser),
		{
			Role: genai.RoleModel,
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

	items, err := adapter.GenAIToResponsesInput(contents)
	if err != nil {
		t.Fatalf("GenAIToResponsesInput err = %v", err)
	}

	if got, want := len(items), 4; got != want {
		t.Fatalf("len(items) = %d, want %d", got, want)
	}

	user := items[0].OfMessage
	if user == nil || user.Content.OfString.Value != "hello" {
		t.Fatalf("user message = %+v", user)
	}

	asst := items[1].OfMessage
	if asst == nil || asst.Role != responses.EasyInputMessageRoleAssistant || asst.Content.OfString.Value != "thinking" {
		t.Fatalf("assistant message = %+v", asst)
	}

	fc := items[2].OfFunctionCall
	if fc == nil {
		t.Fatalf("function call missing")
	}
	if got, want := fc.CallID, "tool_1_1"; got != want {
		t.Fatalf("function call id = %q, want %q", got, want)
	}
	if got, want := fc.Name, "lookup"; got != want {
		t.Fatalf("function call name = %q, want %q", got, want)
	}
	if got, want := fc.Arguments, `{"q":"foo"}`; got != want {
		t.Fatalf("function call args = %q, want %q", got, want)
	}

	fcOut := items[3].OfFunctionCallOutput
	if fcOut == nil {
		t.Fatalf("function call output missing")
	}
	if got, want := fcOut.CallID, "tool_1_2"; got != want {
		t.Fatalf("function call output id = %q, want %q", got, want)
	}
	if got := fcOut.Output.OfString.Value; got == "" || got == "{}" {
		t.Fatalf("function call output empty: %q", got)
	}
}

func TestOpenAIResponseToLLM(t *testing.T) {
	resp := &responses.Response{
		ID:     "resp-1",
		Status: responses.ResponseStatusCompleted,
		Output: []responses.ResponseOutputItemUnion{
			{
				Type: "message",
				Role: constant.ValueOf[constant.Assistant](),
				Content: []responses.ResponseOutputMessageContentUnion{
					{Type: "output_text", Text: "hello"},
				},
			},
			{
				Type:      "function_call",
				CallID:    "call-1",
				Name:      "lookup_city",
				Arguments: `{"city":"Paris"}`,
			},
		},
		Usage: responses.ResponseUsage{
			InputTokens:  3,
			OutputTokens: 5,
			TotalTokens:  8,
		},
	}

	got, err := adapter.OpenAIResponseToLLM(resp, nil)
	if err != nil {
		t.Fatalf("OpenAIResponseToLLM err = %v", err)
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
		TurnComplete: true,
		FinishReason: genai.FinishReasonStop,
	}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("OpenAIResponseToLLM diff (-want +got):\n%s\nparts=%#v", diff, got.Content.Parts)
	}
}

func TestOpenAIResponseToLLM_StopAndFunctionOutput(t *testing.T) {
	resp := &responses.Response{
		ID:     "resp-2",
		Status: responses.ResponseStatusCompleted,
		Output: []responses.ResponseOutputItemUnion{
			{
				Type: "message",
				Role: constant.ValueOf[constant.Assistant](),
				Content: []responses.ResponseOutputMessageContentUnion{
					{Type: "output_text", Text: "Hello<STOP>tail"},
				},
			},
			{
				Type:            "shell_call_output",
				ID:              "out-1",
				CallID:          "call-1",
				MaxOutputLength: 16,
				Output: responses.ResponseOutputItemUnionOutput{
					OfResponseFunctionShellToolCallOutputOutputArray: []responses.ResponseFunctionShellToolCallOutputOutput{
						{
							Outcome: responses.ResponseFunctionShellToolCallOutputOutputOutcomeUnion{
								Type:     string(constant.ValueOf[constant.Exit]()),
								ExitCode: 0,
							},
							Stdout: "done",
							Stderr: "",
						},
					},
				},
			},
		},
	}

	got, err := adapter.OpenAIResponseToLLM(resp, []string{"<STOP>"})
	if err != nil {
		t.Fatalf("OpenAIResponseToLLM err = %v", err)
	}
	if got.FinishReason != genai.FinishReasonStop {
		t.Fatalf("finish reason = %v, want stop", got.FinishReason)
	}
	if got.Content.Parts[0].Text != "Hello" {
		t.Fatalf("trimmed text = %q, want %q", got.Content.Parts[0].Text, "Hello")
	}
	if len(got.Content.Parts) < 2 || got.Content.Parts[1].FunctionResponse == nil {
		t.Fatalf("function response missing: %+v", got.Content.Parts)
	}
	if got.Content.Parts[1].FunctionResponse.Response["output"] == nil {
		t.Fatalf("function response output = %+v", got.Content.Parts[1].FunctionResponse.Response)
	}
}

func TestOpenAIStreamAggregator(t *testing.T) {
	agg := adapter.NewOpenAIStreamAggregator([]string{"<STOP>"})

	partials := agg.Process(&responses.ResponseStreamEventUnion{
		Type:  "response.output_text.delta",
		Delta: "Hel",
	})
	if len(partials) != 1 || !partials[0].Partial || partials[0].Content.Parts[0].Text != "Hel" {
		t.Fatalf("delta partial = %+v", partials)
	}

	agg.Process(&responses.ResponseStreamEventUnion{
		Type:        "response.function_call_arguments.delta",
		ItemID:      "call-1",
		OutputIndex: 0,
		Delta:       `{"city":"Par`,
	})
	agg.Process(&responses.ResponseStreamEventUnion{
		Type:        "response.function_call_arguments.delta",
		ItemID:      "call-1",
		OutputIndex: 0,
		Delta:       `is"}`,
	})

	finalResp := responses.Response{
		ID:     "resp-1",
		Status: responses.ResponseStatusCompleted,
		Output: []responses.ResponseOutputItemUnion{
			{
				Type: "message",
				Role: constant.ValueOf[constant.Assistant](),
				Content: []responses.ResponseOutputMessageContentUnion{
					{Type: "output_text", Text: "Hello"},
				},
			},
			{
				Type:      "function_call",
				CallID:    "call-1",
				Name:      "lookup_city",
				Arguments: `{"city":"Paris"}`,
			},
		},
		Usage: responses.ResponseUsage{
			InputTokens:  4,
			OutputTokens: 2,
			TotalTokens:  6,
		},
	}

	finals := agg.Process(&responses.ResponseStreamEventUnion{
		Type:     "response.completed",
		Response: finalResp,
	})
	if len(finals) != 1 {
		t.Fatalf("expected final response, got %d", len(finals))
	}
	final := finals[0]
	if final.Partial {
		t.Fatalf("final should not be partial: %+v", final)
	}
	if final.FinishReason != genai.FinishReasonStop || !final.TurnComplete {
		t.Fatalf("finish reason/turn complete mismatch: %+v", final)
	}
	if final.UsageMetadata == nil || final.UsageMetadata.TotalTokenCount != 6 {
		t.Fatalf("usage metadata = %+v", final.UsageMetadata)
	}
	if err := agg.Err(); err != nil {
		t.Fatalf("aggregator error = %v", err)
	}
}
