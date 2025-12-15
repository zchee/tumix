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

package adapter

import (
	json "encoding/json/v2"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"unsafe"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"github.com/zchee/tumix/gollm/xai"
	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
)

func TestGenAI2XAIChatOptionsMapsConfig(t *testing.T) {
	tests := map[string]struct {
		config  genai.GenerateContentConfig
		assertf func(t *testing.T, req *xaipb.GetCompletionsRequest)
	}{
		"success: maps all tunables and tools": {
			config: genai.GenerateContentConfig{
				Temperature:      ptrFloat32(0.7),
				TopP:             ptrFloat32(0.9),
				MaxOutputTokens:  123,
				Seed:             ptrInt32(42),
				StopSequences:    []string{"stop"},
				ResponseLogprobs: true,
				Logprobs:         ptrInt32(5),
				FrequencyPenalty: ptrFloat32(0.1),
				PresencePenalty:  ptrFloat32(0.2),
				ResponseJsonSchema: map[string]any{
					"type":       "object",
					"properties": map[string]any{"foo": map[string]any{"type": "string"}},
				},
				Tools: []*genai.Tool{
					{
						FunctionDeclarations: []*genai.FunctionDeclaration{{
							Name:        "fn1",
							Description: "desc",
							Parameters: &genai.Schema{
								Type: genai.TypeObject,
								Properties: map[string]*genai.Schema{
									"foo": {Type: genai.TypeString},
								},
							},
						}},
						CodeExecution: &genai.ToolCodeExecution{},
					},
				},
				ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"fn1"},
				}},
			},
			assertf: func(t *testing.T, req *xaipb.GetCompletionsRequest) {
				t.Helper()

				if got, want := derefInt32(req.MaxTokens), int32(123); got != want {
					t.Fatalf("MaxTokens = %d, want %d", got, want)
				}
				if got, want := derefInt32(req.Seed), int32(42); got != want {
					t.Fatalf("Seed = %d, want %d", got, want)
				}
				if got, want := derefFloat32(req.Temperature), float32(0.7); got != want {
					t.Fatalf("Temperature = %f, want %f", got, want)
				}
				if got, want := derefFloat32(req.TopP), float32(0.9); got != want {
					t.Fatalf("TopP = %f, want %f", got, want)
				}
				if got := req.GetLogprobs(); !got {
					t.Fatalf("Logprobs = false, want true")
				}
				if got, want := derefInt32(req.TopLogprobs), int32(5); got != want {
					t.Fatalf("TopLogprobs = %d, want %d", got, want)
				}
				if got, want := derefFloat32(req.FrequencyPenalty), float32(0.1); got != want {
					t.Fatalf("FrequencyPenalty = %f, want %f", got, want)
				}
				if got, want := derefFloat32(req.PresencePenalty), float32(0.2); got != want {
					t.Fatalf("PresencePenalty = %f, want %f", got, want)
				}
				if diff := cmp.Diff([]string{"stop"}, req.GetStop()); diff != "" {
					t.Fatalf("Stop diff (-want +got):\n%s", diff)
				}

				tools := req.GetTools()
				if len(tools) != 2 {
					t.Fatalf("Tools len = %d, want 2", len(tools))
				}
				fn := tools[0].GetFunction()
				if fn == nil || fn.GetName() != "fn1" {
					t.Fatalf("function tool = %+v", fn)
				}
				var params map[string]any
				if err := json.Unmarshal([]byte(fn.GetParameters()), &params); err != nil {
					t.Fatalf("decode parameters: %v", err)
				}
				if _, ok := params["properties"].(map[string]any)["foo"]; !ok {
					t.Fatalf("parameters missing foo: %+v", params)
				}
				if _, ok := tools[1].GetTool().(*xaipb.Tool_CodeExecution); !ok {
					t.Fatalf("second tool not code execution: %+v", tools[1])
				}

				tc := req.GetToolChoice()
				choice, ok := tc.GetToolChoice().(*xaipb.ToolChoice_FunctionName)
				if !ok {
					t.Fatalf("tool choice type = %T, want FunctionName", tc.GetToolChoice())
				}
				if choice.FunctionName != "fn1" {
					t.Fatalf("tool choice name = %q, want fn1", choice.FunctionName)
				}

				rf := req.GetResponseFormat()
				if rf == nil || rf.GetFormatType() != xaipb.FormatType_FORMAT_TYPE_JSON_SCHEMA {
					t.Fatalf("response format = %+v", rf)
				}
				if rf.GetSchema() == "" {
					t.Fatalf("response schema missing")
				}
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			op := GenAI2XAIChatOptions(&tc.config)
			if op == nil {
				t.Fatalf("ChatOption is nil")
			}

			req := &xaipb.GetCompletionsRequest{}
			op(req, nil)
			tc.assertf(t, req)
		})
	}
}

func TestGenAI2XAIChatOptionsNilOrNoEffect(t *testing.T) {
	tests := map[string]struct {
		config *genai.GenerateContentConfig
	}{
		"success: nil config returns nil":   {config: nil},
		"success: empty config returns nil": {config: &genai.GenerateContentConfig{}},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			op := GenAI2XAIChatOptions(tc.config)
			if op != nil {
				t.Fatalf("ChatOption = %v, want nil", op)
			}
		})
	}
}

func TestXAIResponseToLLM(t *testing.T) {
	tests := map[string]struct {
		resp    *xaipb.GetChatCompletionResponse
		assertf func(t *testing.T, got *genai.Content, llm *genai.GenerateContentResponseUsageMetadata, meta map[string]any, fr genai.FinishReason)
	}{
		"success: maps content, tool call, usage and metadata": {
			resp: &xaipb.GetChatCompletionResponse{
				Outputs: []*xaipb.CompletionOutput{{
					FinishReason: xaipb.FinishReason_REASON_STOP,
					Message: &xaipb.CompletionMessage{
						Role:    xaipb.MessageRole_ROLE_ASSISTANT,
						Content: "hello",
						ToolCalls: []*xaipb.ToolCall{{
							Id: "call1",
							Tool: &xaipb.ToolCall_Function{Function: &xaipb.FunctionCall{
								Name:      "lookup",
								Arguments: `{"city":"Paris"}`,
							}},
						}},
					},
				}},
				Usage: &xaipb.SamplingUsage{
					PromptTokens:           3,
					CompletionTokens:       5,
					TotalTokens:            8,
					CachedPromptTextTokens: 1,
					ReasoningTokens:        0,
				},
				SystemFingerprint: "fp",
				Citations:         []string{"c1"},
			},
			assertf: func(t *testing.T, got *genai.Content, usage *genai.GenerateContentResponseUsageMetadata, meta map[string]any, fr genai.FinishReason) {
				t.Helper()

				if got.Role != genai.RoleModel {
					t.Fatalf("role = %q, want %q", got.Role, genai.RoleModel)
				}
				if len(got.Parts) != 2 || got.Parts[0].Text != "hello" {
					t.Fatalf("parts = %+v", got.Parts)
				}
				fn := got.Parts[1].FunctionCall
				if fn == nil || fn.Name != "lookup" || fn.Args["city"] != "Paris" {
					t.Fatalf("function call = %+v", fn)
				}
				if usage == nil || usage.PromptTokenCount != 3 || usage.CandidatesTokenCount != 5 || usage.TotalTokenCount != 8 {
					t.Fatalf("usage = %+v", usage)
				}
				if fr != genai.FinishReasonStop {
					t.Fatalf("finish reason = %q, want %q", fr, genai.FinishReasonStop)
				}
				if meta["xai_system_fingerprint"] != "fp" || meta["xai_citations"] == nil {
					t.Fatalf("custom meta = %+v", meta)
				}
			},
		},
		"success: invalid tool args captured": {
			resp: &xaipb.GetChatCompletionResponse{
				Outputs: []*xaipb.CompletionOutput{{
					Message: &xaipb.CompletionMessage{
						Role: xaipb.MessageRole_ROLE_ASSISTANT,
						ToolCalls: []*xaipb.ToolCall{{
							Id: "bad1",
							Tool: &xaipb.ToolCall_Function{Function: &xaipb.FunctionCall{
								Name:      "oops",
								Arguments: "{bad",
							}},
						}},
					},
				}},
			},
			assertf: func(t *testing.T, got *genai.Content, _ *genai.GenerateContentResponseUsageMetadata, meta map[string]any, _ genai.FinishReason) {
				t.Helper()
				fn := got.Parts[0].FunctionCall
				if fn == nil {
					t.Fatalf("function call missing")
				}
				if _, ok := fn.Args["raw"]; !ok {
					t.Fatalf("raw args not preserved: %+v", fn.Args)
				}
				errs, ok := meta["tool_call_args_errors"].([]string)
				if !ok || len(errs) != 1 {
					t.Fatalf("tool_call_args_errors = %+v", meta)
				}
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			xresp := newTestXAIResponse(t, tc.resp)
			llm := XAIResponseToLLM(xresp)
			if llm.ErrorCode != "" {
				t.Fatalf("unexpected error: %s", llm.ErrorMessage)
			}
			if llm.Content == nil {
				t.Fatalf("content nil")
			}
			tc.assertf(t, llm.Content, llm.UsageMetadata, llm.CustomMetadata, llm.FinishReason)
		})
	}
}

func TestXAIResponseToLLMNil(t *testing.T) {
	got := XAIResponseToLLM(nil)
	if got.ErrorCode != "NIL_RESPONSE" {
		t.Fatalf("ErrorCode = %q, want NIL_RESPONSE", got.ErrorCode)
	}
}

func TestXAIStreamAggregator_TextAndClose(t *testing.T) {
	aggr := NewXAIStreamAggregator()

	resp1 := newTestXAIResponse(t, &xaipb.GetChatCompletionResponse{
		Outputs: []*xaipb.CompletionOutput{{
			Message: &xaipb.CompletionMessage{Role: xaipb.MessageRole_ROLE_ASSISTANT, Content: "He"},
		}},
	})
	resp2 := newTestXAIResponse(t, &xaipb.GetChatCompletionResponse{
		Outputs: []*xaipb.CompletionOutput{{
			FinishReason: xaipb.FinishReason_REASON_STOP,
			Message:      &xaipb.CompletionMessage{Role: xaipb.MessageRole_ROLE_ASSISTANT, Content: "Hello"},
		}},
	})

	seen := make([]string, 0, 2)
	for llm, err := range aggr.Process(t.Context(), resp1) {
		if err != nil {
			t.Fatalf("process err: %v", err)
		}
		seen = append(seen, llm.Content.Parts[0].Text)
	}
	for llm, err := range aggr.Process(t.Context(), resp2) {
		if err != nil {
			t.Fatalf("process err: %v", err)
		}
		seen = append(seen, llm.Content.Parts[0].Text)
	}

	final := aggr.Close()
	if final == nil {
		t.Fatalf("Close returned nil")
	}
	if got, want := final.Content.Parts[0].Text, "Hello"; got != want {
		t.Fatalf("final text = %q, want %q", got, want)
	}
	if len(seen) != 2 || seen[0] != "He" || seen[1] != "Hello" {
		t.Fatalf("seen sequence = %+v", seen)
	}
}

func TestXAIStreamAggregatorThought(t *testing.T) {
	aggr := NewXAIStreamAggregator()

	resp := newTestXAIResponse(t, &xaipb.GetChatCompletionResponse{
		Outputs: []*xaipb.CompletionOutput{{
			Message: &xaipb.CompletionMessage{
				Role:             xaipb.MessageRole_ROLE_ASSISTANT,
				ReasoningContent: "think",
				Content:          "ok",
			},
		}},
	})

	for llm, err := range aggr.Process(t.Context(), resp) {
		if err != nil {
			t.Fatalf("process err: %v", err)
		}
		if !llm.Content.Parts[0].Thought {
			t.Fatalf("part not marked thought: %+v", llm.Content.Parts[0])
		}
	}
	final := aggr.Close()
	if final == nil || !final.Content.Parts[0].Thought || final.Content.Parts[0].Text != "think" {
		t.Fatalf("aggregated thought incorrect: %+v", final)
	}
}

func TestXAIStreamAggregatorEmptyContent(t *testing.T) {
	aggr := NewXAIStreamAggregator()

	resp := newTestXAIResponse(t, &xaipb.GetChatCompletionResponse{
		Outputs: []*xaipb.CompletionOutput{
			{
				Message: &xaipb.CompletionMessage{
					Role: xaipb.MessageRole_ROLE_ASSISTANT,
				},
			},
		},
	})

	for llm, err := range aggr.Process(t.Context(), resp) {
		if err == nil || llm != nil {
			t.Fatalf("expected error for empty content, got llm=%+v err=%v", llm, err)
		}
	}
}

func TestAppendDelta(t *testing.T) {
	var acc strings.Builder
	acc.WriteString("Hello")
	if delta := appendDeltaToBuilder("Hello world", &acc); delta != " world" {
		t.Fatalf("delta = %q, want %q", delta, " world")
	}
	acc.Reset()
	acc.WriteString("foo")
	if delta := appendDeltaToBuilder("bar", &acc); delta != "bar" {
		t.Fatalf("delta = %q, want bar", delta)
	}
}

func TestIsZeroPart(t *testing.T) {
	if isZeroPart(nil) {
		t.Fatalf("isZeroPart(nil) = true, want false")
	}
	if !isZeroPart(&genai.Part{}) {
		t.Fatalf("isZeroPart(empty Part) = false, want true")
	}
	if isZeroPart(&genai.Part{Text: "x"}) {
		t.Fatalf("isZeroPart(text Part) = true, want false")
	}
}

func BenchmarkXAIStreamAggregator(b *testing.B) {
	ctx := b.Context()
	lengths := []int{
		1_024,
		4_096,
		10_240,
	}

	for _, n := range lengths {
		b.Run(fmt.Sprintf("len=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			payload := strings.Repeat("a", n)
			resp := newTestXAIResponse(b, &xaipb.GetChatCompletionResponse{
				Outputs: []*xaipb.CompletionOutput{
					{
						Message: &xaipb.CompletionMessage{
							Role:    xaipb.MessageRole_ROLE_ASSISTANT,
							Content: payload,
						},
					},
				},
			})

			var size int
			for b.Loop() {
				aggr := NewXAIStreamAggregator()
				for llm, err := range aggr.Process(ctx, resp) {
					if err != nil {
						b.Fatalf("process: %v", err)
					}
					if llm != nil && llm.Content != nil && len(llm.Content.Parts) > 0 {
						size += len(llm.Content.Parts[0].Text)
					}
				}
				if final := aggr.Close(); final != nil && final.Content != nil && len(final.Content.Parts) > 0 {
					size += len(final.Content.Parts[0].Text)
				}
			}
			_ = size
		})
	}
}

func TestMapXAIFinishReason(t *testing.T) {
	tests := map[string]struct {
		in   string
		want genai.FinishReason
	}{
		"success: stop": {
			in:   "REASON_STOP",
			want: genai.FinishReasonStop,
		},
		"success: max_len": {
			in:   "reason_max_len",
			want: genai.FinishReasonMaxTokens,
		},
		"success: invalid": {
			in:   "REASON_INVALID",
			want: genai.FinishReasonUnspecified,
		},
		"success: unknown fallback": {
			in:   "REASON_OTHER",
			want: genai.FinishReasonOther,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			if got := mapXAIFinishReason(tc.in); got != tc.want {
				t.Fatalf("mapXAIFinishReason(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func newTestXAIResponse(tb testing.TB, proto *xaipb.GetChatCompletionResponse) *xai.Response {
	tb.Helper()
	resp := &xai.Response{}
	v := reflect.ValueOf(resp).Elem().FieldByName("proto")
	reflect.NewAt(v.Type(), unsafe.Pointer(v.UnsafeAddr())).Elem().Set(reflect.ValueOf(proto))
	return resp
}

func ptrFloat32(v float32) *float32 {
	return &v
}

func ptrInt32(v int32) *int32 {
	return &v
}

func derefInt32(v *int32) int32 {
	if v == nil {
		return 0
	}
	return *v
}

func derefFloat32(v *float32) float32 {
	if v == nil {
		return 0
	}
	return *v
}
