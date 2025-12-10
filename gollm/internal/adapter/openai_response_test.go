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
	"strings"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/genai"
)

func BenchmarkParseArgs(b *testing.B) {
	json := `{"a":1,"b":{"c":"d","e":[1,2,3]},"long":"` + longJSONFragment + `"}`
	b.ReportAllocs()

	for b.Loop() {
		parseArgs(json)
	}
}

const longJSONFragment = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

func TestTrimAtStop(t *testing.T) {
	t.Parallel()

	if res := trimAtStop("hello", nil); res.text != "hello" || res.hit {
		t.Fatalf("no stop: %+v", res)
	}

	res := trimAtStop("foo!bar", []string{"!"})
	if res.text != "foo" || !res.hit {
		t.Fatalf("stop hit %+v", res)
	}
}

func TestTrimPartsAtStop(t *testing.T) {
	t.Parallel()

	parts := []*genai.Part{
		genai.NewPartFromText("abc!def"),
		genai.NewPartFromText("ghi"),
	}
	if !trimPartsAtStop(parts, []string{"!"}) {
		t.Fatalf("expected trim")
	}
	if parts[0].Text != "abc" || parts[1].Text != "" {
		t.Fatalf("trim result = %+v", parts)
	}
}

func TestParseArgsFallback(t *testing.T) {
	t.Parallel()

	if got := parseArgs(" { \"x\": 1 } "); got["x"] != float64(1) {
		t.Fatalf("parsed json %+v", got)
	}
	if got := parseArgs("not json"); got["raw"] != "not json" {
		t.Fatalf("fallback %+v", got)
	}
	if got := parseArgs(" "); len(got) != 0 {
		t.Fatalf("empty -> %#v", got)
	}
}

func TestResponseOutputUnionToAny(t *testing.T) {
	t.Parallel()

	arr := []responses.ResponseFunctionShellToolCallOutputOutput{{}}
	if got := responseOutputUnionToAny(&responses.ResponseOutputItemUnionOutput{
		OfResponseFunctionShellToolCallOutputOutputArray: arr,
	}); got == nil {
		t.Fatalf("array nil")
	}

	if got := responseOutputUnionToAny(&responses.ResponseOutputItemUnionOutput{
		OfString: "str",
	}); got != "str" {
		t.Fatalf("string got %v", got)
	}
}

func TestMapResponseFinishReason(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		status  responses.ResponseStatus
		details responses.ResponseIncompleteDetails
		want    genai.FinishReason
	}{
		"completed": {
			status:  responses.ResponseStatusCompleted,
			details: responses.ResponseIncompleteDetails{},
			want:    genai.FinishReasonStop,
		},
		"max": {
			status: responses.ResponseStatusIncomplete,
			details: responses.ResponseIncompleteDetails{
				Reason: "max_output_tokens",
			},
			want: genai.FinishReasonMaxTokens,
		},
		"filter": {
			status: responses.ResponseStatusIncomplete,
			details: responses.ResponseIncompleteDetails{
				Reason: "content_filter",
			},
			want: genai.FinishReasonSafety,
		},
		"failed": {
			status:  responses.ResponseStatusFailed,
			details: responses.ResponseIncompleteDetails{},
			want:    genai.FinishReasonOther,
		},
		"default": {
			status:  responses.ResponseStatus("unknown"),
			details: responses.ResponseIncompleteDetails{},
			want:    genai.FinishReasonUnspecified,
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			if got := mapResponseFinishReason(tt.status, tt.details); got != tt.want {
				t.Fatalf("mapResponseFinishReason() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestOpenAIStreamAggregator_FinalFromAccumulated(t *testing.T) {
	t.Parallel()

	agg := NewOpenAIStreamAggregator([]string{"!"})

	// partial text
	partials := agg.Process(&responses.ResponseStreamEventUnion{
		Type:        "response.output_text.delta",
		Delta:       "Hello!",
		OutputIndex: 0,
	})
	if len(partials) != 1 || !partials[0].Partial || partials[0].Content.Parts[0].Text != "Hello!" {
		t.Fatalf("partials = %+v", partials)
	}

	// tool call accumulation
	agg.Process(&responses.ResponseStreamEventUnion{
		Type:        "response.function_call_arguments.delta",
		OutputIndex: 0,
		ItemID:      "tool-1",
		Delta:       `{"x":`,
	})
	agg.Process(&responses.ResponseStreamEventUnion{
		Type:        "response.function_call_arguments.done",
		OutputIndex: 0,
		ItemID:      "tool-1",
		Name:        "lookup",
		Arguments:   `{"x":1}`,
	})

	final := agg.Final()
	if final == nil {
		t.Fatalf("Final() nil")
	}
	if got, want := final.Content.Parts[0].Text, "Hello"; got != want {
		t.Fatalf("text = %q, want %q", got, want)
	}
	if len(final.Content.Parts) != 2 || final.Content.Parts[1].FunctionCall == nil {
		t.Fatalf("function call missing: %+v", final.Content.Parts)
	}
	if final.FinishReason != genai.FinishReasonStop {
		t.Fatalf("finish = %v, want stop", final.FinishReason)
	}
}

func TestOpenAIStreamAggregator_CompletedEvent(t *testing.T) {
	t.Parallel()

	agg := NewOpenAIStreamAggregator(nil)
	resp := responses.Response{
		Status: responses.ResponseStatusCompleted,
		Output: []responses.ResponseOutputItemUnion{{
			Type: "message",
			Role: constant.ValueOf[constant.Assistant](),
			Content: []responses.ResponseOutputMessageContentUnion{
				{
					Type: "output_text",
					Text: "done",
				},
			},
		}},
		Usage: responses.ResponseUsage{
			InputTokens:  1,
			OutputTokens: 2,
			TotalTokens:  3,
		},
	}

	got := agg.Process(&responses.ResponseStreamEventUnion{
		Type:     "response.completed",
		Response: resp,
	})
	if len(got) != 0 {
		t.Fatalf("expected aggregator to retain final until Final(), got %d immediate responses", len(got))
	}
	if agg.Err() != nil {
		t.Fatalf("unexpected err: %v", agg.Err())
	}
	final := agg.Final()
	if final == nil {
		t.Fatalf("Final not set from completed")
	}
	if final.Content == nil || len(final.Content.Parts) == 0 || final.Content.Parts[0].Text != "done" {
		t.Fatalf("final content mismatch: %+v", final)
	}
	if !final.TurnComplete {
		t.Fatalf("final TurnComplete false: %+v", final)
	}
}

func TestOpenAIStreamAggregator_ErrorEvent(t *testing.T) {
	t.Parallel()

	agg := NewOpenAIStreamAggregator(nil)
	agg.Process(&responses.ResponseStreamEventUnion{
		Type:    "response.failed",
		Message: "boom",
	})
	if agg.Err() == nil {
		t.Fatalf("expected error recorded")
	}
	if final := agg.Final(); final != nil {
		t.Fatalf("expected nil final when failed, got %+v", final)
	}
}

func TestOpenAIResponseToLLMError(t *testing.T) {
	t.Parallel()

	if _, err := OpenAIResponseToLLM(nil, nil); err == nil {
		t.Fatalf("expected error on nil response")
	}

	if _, err := OpenAIResponseToLLM(&responses.Response{}, nil); err == nil || !strings.Contains(err.Error(), "empty output") {
		t.Fatalf("expected empty output error, got %v", err)
	}
}
