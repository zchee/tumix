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

package xai

import (
	"iter"
	"slices"
	"strconv"
	"strings"
	"testing"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

func (r *Response) reset() {
	r.proto.Id = ""
	r.proto.Model = ""
	r.proto.Created = nil
	r.proto.SystemFingerprint = ""
	r.proto.Usage = nil
	r.proto.Citations = r.proto.GetCitations()[:0]
	for _, out := range r.proto.GetOutputs() {
		if out == nil {
			continue
		}
		if msg := out.GetMessage(); msg != nil {
			msg.Content = ""
			msg.ReasoningContent = ""
			msg.EncryptedContent = ""
			msg.ToolCalls = msg.GetToolCalls()[:0]
			msg.Role = 0
		}
		out.FinishReason = 0
	}
	r.index = nil
	r.contentBuffers = nil
	r.reasoningBuffers = nil
	r.encryptedBuffers = nil
	r.buffersAreInProto = true
}

func BenchmarkResponseProcessChunkSingle(b *testing.B) {
	chunk := &xaipb.GetChatCompletionChunk{
		Outputs: []*xaipb.CompletionOutputChunk{metadataChunk(0)},
	}

	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			resp := newResponse(&xaipb.GetChatCompletionResponse{}, nil)
			resp.processChunk(chunk)
			_ = resp.Content()
		}
	})
}

func BenchmarkResponseProcessChunkMulti(b *testing.B) {
	chunk := &xaipb.GetChatCompletionChunk{
		Outputs: []*xaipb.CompletionOutputChunk{
			metadataChunk(0),
			metadataChunk(3),
			metadataChunk(5),
		},
	}

	b.ReportAllocs()
	for b.Loop() {
		resp := newResponse(&xaipb.GetChatCompletionResponse{}, nil)
		resp.processChunk(chunk)
		_ = resp.Content()
	}
}

func BenchmarkResponseProcessChunkReuse(b *testing.B) {
	chunk := &xaipb.GetChatCompletionChunk{
		Outputs: []*xaipb.CompletionOutputChunk{
			metadataChunk(0),
			metadataChunk(1),
		},
	}
	resp := newResponse(&xaipb.GetChatCompletionResponse{}, nil)

	b.ReportAllocs()
	for b.Loop() {
		resp.processChunk(chunk)
		_ = resp.Content()
		resp.reset()
	}
}

func BenchmarkResponseProcessChunkHeavy(b *testing.B) {
	chunk := heavyChunk(12, 3, strings.Repeat("delta-", 8))

	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			resp := newResponse(&xaipb.GetChatCompletionResponse{}, nil)
			resp.processChunk(chunk)
			_ = resp.Content()
		}
	})
}

func BenchmarkChunkAccessors(b *testing.B) {
	chunk := heavyChunk(10, 2, strings.Repeat("content-", 6))
	wrapped := newChunk(chunk, nil)

	b.ReportAllocs()
	for b.Loop() {
		_ = wrapped.Content()
		_ = wrapped.ReasoningContent()
		_ = len(wrapped.ToolCalls())
	}
}

func BenchmarkSpanAttributes(b *testing.B) {
	s := &ChatSession{
		request: &xaipb.GetCompletionsRequest{
			Model: "grok-4-1",
			Messages: []*xaipb.Message{
				User("hello"),
				Assistant("world"),
			},
			FrequencyPenalty:    ptr(float32(0.2)),
			PresencePenalty:     ptr(float32(0.1)),
			Temperature:         ptr(float32(0.8)),
			TopP:                ptr(float32(0.9)),
			ParallelToolCalls:   ptr(true),
			StoreMessages:       true,
			UseEncryptedContent: true,
			Logprobs:            true,
			Stop:                []string{"STOP"},
			ResponseFormat: &xaipb.ResponseFormat{
				FormatType: xaipb.FormatType_FORMAT_TYPE_JSON_OBJECT,
			},
			MaxTokens:          ptr(int32(1024)),
			TopLogprobs:        ptr(int32(4)),
			N:                  ptr(int32(2)),
			Seed:               ptr(int32(1234)),
			PreviousResponseId: ptr("prev-id"),
		},
		conversationID: "conv-id",
	}

	b.ReportAllocs()
	for b.Loop() {
		_ = s.makeSpanRequestAttributes()
	}
}

func BenchmarkSpanAttributesManyMessages(b *testing.B) {
	msgs := make([]*xaipb.Message, 0, 32)
	for i := range rangeN(32) {
		msgs = append(msgs, &xaipb.Message{
			Role: xaipb.MessageRole_ROLE_USER,
			Content: []*xaipb.Content{
				TextContent("msg-" + strconv.Itoa(i)),
			},
		})
	}

	s := &ChatSession{
		request: &xaipb.GetCompletionsRequest{
			Model:    "grok-4-1",
			Messages: msgs,
		},
	}

	b.ReportAllocs()
	for b.Loop() {
		_ = s.makeSpanRequestAttributes()
	}
}

func BenchmarkSpanResponseAttributes(b *testing.B) {
	respProto := &xaipb.GetChatCompletionResponse{
		Id:                "resp-1",
		Model:             "grok-4-1",
		SystemFingerprint: "fp",
		Outputs: []*xaipb.CompletionOutput{
			{
				Index: 0,
				Message: &xaipb.CompletionMessage{
					Role:             xaipb.MessageRole_ROLE_ASSISTANT,
					Content:          strings.Repeat("answer", 3),
					ReasoningContent: strings.Repeat("why", 2),
					ToolCalls: []*xaipb.ToolCall{
						{Tool: &xaipb.ToolCall_Function{Function: &xaipb.FunctionCall{Name: "fn", Arguments: `{"a":1}`}}},
					},
				},
				FinishReason: xaipb.FinishReason_REASON_STOP,
			},
			{
				Index: 1,
				Message: &xaipb.CompletionMessage{
					Role:    xaipb.MessageRole_ROLE_ASSISTANT,
					Content: strings.Repeat("extra", 2),
				},
				FinishReason: xaipb.FinishReason_REASON_MAX_LEN,
			},
		},
		Usage: &xaipb.SamplingUsage{
			PromptTokens:     123,
			CompletionTokens: 456,
			TotalTokens:      579,
			ReasoningTokens:  42,
		},
	}
	responses := []*Response{
		newResponse(respProto, nil),
	}

	s := &ChatSession{}

	b.ReportAllocs()
	for b.Loop() {
		_ = s.makeSpanResponseAttributes(responses)
	}
}

func BenchmarkChatSessionStreamAggregate(b *testing.B) {
	chunks := streamingChunks(8, 3, "chunk-")
	resp := newResponse(&xaipb.GetChatCompletionResponse{}, nil)

	b.ReportAllocs()
	for b.Loop() {
		for _, ch := range chunks {
			resp.processChunk(ch)
		}
		_ = resp.Content()
		_ = resp.ToolCalls()
		resp.reset()
	}
}

func BenchmarkResponseToolCalls(b *testing.B) {
	const (
		outputs   = 6
		toolCalls = 10
	)

	buildToolCalls := func(idx int) []*xaipb.ToolCall {
		calls := make([]*xaipb.ToolCall, 0, toolCalls)
		for j := range rangeN(toolCalls) {
			calls = append(calls, &xaipb.ToolCall{
				Id: "call-" + strconv.Itoa(idx*toolCalls+j),
				Tool: &xaipb.ToolCall_Function{
					Function: &xaipb.FunctionCall{
						Name:      "fn",
						Arguments: `{"a":1,"b":2}`,
					},
				},
			})
		}
		return calls
	}

	outs := make([]*xaipb.CompletionOutput, 0, outputs)
	for i := range rangeN(outputs) {
		outs = append(outs, &xaipb.CompletionOutput{
			Index: int32(i),
			Message: &xaipb.CompletionMessage{
				Role:      xaipb.MessageRole_ROLE_ASSISTANT,
				ToolCalls: buildToolCalls(i),
			},
		})
	}
	resp := newResponse(&xaipb.GetChatCompletionResponse{Outputs: outs}, nil)

	b.ReportAllocs()
	for b.Loop() {
		if got := len(resp.ToolCalls()); got == 0 {
			b.Fatalf("unexpected zero tool calls")
		}
	}
}

func streamingChunks(nChunks, outputs int, contentPrefix string) []*xaipb.GetChatCompletionChunk {
	chunks := make([]*xaipb.GetChatCompletionChunk, 0, nChunks)
	for i := range rangeN(nChunks) {
		outs := make([]*xaipb.CompletionOutputChunk, 0, outputs)
		for j := range rangeN(outputs) {
			outs = append(outs, &xaipb.CompletionOutputChunk{
				Index: int32(j),
				Delta: &xaipb.Delta{
					Role:             xaipb.MessageRole_ROLE_ASSISTANT,
					Content:          contentPrefix + strconv.Itoa(i) + "-" + strconv.Itoa(j),
					ReasoningContent: "r-" + strconv.Itoa(j),
					ToolCalls: []*xaipb.ToolCall{
						{Tool: &xaipb.ToolCall_Function{Function: &xaipb.FunctionCall{Name: "fn", Arguments: `{"x":1}`}}},
					},
				},
				FinishReason: xaipb.FinishReason_REASON_STOP,
			})
		}
		chunks = append(chunks, &xaipb.GetChatCompletionChunk{Outputs: outs})
	}

	return chunks
}

func metadataChunk(idx int32) *xaipb.CompletionOutputChunk {
	return &xaipb.CompletionOutputChunk{
		Index: idx,
		Delta: &xaipb.Delta{
			Role:    xaipb.MessageRole_ROLE_ASSISTANT,
			Content: "hello",
		},
	}
}

func heavyChunk(outputs, toolCalls int, content string) *xaipb.GetChatCompletionChunk {
	chunks := slices.Grow(make([]*xaipb.CompletionOutputChunk, 0, outputs), outputs)
	for i := range rangeN(outputs) {
		calls := make([]*xaipb.ToolCall, 0, toolCalls)
		for j := range rangeN(toolCalls) {
			calls = append(calls, &xaipb.ToolCall{
				Id: "call-" + strconv.Itoa(i*toolCalls+j),
				Tool: &xaipb.ToolCall_Function{
					Function: &xaipb.FunctionCall{
						Name:      "fn",
						Arguments: `{"foo": "bar"}`,
					},
				},
			})
		}

		chunks = append(chunks, &xaipb.CompletionOutputChunk{
			Index: int32(i),
			Delta: &xaipb.Delta{
				Role:             xaipb.MessageRole_ROLE_ASSISTANT,
				Content:          content,
				ReasoningContent: content,
				EncryptedContent: content,
				ToolCalls:        calls,
			},
		})
	}

	return &xaipb.GetChatCompletionChunk{Outputs: chunks}
}

func rangeN(n int) iter.Seq[int] {
	return func(yield func(int) bool) {
		for i := range n {
			if !yield(i) {
				return
			}
		}
	}
}
