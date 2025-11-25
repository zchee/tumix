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
	"strconv"
	"strings"
	"testing"

	"iter"
	"slices"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

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
		for i := 0; i < n; i++ {
			if !yield(i) {
				return
			}
		}
	}
}
