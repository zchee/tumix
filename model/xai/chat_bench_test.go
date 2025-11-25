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
	"testing"

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
		resp.contentBuffers = resp.contentBuffers[:0]
		resp.reasoningBuffers = resp.reasoningBuffers[:0]
		resp.encryptedBuffers = resp.encryptedBuffers[:0]
		resp.buffersAreInProto = true
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
