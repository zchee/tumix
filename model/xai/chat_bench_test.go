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
	for i := 0; i < b.N; i++ {
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
	for i := 0; i < b.N; i++ {
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
