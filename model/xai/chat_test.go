package xai

import (
	"testing"

	pb "github.com/zchee/tumix/model/xai/pb/xai/api/v1"
)

func TestResponseProcessChunk(t *testing.T) {
	idx := 0
	resp := newResponse(&pb.GetChatCompletionResponse{}, &idx)
	chunk := &pb.GetChatCompletionChunk{
		Outputs: []*pb.CompletionOutputChunk{
			{
				Index: 0,
				Delta: &pb.Delta{
					Role:             pb.MessageRole_ROLE_ASSISTANT,
					Content:          "Hello",
					ReasoningContent: "Why",
				},
			},
		},
	}

	resp.processChunk(chunk)

	if got := resp.Content(); got != "Hello" {
		t.Fatalf("content mismatch: %q", got)
	}
	if got := resp.ReasoningContent(); got != "Why" {
		t.Fatalf("reasoning mismatch: %q", got)
	}
}
