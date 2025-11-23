package xai

import (
	"testing"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

func TestResponseProcessChunk(t *testing.T) {
	idx := 0
	resp := newResponse(&xaipb.GetChatCompletionResponse{}, &idx)
	chunk := &xaipb.GetChatCompletionChunk{
		Outputs: []*xaipb.CompletionOutputChunk{
			{
				Index: 0,
				Delta: &xaipb.Delta{
					Role:             xaipb.MessageRole_ROLE_ASSISTANT,
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
