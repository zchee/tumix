package xai

import (
	"testing"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

func TestResponseDecodeJSON(t *testing.T) {
	resp := newResponse(&xaipb.GetChatCompletionResponse{
		Outputs: []*xaipb.CompletionOutput{{
			Message: &xaipb.CompletionMessage{Role: xaipb.MessageRole_ROLE_ASSISTANT, Content: `{"foo":123}`},
		}},
	}, nil)
	var out struct{ Foo int `json:"foo"` }
	if err := resp.DecodeJSON(&out); err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	if out.Foo != 123 {
		t.Fatalf("unexpected value: %+v", out)
	}
}

