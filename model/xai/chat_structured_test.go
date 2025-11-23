package xai

import (
	"context"
	"testing"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

type demoStruct struct {
	Name string `json:"name"`
}

func TestWithJSONStruct(t *testing.T) {
	s := chatSessionForTest()
	WithJSONStruct[*demoStruct]()(s.request, nil)
	if s.request.ResponseFormat == nil || s.request.ResponseFormat.GetFormatType() != xaipb.FormatType_FORMAT_TYPE_JSON_SCHEMA {
		t.Fatalf("response format not set")
	}
}

// chatSessionForTest builds a minimal ChatSession with a dummy request.
func chatSessionForTest() *ChatSession {
	return &ChatSession{
		request: &xaipb.GetCompletionsRequest{Model: "grok"},
	}
}

func TestParseIntoCompilation(t *testing.T) {
	// compile-time presence check; do not invoke network
	_ = ParseInto[*demoStruct]
	_ = context.Background()
}
