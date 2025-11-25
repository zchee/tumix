package xai

import (
	json "encoding/json/v2"
	"testing"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

func TestToolCallArguments(t *testing.T) {
	args := map[string]any{"x": 1, "y": "z"}
	bytes, err := json.Marshal(args)
	if err != nil {
		t.Fatal(err)
	}
	tc := &xaipb.ToolCall{
		Tool: &xaipb.ToolCall_Function{
			Function: &xaipb.FunctionCall{
				Arguments: string(bytes),
			},
		},
	}
	var out map[string]any
	if err := ToolCallArguments(tc, &out); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out["y"] != "z" {
		t.Fatalf("unexpected value: %+v", out)
	}
}

func TestToolCallArgumentsMissing(t *testing.T) {
	if err := ToolCallArguments(&xaipb.ToolCall{}, nil); err == nil {
		t.Fatalf("expected error for missing function")
	}
}
