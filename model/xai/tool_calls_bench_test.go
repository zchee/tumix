package xai

import (
	"testing"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

func BenchmarkToolCallArguments(b *testing.B) {
	tc := &xaipb.ToolCall{
		Tool: &xaipb.ToolCall_Function{
			Function: &xaipb.FunctionCall{
				Name:      "lookup",
				Arguments: `{"a":1,"b":"two","c":[1,2,3],"d":{"nested":true}}`,
			},
		},
	}

	b.ReportAllocs()
	for b.Loop() {
		var dst struct {
			A int             `json:"a"`
			B string          `json:"b"`
			C []int           `json:"c"`
			D map[string]bool `json:"d"`
		}
		if err := ToolCallArguments(tc, &dst); err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
	}
}
