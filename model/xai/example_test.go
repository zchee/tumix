package xai

import (
	"fmt"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

// ExampleWithJSONStruct shows how to request structured output without hitting the network.
func ExampleWithJSONStruct() {
	req := &xaipb.GetCompletionsRequest{
		Model: "grok-4",
	}
	WithJSONStruct[struct {
		Name string `json:"name"`
	}]()(req, nil)

	if req.GetResponseFormat() != nil && req.ResponseFormat.Schema != nil {
		fmt.Println("schema set")
	}
	// Output:
	// schema set
}

// ExampleToolCallArguments demonstrates decoding tool call arguments locally.
func ExampleToolCallArguments() {
	tc := &xaipb.ToolCall{
		Tool: &xaipb.ToolCall_Function{
			Function: &xaipb.FunctionCall{
				Name:      "sum",
				Arguments: `{"a":1,"b":2}`,
			},
		},
	}
	var args struct {
		A int `json:"a"`
		B int `json:"b"`
	}
	_ = ToolCallArguments(tc, &args)
	fmt.Printf("%d %d\n", args.A, args.B)
	// Output:
	// 1 2
}
