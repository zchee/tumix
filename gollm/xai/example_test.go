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
	"fmt"

	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
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
