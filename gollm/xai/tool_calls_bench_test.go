// Copyright 2025 The tumix Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
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

	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
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
