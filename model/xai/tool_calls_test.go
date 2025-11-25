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
	"testing"

	"github.com/bytedance/sonic"
	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

func TestToolCallArguments(t *testing.T) {
	args := map[string]any{"x": 1, "y": "z"}
	bytes, err := sonic.ConfigFastest.Marshal(args)
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
