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

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

func TestUsesServerSideTools(t *testing.T) {
	fnTool := &xaipb.Tool{
		Tool: &xaipb.Tool_Function{
			Function: &xaipb.Function{},
		},
	}
	if usesServerSideTools([]*xaipb.Tool{fnTool}) {
		t.Fatalf("function-only tools should not be treated as server-side")
	}

	webTool := &xaipb.Tool{
		Tool: &xaipb.Tool_WebSearch{
			WebSearch: &xaipb.WebSearch{},
		},
	}
	if !usesServerSideTools([]*xaipb.Tool{fnTool, webTool}) {
		t.Fatalf("non-function tools should trigger server-side detection")
	}

	if usesServerSideTools(nil) {
		t.Fatalf("nil tool list should be false")
	}
}

func TestAutoDetectMultiOutput(t *testing.T) {
	idx := int32(0)
	out := autoDetectMultiOutput(&idx, []*xaipb.CompletionOutput{
		{
			Index: 0,
		},
	})
	if out == nil || *out != 0 {
		t.Fatalf("expected index preserved, got %v", out)
	}

	out = autoDetectMultiOutput(&idx, []*xaipb.CompletionOutput{
		{
			Index: 1,
		},
	})
	if out != nil {
		t.Fatalf("expected nil index when multiple outputs present")
	}
}

func TestAutoDetectMultiOutputChunks(t *testing.T) {
	idx := int32(0)
	out := autoDetectMultiOutputChunks(&idx, []*xaipb.CompletionOutputChunk{
		{
			Index: 0,
		},
	})
	if out == nil || *out != 0 {
		t.Fatalf("expected chunk index preserved, got %v", out)
	}

	out = autoDetectMultiOutputChunks(&idx, []*xaipb.CompletionOutputChunk{
		{
			Index: 2,
		},
	})
	if out != nil {
		t.Fatalf("expected nil chunk index when larger index observed")
	}
}
