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
	"time"
)

func TestToolFunction(t *testing.T) {
	tool, err := Tool("foo", "bar", map[string]string{"a": "b"})
	if err != nil {
		t.Fatalf("Tool returned error: %v", err)
	}
	fn := tool.GetFunction()
	if fn == nil || fn.GetName() != "foo" || fn.GetDescription() != "bar" {
		t.Fatalf("unexpected function tool: %+v", fn)
	}
	if fn.GetParameters() == "" {
		t.Fatalf("expected parameters to be serialized")
	}
}

func TestMCPTool(t *testing.T) {
	auth := "token"
	tool := MCPTool("https://server", "label", "desc", []string{"a"}, auth, map[string]string{"h": "v"})
	mcp := tool.GetMcp()
	if mcp == nil {
		t.Fatalf("MCP tool not set")
	}
	if mcp.GetAuthorization() != auth {
		t.Fatalf("authorization not propagated")
	}
}

func TestCollectionsSearchTool(t *testing.T) {
	limit := int32(5)
	tool := CollectionsSearchTool([]string{"c1", "c2"}, limit)
	cs := tool.GetCollectionsSearch()
	if cs == nil || len(cs.GetCollectionIds()) != 2 {
		t.Fatalf("collections search not populated")
	}
	if cs.GetLimit() != limit {
		t.Fatalf("limit not set")
	}
}

func TestXSearchToolDates(t *testing.T) {
	from := time.Unix(0, 0).UTC()
	tool := XSearchTool(&from, nil, nil, nil, false, false)
	if tool.GetXSearch().GetFromDate().AsTime() != from {
		t.Fatalf("from date mismatch")
	}
}

// Ensure ToolChoice RequiredTool helper sets required mode.
func TestRequiredTool(t *testing.T) {
	choice := RequiredTool("foo")
	if choice.GetFunctionName() != "foo" {
		t.Fatalf("expected function name foo, got %s", choice.GetFunctionName())
	}
}
