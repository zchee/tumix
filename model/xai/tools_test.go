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
	if fn == nil || fn.Name != "foo" || fn.Description != "bar" {
		t.Fatalf("unexpected function tool: %+v", fn)
	}
	if fn.Parameters == "" {
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
	if cs == nil || len(cs.CollectionIds) != 2 {
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
