package xai

import (
	"testing"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

func TestUsesServerSideTools(t *testing.T) {
	fnTool := &xaipb.Tool{Tool: &xaipb.Tool_Function{Function: &xaipb.Function{}}}
	if usesServerSideTools([]*xaipb.Tool{fnTool}) {
		t.Fatalf("function-only tools should not be treated as server-side")
	}

	webTool := &xaipb.Tool{Tool: &xaipb.Tool_WebSearch{WebSearch: &xaipb.WebSearch{}}}
	if !usesServerSideTools([]*xaipb.Tool{fnTool, webTool}) {
		t.Fatalf("non-function tools should trigger server-side detection")
	}

	if usesServerSideTools(nil) {
		t.Fatalf("nil tool list should be false")
	}
}

func TestAutoDetectMultiOutput(t *testing.T) {
	idx := 0
	out := autoDetectMultiOutput(&idx, []*xaipb.CompletionOutput{{Index: 0}})
	if out == nil || *out != 0 {
		t.Fatalf("expected index preserved, got %v", out)
	}

	out = autoDetectMultiOutput(&idx, []*xaipb.CompletionOutput{{Index: 1}})
	if out != nil {
		t.Fatalf("expected nil index when multiple outputs present")
	}
}

func TestAutoDetectMultiOutputChunks(t *testing.T) {
	idx := 0
	out := autoDetectMultiOutputChunks(&idx, []*xaipb.CompletionOutputChunk{{Index: 0}})
	if out == nil || *out != 0 {
		t.Fatalf("expected chunk index preserved, got %v", out)
	}

	out = autoDetectMultiOutputChunks(&idx, []*xaipb.CompletionOutputChunk{{Index: 2}})
	if out != nil {
		t.Fatalf("expected nil chunk index when larger index observed")
	}
}

func TestValueOrZeroHelpers(t *testing.T) {
	if got := valueOrZeroFloat32[float32](nil); got != 0 {
		t.Fatalf("nil float32 -> %v", got)
	}
	f := float32(1.5)
	if got := valueOrZeroFloat32(&f); got != 1.5 {
		t.Fatalf("float32 value lost: %v", got)
	}

	if got := valueOrZeroBool(nil); got {
		t.Fatalf("nil bool should be false")
	}
	b := true
	if got := valueOrZeroBool(&b); !got {
		t.Fatalf("bool value lost")
	}
}
