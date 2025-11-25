package xai

import (
	"testing"
)

func TestPtrAndDeref(t *testing.T) {
	v := 42
	if got := ptr(v); got == nil || *got != v {
		t.Fatalf("ptr(%d) = %v", v, got)
	}

	if got := deref(ptr("ok")); got != "ok" {
		t.Fatalf("deref pointer returned %q", got)
	}
	if got := deref[string](nil); got != "" {
		t.Fatalf("deref nil returned %q", got)
	}
}
