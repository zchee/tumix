package xai

import (
	"testing"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

func TestFileOptions(t *testing.T) {
	cfg := &fileUploadConfig{}

	WithFilename("foo.txt")(cfg)
	if cfg.filename != "foo.txt" {
		t.Fatalf("filename not set: %s", cfg.filename)
	}

	called := false
	WithProgress(func(uploaded, total int64) { called = true })(cfg)
	if cfg.progress == nil {
		t.Fatalf("progress func nil")
	}
	cfg.progress(1, 2)
	if !called {
		t.Fatalf("progress func not invoked")
	}
}

func TestFileSortAndOrderToProto(t *testing.T) {
	if got := fileSortByToProto(FileSortByFilename); got == nil || *got != xaipb.FilesSortBy_FILES_SORT_BY_FILENAME {
		t.Fatalf("filename sort mismatch: %v", got)
	}
	if got := fileSortByToProto(FileSortBySize); got == nil || *got != xaipb.FilesSortBy_FILES_SORT_BY_SIZE {
		t.Fatalf("size sort mismatch: %v", got)
	}
	if got := fileSortByToProto("" /* default */); got == nil || *got != xaipb.FilesSortBy_FILES_SORT_BY_CREATED_AT {
		t.Fatalf("default sort mismatch: %v", got)
	}

	if got := orderToProto(OrderAscending); got != xaipb.Ordering_ASCENDING {
		t.Fatalf("ascending order mismatch: %v", got)
	}
	if got := orderToProto(OrderDescending); got != xaipb.Ordering_DESCENDING {
		t.Fatalf("descending order mismatch: %v", got)
	}
	if got := orderToProto(Order("")); got != xaipb.Ordering(0) {
		t.Fatalf("default order mismatch: %v", got)
	}
}
