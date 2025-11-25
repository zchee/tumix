package xai

import (
	"testing"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

func TestWithJSONSchema(t *testing.T) {
	schema := `{"type":"object"}`
	req := &xaipb.GetCompletionsRequest{}
	WithJSONSchema(schema)(req, nil)
	if req.GetResponseFormat() == nil || req.GetResponseFormat().GetFormatType() != xaipb.FormatType_FORMAT_TYPE_JSON_SCHEMA {
		t.Fatalf("response format not set to json schema")
	}
	if req.GetResponseFormat().GetSchema() != schema {
		t.Fatalf("schema not propagated")
	}
}
