package xai

import (
	"testing"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

func TestChunkConfigBuilders(t *testing.T) {
	chars := ChunkConfigChars(100, 10, true, false)
	if got := chars.GetCharsConfiguration().GetMaxChunkSizeChars(); got != 100 {
		t.Fatalf("chars max size = %d", got)
	}
	if !chars.GetStripWhitespace() {
		t.Fatalf("expected strip_whitespace true")
	}

	tokens := ChunkConfigTokens(256, 32, "cl100k", false, true)
	if tokens.GetTokensConfiguration().GetEncodingName() != "cl100k" {
		t.Fatalf("encoding not set")
	}
	if !tokens.GetInjectNameIntoChunks() {
		t.Fatalf("injectNameIntoChunks not set")
	}

	idx := IndexConfig("grok-embed")
	if idx.GetModelName() != "grok-embed" {
		t.Fatalf("index model mismatch")
	}
}

func TestDocumentSearchOptions(t *testing.T) {
	metric := xaipb.RankingMetric_RANKING_METRIC_COSINE_SIMILARITY
	opt := applyDocumentSearchOptions([]DocumentSearchOption{WithSearchLimit(5), WithRankingMetric(metric)})

	if opt.limit == nil || *opt.limit != 5 {
		t.Fatalf("limit not set")
	}
	if opt.rankingMetric == nil || *opt.rankingMetric != metric {
		t.Fatalf("ranking metric not set")
	}
}
