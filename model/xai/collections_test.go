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
