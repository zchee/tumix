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

	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
)

func TestSearchParametersProto(t *testing.T) {
	from := time.Date(2024, 1, 2, 3, 4, 5, 0, time.UTC)
	params := SearchParameters{
		Sources: []*xaipb.Source{
			WebSource("US", []string{"example.com"}, nil, true),
		},
		Mode:             SearchModeOn,
		FromDate:         &from,
		ReturnCitations:  true,
		MaxSearchResults: 7,
	}
	proto := params.Proto()
	if proto.GetMode() != xaipb.SearchMode_ON_SEARCH_MODE {
		t.Fatalf("mode not converted")
	}
	if proto.GetFromDate() == nil || !proto.GetFromDate().AsTime().Equal(from) {
		t.Fatalf("from date mismatch")
	}
	if proto.GetMaxSearchResults() != 7 {
		t.Fatalf("max search results mismatch")
	}
	if len(proto.GetSources()) != 1 || proto.GetSources()[0].GetWeb() == nil {
		t.Fatalf("web source missing")
	}
}

func TestDocumentsSource(t *testing.T) {
	ids := []string{"c1", "c2"}
	s := DocumentsSource(ids)
	if got := s.GetCollectionIds(); len(got) != 2 || got[0] != "c1" || got[1] != "c2" {
		t.Fatalf("collection ids not set: %+v", got)
	}
}

func TestWebAndNewsSources(t *testing.T) {
	ws := WebSource("US", []string{"ex"}, []string{"allow"}, true)
	if web := ws.GetWeb(); web == nil || web.GetCountry() != "US" || !web.GetSafeSearch() {
		t.Fatalf("web source not populated: %+v", web)
	}

	wsNoCountry := WebSource("", nil, nil, false)
	if country := wsNoCountry.GetWeb().GetCountry(); country != "" {
		t.Fatalf("expected empty country when unset, got %q", country)
	}

	news := NewsSource("GB", []string{"ex"}, false)
	if n := news.GetNews(); n == nil || n.GetCountry() != "GB" || n.GetExcludedWebsites()[0] != "ex" || n.GetSafeSearch() {
		t.Fatalf("news source not populated: %+v", n)
	}
}

func TestXAndRSSSources(t *testing.T) {
	x := XSource([]string{"inc"}, []string{"exc"}, 5, 0)
	xpb := x.GetX()
	if xpb == nil || xpb.GetIncludedXHandles()[0] != "inc" || xpb.GetExcludedXHandles()[0] != "exc" {
		t.Fatalf("x source handles not set: %+v", xpb)
	}
	if xpb.GetPostFavoriteCount() != 5 || xpb.GetPostViewCount() != 0 {
		t.Fatalf("x source counters unexpected: %+v", xpb)
	}

	xZero := XSource(nil, nil, 0, 0)
	if xZero.GetX().GetPostFavoriteCount() != 0 || xZero.GetX().GetPostViewCount() != 0 {
		t.Fatalf("zero counters should yield zero values")
	}

	rss := RSSSource([]string{"link1", "link2"})
	if len(rss.GetRss().GetLinks()) != 2 {
		t.Fatalf("rss links not set: %+v", rss.GetRss().GetLinks())
	}
}

func TestSearchModeToProto(t *testing.T) {
	tests := []struct {
		mode     SearchMode
		expected xaipb.SearchMode
	}{
		{SearchModeOn, xaipb.SearchMode_ON_SEARCH_MODE},
		{SearchModeOff, xaipb.SearchMode_OFF_SEARCH_MODE},
		{"", xaipb.SearchMode_AUTO_SEARCH_MODE},
	}

	for _, tt := range tests {
		if got := searchModeToProto(tt.mode); got != tt.expected {
			t.Fatalf("mode %q -> %v, want %v", tt.mode, got, tt.expected)
		}
	}
}
