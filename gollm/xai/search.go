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
	"time"

	"google.golang.org/protobuf/types/known/timestamppb"

	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
)

// SearchMode controls when the model should perform search.
type SearchMode string

const (
	// SearchModeAuto lets the model decide whether to search.
	SearchModeAuto SearchMode = "auto"
	// SearchModeOn forces the model to search.
	SearchModeOn SearchMode = "on"
	// SearchModeOff disables search.
	SearchModeOff SearchMode = "off"
)

// SearchParameters mirrors the Python SDK configuration for search.
type SearchParameters struct {
	Sources          []*xaipb.Source
	Mode             SearchMode
	FromDate         *time.Time
	ToDate           *time.Time
	ReturnCitations  bool
	MaxSearchResults int32
}

// Proto converts the struct into the protobuf message.
func (p SearchParameters) Proto() *xaipb.SearchParameters {
	params := &xaipb.SearchParameters{
		Sources:         p.Sources,
		Mode:            searchModeToProto(p.Mode),
		ReturnCitations: p.ReturnCitations,
	}
	if p.FromDate != nil {
		params.FromDate = timestamppb.New(*p.FromDate)
	}
	if p.ToDate != nil {
		params.ToDate = timestamppb.New(*p.ToDate)
	}
	if p.MaxSearchResults > 0 {
		params.MaxSearchResults = &p.MaxSearchResults
	}
	return params
}

// WebSource builds a web search source.
func WebSource(country string, excludedWebsites, allowedWebsites []string, safeSearch bool) *xaipb.Source {
	var countryPtr *string
	if country != "" {
		countryPtr = &country
	}
	return &xaipb.Source{Source: &xaipb.Source_Web{Web: &xaipb.WebSource{
		Country:          countryPtr,
		ExcludedWebsites: excludedWebsites,
		AllowedWebsites:  allowedWebsites,
		SafeSearch:       safeSearch,
	}}}
}

// NewsSource builds a news search source.
func NewsSource(country string, excludedWebsites []string, safeSearch bool) *xaipb.Source {
	var countryPtr *string
	if country != "" {
		countryPtr = &country
	}
	return &xaipb.Source{
		Source: &xaipb.Source_News{
			News: &xaipb.NewsSource{
				Country:          countryPtr,
				ExcludedWebsites: excludedWebsites,
				SafeSearch:       safeSearch,
			},
		},
	}
}

// XSource builds an X (Twitter) search source.
func XSource(includedHandles, excludedHandles []string, favoriteCount, viewCount int32) *xaipb.Source {
	var favPtr, viewPtr *int32
	if favoriteCount > 0 {
		favPtr = &favoriteCount
	}
	if viewCount > 0 {
		viewPtr = &viewCount
	}
	return &xaipb.Source{
		Source: &xaipb.Source_X{
			X: &xaipb.XSource{
				IncludedXHandles:  includedHandles,
				ExcludedXHandles:  excludedHandles,
				PostFavoriteCount: favPtr,
				PostViewCount:     viewPtr,
			},
		},
	}
}

// RSSSource builds an RSS feed search source.
func RSSSource(links []string) *xaipb.Source {
	return &xaipb.Source{
		Source: &xaipb.Source_Rss{
			Rss: &xaipb.RssSource{
				Links: links,
			},
		},
	}
}

// DocumentsSource builds a documents source for semantic search over collections.
func DocumentsSource(collectionIDs []string) *xaipb.DocumentsSource {
	return &xaipb.DocumentsSource{
		CollectionIds: collectionIDs,
	}
}

func searchModeToProto(mode SearchMode) xaipb.SearchMode {
	switch mode {
	case SearchModeOn:
		return xaipb.SearchMode_ON_SEARCH_MODE
	case SearchModeOff:
		return xaipb.SearchMode_OFF_SEARCH_MODE
	default:
		return xaipb.SearchMode_AUTO_SEARCH_MODE
	}
}
