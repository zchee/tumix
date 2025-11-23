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
	"context"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

// DocumentsClient provides access to the Documents service.
type DocumentsClient struct {
	stub xaipb.DocumentsClient
}

// Search performs semantic search across specified collections.
func (c *DocumentsClient) Search(ctx context.Context, query string, collectionIDs []string, limit int32) (*xaipb.SearchResponse, error) {
	req := &xaipb.SearchRequest{
		Query:  query,
		Source: &xaipb.DocumentsSource{CollectionIds: collectionIDs},
	}
	if limit > 0 {
		req.Limit = &limit
	}
	return c.stub.Search(ctx, req)
}
