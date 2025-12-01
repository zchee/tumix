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
	"errors"
	"fmt"

	"google.golang.org/grpc"

	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
	collectionspb "github.com/zchee/tumix/gollm/xai/api/v1/collectionspb"
	ragpb "github.com/zchee/tumix/gollm/xai/api/v1/ragpb"
	sharedpb "github.com/zchee/tumix/gollm/xai/api/v1/sharedpb"
)

// Order defines sort order for collections/documents/files.
type Order string

const (
	OrderAscending  Order = "asc"
	OrderDescending Order = "desc"
)

// CollectionSortBy defines collection sort fields.
type CollectionSortBy string

const (
	CollectionSortByName CollectionSortBy = "name"
	CollectionSortByAge  CollectionSortBy = "age"
)

// DocumentSortBy defines document sort fields.
type DocumentSortBy string

const (
	DocumentSortByName DocumentSortBy = "name"
	DocumentSortByAge  DocumentSortBy = "age"
	DocumentSortBySize DocumentSortBy = "size"
)

// CollectionsClient provides gRPC access to collections/documents.
type CollectionsClient struct {
	collections collectionspb.CollectionsClient
	documents   xaipb.DocumentsClient
}

type collectionsOption func(*collectionsRequest)

type collectionsRequest struct {
	teamID *string
}

// DocumentSearchOption customizes document search requests.
type DocumentSearchOption func(*documentSearchRequest)

type documentSearchRequest struct {
	limit         *int32
	rankingMetric *xaipb.RankingMetric
}

// WithSearchLimit sets the maximum number of results returned.
func WithSearchLimit(limit int32) DocumentSearchOption {
	return func(r *documentSearchRequest) {
		if limit > 0 {
			r.limit = &limit
		}
	}
}

// WithRankingMetric sets the ranking metric for document search.
func WithRankingMetric(metric xaipb.RankingMetric) DocumentSearchOption {
	return func(r *documentSearchRequest) {
		if metric != xaipb.RankingMetric_RANKING_METRIC_UNKNOWN {
			r.rankingMetric = &metric
		}
	}
}

// WithTeamID sets an explicit team id on management requests.
func WithTeamID(teamID string) collectionsOption {
	return func(r *collectionsRequest) {
		if teamID != "" {
			r.teamID = &teamID
		}
	}
}

func applyCollectionsOptions(opts []collectionsOption) collectionsRequest {
	r := collectionsRequest{}
	for _, opt := range opts {
		opt(&r)
	}
	return r
}

func applyDocumentSearchOptions(opts []DocumentSearchOption) documentSearchRequest {
	r := documentSearchRequest{}
	for _, opt := range opts {
		opt(&r)
	}
	return r
}

// NewCollectionsClient builds a client. managementConn is required for collection mutations.
func NewCollectionsClient(apiConn, managementConn *grpc.ClientConn) *CollectionsClient {
	var collections collectionspb.CollectionsClient
	if managementConn != nil {
		collections = collectionspb.NewCollectionsClient(managementConn)
	}
	return &CollectionsClient{
		collections: collections,
		documents:   xaipb.NewDocumentsClient(apiConn),
	}
}

// IndexConfig builds an IndexConfiguration with the provided model name.
func IndexConfig(modelName string) *ragpb.IndexConfiguration {
	return &ragpb.IndexConfiguration{
		ModelName: modelName,
	}
}

// ChunkConfigChars builds a character-based chunk configuration.
func ChunkConfigChars(maxSize, overlap int32, stripWhitespace, injectName bool) *ragpb.ChunkConfiguration {
	return &ragpb.ChunkConfiguration{
		Config: &ragpb.ChunkConfiguration_CharsConfiguration{
			CharsConfiguration: &ragpb.CharsConfiguration{
				MaxChunkSizeChars: maxSize,
				ChunkOverlapChars: overlap,
			},
		},
		StripWhitespace:      stripWhitespace,
		InjectNameIntoChunks: injectName,
	}
}

// ChunkConfigTokens builds a token-based chunk configuration.
func ChunkConfigTokens(maxTokens, overlapTokens int32, encoding string, stripWhitespace, injectName bool) *ragpb.ChunkConfiguration {
	return &ragpb.ChunkConfiguration{
		Config: &ragpb.ChunkConfiguration_TokensConfiguration{
			TokensConfiguration: &ragpb.TokensConfiguration{
				MaxChunkSizeTokens: maxTokens,
				ChunkOverlapTokens: overlapTokens,
				EncodingName:       encoding,
			},
		},
		StripWhitespace:      stripWhitespace,
		InjectNameIntoChunks: injectName,
	}
}

func (c *CollectionsClient) requireCollectionsStub() error {
	if c.collections == nil {
		return errors.New("management API key required for collections operations")
	}
	return nil
}

// Create makes a new collection.
func (c *CollectionsClient) Create(ctx context.Context, name, modelName string, chunkCfg *ragpb.ChunkConfiguration, opts ...collectionsOption) (*collectionspb.CollectionMetadata, error) {
	if err := c.requireCollectionsStub(); err != nil {
		return nil, err
	}
	opt := applyCollectionsOptions(opts)
	req := &collectionspb.CreateCollectionRequest{
		CollectionName: name,
	}
	if modelName != "" {
		req.IndexConfiguration = &ragpb.IndexConfiguration{
			ModelName: modelName,
		}
	}
	if chunkCfg != nil {
		req.ChunkConfiguration = chunkCfg
	}
	if opt.teamID != nil {
		req.TeamId = opt.teamID
	}
	resp, err := c.collections.CreateCollection(ctx, req)
	return resp, WrapError(err)
}

// List collections with optional ordering and pagination.
func (c *CollectionsClient) List(ctx context.Context, limit int32, order Order, sortBy CollectionSortBy, paginationToken string, opts ...collectionsOption) (*collectionspb.ListCollectionsResponse, error) {
	if err := c.requireCollectionsStub(); err != nil {
		return nil, err
	}
	opt := applyCollectionsOptions(opts)
	req := &collectionspb.ListCollectionsRequest{}
	if limit > 0 {
		req.Limit = &limit
	}
	if ord := orderToShared(order); ord != nil {
		req.Order = ord
	}
	if sb := collectionSortToProto(sortBy); sb != nil {
		req.SortBy = sb
	}
	if opt.teamID != nil {
		req.TeamId = opt.teamID
	}
	if paginationToken != "" {
		req.PaginationToken = &paginationToken
	}
	resp, err := c.collections.ListCollections(ctx, req)
	return resp, WrapError(err)
}

// Get returns collection metadata.
func (c *CollectionsClient) Get(ctx context.Context, collectionID string, opts ...collectionsOption) (*collectionspb.CollectionMetadata, error) {
	if err := c.requireCollectionsStub(); err != nil {
		return nil, err
	}
	opt := applyCollectionsOptions(opts)
	req := &collectionspb.GetCollectionMetadataRequest{
		CollectionId: collectionID,
	}
	if opt.teamID != nil {
		req.TeamId = opt.teamID
	}
	resp, err := c.collections.GetCollectionMetadata(ctx, req)
	return resp, WrapError(err)
}

// Update changes name and/or chunk configuration.
func (c *CollectionsClient) Update(ctx context.Context, collectionID, name string, chunkCfg *ragpb.ChunkConfiguration, opts ...collectionsOption) (*collectionspb.CollectionMetadata, error) {
	if err := c.requireCollectionsStub(); err != nil {
		return nil, err
	}
	if name == "" && chunkCfg == nil {
		return nil, errors.New("either name or chunk configuration must be provided")
	}
	opt := applyCollectionsOptions(opts)
	req := &collectionspb.UpdateCollectionRequest{
		CollectionId:       collectionID,
		CollectionName:     &name,
		ChunkConfiguration: chunkCfg,
	}
	if opt.teamID != nil {
		req.TeamId = opt.teamID
	}
	resp, err := c.collections.UpdateCollection(ctx, req)
	return resp, WrapError(err)
}

// Delete removes a collection.
func (c *CollectionsClient) Delete(ctx context.Context, collectionID string, opts ...collectionsOption) error {
	if err := c.requireCollectionsStub(); err != nil {
		return err
	}
	opt := applyCollectionsOptions(opts)
	req := &collectionspb.DeleteCollectionRequest{
		CollectionId: collectionID,
	}
	if opt.teamID != nil {
		req.TeamId = opt.teamID
	}
	_, err := c.collections.DeleteCollection(ctx, req)
	return WrapError(err)
}

// Search performs semantic search over collections via the Documents service (data plane).
func (c *CollectionsClient) Search(ctx context.Context, query string, collectionIDs []string, opts ...DocumentSearchOption) (*xaipb.SearchResponse, error) {
	params := applyDocumentSearchOptions(opts)
	req := &xaipb.SearchRequest{
		Query: query,
		Source: &xaipb.DocumentsSource{
			CollectionIds: collectionIDs,
		},
	}
	if params.limit != nil {
		req.Limit = params.limit
	}
	if params.rankingMetric != nil {
		req.RankingMetric = params.rankingMetric
	}
	resp, err := c.documents.Search(ctx, req)
	return resp, WrapError(err)
}

// CollectionsSearchTool builds a server-side collections search tool definition for chat requests.
// This mirrors the helper in tools.go but keeps the collections surface discoverable in one place.
func (c *CollectionsClient) CollectionsSearchTool(collectionIDs []string, limit int32) *xaipb.Tool {
	return CollectionsSearchTool(collectionIDs, limit)
}

// UploadDocument uploads raw bytes into a collection.
func (c *CollectionsClient) UploadDocument(ctx context.Context, collectionID, name string, data []byte, contentType string, fields map[string]string, opts ...collectionsOption) (*collectionspb.DocumentMetadata, error) {
	if err := c.requireCollectionsStub(); err != nil {
		return nil, err
	}
	opt := applyCollectionsOptions(opts)
	req := &collectionspb.UploadDocumentRequest{
		CollectionId: collectionID,
		Name:         name,
		Data:         data,
		ContentType:  contentType,
		Fields:       fields,
	}
	if opt.teamID != nil {
		req.TeamId = opt.teamID
	}
	resp, err := c.collections.UploadDocument(ctx, req)
	return resp, WrapError(err)
}

// AddExistingDocument attaches an existing file to a collection.
func (c *CollectionsClient) AddExistingDocument(ctx context.Context, collectionID, fileID string, fields map[string]string, opts ...collectionsOption) error {
	if err := c.requireCollectionsStub(); err != nil {
		return err
	}
	opt := applyCollectionsOptions(opts)
	req := &collectionspb.AddDocumentToCollectionRequest{
		CollectionId: collectionID,
		FileId:       fileID,
		Fields:       fields,
	}
	if opt.teamID != nil {
		req.TeamId = opt.teamID
	}
	_, err := c.collections.AddDocumentToCollection(ctx, req)
	return WrapError(err)
}

// ListDocuments returns documents within a collection.
func (c *CollectionsClient) ListDocuments(ctx context.Context, collectionID string, limit int32, order Order, sortBy DocumentSortBy, paginationToken string, opts ...collectionsOption) (*collectionspb.ListDocumentsResponse, error) {
	if err := c.requireCollectionsStub(); err != nil {
		return nil, err
	}
	opt := applyCollectionsOptions(opts)
	req := &collectionspb.ListDocumentsRequest{
		CollectionId: collectionID,
	}
	if limit > 0 {
		req.Limit = &limit
	}
	if ord := orderToShared(order); ord != nil {
		req.Order = ord
	}
	if sb := documentSortToProto(sortBy); sb != nil {
		req.SortBy = sb
	}
	if opt.teamID != nil {
		req.TeamId = opt.teamID
	}
	if paginationToken != "" {
		req.PaginationToken = &paginationToken
	}
	resp, err := c.collections.ListDocuments(ctx, req)
	return resp, WrapError(err)
}

// GetDocument returns metadata for a file within a collection.
func (c *CollectionsClient) GetDocument(ctx context.Context, collectionID, fileID string, opts ...collectionsOption) (*collectionspb.DocumentMetadata, error) {
	if err := c.requireCollectionsStub(); err != nil {
		return nil, err
	}
	opt := applyCollectionsOptions(opts)
	req := &collectionspb.GetDocumentMetadataRequest{
		CollectionId: collectionID,
		FileId:       fileID,
	}
	if opt.teamID != nil {
		req.TeamId = opt.teamID
	}
	resp, err := c.collections.GetDocumentMetadata(ctx, req)
	return resp, WrapError(err)
}

// BatchGetDocuments fetches metadata for multiple documents.
func (c *CollectionsClient) BatchGetDocuments(ctx context.Context, collectionID string, fileIDs []string, opts ...collectionsOption) (*collectionspb.BatchGetDocumentsResponse, error) {
	if err := c.requireCollectionsStub(); err != nil {
		return nil, err
	}
	opt := applyCollectionsOptions(opts)
	req := &collectionspb.BatchGetDocumentsRequest{
		CollectionId: collectionID,
		FileIds:      fileIDs,
	}
	if opt.teamID != nil {
		req.TeamId = opt.teamID
	}
	resp, err := c.collections.BatchGetDocuments(ctx, req)
	return resp, WrapError(err)
}

// RemoveDocument detaches a document from a collection.
func (c *CollectionsClient) RemoveDocument(ctx context.Context, collectionID, fileID string, opts ...collectionsOption) error {
	if err := c.requireCollectionsStub(); err != nil {
		return err
	}
	opt := applyCollectionsOptions(opts)
	req := &collectionspb.RemoveDocumentFromCollectionRequest{
		CollectionId: collectionID,
		FileId:       fileID,
	}
	if opt.teamID != nil {
		req.TeamId = opt.teamID
	}
	_, err := c.collections.RemoveDocumentFromCollection(ctx, req)
	return WrapError(err)
}

// UpdateDocument updates a document's data/metadata.
func (c *CollectionsClient) UpdateDocument(ctx context.Context, collectionID, fileID, name string, data []byte, contentType string, fields map[string]string, opts ...collectionsOption) (*collectionspb.DocumentMetadata, error) {
	if err := c.requireCollectionsStub(); err != nil {
		return nil, err
	}
	opt := applyCollectionsOptions(opts)
	req := &collectionspb.UpdateDocumentRequest{
		CollectionId: collectionID,
		FileId:       fileID,
		Fields:       fields,
	}
	if opt.teamID != nil {
		req.TeamId = opt.teamID
	}
	if name != "" {
		req.Name = &name
	}
	if len(data) > 0 {
		req.Data = data
	}
	if contentType != "" {
		req.ContentType = &contentType
	}
	resp, err := c.collections.UpdateDocument(ctx, req)
	return resp, WrapError(err)
}

// ReindexDocument reprocesses a document after config changes.
func (c *CollectionsClient) ReindexDocument(ctx context.Context, collectionID, fileID string, opts ...collectionsOption) error {
	if err := c.requireCollectionsStub(); err != nil {
		return err
	}
	opt := applyCollectionsOptions(opts)
	req := &collectionspb.ReIndexDocumentRequest{
		CollectionId: collectionID,
		FileId:       fileID,
	}
	if opt.teamID != nil {
		req.TeamId = opt.teamID
	}
	_, err := c.collections.ReIndexDocument(ctx, req)
	return WrapError(err)
}

func collectionSortToProto(sort CollectionSortBy) *collectionspb.CollectionsSortBy {
	var v collectionspb.CollectionsSortBy
	switch sort {
	case CollectionSortByAge:
		v = collectionspb.CollectionsSortBy_COLLECTIONS_SORT_BY_AGE
	default:
		v = collectionspb.CollectionsSortBy_COLLECTIONS_SORT_BY_NAME
	}
	return &v
}

func documentSortToProto(sort DocumentSortBy) *collectionspb.DocumentsSortBy {
	var v collectionspb.DocumentsSortBy
	switch sort {
	case DocumentSortByAge:
		v = collectionspb.DocumentsSortBy_DOCUMENTS_SORT_BY_AGE
	case DocumentSortBySize:
		v = collectionspb.DocumentsSortBy_DOCUMENTS_SORT_BY_SIZE
	default:
		v = collectionspb.DocumentsSortBy_DOCUMENTS_SORT_BY_NAME
	}
	return &v
}

func orderToShared(o Order) *sharedpb.Ordering {
	var v sharedpb.Ordering
	switch o {
	case OrderAscending:
		v = sharedpb.Ordering_ORDERING_ASCENDING
	case OrderDescending:
		v = sharedpb.Ordering_ORDERING_DESCENDING
	default:
		return nil
	}
	return &v
}

// ValidateOrder ensures incoming order strings match expected values.
func ValidateOrder(order Order) error {
	switch order {
	case "", OrderAscending, OrderDescending:
		return nil
	default:
		return fmt.Errorf("invalid order %q", order)
	}
}
