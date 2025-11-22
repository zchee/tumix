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

/*
import (
	"context"
	"fmt"

	"google.golang.org/grpc"

	pb "github.com/zchee/tumix/model/xai/pb/xai/api/v1"
	sharedpb "github.com/zchee/tumix/model/xai/pb/xai/shared"
)

// Order is used for collection/document listing order.
type Order string

// Collection and document sorting helpers.
const (
	OrderAscending  Order = "asc"
	OrderDescending Order = "desc"
)

type CollectionSortBy string

const (
	CollectionSortByName CollectionSortBy = "name"
	CollectionSortByAge  CollectionSortBy = "age"
)

type DocumentSortBy string

const (
	DocumentSortByName DocumentSortBy = "name"
	DocumentSortByAge  DocumentSortBy = "age"
	DocumentSortBySize DocumentSortBy = "size"
)

// CollectionsClient provides access to the Collections and Documents services.
type CollectionsClient struct {
	collections pb.CollectionsClient
	documents   pb.DocumentsClient
}

// NewCollectionsClient builds a CollectionsClient using the API and management connections.
func NewCollectionsClient(apiConn, managementConn *grpc.ClientConn) *CollectionsClient {
	return &CollectionsClient{
		collections: pb.NewCollectionsClient(managementConn),
		documents:   pb.NewDocumentsClient(apiConn),
	}
}

// Create makes a new collection for document embeddings.
func (c *CollectionsClient) Create(ctx context.Context, name string, modelName string, chunkConfig *pb.ChunkConfiguration) (*pb.CollectionMetadata, error) {
	if c == nil || c.collections == nil {
		return nil, fmt.Errorf("management API key is required for collections operations")
	}

	req := &pb.CreateCollectionRequest{CollectionName: name}
	if modelName != "" {
		req.IndexConfiguration = &pb.IndexConfiguration{ModelName: modelName}
	}
	if chunkConfig != nil {
		req.ChunkConfiguration = chunkConfig
	}

	return c.collections.CreateCollection(ctx, req)
}

// List returns collections with optional pagination and ordering.
func (c *CollectionsClient) List(ctx context.Context, limit int32, order Order, sortBy CollectionSortBy, paginationToken string) (*pb.ListCollectionsResponse, error) {
	if c == nil || c.collections == nil {
		return nil, fmt.Errorf("management API key is required for collections operations")
	}
	req := &pb.ListCollectionsRequest{}
	if limit > 0 {
		req.Limit = &limit
	}
	if order != "" {
		req.Order = orderToProto(order)
	}
	if sortBy != "" {
		req.SortBy = collectionSortByToProto(sortBy)
	}
	if paginationToken != "" {
		req.PaginationToken = &paginationToken
	}
	return c.collections.ListCollections(ctx, req)
}

// Get returns collection metadata.
func (c *CollectionsClient) Get(ctx context.Context, collectionID string) (*pb.CollectionMetadata, error) {
	if c == nil || c.collections == nil {
		return nil, fmt.Errorf("management API key is required for collections operations")
	}
	return c.collections.GetCollectionMetadata(ctx, &pb.GetCollectionMetadataRequest{CollectionId: collectionID})
}

// Update changes the name and/or chunk configuration of a collection.
func (c *CollectionsClient) Update(ctx context.Context, collectionID string, name string, chunkConfig *pb.ChunkConfiguration) (*pb.CollectionMetadata, error) {
	if c == nil || c.collections == nil {
		return nil, fmt.Errorf("management API key is required for collections operations")
	}
	if name == "" && chunkConfig == nil {
		return nil, fmt.Errorf("at least one of name or chunk configuration must be provided")
	}

	req := &pb.UpdateCollectionRequest{CollectionId: collectionID}
	if name != "" {
		req.CollectionName = &name
	}
	if chunkConfig != nil {
		req.ChunkConfiguration = chunkConfig
	}

	return c.collections.UpdateCollection(ctx, req)
}

// Delete removes a collection.
func (c *CollectionsClient) Delete(ctx context.Context, collectionID string) error {
	if c == nil || c.collections == nil {
		return fmt.Errorf("management API key is required for collections operations")
	}
	_, err := c.collections.DeleteCollection(ctx, &pb.DeleteCollectionRequest{CollectionId: collectionID})
	return err
}

// Search performs semantic search across specified collections.
func (c *CollectionsClient) Search(ctx context.Context, query string, collectionIDs []string, limit int32) (*pb.SearchResponse, error) {
	req := &pb.SearchRequest{
		Query:  query,
		Source: &pb.DocumentsSource{CollectionIds: collectionIDs},
	}
	if limit > 0 {
		req.Limit = &limit
	}
	return c.documents.Search(ctx, req)
}

// UploadDocument uploads document bytes to a collection.
func (c *CollectionsClient) UploadDocument(ctx context.Context, collectionID, name string, data []byte, contentType string, fields map[string]string) (*pb.DocumentMetadata, error) {
	if c == nil || c.collections == nil {
		return nil, fmt.Errorf("management API key is required for collections operations")
	}
	return c.collections.UploadDocument(ctx, &pb.UploadDocumentRequest{
		CollectionId: collectionID,
		Name:         name,
		Data:         data,
		ContentType:  contentType,
		Fields:       fields,
	})
}

// AddExistingDocument attaches an existing file to a collection.
func (c *CollectionsClient) AddExistingDocument(ctx context.Context, collectionID, fileID string, fields map[string]string) error {
	if c == nil || c.collections == nil {
		return fmt.Errorf("management API key is required for collections operations")
	}
	_, err := c.collections.AddDocumentToCollection(ctx, &pb.AddDocumentToCollectionRequest{
		CollectionId: collectionID,
		FileId:       fileID,
		Fields:       fields,
	})
	return err
}

// ListDocuments returns documents within a collection.
func (c *CollectionsClient) ListDocuments(ctx context.Context, collectionID string, limit int32, order Order, sortBy DocumentSortBy, paginationToken string) (*pb.ListDocumentsResponse, error) {
	if c == nil || c.collections == nil {
		return nil, fmt.Errorf("management API key is required for collections operations")
	}
	req := &pb.ListDocumentsRequest{CollectionId: collectionID}
	if limit > 0 {
		req.Limit = &limit
	}
	if order != "" {
		req.Order = orderToProto(order)
	}
	if sortBy != "" {
		req.SortBy = documentSortByToProto(sortBy)
	}
	if paginationToken != "" {
		req.PaginationToken = &paginationToken
	}
	return c.collections.ListDocuments(ctx, req)
}

// GetDocument fetches document metadata.
func (c *CollectionsClient) GetDocument(ctx context.Context, collectionID, fileID string) (*pb.DocumentMetadata, error) {
	if c == nil || c.collections == nil {
		return nil, fmt.Errorf("management API key is required for collections operations")
	}
	return c.collections.GetDocumentMetadata(ctx, &pb.GetDocumentMetadataRequest{
		CollectionId: collectionID,
		FileId:       fileID,
	})
}

// BatchGetDocuments fetches multiple document metadata entries by ID.
func (c *CollectionsClient) BatchGetDocuments(ctx context.Context, collectionID string, fileIDs []string) (*pb.BatchGetDocumentsResponse, error) {
	if c == nil || c.collections == nil {
		return nil, fmt.Errorf("management API key is required for collections operations")
	}
	return c.collections.BatchGetDocuments(ctx, &pb.BatchGetDocumentsRequest{CollectionId: collectionID, FileIds: fileIDs})
}

// RemoveDocument detaches a document from a collection.
func (c *CollectionsClient) RemoveDocument(ctx context.Context, collectionID, fileID string) error {
	if c == nil || c.collections == nil {
		return fmt.Errorf("management API key is required for collections operations")
	}
	_, err := c.collections.RemoveDocumentFromCollection(ctx, &pb.RemoveDocumentFromCollectionRequest{CollectionId: collectionID, FileId: fileID})
	return err
}

// UpdateDocument updates document metadata and optionally data.
func (c *CollectionsClient) UpdateDocument(ctx context.Context, collectionID, fileID, name string, data []byte, contentType string, fields map[string]string) (*pb.DocumentMetadata, error) {
	if c == nil || c.collections == nil {
		return nil, fmt.Errorf("management API key is required for collections operations")
	}
	req := &pb.UpdateDocumentRequest{CollectionId: collectionID, FileId: fileID}
	if name != "" {
		req.Name = &name
	}
	if data != nil {
		req.Data = data
	}
	if contentType != "" {
		req.ContentType = &contentType
	}
	if fields != nil {
		req.Fields = fields
	}
	return c.collections.UpdateDocument(ctx, req)
}

// ReindexDocument regenerates indices for a document after config changes.
func (c *CollectionsClient) ReindexDocument(ctx context.Context, collectionID, fileID string) error {
	if c == nil || c.collections == nil {
		return fmt.Errorf("management API key is required for collections operations")
	}
	_, err := c.collections.ReIndexDocument(ctx, &pb.ReIndexDocumentRequest{CollectionId: collectionID, FileId: fileID})
	return err
}

func orderToProto(order Order) *sharedpb.Ordering {
	var v sharedpb.Ordering
	switch order {
	case OrderDescending:
		v = sharedpb.Ordering_ORDERING_DESCENDING
	default:
		v = sharedpb.Ordering_ORDERING_ASCENDING
	}
	return &v
}

func collectionSortByToProto(sortBy CollectionSortBy) *pb.CollectionsSortBy {
	var v pb.CollectionsSortBy
	switch sortBy {
	case CollectionSortByAge:
		v = pb.CollectionsSortBy_COLLECTIONS_SORT_BY_AGE
	default:
		v = pb.CollectionsSortBy_COLLECTIONS_SORT_BY_NAME
	}
	return &v
}

func documentSortByToProto(sortBy DocumentSortBy) *pb.DocumentsSortBy {
	var v pb.DocumentsSortBy
	switch sortBy {
	case DocumentSortByAge:
		v = pb.DocumentsSortBy_DOCUMENTS_SORT_BY_AGE
	case DocumentSortBySize:
		v = pb.DocumentsSortBy_DOCUMENTS_SORT_BY_SIZE
	default:
		v = pb.DocumentsSortBy_DOCUMENTS_SORT_BY_NAME
	}
	return &v
}
*/