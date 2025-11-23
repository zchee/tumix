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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"time"

	"google.golang.org/grpc"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

// Order is used for collection/document listing order.
type Order string

const (
	// OrderAscending sorts results in ascending order.
	OrderAscending Order = "asc"
	// OrderDescending sorts results in descending order.
	OrderDescending Order = "desc"
)

// CollectionSortBy defines the field to sort collections by.
type CollectionSortBy string

const (
	// CollectionSortByName sorts collections by name.
	CollectionSortByName CollectionSortBy = "name"
	// CollectionSortByAge sorts collections by creation time.
	CollectionSortByAge CollectionSortBy = "age"
)

// DocumentSortBy defines the field to sort documents by.
type DocumentSortBy string

const (
	// DocumentSortByName sorts documents by name.
	DocumentSortByName DocumentSortBy = "name"
	// DocumentSortByAge sorts documents by creation time.
	DocumentSortByAge DocumentSortBy = "age"
	// DocumentSortBySize sorts documents by size.
	DocumentSortBySize DocumentSortBy = "size"
)

// CollectionsClient provides access to the Collections and Documents services.
type CollectionsClient struct {
	documents xaipb.DocumentsClient
	apiKey    string
	baseURL   string
	client    *http.Client
}

// NewCollectionsClient builds a CollectionsClient using the API connection and management API credentials.
func NewCollectionsClient(apiConn *grpc.ClientConn, apiKey string, baseURL string) *CollectionsClient {
	return &CollectionsClient{
		documents: xaipb.NewDocumentsClient(apiConn),
		apiKey:    apiKey,
		baseURL:   baseURL,
		client: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

// Collection represents a document collection.
type Collection struct {
	ID                 string              `json:"collection_id"`
	Name               string              `json:"name"`
	IndexConfiguration *IndexConfiguration `json:"index_configuration,omitempty"`
	ChunkConfiguration *ChunkConfiguration `json:"chunk_configuration,omitempty"`
	CreatedAt          int64               `json:"created_at,omitempty"`
	UpdatedAt          int64               `json:"updated_at,omitempty"`
}

// IndexConfiguration defines the indexing parameters for a collection.
type IndexConfiguration struct {
	ModelName string `json:"model_name,omitempty"`
}

// ChunkConfiguration defines the chunking parameters for a collection.
type ChunkConfiguration struct {
	MinTokens int `json:"min_tokens,omitempty"`
	MaxTokens int `json:"max_tokens,omitempty"`
}

type createCollectionRequest struct {
	Name               string              `json:"name"`
	IndexConfiguration *IndexConfiguration `json:"index_configuration,omitempty"`
	ChunkConfiguration *ChunkConfiguration `json:"chunk_configuration,omitempty"`
}

// Create makes a new collection for document embeddings.
func (c *CollectionsClient) Create(ctx context.Context, name string, modelName string) (*Collection, error) {
	reqBody := createCollectionRequest{
		Name: name,
	}
	if modelName != "" {
		reqBody.IndexConfiguration = &IndexConfiguration{
			ModelName: modelName,
		}
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	var resp Collection
	if err := c.doRequest(ctx, http.MethodPost, "/v1/collections", bytes.NewReader(data), &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ListCollectionsResponse is the response from listing collections.
type ListCollectionsResponse struct {
	Collections []Collection `json:"collections"`
	NextToken   string       `json:"next_token,omitempty"`
}

// List returns collections with optional pagination and ordering.
func (c *CollectionsClient) List(ctx context.Context, limit int32, order Order, sortBy CollectionSortBy, paginationToken string) (*ListCollectionsResponse, error) {
	query := make(map[string]string)
	if limit > 0 {
		query["limit"] = fmt.Sprintf("%d", limit)
	}
	if order != "" {
		query["order"] = string(order)
	}
	if sortBy != "" {
		query["sort_by"] = string(sortBy)
	}
	if paginationToken != "" {
		query["pagination_token"] = paginationToken
	}

	var resp ListCollectionsResponse
	if err := c.doRequest(ctx, http.MethodGet, "/v1/collections", nil, &resp, query); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Get returns collection metadata.
func (c *CollectionsClient) Get(ctx context.Context, collectionID string) (*Collection, error) {
	var resp Collection
	if err := c.doRequest(ctx, http.MethodGet, fmt.Sprintf("/v1/collections/%s", collectionID), nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

type updateCollectionRequest struct {
	Name               string              `json:"name,omitempty"`
	ChunkConfiguration *ChunkConfiguration `json:"chunk_configuration,omitempty"`
}

// Update changes the name and/or chunk configuration of a collection.
func (c *CollectionsClient) Update(ctx context.Context, collectionID string, name string) (*Collection, error) {
	reqBody := updateCollectionRequest{
		Name: name,
	}
	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	var resp Collection
	if err := c.doRequest(ctx, http.MethodPatch, fmt.Sprintf("/v1/collections/%s", collectionID), bytes.NewReader(data), &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Delete removes a collection.
func (c *CollectionsClient) Delete(ctx context.Context, collectionID string) error {
	return c.doRequest(ctx, http.MethodDelete, fmt.Sprintf("/v1/collections/%s", collectionID), nil, nil)
}

// Search performs semantic search across specified collections.
func (c *CollectionsClient) Search(ctx context.Context, query string, collectionIDs []string, limit int32) (*xaipb.SearchResponse, error) {
	req := &xaipb.SearchRequest{
		Query:  query,
		Source: &xaipb.DocumentsSource{
			CollectionIds: collectionIDs,
		},
	}
	if limit > 0 {
		req.Limit = &limit
	}
	return c.documents.Search(ctx, req)
}

// Document represents a file within a collection.
type Document struct {
	ID           string            `json:"file_id"`
	Name         string            `json:"name"`
	CollectionID string            `json:"collection_id"`
	Metadata     map[string]string `json:"metadata,omitempty"`
	SizeBytes    int64             `json:"size_bytes,omitempty"`
	Status       string            `json:"status,omitempty"`
	CreatedAt    int64             `json:"created_at,omitempty"`
	UpdatedAt    int64             `json:"updated_at,omitempty"`
}

// UploadDocument uploads document bytes to a collection.
func (c *CollectionsClient) UploadDocument(ctx context.Context, collectionID, name string, data []byte, contentType string) (*Document, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	// Add file
	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename="%s"`, name))
	h.Set("Content-Type", contentType)
	part, err := writer.CreatePart(h)
	if err != nil {
		return nil, err
	}
	if _, err := part.Write(data); err != nil {
		return nil, err
	}

	if err := writer.Close(); err != nil {
		return nil, err
	}

	var resp Document
	if err := c.doRequestWithHeaders(ctx, http.MethodPost, fmt.Sprintf("/v1/collections/%s/documents", collectionID), body, &resp, map[string]string{
		"Content-Type": writer.FormDataContentType(),
	}); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ListDocumentsResponse is the response from listing documents in a collection.
type ListDocumentsResponse struct {
	Documents []Document `json:"documents"`
	NextToken string     `json:"next_token,omitempty"`
}

// ListDocuments returns documents within a collection.
func (c *CollectionsClient) ListDocuments(ctx context.Context, collectionID string, limit int32, order Order, sortBy DocumentSortBy, paginationToken string) (*ListDocumentsResponse, error) {
	query := make(map[string]string)
	if limit > 0 {
		query["limit"] = fmt.Sprintf("%d", limit)
	}
	if order != "" {
		query["order"] = string(order)
	}
	if sortBy != "" {
		query["sort_by"] = string(sortBy)
	}
	if paginationToken != "" {
		query["pagination_token"] = paginationToken
	}

	var resp ListDocumentsResponse
	if err := c.doRequest(ctx, http.MethodGet, fmt.Sprintf("/v1/collections/%s/documents", collectionID), nil, &resp, query); err != nil {
		return nil, err
	}
	return &resp, nil
}

// GetDocument fetches document metadata.
func (c *CollectionsClient) GetDocument(ctx context.Context, collectionID, fileID string) (*Document, error) {
	var resp Document
	if err := c.doRequest(ctx, http.MethodGet, fmt.Sprintf("/v1/collections/%s/documents/%s", collectionID, fileID), nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// RemoveDocument detaches a document from a collection.
func (c *CollectionsClient) RemoveDocument(ctx context.Context, collectionID, fileID string) error {
	return c.doRequest(ctx, http.MethodDelete, fmt.Sprintf("/v1/collections/%s/documents/%s", collectionID, fileID), nil, nil)
}

func (c *CollectionsClient) doRequest(ctx context.Context, method, path string, body io.Reader, result any, queryParams ...map[string]string) error {
	return c.doRequestWithHeaders(ctx, method, path, body, result, nil, queryParams...)
}

func (c *CollectionsClient) doRequestWithHeaders(ctx context.Context, method, path string, body io.Reader, result any, headers map[string]string, queryParams ...map[string]string) error {
	url := c.baseURL + path
	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return err
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	if headers != nil {
		for k, v := range headers {
			req.Header.Set(k, v)
		}
	} else if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	if len(queryParams) > 0 {
		q := req.URL.Query()
		for _, params := range queryParams {
			for k, v := range params {
				q.Set(k, v)
			}
		}
		req.URL.RawQuery = q.Encode()
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(respBody))
	}

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return err
		}
	}

	return nil
}
