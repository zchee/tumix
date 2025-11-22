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
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"

	pb "github.com/zchee/tumix/model/xai/pb/xai/api/v1"
)

const uploadChunkSize = 3 << 20 // 3 MiB

// UploadProgress reports progress of a single upload.
type UploadProgress func(uploaded, total int64)

type uploadOptions struct {
	filename string
	progress UploadProgress
	size     int64
}

// UploadOption configures upload behaviour.
type UploadOption func(*uploadOptions)

// WithUploadFilename overrides the inferred filename (required for byte slices/readers).
func WithUploadFilename(name string) UploadOption {
	return func(o *uploadOptions) { o.filename = name }
}

// WithUploadProgress sets a progress callback.
func WithUploadProgress(fn UploadProgress) UploadOption {
	return func(o *uploadOptions) { o.progress = fn }
}

// WithKnownSize hints the total size when uploading from a reader.
func WithKnownSize(size int64) UploadOption {
	return func(o *uploadOptions) { o.size = size }
}

// FileOrder controls list ordering.
type FileOrder string

// FileSortBy controls list sorting field.
type FileSortBy string

const (
	FileOrderAsc  FileOrder = "asc"
	FileOrderDesc FileOrder = "desc"

	FileSortCreatedAt FileSortBy = "created_at"
	FileSortFilename  FileSortBy = "filename"
	FileSortSize      FileSortBy = "size"
)

// FilesClient provides access to the Files service.
type FilesClient struct {
	stub pb.FilesClient
}

// Upload streams a file to xAI's servers. The source can be a file path (string), a byte slice, or an io.Reader.
func (c *FilesClient) Upload(ctx context.Context, src any, opts ...UploadOption) (*pb.File, error) {
	if c == nil {
		return nil, fmt.Errorf("files client is nil")
	}
	config := uploadOptions{}
	for _, opt := range opts {
		opt(&config)
	}

	switch v := src.(type) {
	case string:
		return c.uploadPath(ctx, v, &config)
	case []byte:
		if config.filename == "" {
			return nil, fmt.Errorf("filename is required when uploading raw bytes")
		}
		return c.uploadReader(ctx, config.filename, config.sizeOr(int64(len(v))), bytesReader(v), &config)
	case io.Reader:
		if config.filename == "" {
			return nil, fmt.Errorf("filename is required when uploading a reader")
		}
		return c.uploadReader(ctx, config.filename, config.sizeOr(-1), v, &config)
	default:
		return nil, fmt.Errorf("unsupported upload source type %T", src)
	}
}

// UploadPath uploads a file from disk.
func (c *FilesClient) UploadPath(ctx context.Context, path string, progress UploadProgress) (*pb.File, error) {
	return c.Upload(ctx, path, WithUploadProgress(progress))
}

// UploadBytes uploads raw bytes with the provided filename.
func (c *FilesClient) UploadBytes(ctx context.Context, filename string, data []byte, progress UploadProgress) (*pb.File, error) {
	return c.Upload(ctx, data, WithUploadFilename(filename), WithUploadProgress(progress), WithKnownSize(int64(len(data))))
}

// UploadReader uploads from an io.Reader with a known filename and optional size.
func (c *FilesClient) UploadReader(ctx context.Context, filename string, r io.Reader, size int64, progress UploadProgress) (*pb.File, error) {
	return c.Upload(ctx, r, WithUploadFilename(filename), WithKnownSize(size), WithUploadProgress(progress))
}

func (c *FilesClient) uploadPath(ctx context.Context, path string, config *uploadOptions) (*pb.File, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	filename := config.filename
	if filename == "" {
		filename = filepath.Base(path)
	}
	return c.uploadReader(ctx, filename, info.Size(), file, config)
}

func (c *FilesClient) uploadReader(ctx context.Context, filename string, size int64, r io.Reader, config *uploadOptions) (*pb.File, error) {
	stream, err := c.stub.UploadFile(ctx)
	if err != nil {
		return nil, err
	}

	initChunk := &pb.UploadFileChunk{Chunk: &pb.UploadFileChunk_Init{
		Init: &pb.UploadFileInit{Name: filename},
	}}
	if err := stream.Send(initChunk); err != nil {
		return nil, err
	}

	buf := make([]byte, uploadChunkSize)
	var uploaded int64
	for {
		read, readErr := io.ReadFull(r, buf)
		if readErr == io.EOF || readErr == io.ErrUnexpectedEOF {
			if read > 0 {
				if err := stream.Send(&pb.UploadFileChunk{Chunk: &pb.UploadFileChunk_Data{Data: buf[:read]}}); err != nil {
					return nil, err
				}
				uploaded += int64(read)
				invokeProgress(config.progress, uploaded, size)
			}
			break
		}
		if readErr != nil {
			return nil, readErr
		}

		if err := stream.Send(&pb.UploadFileChunk{Chunk: &pb.UploadFileChunk_Data{Data: buf[:read]}}); err != nil {
			return nil, err
		}
		uploaded += int64(read)
		invokeProgress(config.progress, uploaded, size)
	}

	resp, err := stream.CloseAndRecv()
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// UploadRequest represents a single item in a batch upload.
type UploadRequest struct {
	Source   any
	Filename string
	Size     int64
	Progress UploadProgress
}

// BatchUpload uploads multiple files concurrently. Results are returned in index order.
func (c *FilesClient) BatchUpload(ctx context.Context, reqs []UploadRequest, batchSize int, onComplete func(int, UploadRequest, *pb.File, error)) map[int]error {
	if batchSize <= 0 {
		batchSize = 4
	}

	sem := make(chan struct{}, batchSize)
	var wg sync.WaitGroup
	results := make(map[int]error)
	var mu sync.Mutex

	for idx, req := range reqs {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, r UploadRequest) {
			defer wg.Done()
			defer func() { <-sem }()

			file, err := c.Upload(ctx, r.Source,
				WithUploadFilename(r.Filename),
				WithKnownSize(r.Size),
				WithUploadProgress(r.Progress),
			)
			mu.Lock()
			results[i] = err
			mu.Unlock()

			if onComplete != nil {
				onComplete(i, r, file, err)
			}
		}(idx, req)
	}
	wg.Wait()
	return results
}

// List lists files with optional pagination and sorting.
func (c *FilesClient) List(ctx context.Context, limit int32, order FileOrder, sortBy FileSortBy, paginationToken string) (*pb.ListFilesResponse, error) {
	req := &pb.ListFilesRequest{}
	if limit > 0 {
		req.Limit = limit
	}
	if order != "" {
		req.Order = fileOrderToProto(order)
	}
	if sortBy != "" {
		req.SortBy = fileSortByToProto(sortBy)
	}
	if paginationToken != "" {
		req.PaginationToken = &paginationToken
	}
	return c.stub.ListFiles(ctx, req)
}

// Get returns file metadata.
func (c *FilesClient) Get(ctx context.Context, fileID string) (*pb.File, error) {
	return c.stub.RetrieveFile(ctx, &pb.RetrieveFileRequest{FileId: fileID})
}

// Delete removes a file.
func (c *FilesClient) Delete(ctx context.Context, fileID string) error {
	_, err := c.stub.DeleteFile(ctx, &pb.DeleteFileRequest{FileId: fileID})
	return err
}

// Content downloads the full content of a file.
func (c *FilesClient) Content(ctx context.Context, fileID string) ([]byte, error) {
	stream, err := c.stub.RetrieveFileContent(ctx, &pb.RetrieveFileContentRequest{FileId: fileID})
	if err != nil {
		return nil, err
	}
	var data []byte
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		data = append(data, chunk.Data...)
	}
	return data, nil
}

func invokeProgress(fn UploadProgress, uploaded, total int64) {
	if fn != nil {
		fn(uploaded, total)
	}
}

func (o *uploadOptions) sizeOr(defaultSize int64) int64 {
	if o.size > 0 {
		return o.size
	}
	return defaultSize
}

func fileOrderToProto(order FileOrder) pb.Ordering {
	switch order {
	case FileOrderDesc:
		return pb.Ordering_DESCENDING
	default:
		return pb.Ordering_ASCENDING
	}
}

func fileSortByToProto(sortBy FileSortBy) *pb.FilesSortBy {
	var v pb.FilesSortBy
	switch sortBy {
	case FileSortFilename:
		v = pb.FilesSortBy_FILES_SORT_BY_FILENAME
	case FileSortSize:
		v = pb.FilesSortBy_FILES_SORT_BY_SIZE
	case FileSortCreatedAt:
		v = pb.FilesSortBy_FILES_SORT_BY_CREATED_AT
	default:
		v = pb.FilesSortBy_FILES_SORT_BY_CREATED_AT
	}
	return &v
}

// bytesReader adapts a byte slice to io.Reader without extra allocation.
func bytesReader(b []byte) io.Reader { return bytes.NewReader(b) }
*/