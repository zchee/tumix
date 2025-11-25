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
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

const uploadChunkSize = 3 << 20 // 3 MiB, matches Python SDK

// ProgressFunc mirrors the Python progress callback signature (uploaded, total).
type ProgressFunc func(uploaded, total int64)

type fileUploadConfig struct {
	filename string
	progress ProgressFunc
}

// FileOption customizes upload behavior.
type FileOption func(*fileUploadConfig)

// WithFilename overrides the filename used for non-path uploads (e.g. bytes, readers).
func WithFilename(name string) FileOption {
	return func(cfg *fileUploadConfig) { cfg.filename = name }
}

// WithProgress registers a progress callback invoked after each chunk is sent.
func WithProgress(fn ProgressFunc) FileOption {
	return func(cfg *fileUploadConfig) { cfg.progress = fn }
}

// FilesClient wraps the Files service.
type FilesClient struct {
	files xaipb.FilesClient
}

// Upload sends a file using the client-streaming UploadFile RPC.
//
// Supported `src`:
//   - string: path on disk
//   - []byte: raw data (requires WithFilename)
//   - io.Reader: streaming reader (requires WithFilename; total size optional)
func (c *FilesClient) Upload(ctx context.Context, src any, opts ...FileOption) (*xaipb.File, error) {
	if c == nil || c.files == nil {
		return nil, errors.New("files client not initialized")
	}
	cfg := fileUploadConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}

	var (
		reader    io.Reader
		totalSize int64 = -1
		filename  string
	)

	switch v := src.(type) {
	case string:
		info, err := os.Stat(v)
		if err != nil {
			return nil, err
		}
		if info.IsDir() {
			return nil, fmt.Errorf("%s is a directory", v)
		}
		file, err := os.Open(v)
		if err != nil {
			return nil, err
		}
		reader = file
		totalSize = info.Size()
		filename = filepath.Base(v)
		defer file.Close()
	case []byte:
		if cfg.filename == "" {
			return nil, errors.New("filename required when uploading raw bytes")
		}
		reader = bytes.NewReader(v)
		totalSize = int64(len(v))
		filename = cfg.filename
	case io.Reader:
		if cfg.filename == "" {
			return nil, errors.New("filename required when uploading from reader")
		}
		reader = v
		filename = cfg.filename
	default:
		return nil, fmt.Errorf("unsupported upload source type %T", src)
	}

	stream, err := c.files.UploadFile(ctx)
	if err != nil {
		return nil, WrapError(err)
	}

	init := &xaipb.UploadFileChunk{
		Chunk: &xaipb.UploadFileChunk_Init{
			Init: &xaipb.UploadFileInit{
				Name:    filename,
				Purpose: "", // parity with Python client
			},
		},
	}
	if err := stream.Send(init); err != nil {
		return nil, err
	}

	var uploaded int64
	buf := make([]byte, uploadChunkSize)
	for {
		n, readErr := reader.Read(buf)
		if n > 0 {
			chunk := &xaipb.UploadFileChunk{
				Chunk: &xaipb.UploadFileChunk_Data{
					Data: buf[:n],
				},
			}
			if err := stream.Send(chunk); err != nil {
				return nil, err
			}
			uploaded += int64(n)
			if cfg.progress != nil {
				cfg.progress(uploaded, totalSize)
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			return nil, readErr
		}
	}

	resp, err := stream.CloseAndRecv()
	if err != nil {
		return nil, WrapError(err)
	}
	return resp, nil
}

// BatchUpload uploads multiple files concurrently. Sources must be paths.
// Results are returned in index order; errors are stored per entry.
func (c *FilesClient) BatchUpload(ctx context.Context, paths []string, concurrency int, progress ProgressFunc) ([]*xaipb.File, []error) {
	if concurrency <= 0 {
		concurrency = 10
	}
	results := make([]*xaipb.File, len(paths))
	errs := make([]error, len(paths))

	var wg sync.WaitGroup
	sem := make(chan struct{}, concurrency)
	for i, p := range paths {
		sem <- struct{}{}
		wg.Go(func() {
			defer wg.Done()
			file, err := c.Upload(ctx, p, WithProgress(progress))
			results[i] = file
			errs[i] = err
			<-sem
		})
	}
	wg.Wait()
	return results, errs
}

// List returns file metadata.
func (c *FilesClient) List(ctx context.Context, limit int32, order Order, sortBy FileSortBy, paginationToken string) (*xaipb.ListFilesResponse, error) {
	req := &xaipb.ListFilesRequest{}
	if limit > 0 {
		req.Limit = limit
	}
	req.Order = orderToProto(order)
	if sb := fileSortByToProto(sortBy); sb != nil {
		req.SortBy = sb
	}
	if paginationToken != "" {
		req.PaginationToken = &paginationToken
	}
	return c.files.ListFiles(ctx, req)
}

// Get retrieves metadata for a file.
func (c *FilesClient) Get(ctx context.Context, fileID string) (*xaipb.File, error) {
	return c.files.RetrieveFile(ctx, &xaipb.RetrieveFileRequest{
		FileId: fileID,
	})
}

// Delete removes a file by ID.
func (c *FilesClient) Delete(ctx context.Context, fileID string) (*xaipb.DeleteFileResponse, error) {
	return c.files.DeleteFile(ctx, &xaipb.DeleteFileRequest{
		FileId: fileID,
	})
}

// Content downloads the full file content.
func (c *FilesClient) Content(ctx context.Context, fileID string) ([]byte, error) {
	stream, err := c.files.RetrieveFileContent(ctx, &xaipb.RetrieveFileContentRequest{
		FileId: fileID,
	})
	if err != nil {
		return nil, WrapError(err)
	}
	var buf bytes.Buffer
	for {
		chunk, err := stream.Recv()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}
		buf.Write(chunk.GetData())
	}
	return buf.Bytes(), nil
}

// URL retrieves a signed URL for file download when available.
func (c *FilesClient) URL(ctx context.Context, fileID string) (string, error) {
	resp, err := c.files.RetrieveFileURL(ctx, &xaipb.RetrieveFileURLRequest{
		FileId: fileID,
	})
	if err != nil {
		return "", WrapError(err)
	}
	return resp.GetUrl(), nil
}

// FileSortBy controls sorting in List.
type FileSortBy string

const (
	FileSortByCreatedAt FileSortBy = "created_at"
	FileSortByFilename  FileSortBy = "filename"
	FileSortBySize      FileSortBy = "size"
)

func fileSortByToProto(sort FileSortBy) *xaipb.FilesSortBy {
	var v xaipb.FilesSortBy
	switch sort {
	case FileSortByFilename:
		v = xaipb.FilesSortBy_FILES_SORT_BY_FILENAME
	case FileSortBySize:
		v = xaipb.FilesSortBy_FILES_SORT_BY_SIZE
	case FileSortByCreatedAt:
		fallthrough
	default:
		v = xaipb.FilesSortBy_FILES_SORT_BY_CREATED_AT
	}
	return &v
}

func orderToProto(o Order) xaipb.Ordering {
	switch o {
	case OrderAscending:
		return xaipb.Ordering_ASCENDING
	case OrderDescending:
		return xaipb.Ordering_DESCENDING
	default:
		return xaipb.Ordering(0)
	}
}
