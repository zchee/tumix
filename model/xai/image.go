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
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

// ImageFormat selects the image response format.
type ImageFormat string

const (
	// ImageFormatURL returns the image as a URL.
	ImageFormatURL ImageFormat = "url"
	// ImageFormatBase64 returns the image as a base64-encoded string.
	ImageFormatBase64 ImageFormat = "base64"
)

// ImageOption customizes image generation requests.
type ImageOption func(*xaipb.GenerateImageRequest)

// WithImageUser sets the end-user id.
func WithImageUser(user string) ImageOption {
	return func(req *xaipb.GenerateImageRequest) { req.User = user }
}

// WithImageFormat sets the desired return format.
func WithImageFormat(format ImageFormat) ImageOption {
	return func(req *xaipb.GenerateImageRequest) { req.Format = imageFormatToProto(format) }
}

// ImageClient wraps the Image service.
type ImageClient struct {
	image xaipb.ImageClient
}

// Sample generates a single image.
func (c *ImageClient) Sample(ctx context.Context, prompt, model string, opts ...ImageOption) (*ImageResponse, error) {
	resp, err := c.SampleBatch(ctx, prompt, model, 1, opts...)
	if err != nil {
		return nil, err
	}
	return resp[0], nil
}

// SampleBatch generates n images.
func (c *ImageClient) SampleBatch(ctx context.Context, prompt, model string, n int, opts ...ImageOption) ([]*ImageResponse, error) {
	req := &xaipb.GenerateImageRequest{
		Prompt: prompt,
		Model:  model,
		N:      int32Ptr(int32(n)),
		Format: xaipb.ImageFormat_IMG_FORMAT_URL,
	}
	for _, opt := range opts {
		opt(req)
	}

	resp, err := c.image.GenerateImage(ctx, req)
	if err != nil {
		return nil, err
	}
	images := make([]*ImageResponse, len(resp.Images))
	for i := range resp.Images {
		images[i] = &ImageResponse{
			proto: resp,
			index: i,
		}
	}
	return images, nil
}

// ImageResponse provides helpers around GenerateImageResponse.
type ImageResponse struct {
	proto *xaipb.ImageResponse
	index int
}

// Prompt returns the prompt actually used to generate the image.
func (r *ImageResponse) Prompt() string { return r.image().GetUpSampledPrompt() }

// URL returns the image URL if present.
func (r *ImageResponse) URL() (string, error) {
	url := r.image().GetUrl()
	if url == "" {
		return "", fmt.Errorf("image was not returned via URL")
	}
	return url, nil
}

// Base64 returns the base64 representation if present.
func (r *ImageResponse) Base64() (string, error) {
	if r.image().GetBase64() == "" {
		return "", fmt.Errorf("image was not returned via base64")
	}
	return r.image().GetBase64(), nil
}

// Data returns the raw bytes of the image, downloading if needed.
func (r *ImageResponse) Data(ctx context.Context) ([]byte, error) {
	if r.image().GetBase64() != "" {
		data := r.image().GetBase64()
		if comma := strings.Index(data, "base64,"); comma >= 0 {
			data = data[comma+7:]
		}
		return base64.StdEncoding.DecodeString(data)
	}
	url, err := r.URL()
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "xai-go-sdk")
	client := &http.Client{
		Timeout: 5 * time.Second,
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func (r *ImageResponse) image() *xaipb.GeneratedImage { return r.proto.Images[r.index] }

func imageFormatToProto(f ImageFormat) xaipb.ImageFormat {
	switch f {
	case ImageFormatBase64:
		return xaipb.ImageFormat_IMG_FORMAT_BASE64
	case ImageFormatURL:
		fallthrough
	default:
		return xaipb.ImageFormat_IMG_FORMAT_URL
	}
}

func int32Ptr(v int32) *int32 { return &v }
