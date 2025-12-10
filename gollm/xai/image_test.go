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
	"encoding/base64"
	"testing"

	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
)

func TestImageOptions(t *testing.T) {
	req := &xaipb.GenerateImageRequest{}

	WithImageUser("user1")(req)
	WithImageFormat(ImageFormatBase64)(req)

	if req.GetUser() != "user1" {
		t.Fatalf("user not set: %s", req.GetUser())
	}
	if req.GetFormat() != xaipb.ImageFormat_IMG_FORMAT_BASE64 {
		t.Fatalf("format not converted: %v", req.GetFormat())
	}
}

func TestImageResponseHelpers(t *testing.T) {
	raw := []byte("pngdata")
	b64 := "data:image/png;base64," + base64.StdEncoding.EncodeToString(raw)
	proto := &xaipb.ImageResponse{
		Images: []*xaipb.GeneratedImage{
			{
				Image:           &xaipb.GeneratedImage_Base64{Base64: b64},
				UpSampledPrompt: "refined",
			},
		},
	}

	resp := &ImageResponse{proto: proto, index: 0}

	if got, err := resp.Base64(); err != nil || got != b64 {
		t.Fatalf("base64 helper failed: got %q err=%v", got, err)
	}

	if got := resp.Prompt(); got != "refined" {
		t.Fatalf("prompt mismatch: %s", got)
	}

	data, err := resp.Data(t.Context())
	if err != nil {
		t.Fatalf("data decode error: %v", err)
	}
	if !bytes.Equal(data, raw) {
		t.Fatalf("data mismatch: %q", data)
	}

	// Clear base64 to exercise URL error path.
	proto.Images[0].Image = &xaipb.GeneratedImage_Url{Url: ""}
	if _, err := resp.URL(); err == nil {
		t.Fatalf("expected URL error when url empty")
	}
	if _, err := resp.Base64(); err == nil {
		t.Fatalf("expected base64 error when empty")
	}
}
