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

// SamplerClient exposes the Sample service for low-level text sampling.
type SamplerClient struct {
	sample xaipb.SampleClient
}

// SampleText performs a unary sampling request.
func (c *SamplerClient) SampleText(ctx context.Context, req *xaipb.SampleTextRequest) (*xaipb.SampleTextResponse, error) {
	resp, err := c.sample.SampleText(ctx, req)
	if err != nil {
		return nil, WrapError(err)
	}
	return resp, nil
}

// SampleTextStreaming opens a server-streaming sampling request.
func (c *SamplerClient) SampleTextStreaming(ctx context.Context, req *xaipb.SampleTextRequest) (xaipb.Sample_SampleTextStreamingClient, error) {
	return c.sample.SampleTextStreaming(ctx, req)
}
