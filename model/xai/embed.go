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

	pb "github.com/zchee/tumix/model/xai/pb/xai/api/v1"
)

// EmbedClient provides access to the Embeddings service.
type EmbedClient struct {
	stub pb.EmbedderClient
}

// Create generates embeddings for the provided inputs.
func (c *EmbedClient) Create(ctx context.Context, req *pb.EmbedRequest) (*pb.EmbedResponse, error) {
	return c.stub.Embed(ctx, req)
}

// CreateStrings generates embeddings for a list of text strings.
func (c *EmbedClient) CreateStrings(ctx context.Context, model string, texts []string) (*pb.EmbedResponse, error) {
	inputs := make([]*pb.EmbedInput, len(texts))
	for i, t := range texts {
		inputs[i] = &pb.EmbedInput{
			Input: &pb.EmbedInput_String_{String_: t},
		}
	}
	return c.Create(ctx, &pb.EmbedRequest{
		Model: model,
		Input: inputs,
	})
}
