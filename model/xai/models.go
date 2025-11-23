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

	"google.golang.org/protobuf/types/known/emptypb"

	pb "github.com/zchee/tumix/model/xai/pb/xai/api/v1"
)

// ModelsClient provides access to the Models service.
type ModelsClient struct {
	stub pb.ModelsClient
}

// ListLanguageModels lists available language models.
func (c *ModelsClient) ListLanguageModels(ctx context.Context) (*pb.ListLanguageModelsResponse, error) {
	return c.stub.ListLanguageModels(ctx, &emptypb.Empty{})
}

// GetLanguageModel retrieves information about a specific language model.
func (c *ModelsClient) GetLanguageModel(ctx context.Context, name string) (*pb.LanguageModel, error) {
	return c.stub.GetLanguageModel(ctx, &pb.GetModelRequest{Name: name})
}

// ListEmbeddingModels lists available embedding models.
func (c *ModelsClient) ListEmbeddingModels(ctx context.Context) (*pb.ListEmbeddingModelsResponse, error) {
	return c.stub.ListEmbeddingModels(ctx, &emptypb.Empty{})
}

// GetEmbeddingModel retrieves information about a specific embedding model.
func (c *ModelsClient) GetEmbeddingModel(ctx context.Context, name string) (*pb.EmbeddingModel, error) {
	return c.stub.GetEmbeddingModel(ctx, &pb.GetModelRequest{Name: name})
}

// ListImageGenerationModels lists available image generation models.
func (c *ModelsClient) ListImageGenerationModels(ctx context.Context) (*pb.ListImageGenerationModelsResponse, error) {
	return c.stub.ListImageGenerationModels(ctx, &emptypb.Empty{})
}

// GetImageGenerationModel retrieves information about a specific image generation model.
func (c *ModelsClient) GetImageGenerationModel(ctx context.Context, name string) (*pb.ImageGenerationModel, error) {
	return c.stub.GetImageGenerationModel(ctx, &pb.GetModelRequest{Name: name})
}
