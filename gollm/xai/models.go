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

	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
)

// ModelsClient provides access to the Models service.
type ModelsClient struct {
	models xaipb.ModelsClient
}

// ListLanguageModels lists available language models.
func (c *ModelsClient) ListLanguageModels(ctx context.Context) (*xaipb.ListLanguageModelsResponse, error) {
	resp, err := c.models.ListLanguageModels(ctx, &emptypb.Empty{})
	if err != nil {
		return nil, WrapError(err)
	}
	return resp, nil
}

// GetLanguageModel retrieves information about a specific language model.
func (c *ModelsClient) GetLanguageModel(ctx context.Context, name string) (*xaipb.LanguageModel, error) {
	resp, err := c.models.GetLanguageModel(ctx, &xaipb.GetModelRequest{
		Name: name,
	})
	return resp, WrapError(err)
}

// ListEmbeddingModels lists available embedding models.
func (c *ModelsClient) ListEmbeddingModels(ctx context.Context) (*xaipb.ListEmbeddingModelsResponse, error) {
	resp, err := c.models.ListEmbeddingModels(ctx, &emptypb.Empty{})
	return resp, WrapError(err)
}

// GetEmbeddingModel retrieves information about a specific embedding model.
func (c *ModelsClient) GetEmbeddingModel(ctx context.Context, name string) (*xaipb.EmbeddingModel, error) {
	resp, err := c.models.GetEmbeddingModel(ctx, &xaipb.GetModelRequest{
		Name: name,
	})
	return resp, WrapError(err)
}

// ListImageGenerationModels lists available image generation models.
func (c *ModelsClient) ListImageGenerationModels(ctx context.Context) (*xaipb.ListImageGenerationModelsResponse, error) {
	resp, err := c.models.ListImageGenerationModels(ctx, &emptypb.Empty{})
	return resp, WrapError(err)
}

// GetImageGenerationModel retrieves information about a specific image generation model.
func (c *ModelsClient) GetImageGenerationModel(ctx context.Context, name string) (*xaipb.ImageGenerationModel, error) {
	resp, err := c.models.GetImageGenerationModel(ctx, &xaipb.GetModelRequest{
		Name: name,
	})
	return resp, WrapError(err)
}
