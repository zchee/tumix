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

// ChatClient handles chat operations.
type ChatClient struct {
	chat xaipb.ChatClient
}

// Create initializes a new chat session for the specified model.
func (c *ChatClient) Create(model string, opts ...ChatOption) *ChatSession {
	req := &xaipb.GetCompletionsRequest{
		Model: model,
	}
	session := &ChatSession{
		stub:    c.chat,
		request: req,
	}
	for _, opt := range opts {
		opt(req, session)
	}

	return session
}

// GetStoredCompletion retrieves a stored response using the response ID.
func (c *ChatClient) GetStoredCompletion(ctx context.Context, responseID string) (*xaipb.GetChatCompletionResponse, error) {
	req := &xaipb.GetStoredCompletionRequest{
		ResponseId: responseID,
	}
	return c.chat.GetStoredCompletion(ctx, req)
}

// DeleteStoredCompletion deletes a stored response using the response ID.
func (c *ChatClient) DeleteStoredCompletion(ctx context.Context, responseID string) error {
	req := &xaipb.DeleteStoredCompletionRequest{
		ResponseId: responseID,
	}
	_, err := c.chat.DeleteStoredCompletion(ctx, req)
	return err
}

// StartDeferredCompletion starts sampling of the model and immediately returns a response containing a request id.
func (c *ChatClient) StartDeferredCompletion(ctx context.Context, req *xaipb.GetCompletionsRequest) (*xaipb.StartDeferredResponse, error) {
	return c.chat.StartDeferredCompletion(ctx, req)
}

// GetDeferredCompletion gets the result of a deferred completion.
func (c *ChatClient) GetDeferredCompletion(ctx context.Context, requestID string) (*xaipb.GetDeferredCompletionResponse, error) {
	req := &xaipb.GetDeferredRequest{
		RequestId: requestID,
	}
	return c.chat.GetDeferredCompletion(ctx, req)
}

// ParseInto is a generic convenience for structured outputs into type T.
func ParseInto[T any](ctx context.Context, s *ChatSession) (*Response, *T, error) {
	var out T
	resp, err := s.Parse(ctx, &out)
	return resp, &out, err
}
