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
	"crypto/tls"
	"fmt"
	"os"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"

	pb "github.com/zchee/tumix/model/xai/pb/xai/api/v1"
)

const defaultServiceConfig = `{"methodConfig":[{"name":[{}],"retryPolicy":{"maxAttempts":5,"initialBackoff":"0.1s","maxBackoff":"1s","backoffMultiplier":2,"retryableStatusCodes":["UNAVAILABLE"]}}]}`

// Client aggregates all xAI service clients.
type Client struct {
	apiConn        *grpc.ClientConn
	managementConn *grpc.ClientConn

	Auth        *AuthClient
	Chat        *ChatClient
	Collections *CollectionsClient
	Documents   *DocumentsClient
	Embed       *EmbedClient
	// Files       *FilesClient
	Image     *ImageClient
	Models    *ModelsClient
	Sampler   *SamplerClient
	Tokenizer *TokenizerClient
}

// NewClient creates a new xAI API client with optional configuration.
func NewClient(ctx context.Context, apiKey string, optFns ...ClientOption) (*Client, error) {
	opts := defaultClientOptions()
	opts.apiKey = apiKey
	for _, fn := range optFns {
		fn(opts)
	}

	if opts.apiKey == "" {
		opts.apiKey = os.Getenv("XAI_API_KEY")
	}
	if opts.apiKey == "" {
		return nil, fmt.Errorf("API key is required")
	}
	if opts.managementKey == "" {
		opts.managementKey = os.Getenv("XAI_MANAGEMENT_KEY")
	}

	apiConn, err := grpc.DialContext(ctx, opts.apiHost, buildDialOptions(opts, opts.apiKey)...)
	if err != nil {
		return nil, err
	}

	var mgmtConn *grpc.ClientConn
	if opts.managementKey != "" {
		mgmtConn, err = grpc.DialContext(ctx, opts.managementHost, buildDialOptions(opts, opts.managementKey)...)
		if err != nil {
			apiConn.Close()
			return nil, err
		}
	}

	client := &Client{
		apiConn:        apiConn,
		managementConn: mgmtConn,
		Auth:           &AuthClient{stub: pb.NewAuthClient(apiConn)},
		Chat:           &ChatClient{stub: pb.NewChatClient(apiConn)},
		Documents:      &DocumentsClient{stub: pb.NewDocumentsClient(apiConn)},
		Embed:          &EmbedClient{stub: pb.NewEmbedderClient(apiConn)},
		// Files:          &FilesClient{stub: pb.NewFilesClient(apiConn)},
		Image:     &ImageClient{stub: pb.NewImageClient(apiConn)},
		Models:    &ModelsClient{stub: pb.NewModelsClient(apiConn)},
		Sampler:   &SamplerClient{stub: pb.NewSampleClient(apiConn)},
		Tokenizer: &TokenizerClient{stub: pb.NewTokenizeClient(apiConn)},
	}

	client.Collections = NewCollectionsClient(apiConn, opts.managementKey, "https://"+opts.managementHost)

	return client, nil
}

// Close closes all underlying connections.
func (c *Client) Close() error {
	if c == nil {
		return nil
	}
	var firstErr error
	if c.managementConn != nil {
		if err := c.managementConn.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if c.apiConn != nil {
		if err := c.apiConn.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

func buildDialOptions(opts *clientOptions, token string) []grpc.DialOption {
	creds := credentials.NewTLS(&tls.Config{})
	if opts.useInsecure {
		creds = insecure.NewCredentials()
	}

	base := []grpc.DialOption{
		grpc.WithTransportCredentials(creds),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallSendMsgSize(defaultMaxMessageBytes),
			grpc.MaxCallRecvMsgSize(defaultMaxMessageBytes),
		),
		grpc.WithDefaultServiceConfig(defaultServiceConfig),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                30 * time.Second,
			Timeout:             10 * time.Second,
			PermitWithoutStream: true,
		}),
		grpc.WithChainUnaryInterceptor(
			authUnaryInterceptor(token, opts.metadata),
			timeoutUnaryInterceptor(opts.timeout),
		),
		grpc.WithChainStreamInterceptor(
			authStreamInterceptor(token, opts.metadata),
			timeoutStreamInterceptor(opts.timeout),
		),
	}

	if len(opts.dialOptions) > 0 {
		base = append(base, opts.dialOptions...)
	}

	return base
}

func authUnaryInterceptor(token string, md map[string]string) grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, callOpts ...grpc.CallOption) error {
		ctx = attachMetadata(ctx, token, md)
		return invoker(ctx, method, req, reply, cc, callOpts...)
	}
}

func authStreamInterceptor(token string, md map[string]string) grpc.StreamClientInterceptor {
	return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, callOpts ...grpc.CallOption) (grpc.ClientStream, error) {
		ctx = attachMetadata(ctx, token, md)
		return streamer(ctx, desc, cc, method, callOpts...)
	}
}

func timeoutUnaryInterceptor(timeout time.Duration) grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, callOpts ...grpc.CallOption) error {
		if _, ok := ctx.Deadline(); !ok && timeout > 0 {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, timeout)
			defer cancel()
		}
		return invoker(ctx, method, req, reply, cc, callOpts...)
	}
}

func timeoutStreamInterceptor(timeout time.Duration) grpc.StreamClientInterceptor {
	return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, callOpts ...grpc.CallOption) (grpc.ClientStream, error) {
		if _, ok := ctx.Deadline(); !ok && timeout > 0 {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, timeout)
			defer cancel()
		}
		return streamer(ctx, desc, cc, method, callOpts...)
	}
}

func attachMetadata(ctx context.Context, token string, static map[string]string) context.Context {
	pairs := []string{"authorization", "Bearer " + token}
	for k, v := range static {
		pairs = append(pairs, k, v)
	}
	return metadata.AppendToOutgoingContext(ctx, pairs...)
}
