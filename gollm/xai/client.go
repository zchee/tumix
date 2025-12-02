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
	"crypto/tls"
	"errors"
	"os"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"

	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
	billingpb "github.com/zchee/tumix/gollm/xai/management_api/v1"
)

const defaultServiceConfig = `{
	"methodConfig":
		[
			{
				"name":[{}],
				"retryPolicy":{
					"maxAttempts":5,
					"initialBackoff":"0.1s",
					"maxBackoff":"1s",
					"backoffMultiplier":2,
					"retryableStatusCodes":["UNAVAILABLE"]
				}
			}
		]
	}`

// Client aggregates all xAI service clients.
type Client struct {
	apiConn        *grpc.ClientConn
	managementConn *grpc.ClientConn

	Auth        *AuthClient
	Billing     *BillingClient
	Chat        *ChatClient
	Collections *CollectionsClient
	Files       *FilesClient
	Embed       *EmbedClient
	Image       *ImageClient
	Models      *ModelsClient
	Sampler     *SamplerClient
	Tokenizer   *TokenizerClient
}

// NewClient creates a new xAI API client with optional configuration.
func NewClient(apiKey string, optFns ...ClientOption) (*Client, error) {
	opts := DefaultClientOptions()
	opts.apiKey = apiKey
	for _, fn := range optFns {
		fn(opts)
	}

	if opts.apiKey == "" {
		opts.apiKey = os.Getenv("XAI_API_KEY")
	}
	if opts.apiKey == "" {
		return nil, errors.New("API key is required")
	}
	if opts.managementKey == "" {
		opts.managementKey = os.Getenv("XAI_MANAGEMENT_KEY")
	}

	apiConn := opts.apiConn
	var err error
	if apiConn == nil {
		apiConn, err = grpc.NewClient(opts.apiHost, BuildDialOptions(opts, opts.apiKey)...)
		if err != nil {
			return nil, err
		}
	}

	client := &Client{
		apiConn: apiConn,
		Auth: &AuthClient{
			auth: xaipb.NewAuthClient(apiConn),
		},
		Chat: &ChatClient{
			chat: xaipb.NewChatClient(apiConn),
		},
		Files: &FilesClient{
			files: xaipb.NewFilesClient(apiConn),
		},
		Embed: &EmbedClient{
			embedder: xaipb.NewEmbedderClient(apiConn),
		},
		Image: &ImageClient{
			image: xaipb.NewImageClient(apiConn),
		},
		Models: &ModelsClient{
			models: xaipb.NewModelsClient(apiConn),
		},
		Sampler: &SamplerClient{
			sample: xaipb.NewSampleClient(apiConn),
		},
		Tokenizer: &TokenizerClient{
			tokenize: xaipb.NewTokenizeClient(apiConn),
		},
	}

	if opts.managementKey != "" { //nolint:nestif // TODO(zchee): fix nolint
		client.managementConn = opts.managementConn
		if client.managementConn == nil {
			client.managementConn, err = grpc.NewClient(opts.managementHost, BuildDialOptions(opts, opts.managementKey)...)
			if err != nil {
				if cloneErr := apiConn.Close(); cloneErr != nil {
					err = errors.Join(err, cloneErr)
				}
				return nil, err
			}
		}
		client.Billing = &BillingClient{
			uisvc: billingpb.NewUISvcClient(client.managementConn),
		}

		client.Collections = NewCollectionsClient(apiConn, client.managementConn)
	}

	return client, nil
}

// Close closes all underlying connections.
func (c *Client) Close() error {
	if c == nil {
		return nil
	}
	var firstErr error
	if c.managementConn != nil {
		if err := c.managementConn.Close(); err != nil {
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

// BuildDialOptions builds gRPC dial options based on the provided client options and token.
func BuildDialOptions(opts *clientOptions, token string) []grpc.DialOption {
	creds := credentials.NewTLS(&tls.Config{
		MinVersion: tls.VersionTLS12,
	})
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
			AuthUnaryInterceptor(token, opts.metadata),
			TimeoutUnaryInterceptor(opts.timeout),
		),
		grpc.WithChainStreamInterceptor(
			AuthStreamInterceptor(token, opts.metadata),
			TimeoutStreamInterceptor(opts.timeout),
		),
	}

	if len(opts.dialOptions) > 0 {
		base = append(base, opts.dialOptions...)
	}

	return base
}
