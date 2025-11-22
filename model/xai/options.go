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
	"runtime"
	"time"

	"google.golang.org/grpc"
)

const (
	DefaultAPIHost                         = "api.x.ai:443"
	DefaultManagementAPIHost               = "management-api.x.ai:443"
	defaultMaxMessageBytes   int           = 20 << 20 // 20 MiB
	defaultTimeout           time.Duration = 27 * time.Minute
)

// ClientOption configures the xAI client.
type ClientOption func(*clientOptions)

type clientOptions struct {
	apiKey         string
	managementKey  string
	apiHost        string
	managementHost string
	metadata       map[string]string
	dialOptions    []grpc.DialOption
	useInsecure    bool
	timeout        time.Duration
}

func defaultClientOptions() *clientOptions {
	return &clientOptions{
		apiHost:        DefaultAPIHost,
		managementHost: DefaultManagementAPIHost,
		metadata: map[string]string{
			"xai-sdk-version":  "go/dev", // SDK version placeholder
			"xai-sdk-language": "go/" + runtime.Version(),
		},
		timeout: defaultTimeout,
	}
}

// WithAPIKey sets the API key used for the data plane.
func WithAPIKey(key string) ClientOption {
	return func(o *clientOptions) {
		o.apiKey = key
	}
}

// WithManagementAPIKey sets the management API key used for collection administration.
func WithManagementAPIKey(key string) ClientOption {
	return func(o *clientOptions) {
		o.managementKey = key
	}
}

// WithAPIHost overrides the default API host.
func WithAPIHost(host string) ClientOption {
	return func(o *clientOptions) {
		if host != "" {
			o.apiHost = host
		}
	}
}

// WithManagementAPIHost overrides the default management API host.
func WithManagementAPIHost(host string) ClientOption {
	return func(o *clientOptions) {
		if host != "" {
			o.managementHost = host
		}
	}
}

// WithMetadata attaches static metadata to every RPC.
func WithMetadata(md map[string]string) ClientOption {
	return func(o *clientOptions) {
		for k, v := range md {
			o.metadata[k] = v
		}
	}
}

// WithDialOptions appends grpc.DialOptions to the client configuration.
func WithDialOptions(opts ...grpc.DialOption) ClientOption {
	return func(o *clientOptions) {
		o.dialOptions = append(o.dialOptions, opts...)
	}
}

// WithInsecure enables plaintext transport (useful for local development).
func WithInsecure() ClientOption {
	return func(o *clientOptions) {
		o.useInsecure = true
	}
}

// WithTimeout sets the default RPC timeout applied when no deadline is present on the context.
func WithTimeout(timeout time.Duration) ClientOption {
	return func(o *clientOptions) {
		if timeout > 0 {
			o.timeout = timeout
		}
	}
}
