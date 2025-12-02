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
	"testing"
	"time"

	"google.golang.org/grpc"
)

func TestDefaultClientOptions(t *testing.T) {
	opts := DefaultClientOptions()

	if opts.apiHost != APIHost {
		t.Fatalf("apiHost = %s, want %s", opts.apiHost, APIHost)
	}
	if opts.managementHost != ManagementAPIHost {
		t.Fatalf("managementHost = %s, want %s", opts.managementHost, ManagementAPIHost)
	}
	if opts.timeout != defaultTimeout {
		t.Fatalf("timeout = %v, want %v", opts.timeout, defaultTimeout)
	}

	version := opts.metadata["xai-sdk-version"]
	if version == "" {
		t.Fatalf("xai-sdk-version metadata not set")
	}
	language := opts.metadata["xai-sdk-language"]
	if language == "" {
		t.Fatalf("xai-sdk-language metadata not set")
	}
}

func TestClientOptionMutators(t *testing.T) {
	opts := DefaultClientOptions()

	WithAPIKey("k1")(opts)
	if opts.apiKey != "k1" {
		t.Fatalf("apiKey = %s", opts.apiKey)
	}

	WithManagementAPIKey("mk")(opts)
	if opts.managementKey != "mk" {
		t.Fatalf("managementKey = %s", opts.managementKey)
	}

	WithAPIHost("custom:1")(opts)
	if opts.apiHost != "custom:1" {
		t.Fatalf("apiHost not updated: %s", opts.apiHost)
	}
	WithAPIHost("")(opts)
	if opts.apiHost != "custom:1" {
		t.Fatalf("apiHost changed on empty override")
	}

	WithManagementAPIHost("mgt:2")(opts)
	if opts.managementHost != "mgt:2" {
		t.Fatalf("managementHost not updated: %s", opts.managementHost)
	}
	WithManagementAPIHost("")(opts)
	if opts.managementHost != "mgt:2" {
		t.Fatalf("managementHost changed on empty override")
	}

	WithMetadata(map[string]string{"a": "b"})(opts)
	if opts.metadata["a"] != "b" {
		t.Fatalf("metadata merge failed: %+v", opts.metadata)
	}

	WithSDKVersion("1.2.3")(opts)
	if got := opts.metadata["xai-sdk-version"]; got != "go/1.2.3" {
		t.Fatalf("xai-sdk-version = %s", got)
	}

	WithDialOptions(grpc.EmptyDialOption{})(opts)
	if len(opts.dialOptions) == 0 {
		t.Fatalf("dial options not appended")
	}

	WithInsecure()(opts)
	if !opts.useInsecure {
		t.Fatalf("useInsecure not set")
	}

	WithTimeout(2 * time.Second)(opts)
	if opts.timeout != 2*time.Second {
		t.Fatalf("timeout not updated: %v", opts.timeout)
	}

	WithTimeout(0)(opts)
	if opts.timeout != 2*time.Second {
		t.Fatalf("timeout changed on non-positive value: %v", opts.timeout)
	}
}
