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
	"maps"
	"testing"
	"time"

	"google.golang.org/grpc"
)

var dialOptionsSink []grpc.DialOption

func BenchmarkBuildDialOptions(b *testing.B) {
	base := DefaultClientOptions()
	base.metadata = maps.Clone(base.metadata)

	b.Run("default_tls", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			dialOptionsSink = BuildDialOptions(base, "api-token")
		}
		if len(dialOptionsSink) == 0 {
			b.Fatalf("expected dial options")
		}
	})

	b.Run("insecure_with_overrides", func(b *testing.B) {
		opts := *base
		opts.metadata = maps.Clone(base.metadata)
		opts.metadata["x-custom-header"] = "bench"
		opts.useInsecure = true
		opts.dialOptions = []grpc.DialOption{
			grpc.WithAuthority("localhost:8080"),
			grpc.WithUserAgent("xai-bench"),
			grpc.WithIdleTimeout(30 * time.Second),
		}

		b.ReportAllocs()
		for b.Loop() {
			dialOptionsSink = BuildDialOptions(&opts, "management-token")
		}
		if len(dialOptionsSink) == 0 {
			b.Fatalf("expected dial options")
		}
	})
}
