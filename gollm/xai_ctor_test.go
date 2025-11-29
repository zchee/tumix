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

package gollm

import (
	"context"
	"net"
	"testing"

	"google.golang.org/grpc"

	"github.com/zchee/tumix/gollm/xai"
)

func TestNewXAILLMConstructor(t *testing.T) {
	t.Parallel()

	llm, err := NewXAILLM(t.Context(), AuthMethodAPIKey("dummy"), "grok-test",
		xai.WithAPIHost("127.0.0.1:0"), // no outbound traffic in ctor
		xai.WithInsecure(),
		xai.WithTimeout(0),
		xai.WithDialOptions(grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return nil, nil
		})),
	)
	if err != nil {
		t.Fatalf("NewXAILLM error: %v", err)
	}
	if got := llm.Name(); got != "grok-test" {
		t.Fatalf("Name() = %q, want %q", got, "grok-test")
	}
}
