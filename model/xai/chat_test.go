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

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

func TestResponseProcessChunk(t *testing.T) {
	idx := int32(0)
	resp := newResponse(&xaipb.GetChatCompletionResponse{}, &idx)
	chunk := &xaipb.GetChatCompletionChunk{
		Outputs: []*xaipb.CompletionOutputChunk{
			{
				Index: 0,
				Delta: &xaipb.Delta{
					Role:             xaipb.MessageRole_ROLE_ASSISTANT,
					Content:          "Hello",
					ReasoningContent: "Why",
				},
			},
		},
	}

	resp.processChunk(chunk)

	if got := resp.Content(); got != "Hello" {
		t.Fatalf("content mismatch: %q", got)
	}
	if got := resp.ReasoningContent(); got != "Why" {
		t.Fatalf("reasoning mismatch: %q", got)
	}
}
