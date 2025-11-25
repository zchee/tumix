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

func TestResponseDecodeJSON(t *testing.T) {
	resp := newResponse(&xaipb.GetChatCompletionResponse{
		Outputs: []*xaipb.CompletionOutput{{
			Message: &xaipb.CompletionMessage{
				Role:    xaipb.MessageRole_ROLE_ASSISTANT,
				Content: `{"foo":123}`,
			},
		}},
	}, nil)
	var out struct {
		Foo int `json:"foo"`
	}
	if err := resp.DecodeJSON(&out); err != nil {
		t.Fatalf("decode failed: %v", err)
	}
	if out.Foo != 123 {
		t.Fatalf("unexpected value: %+v", out)
	}
}
