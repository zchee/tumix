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

	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
)

type demoStruct struct {
	Name string `json:"name"`
}

func TestWithJSONStruct(t *testing.T) {
	s := chatSessionForTest()
	WithJSONStruct[*demoStruct]()(s.request, nil)
	if s.request.GetResponseFormat() == nil || s.request.GetResponseFormat().GetFormatType() != xaipb.FormatType_FORMAT_TYPE_JSON_SCHEMA {
		t.Fatalf("response format not set")
	}
}

// chatSessionForTest builds a minimal ChatSession with a dummy request.
func chatSessionForTest() *ChatSession {
	return &ChatSession{
		request: &xaipb.GetCompletionsRequest{
			Model: "grok-4-1-fast-reasoning",
		},
	}
}
