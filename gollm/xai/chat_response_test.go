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
	"encoding/json/jsontext"
	"reflect"
	"testing"

	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
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

func TestSchemaBytesCacheReused(t *testing.T) {
	type sample struct {
		Foo int `json:"foo"`
	}

	first, err := schemaBytesForType(reflect.TypeFor[sample]())
	if err != nil {
		t.Fatalf("first schema err: %v", err)
	}
	second, err := schemaBytesForType(reflect.TypeFor[sample]())
	if err != nil {
		t.Fatalf("second schema err: %v", err)
	}

	if hdr1, hdr2 := reflect.ValueOf(first).Pointer(), reflect.ValueOf(second).Pointer(); hdr1 != hdr2 {
		t.Fatalf("expected cached schema slice to be reused")
	}
	if !jsontext.Value(first).IsValid() {
		t.Fatalf("schema is not valid JSON: %s", string(first))
	}
}

func TestResponseProcessChunkStreamingAggregation(t *testing.T) {
	resp := newResponse(&xaipb.GetChatCompletionResponse{}, nil)

	chunk1 := &xaipb.GetChatCompletionChunk{
		Outputs: []*xaipb.CompletionOutputChunk{{
			Index: 0,
			Delta: &xaipb.Delta{
				Role:    xaipb.MessageRole_ROLE_ASSISTANT,
				Content: "hel",
			},
		}},
	}
	chunk2 := &xaipb.GetChatCompletionChunk{
		Outputs: []*xaipb.CompletionOutputChunk{{
			Index: 0,
			Delta: &xaipb.Delta{
				Role:             xaipb.MessageRole_ROLE_ASSISTANT,
				Content:          "lo",
				ReasoningContent: "why",
			},
		}},
	}

	resp.processChunk(chunk1)
	resp.processChunk(chunk2)

	if got := resp.Content(); got != "hello" {
		t.Fatalf("content aggregation mismatch: %q", got)
	}
	if got := resp.ReasoningContent(); got != "why" {
		t.Fatalf("reasoning aggregation mismatch: %q", got)
	}
}
