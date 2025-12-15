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

	"go.opentelemetry.io/otel/attribute"

	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
)

func findAttr(attrs []attribute.KeyValue, key string) (attribute.Value, bool) {
	for _, kv := range attrs {
		if string(kv.Key) == key {
			return kv.Value, true
		}
	}
	return attribute.Value{}, false
}

func TestMakeSpanRequestAttributesContentJoin(t *testing.T) {
	s := &ChatSession{
		request: &xaipb.GetCompletionsRequest{
			Model: "grok-4-1",
			Messages: []*xaipb.Message{
				{
					Role: xaipb.MessageRole_ROLE_USER,
					Content: []*xaipb.Content{
						TextContent("a"),
						TextContent("b"),
					},
				},
			},
		},
	}

	attrs := s.makeSpanRequestAttributes()
	if got, ok := findAttr(attrs, "gen_ai.prompt.0.content"); !ok || got.AsString() != "ab" {
		t.Fatalf("gen_ai.prompt.0.content=%q ok=%v, want %q", got.AsString(), ok, "ab")
	}
}

func TestMakeSpanRequestAttributesCachingAndInvalidation(t *testing.T) {
	s := &ChatSession{
		request: &xaipb.GetCompletionsRequest{
			Model: "grok-4-1",
			Messages: []*xaipb.Message{
				User("hello"),
			},
		},
	}

	attrs1 := s.makeSpanRequestAttributes()
	attrs2 := s.makeSpanRequestAttributes()
	if len(attrs1) == 0 || len(attrs2) == 0 {
		t.Fatalf("unexpected empty attrs: len1=%d len2=%d", len(attrs1), len(attrs2))
	}
	if &attrs1[0] != &attrs2[0] {
		t.Fatalf("expected cached attrs reuse, got different backing arrays")
	}

	s.Append(User("world"))
	attrs3 := s.makeSpanRequestAttributes()
	if len(attrs3) <= len(attrs1) {
		t.Fatalf("expected attrs to grow after Append: len=%d want > %d", len(attrs3), len(attrs1))
	}
	if _, ok := findAttr(attrs3, "gen_ai.prompt.1.role"); !ok {
		t.Fatalf("expected gen_ai.prompt.1.role after Append")
	}
}
