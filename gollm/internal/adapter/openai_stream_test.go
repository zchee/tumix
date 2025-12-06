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

package adapter

import (
	"testing"

	"github.com/openai/openai-go/v3/responses"
)

func BenchmarkOpenAIStreamAggregator(b *testing.B) {
	event := responses.ResponseStreamEventUnion{
		Type:  "response.output_text.delta",
		Delta: "hello world",
	}

	b.ReportAllocs()
	for b.Loop() {
		agg := NewOpenAIStreamAggregator(nil)
		if out := agg.Process(&event); len(out) != 1 {
			b.Fatalf("got %d partials, want 1", len(out))
		}
		if final := agg.Final(); final == nil {
			b.Fatalf("Final() returned nil")
		}
	}
}
