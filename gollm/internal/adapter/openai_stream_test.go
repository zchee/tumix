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
	"strconv"
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

func BenchmarkOpenAIStreamAggregatorToolCalls(b *testing.B) {
	tests := []struct {
		name          string
		toolCalls     int
		deltasPerCall int
	}{
		{
			name:          "calls=4",
			toolCalls:     4,
			deltasPerCall: 4,
		},
		{
			name:          "calls=16",
			toolCalls:     16,
			deltasPerCall: 4,
		},
		{
			name:          "calls=64",
			toolCalls:     64,
			deltasPerCall: 4,
		},
		{
			name:          "calls=256",
			toolCalls:     256,
			deltasPerCall: 4,
		},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			events := make([]responses.ResponseStreamEventUnion, 0, tt.toolCalls*(tt.deltasPerCall+1))
			for i := range tt.toolCalls {
				id := "tool-" + strconv.Itoa(i)
				for range tt.deltasPerCall {
					events = append(events, responses.ResponseStreamEventUnion{
						Type:        "response.function_call_arguments.delta",
						OutputIndex: int64(i),
						ItemID:      id,
						Delta:       `{"x":`,
					})
				}
				events = append(events, responses.ResponseStreamEventUnion{
					Type:        "response.function_call_arguments.done",
					OutputIndex: int64(i),
					ItemID:      id,
					Name:        "fn",
					Arguments:   `{"x":1}`,
				})
			}

			b.ReportAllocs()
			for b.Loop() {
				agg := NewOpenAIStreamAggregator(nil)
				for i := range events {
					agg.Process(&events[i])
				}
				if got := len(agg.toolCalls); got != tt.toolCalls {
					b.Fatalf("len(toolCalls)=%d, want %d", got, tt.toolCalls)
				}
				if agg.Err() != nil {
					b.Fatalf("unexpected err: %v", agg.Err())
				}
			}
		})
	}
}
