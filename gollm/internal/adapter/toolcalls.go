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
	json "encoding/json/v2"
	"slices"
	"strings"

	"google.golang.org/genai"
)

type ToolCallState struct {
	ID    string
	Name  string
	Index int64
	args  strings.Builder
}

// ToolCallAccumulator orders tool calls by index and builds genai FunctionCall parts.
type ToolCallAccumulator struct {
	byIndex map[int64]*ToolCallState
	order   []int64
}

func NewToolCallAccumulator() *ToolCallAccumulator {
	return &ToolCallAccumulator{byIndex: make(map[int64]*ToolCallState)}
}

func (a *ToolCallAccumulator) Ensure(idx int64, id string) *ToolCallState {
	if state, ok := a.byIndex[idx]; ok {
		if id != "" && state.ID == "" {
			state.ID = id
		}
		return state
	}
	state := &ToolCallState{Index: idx, ID: id}
	a.byIndex[idx] = state
	a.order = append(a.order, idx)
	return state
}

// Parts returns ordered FunctionCall parts.
func (a *ToolCallAccumulator) Parts() []*genai.Part {
	if len(a.byIndex) == 0 {
		return nil
	}
	if len(a.order) > 1 {
		slices.Sort(a.order)
	}

	parts := make([]*genai.Part, 0, len(a.order))
	for _, idx := range a.order {
		tc := a.byIndex[idx]
		parts = append(parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   tc.ID,
				Name: tc.Name,
				Args: ParseArgs(tc.args.String()),
			},
		})
	}
	return parts
}

// ParseArgs converts a JSON object string into a map, returning {"raw": raw} on error.
func ParseArgs(raw string) map[string]any {
	if strings.TrimSpace(raw) == "" {
		return map[string]any{}
	}

	var out map[string]any
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return map[string]any{"raw": raw}
	}
	return out
}

func (s *ToolCallState) ArgsBuilder() *strings.Builder {
	return &s.args
}
