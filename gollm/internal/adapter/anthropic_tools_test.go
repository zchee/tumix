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

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

func TestGenAIToolsToAnthropic(t *testing.T) {
	t.Parallel()

	desc := "Return weather for a city"
	params := &genai.Schema{
		Type: "object",
		Properties: map[string]*genai.Schema{
			"city": {Type: "string"},
		},
	}

	tests := map[string]struct {
		tools []*genai.Tool
		cfg   *genai.ToolConfig
		check func(t *testing.T, got []anthropic.ToolUnionParam, tc *anthropic.ToolChoiceUnionParam)
	}{
		"empty tools returns nil": {
			tools: nil,
			cfg:   nil,
			check: func(t *testing.T, got []anthropic.ToolUnionParam, tc *anthropic.ToolChoiceUnionParam) {
				t.Helper()
				if got != nil {
					t.Fatalf("tools = %+v, want nil", got)
				}
				if tc != nil {
					t.Fatalf("tool choice = %+v, want nil", tc)
				}
			},
		},
		"skips empty declarations": {
			tools: []*genai.Tool{
				{FunctionDeclarations: []*genai.FunctionDeclaration{{}, {Name: ""}, nil}},
			},
			check: func(t *testing.T, got []anthropic.ToolUnionParam, _ *anthropic.ToolChoiceUnionParam) {
				t.Helper()
				if len(got) != 0 {
					t.Fatalf("len(tools) = %d, want 0", len(got))
				}
			},
		},
		"maps tool and sets none choice": {
			tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{Name: "lookup_weather", Description: desc, Parameters: params},
					},
				},
			},
			cfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeNone},
			},
			check: func(t *testing.T, got []anthropic.ToolUnionParam, tc *anthropic.ToolChoiceUnionParam) {
				t.Helper()

				if len(got) != 1 {
					t.Fatalf("len(tools) = %d, want 1", len(got))
				}
				tool := got[0].OfTool
				if tool == nil {
					t.Fatal("tool nil")
				}
				if tool.Name != "lookup_weather" {
					t.Fatalf("tool.Name = %q, want lookup_weather", tool.Name)
				}
				if tool.Type != anthropic.ToolTypeCustom {
					t.Fatalf("tool.Type = %q, want %q", tool.Type, anthropic.ToolTypeCustom)
				}
				if tool.InputSchema.Type != constant.ValueOf[constant.Object]() {
					t.Fatalf("schema type = %q, want object", tool.InputSchema.Type)
				}
				if diff := cmp.Diff(params, tool.InputSchema.Properties); diff != "" {
					t.Fatalf("schema properties diff (-want +got):\n%s", diff)
				}
				if !tool.Description.Valid() || tool.Description.Value != desc {
					t.Fatalf("description = %+v, want %q", tool.Description, desc)
				}

				if tc == nil || tc.OfNone == nil {
					t.Fatalf("tool choice none not set: %+v", tc)
				}
			},
		},
		"sets auto choice for any/auto mode": {
			tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{Name: "ping"},
					},
				},
			},
			cfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAny},
			},
			check: func(t *testing.T, _ []anthropic.ToolUnionParam, tc *anthropic.ToolChoiceUnionParam) {
				t.Helper()
				if tc == nil || tc.OfAuto == nil {
					t.Fatalf("tool choice auto not set: %+v", tc)
				}
				want := constant.ValueOf[constant.Auto]()
				if tc.OfAuto.Type != want {
					t.Fatalf("auto type = %q, want %q", tc.OfAuto.Type, want)
				}
			},
		},
	}

	for name, tc := range tests {
		tc := tc
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			got, choice := GenAIToolsToAnthropic(tc.tools, tc.cfg)
			if tc.check == nil {
				t.Fatalf("test %s missing check", name)
			}
			tc.check(t, got, choice)
		})
	}
}
