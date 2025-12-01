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

	"github.com/google/go-cmp/cmp"
	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/genai"
)

func TestGenAIToolsToOpenAI(t *testing.T) {
	t.Parallel()

	schema := shared.FunctionParameters{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{"type": "string"},
		},
	}

	tests := map[string]struct {
		tools []*genai.Tool
		cfg   *genai.ToolConfig
		check func(t *testing.T, params []openai.ChatCompletionToolUnionParam, choice *openai.ChatCompletionToolChoiceOptionUnionParam)
	}{
		"empty tools returns nil": {
			tools: nil,
			check: func(t *testing.T, params []openai.ChatCompletionToolUnionParam, choice *openai.ChatCompletionToolChoiceOptionUnionParam) {
				t.Helper()
				if params != nil {
					t.Fatalf("params = %+v, want nil", params)
				}
				if choice != nil {
					t.Fatalf("choice = %+v, want nil", choice)
				}
			},
		},
		"maps function definition and prefers json schema": {
			tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{
							Name:                 "lookup_weather",
							Description:          "  get weather ",
							ParametersJsonSchema: schema,
						},
					},
				},
			},
			cfg: nil,
			check: func(t *testing.T, params []openai.ChatCompletionToolUnionParam, choice *openai.ChatCompletionToolChoiceOptionUnionParam) {
				t.Helper()
				if len(params) != 1 {
					t.Fatalf("len(params) = %d, want 1", len(params))
				}
				fn := params[0].OfFunction
				if fn == nil {
					t.Fatal("function tool nil")
				}
				if fn.Function.Name != "lookup_weather" {
					t.Fatalf("name = %q", fn.Function.Name)
				}
				if !fn.Function.Description.Valid() || fn.Function.Description.Value != "get weather" {
					t.Fatalf("description = %+v, want trimmed", fn.Function.Description)
				}
				if diff := cmp.Diff(schema, fn.Function.Parameters); diff != "" {
					t.Fatalf("parameters diff (-want +got):\n%s", diff)
				}
				if choice != nil {
					t.Fatalf("choice = %+v, want nil", choice)
				}
			},
		},
		"falls back to parameters and builds none choice": {
			tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{
							Name:        "echo",
							Description: "",
							Parameters: &genai.Schema{
								Type: "object",
								Properties: map[string]*genai.Schema{
									"msg": {Type: "string"},
								},
							},
						},
					},
				},
			},
			cfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeNone},
			},
			check: func(t *testing.T, params []openai.ChatCompletionToolUnionParam, choice *openai.ChatCompletionToolChoiceOptionUnionParam) {
				t.Helper()
				if len(params) != 1 {
					t.Fatalf("len(params) = %d, want 1", len(params))
				}
				fn := params[0].OfFunction
				if fn == nil {
					t.Fatal("function tool nil")
				}
				if fn.Function.Parameters == nil {
					t.Fatalf("parameters nil")
				}
				if choice == nil || !choice.OfAuto.Valid() || choice.OfAuto.Value != "none" {
					t.Fatalf("choice = %+v, want auto none", choice)
				}
			},
		},
		"single allowed function sets choice function": {
			tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "f1"}},
				},
			},
			cfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"f1"},
				},
			},
			check: func(t *testing.T, _ []openai.ChatCompletionToolUnionParam, choice *openai.ChatCompletionToolChoiceOptionUnionParam) {
				t.Helper()
				if choice == nil || choice.OfFunctionToolChoice == nil {
					t.Fatalf("choice missing function: %+v", choice)
				}
				if choice.OfFunctionToolChoice.Function.Name != "f1" {
					t.Fatalf("function choice name = %q", choice.OfFunctionToolChoice.Function.Name)
				}
				want := constant.ValueOf[constant.Function]()
				if choice.OfFunctionToolChoice.Type != want {
					t.Fatalf("choice type = %q, want %q", choice.OfFunctionToolChoice.Type, want)
				}
			},
		},
		"multiple allowed functions leaves choice nil": {
			tools: []*genai.Tool{
				{FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "f1"}}},
			},
			cfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"f1", "f2"},
				},
			},
			check: func(t *testing.T, _ []openai.ChatCompletionToolUnionParam, choice *openai.ChatCompletionToolChoiceOptionUnionParam) {
				t.Helper()
				if choice != nil {
					t.Fatalf("choice = %+v, want nil", choice)
				}
			},
		},
	}

	for name, tc := range tests {
		tc := tc
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			params, choice := GenAIToolsToOpenAI(tc.tools, tc.cfg)
			if tc.check == nil {
				t.Fatalf("missing check function")
			}
			tc.check(t, params, choice)
		})
	}
}

func TestToFunctionParameters(t *testing.T) {
	t.Parallel()

	t.Run("map passthrough", func(t *testing.T) {
		t.Parallel()
		in := shared.FunctionParameters{"foo": "bar"}
		got, err := toFunctionParameters(in)
		if err != nil {
			t.Fatalf("toFunctionParameters error: %v", err)
		}
		if diff := cmp.Diff(in, got); diff != "" {
			t.Fatalf("diff (-want +got):\n%s", diff)
		}
	})

	t.Run("bytes json", func(t *testing.T) {
		t.Parallel()
		got, err := toFunctionParameters([]byte(`{"a":1}`))
		if err != nil {
			t.Fatalf("toFunctionParameters error: %v", err)
		}
		if diff := cmp.Diff(shared.FunctionParameters{"a": float64(1)}, got); diff != "" {
			t.Fatalf("diff (-want +got):\n%s", diff)
		}
	})

	t.Run("struct marshaling", func(t *testing.T) {
		t.Parallel()
		type payload struct {
			ID   string `json:"id"`
			Flag bool   `json:"flag"`
		}
		got, err := toFunctionParameters(payload{ID: "123", Flag: true})
		if err != nil {
			t.Fatalf("toFunctionParameters error: %v", err)
		}
		if diff := cmp.Diff(shared.FunctionParameters{"id": "123", "flag": true}, got); diff != "" {
			t.Fatalf("diff (-want +got):\n%s", diff)
		}
	})

	t.Run("invalid json bytes", func(t *testing.T) {
		t.Parallel()
		if _, err := toFunctionParameters([]byte(`{"bad"`)); err == nil {
			t.Fatalf("expected error for invalid json")
		}
	})

	t.Run("nil input returns nil", func(t *testing.T) {
		t.Parallel()
		got, err := toFunctionParameters(nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != nil {
			t.Fatalf("got = %+v, want nil", got)
		}
	})
}
