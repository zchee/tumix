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

package gollm

import (
	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/adk/model"

	"github.com/zchee/tumix/gollm/xai"
)

const providerParamsKey = "gollm:provider-params"

// ProviderParams carries provider-specific tuning knobs that are not represented
// in the generic genai.GenerateContentConfig.
type ProviderParams struct {
	Anthropic *AnthropicProviderParams
	OpenAI    *OpenAIProviderParams
	XAI       *XAIProviderParams
}

// AnthropicParamMutator mutates the Anthropic message params after defaults are applied.
type AnthropicParamMutator func(*anthropic.MessageNewParams)

// AnthropicProviderParams contains Anthropic-specific overrides.
type AnthropicProviderParams struct {
	Mutate []AnthropicParamMutator
}

// OpenAIParamMutator mutates the OpenAI chat completion params after defaults are applied.
type OpenAIParamMutator func(*responses.ResponseNewParams)

// OpenAIProviderParams contains OpenAI-specific overrides.
type OpenAIProviderParams struct {
	Mutate []OpenAIParamMutator
}

// XAIProviderParams contains extra chat options for xAI requests.
type XAIProviderParams struct {
	Options []xai.ChatOption
}

// SetProviderParams attaches provider params to a request using the reserved tools slot.
func SetProviderParams(req *model.LLMRequest, params *ProviderParams) {
	if req == nil || params == nil {
		return
	}
	if req.Tools == nil {
		req.Tools = make(map[string]any)
	}
	req.Tools[providerParamsKey] = params
}

func providerParams(req *model.LLMRequest) (*ProviderParams, bool) {
	if req == nil || req.Tools == nil {
		return nil, false
	}

	raw, ok := req.Tools[providerParamsKey]
	if !ok || raw == nil {
		return nil, false
	}

	switch v := raw.(type) {
	case *ProviderParams:
		return v, true
	case ProviderParams:
		return &v, true
	default:
		return nil, false
	}
}
