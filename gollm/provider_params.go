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
type AnthropicParamMutator func(*anthropic.BetaMessageNewParams)

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

// effectiveProviderParams returns a merged view of request-scoped and default provider params.
//
// Defaults are applied first, then request params; later mutators/options can override earlier ones.
// If neither defaults nor request params are present, it returns (nil, false).
func effectiveProviderParams(req *model.LLMRequest, defaults *ProviderParams) (*ProviderParams, bool) {
	reqParams, _ := providerParams(req)

	merged := mergeProviderParams(defaults, reqParams)
	if merged == nil {
		return nil, false
	}

	return merged, true
}

func mergeProviderParams(defaults, overrides *ProviderParams) *ProviderParams {
	if defaults == nil && overrides == nil {
		return nil
	}

	merged := ProviderParams{
		Anthropic: mergeAnthropicParams(defaults, overrides),
		OpenAI:    mergeOpenAIParams(defaults, overrides),
		XAI:       mergeXAIParams(defaults, overrides),
	}

	if merged.Anthropic == nil && merged.OpenAI == nil && merged.XAI == nil {
		return nil
	}

	return &merged
}

func mergeAnthropicParams(defaults, overrides *ProviderParams) *AnthropicProviderParams {
	base := copyAnthropicParams(defaults)
	add := copyAnthropicParams(overrides)

	switch {
	case base == nil && add == nil:
		return nil
	case base == nil:
		return add
	case add == nil:
		return base
	default:
		merged := make([]AnthropicParamMutator, 0, len(base.Mutate)+len(add.Mutate))
		merged = append(merged, base.Mutate...)
		merged = append(merged, add.Mutate...)
		return &AnthropicProviderParams{Mutate: merged}
	}
}

func mergeOpenAIParams(defaults, overrides *ProviderParams) *OpenAIProviderParams {
	base := copyOpenAIParams(defaults)
	add := copyOpenAIParams(overrides)

	switch {
	case base == nil && add == nil:
		return nil
	case base == nil:
		return add
	case add == nil:
		return base
	default:
		merged := make([]OpenAIParamMutator, 0, len(base.Mutate)+len(add.Mutate))
		merged = append(merged, base.Mutate...)
		merged = append(merged, add.Mutate...)
		return &OpenAIProviderParams{Mutate: merged}
	}
}

func mergeXAIParams(defaults, overrides *ProviderParams) *XAIProviderParams {
	base := copyXAIParams(defaults)
	add := copyXAIParams(overrides)

	switch {
	case base == nil && add == nil:
		return nil
	case base == nil:
		return add
	case add == nil:
		return base
	default:
		merged := make([]xai.ChatOption, 0, len(base.Options)+len(add.Options))
		merged = append(merged, base.Options...)
		merged = append(merged, add.Options...)
		return &XAIProviderParams{Options: merged}
	}
}

func copyAnthropicParams(params *ProviderParams) *AnthropicProviderParams {
	if params == nil || params.Anthropic == nil {
		return nil
	}
	mutate := make([]AnthropicParamMutator, len(params.Anthropic.Mutate))
	copy(mutate, params.Anthropic.Mutate)
	return &AnthropicProviderParams{Mutate: mutate}
}

func copyOpenAIParams(params *ProviderParams) *OpenAIProviderParams {
	if params == nil || params.OpenAI == nil {
		return nil
	}
	mutate := make([]OpenAIParamMutator, len(params.OpenAI.Mutate))
	copy(mutate, params.OpenAI.Mutate)
	return &OpenAIProviderParams{Mutate: mutate}
}

func copyXAIParams(params *ProviderParams) *XAIProviderParams {
	if params == nil || params.XAI == nil {
		return nil
	}
	opts := make([]xai.ChatOption, len(params.XAI.Options))
	copy(opts, params.XAI.Options)
	return &XAIProviderParams{Options: opts}
}
