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
//
// This remains a defined type (not an alias) to preserve type identity for
// downstream consumers that may reflect on it.
type ProviderParams struct {
	Anthropic *AnthropicProviderParams
	OpenAI    *OpenAIProviderParams
	XAI       *XAIProviderParams
}

// ProviderMutator mutates provider-specific request params after defaults are applied.
type ProviderMutator[T any] func(*T)

// ProviderMutators groups mutators for a provider.
type ProviderMutators[T any] struct {
	Mutate []ProviderMutator[T]
}

// ProviderOptions groups provider-specific options to append to a request.
type ProviderOptions[T any] struct {
	Options []T
}

// AnthropicParamMutator mutates the Anthropic message params after defaults are applied.
type AnthropicParamMutator = ProviderMutator[anthropic.BetaMessageNewParams]

// AnthropicProviderParams contains Anthropic-specific overrides.
type AnthropicProviderParams = ProviderMutators[anthropic.BetaMessageNewParams]

// OpenAIParamMutator mutates the OpenAI chat completion params after defaults are applied.
type OpenAIParamMutator = ProviderMutator[responses.ResponseNewParams]

// OpenAIProviderParams contains OpenAI-specific overrides.
type OpenAIProviderParams = ProviderMutators[responses.ResponseNewParams]

// XAIProviderParams contains extra chat options for xAI requests.
type XAIProviderParams = ProviderOptions[xai.ChatOption]

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

	var (
		defAnthropic *AnthropicProviderParams
		defOpenAI    *OpenAIProviderParams
		defXAI       *XAIProviderParams
		ovAnthropic  *AnthropicProviderParams
		ovOpenAI     *OpenAIProviderParams
		ovXAI        *XAIProviderParams
	)

	if defaults != nil {
		defAnthropic = defaults.Anthropic
		defOpenAI = defaults.OpenAI
		defXAI = defaults.XAI
	}
	if overrides != nil {
		ovAnthropic = overrides.Anthropic
		ovOpenAI = overrides.OpenAI
		ovXAI = overrides.XAI
	}

	merged := ProviderParams{
		Anthropic: mergeMutators(copyMutators(defAnthropic), copyMutators(ovAnthropic)),
		OpenAI:    mergeMutators(copyMutators(defOpenAI), copyMutators(ovOpenAI)),
		XAI:       mergeOptions(copyOptions(defXAI), copyOptions(ovXAI)),
	}

	if merged.Anthropic == nil && merged.OpenAI == nil && merged.XAI == nil {
		return nil
	}

	return &merged
}

func mergeMutators[T any](base, add *ProviderMutators[T]) *ProviderMutators[T] {
	switch {
	case base == nil && add == nil:
		return nil
	case base == nil:
		return add
	case add == nil:
		return base
	default:
		merged := make([]ProviderMutator[T], 0, len(base.Mutate)+len(add.Mutate))
		merged = append(merged, base.Mutate...)
		merged = append(merged, add.Mutate...)
		return &ProviderMutators[T]{Mutate: merged}
	}
}

func mergeOptions[T any](base, add *ProviderOptions[T]) *ProviderOptions[T] {
	switch {
	case base == nil && add == nil:
		return nil
	case base == nil:
		return add
	case add == nil:
		return base
	default:
		merged := make([]T, 0, len(base.Options)+len(add.Options))
		merged = append(merged, base.Options...)
		merged = append(merged, add.Options...)
		return &ProviderOptions[T]{Options: merged}
	}
}

func copyMutators[T any](params *ProviderMutators[T]) *ProviderMutators[T] {
	if params == nil {
		return nil
	}
	mutate := make([]ProviderMutator[T], len(params.Mutate))
	copy(mutate, params.Mutate)
	return &ProviderMutators[T]{Mutate: mutate}
}

func copyOptions[T any](params *ProviderOptions[T]) *ProviderOptions[T] {
	if params == nil {
		return nil
	}
	opts := make([]T, len(params.Options))
	copy(opts, params.Options)
	return &ProviderOptions[T]{Options: opts}
}
