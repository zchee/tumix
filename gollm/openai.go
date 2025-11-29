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
	"cmp"
	"context"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"iter"
	"slices"
	"strings"
	"time"

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/zchee/tumix/internal/version"
)

// openAILLM implements the adk [model.LLM] interface using OpenAI SDK.
type openAILLM struct {
	client    openai.Client
	name      string
	userAgent string
}

var _ model.LLM = (*openAILLM)(nil)

// NewOpenAILLM creates a new OpenAI-backed LLM.
//
// If authKey is nil, the OpenAI SDK falls back to the OPENAI_API_KEY environment variable.
//
//nolint:unparam
func NewOpenAILLM(_ context.Context, authKey AuthMethod, modelName string, opts ...option.RequestOption) (model.LLM, error) {
	userAgent := version.UserAgent("openai")

	// TODO(zchee): Ues [option.WithHTTPClient] with OTel tracing transport
	ropts := []option.RequestOption{
		option.WithHeader("User-Agent", userAgent),
		option.WithMaxRetries(2),
		option.WithRequestTimeout(3 * time.Minute),
	}
	if authKey != nil {
		ropts = append(ropts, option.WithAPIKey(authKey.value()))
	}

	// opts are allowed to override by order
	opts = append(ropts, opts...)
	client := openai.NewClient(opts...)

	return &openAILLM{
		client:    client,
		name:      modelName,
		userAgent: userAgent,
	}, nil
}

// Name implements [model.LLM].
func (m *openAILLM) Name() string { return m.name }

// GenerateContent implements [model.LLM].
func (m *openAILLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	ensureUserContent(req)
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if req.Config.HTTPOptions == nil {
		req.Config.HTTPOptions = &genai.HTTPOptions{}
	}
	if req.Config.HTTPOptions.Headers == nil {
		req.Config.HTTPOptions.Headers = make(map[string][]string)
	}
	req.Config.HTTPOptions.Headers["User-Agent"] = []string{m.userAgent}

	params, err := m.chatCompletionParams(req)
	if err != nil {
		return func(yield func(*model.LLMResponse, error) bool) {
			yield(nil, err)
		}
	}

	if stream {
		return m.stream(ctx, params)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.client.Chat.Completions.New(ctx, *params)
		if err != nil {
			yield(nil, err)
			return
		}

		llmResp, err := openAIResponseToLLM(resp)
		if err != nil {
			yield(nil, err)
			return
		}
		yield(llmResp, nil)
	}
}

func (m *openAILLM) chatCompletionParams(req *model.LLMRequest) (*openai.ChatCompletionNewParams, error) {
	msgs, err := genaiToOpenAIMessages(req.Contents)
	if err != nil {
		return nil, err
	}
	if len(msgs) == 0 {
		return nil, fmt.Errorf("no messages to send")
	}

	params := openai.ChatCompletionNewParams{
		Model:    resolveModelName(req, m.name),
		Messages: msgs,
	}

	cfg := req.Config
	if cfg.Temperature != nil {
		params.Temperature = openai.Float(float64(*cfg.Temperature))
	}
	if cfg.TopP != nil {
		params.TopP = openai.Float(float64(*cfg.TopP))
	}
	if cfg.MaxOutputTokens > 0 {
		params.MaxTokens = openai.Int(int64(cfg.MaxOutputTokens))
		params.MaxCompletionTokens = openai.Int(int64(cfg.MaxOutputTokens))
	}
	if cfg.CandidateCount > 0 {
		params.N = openai.Int(int64(cfg.CandidateCount))
	}
	if len(cfg.StopSequences) > 0 {
		// OpenAI stop accepts string or []string; we set []string.
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: cfg.StopSequences}
	}
	if cfg.Seed != nil {
		params.Seed = openai.Int(int64(*cfg.Seed))
	}
	switch {
	case cfg.Logprobs != nil:
		params.Logprobs = openai.Bool(true)
		params.TopLogprobs = openai.Int(int64(*cfg.Logprobs))
	case cfg.ResponseLogprobs:
		params.Logprobs = openai.Bool(true)
	}
	if cfg.FrequencyPenalty != nil {
		params.FrequencyPenalty = openai.Float(float64(*cfg.FrequencyPenalty))
	}
	if cfg.PresencePenalty != nil {
		params.PresencePenalty = openai.Float(float64(*cfg.PresencePenalty))
	}

	if len(cfg.Tools) > 0 {
		tools, tc := genaiToolsToOpenAI(cfg.Tools, cfg.ToolConfig)
		params.Tools = tools
		if tc != nil {
			params.ToolChoice = *tc
		}
	}

	return &params, nil
}

func (m *openAILLM) stream(ctx context.Context, params *openai.ChatCompletionNewParams) iter.Seq2[*model.LLMResponse, error] {
	stream := m.client.Chat.Completions.NewStreaming(ctx, *params)
	agg := newOpenAIStreamAggregator()

	return func(yield func(*model.LLMResponse, error) bool) {
		defer stream.Close()

		for stream.Next() {
			chunk := stream.Current()

			for _, resp := range agg.Process(&chunk) {
				if !yield(resp, nil) {
					return
				}
			}
		}

		if err := stream.Err(); err != nil && !errors.Is(err, io.EOF) {
			yield(nil, err)
			return
		}

		if final := agg.Final(); final != nil {
			yield(final, nil)
		}
	}
}

func genaiToOpenAIMessages(contents []*genai.Content) ([]openai.ChatCompletionMessageParamUnion, error) {
	var msgs []openai.ChatCompletionMessageParamUnion
	var text strings.Builder

	for i, c := range contents {
		if c == nil {
			continue
		}

		text.Reset()
		toolCalls := make([]openai.ChatCompletionMessageToolCallUnionParam, 0, len(c.Parts))
		for j, part := range c.Parts {
			if part == nil {
				continue
			}

			switch {
			case part.Text != "":
				text.WriteString(part.Text)

			case part.FunctionCall != nil:
				fc := part.FunctionCall
				if fc.Name == "" {
					return nil, fmt.Errorf("content[%d] part[%d]: function call missing name", i, j)
				}
				argsJSON, err := json.Marshal(fc.Args)
				if err != nil {
					return nil, fmt.Errorf("content[%d] part[%d]: marshal function args: %w", i, j, err)
				}

				toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnionParam{
					OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
						ID:   toolID(fc.ID, i, j),
						Type: constant.ValueOf[constant.Function](),
						Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
							Name:      fc.Name,
							Arguments: string(argsJSON),
						},
					},
				})

			case part.FunctionResponse != nil:
				fr := part.FunctionResponse
				if fr.Name == "" {
					return nil, fmt.Errorf("content[%d] part[%d]: function response missing name", i, j)
				}
				data, err := json.Marshal(fr.Response)
				if err != nil {
					return nil, fmt.Errorf("content[%d] part[%d]: marshal function response: %w", i, j, err)
				}
				msgs = append(msgs,
					openai.ToolMessage(string(data), toolID(fr.ID, i, j)),
				)

			default:
				return nil, fmt.Errorf("content[%d] part[%d]: unsupported part", i, j)
			}
		}

		role := strings.ToLower(strings.TrimSpace(c.Role))
		switch role {
		case genai.RoleUser:
			msgs = append(msgs, openai.UserMessage(text.String()))

		case genai.RoleModel:
			var modelParam openai.ChatCompletionAssistantMessageParam
			if text.Len() > 0 {
				modelParam.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String(text.String()),
				}
			}
			if len(toolCalls) > 0 {
				modelParam.ToolCalls = toolCalls
			}
			// [openai.ChatCompletionAssistantMessageParam.Role] can be elided but just in case
			modelParam.Role = constant.ValueOf[constant.Assistant]()

			msgs = append(msgs,
				openai.ChatCompletionMessageParamUnion{
					OfAssistant: &modelParam,
				},
			)

		default:
			return nil, fmt.Errorf("content[%d]: unsupported role %q", i, role)
		}
	}

	return msgs, nil
}

func genaiToolsToOpenAI(tools []*genai.Tool, cfg *genai.ToolConfig) (params []openai.ChatCompletionToolUnionParam, choiceOpt *openai.ChatCompletionToolChoiceOptionUnionParam) {
	if len(tools) == 0 {
		return nil, nil
	}

	params = make([]openai.ChatCompletionToolUnionParam, 0, len(tools))
	for _, t := range tools {
		for _, decl := range t.FunctionDeclarations {
			if decl == nil || decl.Name == "" {
				continue
			}

			fn := shared.FunctionDefinitionParam{
				Name: decl.Name,
			}
			if desc := strings.TrimSpace(decl.Description); desc != "" {
				fn.Description = openai.String(desc)
			}

			if params, err := toFunctionParameters(decl.ParametersJsonSchema); err == nil && params != nil {
				fn.Parameters = params
			} else if params, err := toFunctionParameters(decl.Parameters); err == nil && params != nil {
				fn.Parameters = params
			}

			params = append(params, openai.ChatCompletionFunctionTool(fn))
		}
	}

	if cfg != nil && cfg.FunctionCallingConfig != nil {
		switch cfg.FunctionCallingConfig.Mode {
		case genai.FunctionCallingConfigModeNone:
			choiceOpt = &openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String("none"),
			}

		case genai.FunctionCallingConfigModeAny:
			if len(cfg.FunctionCallingConfig.AllowedFunctionNames) == 1 {
				choiceOpt = &openai.ChatCompletionToolChoiceOptionUnionParam{
					OfFunctionToolChoice: &openai.ChatCompletionNamedToolChoiceParam{
						Function: openai.ChatCompletionNamedToolChoiceFunctionParam{
							Name: cfg.FunctionCallingConfig.AllowedFunctionNames[0],
						},
						// [openai.ChatCompletionToolChoiceOptionUnionParam.Type] can be elided but just in case
						Type: constant.ValueOf[constant.Function](),
					},
				}
			}
		}
	}

	return params, choiceOpt
}

func toFunctionParameters(src any) (shared.FunctionParameters, error) {
	if src == nil {
		return nil, nil //nolint:nilnil
	}

	raw, err := json.Marshal(src)
	if err != nil {
		return nil, fmt.Errorf("marshal json: %w", err)
	}

	var params map[string]any
	if err := json.Unmarshal(raw, &params); err != nil {
		return nil, fmt.Errorf("unmarshal json: %w", err)
	}

	return params, nil
}

func toolCallsFromMessage(msg *openai.ChatCompletionMessage) []openai.ChatCompletionMessageToolCallUnion {
	if msg == nil {
		return nil
	}

	if len(msg.ToolCalls) > 0 {
		return msg.ToolCalls
	}

	raw := msg.JSON.FunctionCall.Raw()
	if !msg.JSON.FunctionCall.Valid() || raw == "" {
		return nil
	}

	var legacyFunctionCall struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(raw), &legacyFunctionCall); err != nil {
		return nil
	}
	if legacyFunctionCall.Name == "" {
		return nil
	}

	return []openai.ChatCompletionMessageToolCallUnion{
		{
			Type: string(constant.ValueOf[constant.Function]()),
			Function: openai.ChatCompletionMessageFunctionToolCallFunction{
				Name:      legacyFunctionCall.Name,
				Arguments: legacyFunctionCall.Arguments,
			},
		},
	}
}

func openAIResponseToLLM(resp *openai.ChatCompletion) (*model.LLMResponse, error) {
	if resp == nil {
		return nil, fmt.Errorf("nil openai response")
	}
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("empty choices")
	}

	msg := resp.Choices[0].Message
	parts := make([]*genai.Part, 0, 2)
	if msg.Content != "" {
		parts = append(parts, genai.NewPartFromText(msg.Content))
	}

	toolCalls := toolCallsFromMessage(&msg)
	for tc := range slices.Values(toolCalls) {
		var fn openai.ChatCompletionMessageFunctionToolCallFunction
		switch v := tc.AsAny().(type) {
		case openai.ChatCompletionMessageFunctionToolCall:
			fn = v.Function
		default:
			fn = tc.Function
		}
		if fn.Name == "" {
			fn = tc.Function
		}

		parts = append(parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   tc.ID,
				Name: fn.Name,
				Args: parseArgs(fn.Arguments),
			},
		})
	}

	return &model.LLMResponse{
		Content: &genai.Content{
			Parts: parts,
			Role:  genai.RoleModel,
		},
		UsageMetadata: openAIUsage(&resp.Usage),
		FinishReason:  mapOpenAIFinishReason(resp.Choices[0].FinishReason),
	}, nil
}

type toolCallState struct {
	id    string
	name  string
	index int64
	args  strings.Builder
}

type openAIStreamAggregator struct {
	text         strings.Builder
	toolCalls    []*toolCallState
	finishReason string
	usage        *openai.CompletionUsage
}

func newOpenAIStreamAggregator() *openAIStreamAggregator {
	return &openAIStreamAggregator{}
}

func (a *openAIStreamAggregator) Process(chunk *openai.ChatCompletionChunk) []*model.LLMResponse {
	if chunk == nil || len(chunk.Choices) == 0 {
		return nil
	}

	choice := chunk.Choices[0]
	if choice.FinishReason != "" {
		a.finishReason = choice.FinishReason
	}
	if chunk.JSON.Usage.Valid() {
		a.usage = openai.Ptr(chunk.Usage)
	}

	var out []*model.LLMResponse
	delta := choice.Delta

	if delta.Content != "" {
		a.text.WriteString(delta.Content)
		out = append(out, &model.LLMResponse{
			Content: &genai.Content{
				Role:  string(genai.RoleModel),
				Parts: []*genai.Part{genai.NewPartFromText(delta.Content)},
			},
			Partial: true,
		})
	}

	for tc := range slices.Values(delta.ToolCalls) {
		state := a.ensureToolCall(tc.Index, tc.ID)
		if tc.Function.Name != "" {
			state.name = tc.Function.Name
		}
		if tc.Function.Arguments != "" {
			state.args.WriteString(tc.Function.Arguments)
		}
		if tc.ID != "" {
			state.id = tc.ID
		}
	}

	return out
}

func (a *openAIStreamAggregator) Final() *model.LLMResponse {
	if a.text.Len() == 0 && len(a.toolCalls) == 0 && a.finishReason == "" && a.usage == nil {
		return nil
	}

	parts := make([]*genai.Part, 0, 1+len(a.toolCalls))
	if a.text.Len() > 0 {
		parts = append(parts, genai.NewPartFromText(a.text.String()))
	}

	if len(a.toolCalls) > 0 {
		slices.SortFunc(a.toolCalls, func(a, b *toolCallState) int {
			return cmp.Compare(a.index, b.index)
		})

		for _, tc := range a.toolCalls {
			args := parseArgs(tc.args.String())
			parts = append(parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   tc.id,
					Name: tc.name,
					Args: args,
				},
			})
		}
	}

	return &model.LLMResponse{
		Content: &genai.Content{
			Parts: parts,
			Role:  genai.RoleModel,
		},
		UsageMetadata: openAIUsage(a.usage),
		TurnComplete:  a.finishReason != "",
		FinishReason:  mapOpenAIFinishReason(a.finishReason),
	}
}

func (a *openAIStreamAggregator) ensureToolCall(idx int64, id string) *toolCallState {
	for _, tc := range a.toolCalls {
		if tc.index == idx || (id != "" && tc.id == id) {
			return tc
		}
	}

	tc := &toolCallState{
		id:    id,
		index: idx,
	}
	a.toolCalls = append(a.toolCalls, tc)

	return tc
}

func parseArgs(raw string) map[string]any {
	if strings.TrimSpace(raw) == "" {
		return map[string]any{}
	}

	var out map[string]any
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return map[string]any{
			"raw": raw,
		}
	}

	return out
}

func openAIUsage(u *openai.CompletionUsage) *genai.GenerateContentResponseUsageMetadata {
	if u == nil {
		return nil
	}

	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     int32(u.PromptTokens),
		CandidatesTokenCount: int32(u.CompletionTokens),
		TotalTokenCount:      int32(u.TotalTokens),
	}
}

// mapOpenAIFinishReason maps [openai.ChatCompletionChoice.FinishReason] to [genai.FinishReason].
//
// The [openai.ChatCompletionChoice.FinishReason] is any of:
//
//   - stop
//   - length
//   - tool_calls
//   - content_filter
//   - function_call (deprecated)
func mapOpenAIFinishReason(reason string) genai.FinishReason {
	switch strings.ToLower(reason) {
	case "stop":
		return genai.FinishReasonStop
	case "length":
		return genai.FinishReasonMaxTokens
	case "content_filter":
		return genai.FinishReasonSafety
	case "tool_calls", "function_call":
		return genai.FinishReasonOther
	default:
		return genai.FinishReasonUnspecified
	}
}
