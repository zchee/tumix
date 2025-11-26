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

package model

import (
	"context"
	json "encoding/json/v2"
	"fmt"
	"iter"
	"net/http"
	"runtime"
	"slices"
	"strings"

	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/zchee/tumix/internal/version"
	"github.com/zchee/tumix/model/xai"
	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

type xaiModel struct {
	client             *xai.Client
	name               string
	versionHeaderValue string
}

var _ model.LLM = (*xaiModel)(nil)

func NewXAIModel(_ context.Context, modelName string, opts ...xai.ClientOption) (model.LLM, error) {
	client, err := xai.NewClient("", opts...)
	if err != nil {
		return nil, fmt.Errorf("new xAI client: %w", err)
	}

	// Create header value once, when the model is created
	headerValue := fmt.Sprintf("tumix/%s %s", version.Version, strings.TrimPrefix(runtime.Version(), "go"))

	return &xaiModel{
		client:             client,
		name:               modelName,
		versionHeaderValue: headerValue,
	}, nil
}

// Name implements [model.LLM].
func (m *xaiModel) Name() string {
	return m.name
}

// GenerateContent implements [model.LLM].
func (m *xaiModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	m.maybeAppendUserContent(req)
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if req.Config.HTTPOptions == nil {
		req.Config.HTTPOptions = &genai.HTTPOptions{}
	}
	if req.Config.HTTPOptions.Headers == nil {
		req.Config.HTTPOptions.Headers = make(http.Header)
	}
	m.addHeaders(req.Config.HTTPOptions.Headers)

	if stream {
		return m.generateStream(ctx, req)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.generate(ctx, req)
		yield(resp, err)
	}
}

// addHeaders sets the user-agent header.
func (m *xaiModel) addHeaders(headers http.Header) {
	headers.Set("User-Agent", m.versionHeaderValue)
}

// generate calls the model synchronously returning result from the first candidate.
func (m *xaiModel) generate(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	msgs := make([]*xaipb.Message, len(req.Contents))

	var sb strings.Builder
	for i, content := range req.Contents {
		sb.Reset()
		for part := range slices.Values(content.Parts) {
			sb.WriteString(part.Text)
		}

		msgs[i] = &xaipb.Message{
			Content: []*xaipb.Content{
				{
					Content: &xaipb.Content_Text{
						Text: sb.String(),
					},
				},
			},
		}
	}
	options := []xai.ChatOption{
		xai.WithMessages(msgs...),
		genAI2XAIChatOptions(req.Config),
	}
	sess := m.client.Chat.Create(m.name, options...)

	resp, err := sess.Completion(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to call model: %w", err)
	}

	if resp.Content() == "" {
		// shouldn't happen?
		return nil, fmt.Errorf("empty response")
	}

	return xai2LLMResponse(resp), nil
}

// generateStream returns a stream of responses from the model.
func (m *xaiModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	aggregator := NewStreamingResponseAggregator()

	return func(yield func(*model.LLMResponse, error) bool) {
		msgs := make([]*xaipb.Message, len(req.Contents))
		var sb strings.Builder
		for i, content := range req.Contents {
			sb.Reset()
			for part := range slices.Values(content.Parts) {
				sb.WriteString(part.Text)
			}

			msgs[i] = &xaipb.Message{
				Content: []*xaipb.Content{
					{
						Content: &xaipb.Content_Text{
							Text: sb.String(),
						},
					},
				},
			}
		}

		options := []xai.ChatOption{
			xai.WithMessages(msgs...),
			genAI2XAIChatOptions(req.Config),
		}
		sess := m.client.Chat.Create(m.name, options...)

		stream, err := sess.Stream(ctx)
		if err != nil {
			yield(nil, err)
			return
		}
		for resp, err := range stream.Recv() {
			if err != nil {
				yield(nil, err)
				return
			}

			for llmResponse, err := range aggregator.ProcessResponse(ctx, resp) {
				if !yield(llmResponse, err) {
					return // Consumer stopped
				}
			}
		}
		if closeResult := aggregator.Close(); closeResult != nil {
			yield(closeResult, nil)
		}
	}
}

// maybeAppendUserContent appends a user content, so that model can continue to output.
func (m *xaiModel) maybeAppendUserContent(req *model.LLMRequest) {
	if len(req.Contents) == 0 {
		req.Contents = append(req.Contents, genai.NewContentFromText("Handle the requests as specified in the System Instruction.", genai.RoleUser))
	}

	if last := req.Contents[len(req.Contents)-1]; last != nil && last.Role != genai.RoleUser {
		req.Contents = append(req.Contents, genai.NewContentFromText("Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.", genai.RoleUser))
	}
}

func genAI2XAIChatOptions(config *genai.GenerateContentConfig) xai.ChatOption {
	if config == nil {
		return nil
	}

	marshalJSON := func(v any) (string, bool) {
		if v == nil {
			return "", false
		}

		b, err := json.Marshal(v)
		if err != nil {
			return "", false
		}

		return string(b), true
	}

	var (
		temperature      *float32
		topP             *float32
		maxTokens        *int32
		seed             *int32
		stop             []string
		logprobs         bool
		topLogprobs      *int32
		frequencyPenalty *float32
		presencePenalty  *float32
		tools            []*xaipb.Tool
		toolChoice       *xaipb.ToolChoice
		responseFormat   *xaipb.ResponseFormat
	)

	if config.Temperature != nil {
		v := *config.Temperature
		temperature = &v
	}
	if config.TopP != nil {
		v := *config.TopP
		topP = &v
	}
	if config.MaxOutputTokens > 0 {
		v := config.MaxOutputTokens
		maxTokens = &v
	}
	if config.Seed != nil {
		v := *config.Seed
		seed = &v
	}
	if len(config.StopSequences) > 0 {
		stop = slices.Clone(config.StopSequences)
	}
	if config.Logprobs != nil {
		v := *config.Logprobs
		topLogprobs = &v
	}
	logprobs = config.ResponseLogprobs

	if config.FrequencyPenalty != nil {
		v := *config.FrequencyPenalty
		frequencyPenalty = &v
	}
	if config.PresencePenalty != nil {
		v := *config.PresencePenalty
		presencePenalty = &v
	}

	for tool := range slices.Values(config.Tools) {
		if tool == nil {
			continue
		}

		if len(tool.FunctionDeclarations) > 0 {
			for decl := range slices.Values(tool.FunctionDeclarations) {
				if decl == nil || decl.Name == "" {
					continue
				}

				params := ""
				switch {
				case decl.ParametersJsonSchema != nil:
					if raw, ok := marshalJSON(decl.ParametersJsonSchema); ok {
						params = raw
					}
				case decl.Parameters != nil:
					if raw, ok := marshalJSON(decl.Parameters); ok {
						params = raw
					}
				}

				tools = append(tools, &xaipb.Tool{
					Tool: &xaipb.Tool_Function{
						Function: &xaipb.Function{
							Name:        decl.Name,
							Description: decl.Description,
							Parameters:  params,
						},
					},
				})
			}
		}
		if tool.CodeExecution != nil {
			tools = append(tools, xai.CodeExecutionTool())
		}
	}

	if tc := config.ToolConfig; tc != nil && tc.FunctionCallingConfig != nil {
		fc := tc.FunctionCallingConfig
		switch fc.Mode {
		case genai.FunctionCallingConfigModeNone:
			toolChoice = &xaipb.ToolChoice{ToolChoice: &xaipb.ToolChoice_Mode{Mode: xaipb.ToolMode_TOOL_MODE_NONE}}
		case genai.FunctionCallingConfigModeAny:
			if len(fc.AllowedFunctionNames) == 1 && fc.AllowedFunctionNames[0] != "" {
				toolChoice = &xaipb.ToolChoice{ToolChoice: &xaipb.ToolChoice_FunctionName{FunctionName: fc.AllowedFunctionNames[0]}}
			} else {
				toolChoice = &xaipb.ToolChoice{ToolChoice: &xaipb.ToolChoice_Mode{Mode: xaipb.ToolMode_TOOL_MODE_REQUIRED}}
			}
		case genai.FunctionCallingConfigModeAuto:
			toolChoice = &xaipb.ToolChoice{ToolChoice: &xaipb.ToolChoice_Mode{Mode: xaipb.ToolMode_TOOL_MODE_AUTO}}
		}
	}

	switch {
	case config.ResponseJsonSchema != nil:
		if schema, ok := marshalJSON(config.ResponseJsonSchema); ok {
			responseFormat = &xaipb.ResponseFormat{
				FormatType: xaipb.FormatType_FORMAT_TYPE_JSON_SCHEMA,
				Schema:     &schema,
			}
		}
	case config.ResponseSchema != nil:
		if schema, ok := marshalJSON(config.ResponseSchema); ok {
			responseFormat = &xaipb.ResponseFormat{
				FormatType: xaipb.FormatType_FORMAT_TYPE_JSON_SCHEMA,
				Schema:     &schema,
			}
		}
	default:
		mime := strings.ToLower(strings.TrimSpace(config.ResponseMIMEType))
		if strings.HasPrefix(mime, "application/json") {
			responseFormat = &xaipb.ResponseFormat{FormatType: xaipb.FormatType_FORMAT_TYPE_JSON_OBJECT}
		}
	}

	if logprobs || topLogprobs != nil {
		logprobs = true
	}

	hasEffect := temperature != nil || topP != nil || maxTokens != nil || seed != nil || len(stop) > 0 || logprobs || topLogprobs != nil || frequencyPenalty != nil || presencePenalty != nil || len(tools) > 0 || toolChoice != nil || responseFormat != nil
	if !hasEffect {
		return nil
	}

	opt := xai.ChatOption(func(req *xaipb.GetCompletionsRequest, _ *xai.ChatSession) {
		if maxTokens != nil {
			req.MaxTokens = maxTokens
		}
		if seed != nil {
			req.Seed = seed
		}
		if temperature != nil {
			req.Temperature = temperature
		}
		if topP != nil {
			req.TopP = topP
		}
		if len(stop) > 0 {
			req.Stop = append(req.Stop, stop...)
		}
		if logprobs {
			req.Logprobs = true
		}
		if topLogprobs != nil {
			req.TopLogprobs = topLogprobs
		}
		if frequencyPenalty != nil {
			req.FrequencyPenalty = frequencyPenalty
		}
		if presencePenalty != nil {
			req.PresencePenalty = presencePenalty
		}
		if len(tools) > 0 {
			req.Tools = append(req.Tools, tools...)
		}
		if toolChoice != nil {
			req.ToolChoice = toolChoice
		}
		if responseFormat != nil {
			req.ResponseFormat = responseFormat
		}
	})

	return opt
}

func xai2LLMResponse(resp *xai.Response) *model.LLMResponse {
	if resp == nil {
		return &model.LLMResponse{
			ErrorCode:    "NIL_RESPONSE",
			ErrorMessage: "xAI response is nil",
		}
	}

	usage := resp.Usage()
	var usageMetadata *genai.GenerateContentResponseUsageMetadata
	if usage != nil {
		usageMetadata = &genai.GenerateContentResponseUsageMetadata{
			CachedContentTokenCount: usage.GetCachedPromptTextTokens(),
			CandidatesTokenCount:    usage.GetCompletionTokens(),
			PromptTokenCount:        usage.GetPromptTokens(),
			ThoughtsTokenCount:      usage.GetReasoningTokens(),
			TotalTokenCount:         usage.GetTotalTokens(),
		}
	}

	parts := make([]*genai.Part, 0, 3)

	if reasoning := resp.ReasoningContent(); reasoning != "" {
		parts = append(parts, &genai.Part{Text: reasoning, Thought: true})
	}

	if content := resp.Content(); content != "" {
		parts = append(parts, genai.NewPartFromText(content))
	}

	var argErrors []string
	if toolCalls := resp.ToolCalls(); len(toolCalls) > 0 { //nolint:nestif // TODO(zchee): fix nolint
		for _, call := range toolCalls {
			fc := call.GetFunction()
			if fc == nil {
				continue
			}

			args := map[string]any{}
			rawArgs := fc.GetArguments()
			if rawArgs != "" {
				var obj map[string]any
				if err := json.Unmarshal([]byte(rawArgs), &obj); err == nil {
					args = obj
				} else {
					var generic any
					if err2 := json.Unmarshal([]byte(rawArgs), &generic); err2 == nil {
						args = map[string]any{"value": generic}
					} else {
						args["raw"] = rawArgs
						argErrors = append(argErrors, err.Error())
					}
				}
			}

			parts = append(parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   call.GetId(),
					Name: fc.GetName(),
					Args: args,
				},
			})
		}
	}

	role := strings.ToLower(strings.TrimPrefix(resp.Role(), "ROLE_"))
	switch role {
	case "user":
		role = string(genai.RoleUser)
	default:
		role = string(genai.RoleModel)
	}

	finishReason := mapXAIFinishReason(resp.FinishReason())

	custom := map[string]any{}
	if raw := resp.FinishReason(); raw != "" {
		custom["xai_finish_reason"] = raw
	}
	if fp := resp.SystemFingerprint(); fp != "" {
		custom["xai_system_fingerprint"] = fp
	}
	if citations := resp.Citations(); len(citations) > 0 {
		custom["xai_citations"] = slices.Clone(citations)
	}
	if len(argErrors) > 0 {
		custom["tool_call_args_errors"] = slices.Clone(argErrors)
	}
	if len(custom) == 0 {
		custom = nil
	}

	return &model.LLMResponse{
		Content: &genai.Content{
			Role:  role,
			Parts: parts,
		},
		CustomMetadata: custom,
		UsageMetadata:  usageMetadata,
		FinishReason:   finishReason,
	}
}

func mapXAIFinishReason(fr string) genai.FinishReason {
	switch strings.TrimPrefix(strings.ToUpper(fr), "REASON_") {
	case "", "INVALID":
		return genai.FinishReasonUnspecified
	case "STOP":
		return genai.FinishReasonStop
	case "MAX_LEN", "MAX_CONTEXT":
		return genai.FinishReasonMaxTokens
	case "TOOL_CALLS":
		return genai.FinishReasonOther
	case "TIME_LIMIT":
		return genai.FinishReasonOther
	default:
		return genai.FinishReasonOther
	}
}
