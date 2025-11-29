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
	"context"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"slices"
	"strings"
	"time"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/zchee/tumix/internal/version"
)

// anthropicLLM implements the adk [model.LLM] interface using Anthropics SDK.
type anthropicLLM struct {
	client    *anthropic.Client
	name      string
	userAgent string
}

var _ model.LLM = (*anthropicLLM)(nil)

// NewAnthropicLLM creates a new Anthropics-backed LLM.
//
// If authKey is nil, the Anthropics SDK will fall back to ANTHROPIC_API_KEY environment variable.
//
//nolint:unparam
func NewAnthropicLLM(_ context.Context, authKey AuthMethod, modelName string, opts ...option.RequestOption) (model.LLM, error) {
	userAgent := version.UserAgent("anthropic")

	// TODO(zchee): Ues [option.WithHTTPClient] with OTel tracing transport
	ropts := []option.RequestOption{
		option.WithHeader("User-Agent", userAgent),
		option.WithMaxRetries(2),
		option.WithRequestTimeout(3 * time.Minute),
	}
	switch authKey := authKey.(type) {
	case AuthMethodAPIKey:
		ropts = append(ropts, option.WithAPIKey(authKey.value()))
	case AuthMethodAPIToken:
		ropts = append(ropts, option.WithAuthToken(authKey.value()))
	}

	// opts are allowed to override by order
	opts = append(ropts, opts...)
	client := anthropic.NewClient(opts...)

	return &anthropicLLM{
		client:    &client,
		name:      modelName,
		userAgent: userAgent,
	}, nil
}

// Name implements [model.LLM].
func (m *anthropicLLM) Name() string { return m.name }

// GenerateContent implements [model.LLM].
func (m *anthropicLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	m.ensureUserContent(req)
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if req.Config.HTTPOptions == nil {
		req.Config.HTTPOptions = &genai.HTTPOptions{}
	}
	if req.Config.HTTPOptions.Headers == nil {
		req.Config.HTTPOptions.Headers = make(http.Header)
	}
	// Keep the same user-agent used during client construction to satisfy ADK expectations.
	req.Config.HTTPOptions.Headers.Set("User-Agent", m.userAgent)

	system, msgs, err := genaiToAnthropicMessages(req.Config.SystemInstruction, req.Contents)
	if err != nil {
		return func(yield func(*model.LLMResponse, error) bool) {
			yield(nil, err)
		}
	}

	params, err := m.buildParams(req, system, msgs)
	if err != nil {
		return func(yield func(*model.LLMResponse, error) bool) {
			yield(nil, err)
		}
	}

	if stream {
		return m.stream(ctx, params)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.client.Messages.New(ctx, *params)
		if err != nil {
			yield(nil, err)
			return
		}
		llmResp, convErr := anthropicMessageToLLMResponse(resp)
		if convErr != nil {
			yield(nil, convErr)
			return
		}
		yield(llmResp, nil)
	}
}

func (m *anthropicLLM) buildParams(req *model.LLMRequest, system []anthropic.TextBlockParam, msgs []anthropic.MessageParam) (*anthropic.MessageNewParams, error) {
	if len(msgs) == 0 {
		return nil, errors.New("no messages")
	}

	params := &anthropic.MessageNewParams{
		Model:         anthropic.Model(m.modelName(req)),
		Messages:      msgs,
		System:        system,
		MaxTokens:     int64(req.Config.MaxOutputTokens),
		StopSequences: req.Config.StopSequences,
	}

	if params.MaxTokens == 0 {
		// Anthropic requires max_tokens; fall back to a conservative default.
		params.MaxTokens = 1024
	}
	if req.Config.Temperature != nil {
		params.Temperature = param.NewOpt(float64(*req.Config.Temperature))
	}
	if req.Config.TopP != nil {
		params.TopP = param.NewOpt(float64(*req.Config.TopP))
	}
	if req.Config.TopK != nil {
		params.TopK = param.NewOpt(int64(*req.Config.TopK))
	}
	if len(req.Config.Tools) > 0 {
		tools, tc := genaiToolsToAnthropic(req.Config.Tools, req.Config.ToolConfig)
		params.Tools = tools
		if tc != nil {
			params.ToolChoice = *tc
		}
	}

	return params, nil
}

func (m *anthropicLLM) stream(ctx context.Context, params *anthropic.MessageNewParams) iter.Seq2[*model.LLMResponse, error] {
	stream := m.client.Messages.NewStreaming(ctx, *params)
	acc := &anthropic.Message{}

	return func(yield func(*model.LLMResponse, error) bool) {
		defer stream.Close()

		for stream.Next() {
			event := stream.Current()
			if err := acc.Accumulate(event); err != nil {
				yield(nil, err)
				return
			}

			switch ev := event.AsAny().(type) {
			case anthropic.ContentBlockDeltaEvent:
				if delta := ev.Delta.AsAny(); delta != nil {
					if t, ok := delta.(anthropic.TextDelta); ok && t.Text != "" {
						if !yield(&model.LLMResponse{
							Content: &genai.Content{
								Role:  string(genai.RoleModel),
								Parts: []*genai.Part{genai.NewPartFromText(accText(acc))},
							},
							Partial: true,
						}, nil) {
							return
						}
					}
				}
			case anthropic.MessageStopEvent:
				resp, err := anthropicMessageToLLMResponse(acc)
				if err != nil {
					yield(nil, err)
					return
				}
				if !yield(resp, nil) {
					return
				}
			}
		}

		if err := stream.Err(); err != nil {
			yield(nil, err)
		}
	}
}

func accText(msg *anthropic.Message) string {
	if msg == nil {
		return ""
	}
	var sb strings.Builder
	for block := range slices.Values(msg.Content) {
		if v, ok := block.AsAny().(anthropic.TextBlock); ok {
			sb.WriteString(v.Text)
		}
	}
	return sb.String()
}

func (m *anthropicLLM) modelName(req *model.LLMRequest) string {
	if req != nil && strings.TrimSpace(req.Model) != "" {
		return strings.TrimSpace(req.Model)
	}
	return m.name
}

// Convert a non-streaming Anthropics message to an ADK response.
func anthropicMessageToLLMResponse(msg *anthropic.Message) (*model.LLMResponse, error) {
	if msg == nil {
		return nil, errors.New("nil anthropic message")
	}

	parts := make([]*genai.Part, 0, len(msg.Content))
	for block := range slices.Values(msg.Content) {
		switch v := block.AsAny().(type) {
		case anthropic.TextBlock:
			parts = append(parts, genai.NewPartFromText(v.Text))
		case anthropic.ToolUseBlock:
			args := map[string]any{}
			if len(v.Input) > 0 {
				if err := json.Unmarshal(v.Input, &args); err != nil {
					return nil, fmt.Errorf("unmarshal json: %w", err)
				}
			}
			parts = append(parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   v.ID,
					Name: v.Name,
					Args: args,
				},
			})
		}
	}

	usage := msg.Usage
	llmUsage := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     int32(usage.InputTokens),
		CandidatesTokenCount: int32(usage.OutputTokens),
		TotalTokenCount:      int32(usage.InputTokens + usage.OutputTokens),
	}

	return &model.LLMResponse{
		Content: &genai.Content{
			Role:  string(genai.RoleModel),
			Parts: parts,
		},
		UsageMetadata: llmUsage,
		FinishReason:  mapAnthropicFinishReason(msg.StopReason),
	}, nil
}

func mapAnthropicFinishReason(reason anthropic.StopReason) genai.FinishReason {
	switch reason {
	case anthropic.StopReasonStopSequence, anthropic.StopReasonEndTurn:
		return genai.FinishReasonStop
	case anthropic.StopReasonMaxTokens:
		return genai.FinishReasonMaxTokens
	case anthropic.StopReasonToolUse:
		return genai.FinishReasonOther
	default:
		return genai.FinishReasonUnspecified
	}
}

// Convert ADK genai content to Anthropics message params.
func genaiToAnthropicMessages(system *genai.Content, contents []*genai.Content) ([]anthropic.TextBlockParam, []anthropic.MessageParam, error) {
	var systemBlocks []anthropic.TextBlockParam
	if system != nil {
		text := joinTextParts(system.Parts)
		if text != "" {
			systemBlocks = append(systemBlocks, anthropic.TextBlockParam{
				Type: constant.ValueOf[constant.Text](),
				Text: text,
			})
		}
	}

	msgs := make([]anthropic.MessageParam, 0, len(contents))
	for idx, c := range contents {
		if c == nil {
			continue
		}
		role := strings.ToLower(c.Role)
		mp := anthropic.MessageParam{
			Content: make([]anthropic.ContentBlockParamUnion, 0, len(c.Parts)),
		}
		if role == genai.RoleUser {
			mp.Role = anthropic.MessageParamRoleUser
		} else {
			// treat everything else as assistant to satisfy API constraint (only user/assistant allowed).
			mp.Role = anthropic.MessageParamRoleAssistant
		}

		for pi, part := range c.Parts {
			if part == nil {
				continue
			}
			switch {
			case part.Text != "":
				mp.Content = append(mp.Content, anthropic.NewTextBlock(part.Text))

			case part.FunctionCall != nil:
				fc := part.FunctionCall
				if fc.Name == "" {
					return nil, nil, fmt.Errorf("content[%d] part[%d]: function call missing name", idx, pi)
				}
				args := fc.Args
				if args == nil {
					args = map[string]any{}
				}
				mp.Content = append(mp.Content, anthropic.ContentBlockParamUnion{
					OfToolUse: &anthropic.ToolUseBlockParam{
						ID:    toolID(fc.ID, idx, pi),
						Name:  fc.Name,
						Input: args,
						Type:  constant.ValueOf[constant.ToolUse](),
					},
				})

			case part.FunctionResponse != nil:
				fr := part.FunctionResponse
				if fr.Name == "" {
					return nil, nil, fmt.Errorf("content[%d] part[%d]: function response missing name", idx, pi)
				}
				contentJSON, err := json.Marshal(fr.Response)
				if err != nil {
					return nil, nil, fmt.Errorf("marshal json: %w", err)
				}
				mp.Content = append(mp.Content, anthropic.ContentBlockParamUnion{
					OfToolResult: &anthropic.ToolResultBlockParam{
						ToolUseID: toolID(fr.ID, idx, pi),
						Content: []anthropic.ToolResultBlockParamContentUnion{
							{OfText: &anthropic.TextBlockParam{Type: constant.ValueOf[constant.Text](), Text: string(contentJSON)}},
						},
						Type: constant.ValueOf[constant.ToolResult](),
					},
				})

			default:
				return nil, nil, fmt.Errorf("content[%d] part[%d]: unsupported part", idx, pi)
			}
		}

		if len(mp.Content) == 0 {
			return nil, nil, fmt.Errorf("content[%d]: empty parts", idx)
		}
		msgs = append(msgs, mp)
	}

	return systemBlocks, msgs, nil
}

func joinTextParts(parts []*genai.Part) string {
	var sb strings.Builder
	for _, p := range parts {
		if p == nil {
			continue
		}
		sb.WriteString(p.Text)
	}
	return sb.String()
}

func toolID(id string, contentIdx, partIdx int) string {
	if strings.TrimSpace(id) != "" {
		return id
	}
	return fmt.Sprintf("tool_%d_%d", contentIdx, partIdx)
}

// Convert ADK tool declarations to Anthropics definitions.
func genaiToolsToAnthropic(tools []*genai.Tool, cfg *genai.ToolConfig) ([]anthropic.ToolUnionParam, *anthropic.ToolChoiceUnionParam) {
	if len(tools) == 0 {
		return nil, nil
	}

	out := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		for _, decl := range t.FunctionDeclarations {
			if decl == nil || decl.Name == "" {
				continue
			}
			out = append(out, anthropic.ToolUnionParam{
				OfTool: &anthropic.ToolParam{
					Name:        decl.Name,
					Description: param.NewOpt(decl.Description),
					InputSchema: anthropic.ToolInputSchemaParam{
						Type:       constant.ValueOf[constant.Object](),
						Properties: decl.Parameters,
					},
					Type: anthropic.ToolTypeCustom,
				},
			})
		}
	}

	var tc *anthropic.ToolChoiceUnionParam
	if cfg != nil && cfg.FunctionCallingConfig != nil {
		switch cfg.FunctionCallingConfig.Mode {
		case genai.FunctionCallingConfigModeNone:
			none := anthropic.NewToolChoiceNoneParam()
			tc = &anthropic.ToolChoiceUnionParam{OfNone: &none}
		case genai.FunctionCallingConfigModeAny, genai.FunctionCallingConfigModeAuto:
			tc = &anthropic.ToolChoiceUnionParam{OfAuto: &anthropic.ToolChoiceAutoParam{Type: constant.ValueOf[constant.Auto]()}}
		}
	}

	return out, tc
}

// ensureUserContent aligns with ADK behavior of ending with a user turn.
func (m *anthropicLLM) ensureUserContent(req *model.LLMRequest) {
	if len(req.Contents) == 0 {
		req.Contents = append(req.Contents, genai.NewContentFromText("Handle the requests as specified in the System Instruction.", genai.RoleUser))
		return
	}

	if last := req.Contents[len(req.Contents)-1]; last != nil && last.Role != genai.RoleUser {
		req.Contents = append(req.Contents, genai.NewContentFromText("Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.", genai.RoleUser))
	}
}
