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

package xai

import (
	"strconv"
	"strings"

	"github.com/bytedance/sonic"
	"go.opentelemetry.io/otel/attribute"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

//nolint:cyclop,gocyclo,gocognit // TODO(zchee): fix nolint
func (s *ChatSession) makeSpanRequestAttributes() []attribute.KeyValue {
	msgs := s.request.GetMessages()
	attrs := make([]attribute.KeyValue, 0, 18+len(msgs)*6)

	attrs = append(attrs,
		attribute.String("gen_ai.operation.name", "chat"),
		attribute.String("gen_ai.system", "xai"),
		attribute.String("gen_ai.output.type", "text"),
		attribute.String("gen_ai.request.model", s.request.GetModel()),
		attribute.Int("server.port", 443),
		attribute.Float64("gen_ai.request.frequency_penalty", float64(s.request.GetFrequencyPenalty())),
		attribute.Float64("gen_ai.request.presence_penalty", float64(s.request.GetPresencePenalty())),
		attribute.Float64("gen_ai.request.temperature", float64(s.request.GetTemperature())),
		attribute.Bool("gen_ai.request.parallel_tool_calls", s.request.GetParallelToolCalls()),
		attribute.Bool("gen_ai.request.store_messages", s.request.GetStoreMessages()),
		attribute.Bool("gen_ai.request.use_encrypted_content", s.request.GetUseEncryptedContent()),
		attribute.Bool("gen_ai.request.logprobs", s.request.GetLogprobs()),
	)

	if s.request.TopP != nil {
		attrs = append(attrs, attribute.Float64("gen_ai.request.top_p", float64(s.request.GetTopP())))
	}
	if s.request.N != nil {
		attrs = append(attrs, attribute.Int("gen_ai.request.choice.count", int(s.request.GetN())))
	}
	if s.request.Seed != nil {
		attrs = append(attrs, attribute.Int("gen_ai.request.seed", int(s.request.GetSeed())))
	}
	if s.request.MaxTokens != nil {
		attrs = append(attrs, attribute.Int("gen_ai.request.max_tokens", int(s.request.GetMaxTokens())))
	}
	if s.request.TopLogprobs != nil {
		attrs = append(attrs, attribute.Int("gen_ai.request.top_logprobs", int(s.request.GetTopLogprobs())))
	}

	if s.conversationID != "" {
		attrs = append(attrs, attribute.String("gen_ai.conversation.id", s.conversationID))
	}
	if stops := s.request.GetStop(); len(stops) > 0 {
		attrs = append(attrs, attribute.StringSlice("gen_ai.request.stop_sequences", stops))
	}
	if rf := s.request.GetResponseFormat(); rf != nil {
		attrs = append(attrs, attribute.String("gen_ai.output.type", formatTypeLower(rf.GetFormatType())))
	}
	if re := s.request.ReasoningEffort; re != nil {
		attrs = append(attrs, attribute.String("gen_ai.request.reasoning_effort", reasoningEffortLower(*re)))
	}
	if user := s.request.GetUser(); user != "" {
		attrs = append(attrs, attribute.String("user_id", user))
	}
	if prev := s.request.PreviousResponseId; prev != nil {
		attrs = append(attrs, attribute.String("gen_ai.request.previous_response_id", s.request.GetPreviousResponseId()))
	}

	var contentBuf strings.Builder
	for i, msg := range msgs {
		prefix := "gen_ai.prompt." + strconv.Itoa(i)
		role := messageRoleLower(msg.GetRole())
		attrs = append(attrs, attribute.String(prefix+".role", role))

		contentBuf.Reset()
		if parts := msg.GetContent(); len(parts) > 0 {
			total := 0
			for _, c := range parts {
				total += len(c.GetText())
			}
			if total > 0 {
				contentBuf.Grow(total)
				for _, c := range parts {
					if txt := c.GetText(); txt != "" {
						contentBuf.WriteString(txt)
					}
				}
			}
		}
		attrs = append(attrs, attribute.String(prefix+".content", contentBuf.String()))

		if msg.GetRole() == xaipb.MessageRole_ROLE_ASSISTANT {
			if tcs := msg.GetToolCalls(); len(tcs) > 0 {
				if encoded := encodeToolCalls(tcs); encoded != "" {
					attrs = append(attrs, attribute.String(prefix+".tool_calls", encoded))
				}
			}
			if enc := msg.GetEncryptedContent(); enc != "" {
				attrs = append(attrs, attribute.String(prefix+".encrypted_content", enc))
			}
		}
	}

	return attrs
}

func (s *ChatSession) makeSpanResponseAttributes(responses []*Response) []attribute.KeyValue {
	if len(responses) == 0 {
		return nil
	}

	first := responses[0]
	usage := first.Usage()

	toolCallAttrs := 0
	for _, resp := range responses {
		if len(resp.ToolCalls()) > 0 {
			toolCallAttrs++
		}
	}
	attrs := make([]attribute.KeyValue, 0, 12+len(responses)*5+toolCallAttrs)
	attrs = append(attrs,
		attribute.String("gen_ai.response.id", first.proto.GetId()),
		attribute.String("gen_ai.response.model", first.proto.GetModel()),
		attribute.String("gen_ai.response.system_fingerprint", first.proto.GetSystemFingerprint()),
	)

	if usage != nil {
		attrs = append(attrs,
			attribute.Int("gen_ai.usage.input_tokens", int(usage.GetPromptTokens())),
			attribute.Int("gen_ai.usage.output_tokens", int(usage.GetCompletionTokens())),
			attribute.Int("gen_ai.usage.total_tokens", int(usage.GetTotalTokens())),
			attribute.Int("gen_ai.usage.reasoning_tokens", int(usage.GetReasoningTokens())),
		)
	}

	finishReasons := make([]string, len(responses))
	for i, resp := range responses {
		out := resp.outputNoFlush()
		if out == nil {
			continue
		}

		msg := out.GetMessage()
		finishReasons[i] = finishReasonLower(out.GetFinishReason())

		prefix := "gen_ai.completion." + strconv.Itoa(i)
		role := messageRoleLower(msg.GetRole())
		attrs = append(attrs,
			attribute.String(prefix+".role", role),
			attribute.String(prefix+".content", msg.GetContent()),
		)

		if rc := msg.GetReasoningContent(); rc != "" {
			attrs = append(attrs, attribute.String(prefix+".reasoning_content", rc))
		}

		if tcs := msg.GetToolCalls(); len(tcs) > 0 {
			if encoded := encodeToolCalls(tcs); encoded != "" {
				attrs = append(attrs, attribute.String(prefix+".tool_calls", encoded))
			}
		}
	}
	attrs = append(attrs, attribute.StringSlice("gen_ai.response.finish_reasons", finishReasons))

	return attrs
}

func messageRoleLower(role xaipb.MessageRole) string {
	if int(role) < len(messageRoleStrings) {
		return messageRoleStrings[role]
	}

	return strings.ToLower(strings.TrimPrefix(role.String(), "ROLE_"))
}

func finishReasonLower(reason xaipb.FinishReason) string {
	if int(reason) < len(finishReasonStrings) {
		return finishReasonStrings[reason]
	}

	return strings.ToLower(strings.TrimPrefix(reason.String(), "REASON_"))
}

func formatTypeLower(format xaipb.FormatType) string {
	if int(format) < len(formatTypeStrings) {
		return formatTypeStrings[format]
	}

	return strings.ToLower(strings.TrimPrefix(format.String(), "FORMAT_TYPE_"))
}

func reasoningEffortLower(effort xaipb.ReasoningEffort) string {
	if int(effort) < len(reasoningEffortStrings) {
		return reasoningEffortStrings[effort]
	}

	return strings.ToLower(strings.TrimPrefix(effort.String(), "EFFORT_"))
}

var (
	messageRoleStrings = [...]string{
		xaipb.MessageRole_INVALID_ROLE:   "invalid_role",
		xaipb.MessageRole_ROLE_USER:      "user",
		xaipb.MessageRole_ROLE_ASSISTANT: "assistant",
		xaipb.MessageRole_ROLE_SYSTEM:    "system",
		xaipb.MessageRole_ROLE_TOOL:      "tool",
	}

	finishReasonStrings = [...]string{
		xaipb.FinishReason_REASON_INVALID:     "reason_invalid",
		xaipb.FinishReason_REASON_MAX_LEN:     "reason_max_len",
		xaipb.FinishReason_REASON_MAX_CONTEXT: "reason_max_context",
		xaipb.FinishReason_REASON_STOP:        "reason_stop",
		xaipb.FinishReason_REASON_TOOL_CALLS:  "reason_tool_calls",
		xaipb.FinishReason_REASON_TIME_LIMIT:  "reason_time_limit",
	}

	formatTypeStrings = [...]string{
		xaipb.FormatType_FORMAT_TYPE_INVALID:     "format_type_invalid",
		xaipb.FormatType_FORMAT_TYPE_TEXT:        "format_type_text",
		xaipb.FormatType_FORMAT_TYPE_JSON_OBJECT: "format_type_json_object",
		xaipb.FormatType_FORMAT_TYPE_JSON_SCHEMA: "format_type_json_schema",
	}

	reasoningEffortStrings = [...]string{
		xaipb.ReasoningEffort_INVALID_EFFORT: "invalid_effort",
		xaipb.ReasoningEffort_EFFORT_LOW:     "effort_low",
		xaipb.ReasoningEffort_EFFORT_MEDIUM:  "effort_medium",
		xaipb.ReasoningEffort_EFFORT_HIGH:    "effort_high",
	}
)

func encodeToolCalls(tc []*xaipb.ToolCall) string {
	if len(tc) == 0 {
		return ""
	}

	data, err := sonic.ConfigFastest.MarshalToString(tc)
	if err != nil {
		return ""
	}

	return data
}
