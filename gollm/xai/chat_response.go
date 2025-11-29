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
	json "encoding/json/v2"
	"slices"
	"strings"
	"sync"

	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
)

// Response wraps GetChatCompletionResponse with convenience accessors.
type Response struct {
	proto             *xaipb.GetChatCompletionResponse
	index             *int32
	contentBuffers    []*strings.Builder
	reasoningBuffers  []*strings.Builder
	encryptedBuffers  []*strings.Builder
	toolCallScratch   [][]*xaipb.ToolCall
	buffersAreInProto bool
}

func newResponse(protoResp *xaipb.GetChatCompletionResponse, index *int32) *Response {
	return &Response{
		proto:             protoResp,
		index:             index,
		contentBuffers:    nil,
		reasoningBuffers:  nil,
		encryptedBuffers:  nil,
		toolCallScratch:   nil,
		buffersAreInProto: true,
	}
}

// Proto returns the underlying protobuf message (materializing buffered chunks).
func (r *Response) Proto() *xaipb.GetChatCompletionResponse {
	r.flushBuffers()
	return r.proto
}

// Content returns the content string for the selected output(s).
func (r *Response) Content() string {
	r.flushBuffers()
	if out := r.output(); out != nil {
		return out.GetMessage().GetContent()
	}

	return ""
}

// DecodeJSON unmarshals the response content into the provided destination.
// Useful when using structured outputs or JSON response_format.
func (r *Response) DecodeJSON(out any) error {
	return json.Unmarshal([]byte(r.Content()), out)
}

// ReasoningContent returns any reasoning trace text.
func (r *Response) ReasoningContent() string {
	r.flushBuffers()
	if out := r.output(); out != nil {
		return out.GetMessage().GetReasoningContent()
	}

	return ""
}

// EncryptedContent returns encrypted reasoning content when present.
func (r *Response) EncryptedContent() string {
	r.flushBuffers()
	if out := r.output(); out != nil {
		return out.GetMessage().GetEncryptedContent()
	}

	return ""
}

// Role returns the assistant role string.
func (r *Response) Role() string {
	r.flushBuffers()
	if out := r.output(); out != nil {
		return xaipb.MessageRole_name[int32(out.GetMessage().GetRole())]
	}

	return ""
}

// ToolCalls returns tool calls from all assistant outputs.
func (r *Response) ToolCalls() []*xaipb.ToolCall {
	r.flushBuffers()
	outputs := r.proto.GetOutputs()
	if len(outputs) == 0 {
		return nil
	}

	total := 0
	for out := range slices.Values(outputs) {
		msg := out.GetMessage()
		if msg == nil || msg.GetRole() != xaipb.MessageRole_ROLE_ASSISTANT {
			continue
		}
		total += len(msg.GetToolCalls())
	}

	if total == 0 {
		return nil
	}

	calls := make([]*xaipb.ToolCall, 0, total)
	for out := range slices.Values(outputs) {
		msg := out.GetMessage()
		if msg == nil || msg.GetRole() != xaipb.MessageRole_ROLE_ASSISTANT {
			continue
		}
		calls = append(calls, msg.GetToolCalls()...)
	}

	return calls
}

// FinishReason returns the finish reason string.
func (r *Response) FinishReason() string {
	if out := r.outputNoFlush(); out != nil {
		return xaipb.FinishReason_name[int32(out.GetFinishReason())]
	}
	return ""
}

// Usage returns token usage.
func (r *Response) Usage() *xaipb.SamplingUsage {
	return r.proto.GetUsage()
}

// Citations returns any citations returned by the model.
func (r *Response) Citations() []string {
	return r.proto.GetCitations()
}

// SystemFingerprint returns system fingerprint.
func (r *Response) SystemFingerprint() string {
	return r.proto.GetSystemFingerprint()
}

func (r *Response) output() *xaipb.CompletionOutput {
	r.flushBuffers()
	return r.outputNoFlush()
}

func (r *Response) outputNoFlush() *xaipb.CompletionOutput {
	var last *xaipb.CompletionOutput
	idx, hasIdx := deref(r.index), r.index != nil
	for out := range slices.Values(r.proto.GetOutputs()) {
		if out == nil || out.GetMessage() == nil {
			continue
		}
		if out.GetMessage().GetRole() != xaipb.MessageRole_ROLE_ASSISTANT {
			continue
		}
		if hasIdx && out.GetIndex() != idx {
			continue
		}
		last = out
	}

	return last
}

func (r *Response) flushBuffers() {
	if r.buffersAreInProto {
		return
	}

	for idx, b := range r.contentBuffers {
		if b != nil && idx < len(r.proto.GetOutputs()) {
			r.proto.Outputs[idx].Message.Content = b.String()
		}
	}

	for idx, b := range r.reasoningBuffers {
		if b != nil && idx < len(r.proto.GetOutputs()) {
			r.proto.Outputs[idx].Message.ReasoningContent = b.String()
		}
	}

	for idx, b := range r.encryptedBuffers {
		if b != nil && idx < len(r.proto.GetOutputs()) {
			r.proto.Outputs[idx].Message.EncryptedContent = b.String()
		}
	}

	r.buffersAreInProto = true
	releaseBuilders(&r.contentBuffers)
	releaseBuilders(&r.reasoningBuffers)
	releaseBuilders(&r.encryptedBuffers)
}

//nolint:cyclop,gocognit,funlen,gocyclo // TODO(zchee): fix nolint.
func (r *Response) processChunk(chunk *xaipb.GetChatCompletionChunk) {
	r.proto.Usage = chunk.GetUsage()
	r.proto.Created = chunk.GetCreated()
	r.proto.Id = chunk.GetId()
	r.proto.Model = chunk.GetModel()
	r.proto.SystemFingerprint = chunk.GetSystemFingerprint()

	if citations := chunk.GetCitations(); len(citations) > 0 {
		r.proto.Citations = append(slices.Grow(r.proto.GetCitations(), len(citations)), citations...)
	}

	maxOutputIdx := -1
	for _, c := range chunk.GetOutputs() {
		idx := int(c.GetIndex())
		if idx > maxOutputIdx {
			maxOutputIdx = idx
		}
	}

	if len(r.proto.GetOutputs()) == 0 && maxOutputIdx >= 0 {
		r.proto.Outputs = make([]*xaipb.CompletionOutput, maxOutputIdx+1)
		r.contentBuffers = make([]*strings.Builder, maxOutputIdx+1)
		r.reasoningBuffers = make([]*strings.Builder, maxOutputIdx+1)
		r.encryptedBuffers = make([]*strings.Builder, maxOutputIdx+1)
		r.toolCallScratch = make([][]*xaipb.ToolCall, maxOutputIdx+1)

		i := int32(0)
		for range r.proto.GetOutputs() {
			r.proto.Outputs[i] = &xaipb.CompletionOutput{
				Index:   i,
				Message: &xaipb.CompletionMessage{},
			}
			i++
		}
	}

	for _, c := range chunk.GetOutputs() {
		idx := int(c.GetIndex())
		delta := c.GetDelta()
		target := r.ensureOutput(idx)
		msg := target.GetMessage()
		target.Index = c.GetIndex()
		msg.Role = delta.GetRole()
		if calls := delta.GetToolCalls(); len(calls) > 0 {
			existing := msg.GetToolCalls()
			if len(existing) == 0 {
				r.ensureToolCallSlot(idx)
				if len(calls) == 1 {
					buf := make([]*xaipb.ToolCall, 1, 8)
					buf[0] = calls[0]
					r.toolCallScratch[idx] = buf
					msg.ToolCalls = buf
					continue
				}

				r.toolCallScratch[idx] = calls
				msg.ToolCalls = calls
				continue
			}

			need := len(existing) + len(calls)
			spare := growthSpareToolCalls(len(existing))
			if cap(existing) < need {
				existing = slices.Grow(existing, len(calls)+spare)
			}
			existing = append(existing, calls...)
			msg.ToolCalls = existing
			r.ensureToolCallSlot(idx)
			r.toolCallScratch[idx] = existing
		}
		target.FinishReason = c.GetFinishReason()

		//nolint:nestif // TODO(zchee): fix nolint
		if content := delta.GetContent(); content != "" {
			if r.buffersAreInProto {
				if msg.GetContent() == "" {
					msg.Content = content
				} else {
					buf := ensureBuilder(&r.contentBuffers, idx)
					buf.Grow(len(msg.GetContent()) + len(content))
					buf.WriteString(msg.GetContent())
					buf.WriteString(content)
					r.buffersAreInProto = false
				}
			} else {
				buf := ensureBuilder(&r.contentBuffers, idx)
				buf.Grow(len(content))
				buf.WriteString(content)
			}
		}

		//nolint:nestif // TODO(zchee): fix nolint
		if reasoning := delta.GetReasoningContent(); reasoning != "" {
			if r.buffersAreInProto {
				if msg.GetReasoningContent() == "" {
					msg.ReasoningContent = reasoning
				} else {
					buf := ensureBuilder(&r.reasoningBuffers, idx)
					buf.Grow(len(msg.GetReasoningContent()) + len(reasoning))
					buf.WriteString(msg.GetReasoningContent())
					buf.WriteString(reasoning)
					r.buffersAreInProto = false
				}
			} else {
				buf := ensureBuilder(&r.reasoningBuffers, idx)
				buf.Grow(len(reasoning))
				buf.WriteString(reasoning)
			}
		}

		//nolint:nestif // TODO(zchee): fix nolint
		if encrypted := delta.GetEncryptedContent(); encrypted != "" {
			if r.buffersAreInProto {
				if msg.GetEncryptedContent() == "" {
					msg.EncryptedContent = encrypted
				} else {
					buf := ensureBuilder(&r.encryptedBuffers, idx)
					buf.Grow(len(msg.GetEncryptedContent()) + len(encrypted))
					buf.WriteString(msg.GetEncryptedContent())
					buf.WriteString(encrypted)
					r.buffersAreInProto = false
				}
			} else {
				buf := ensureBuilder(&r.encryptedBuffers, idx)
				buf.Grow(len(encrypted))
				buf.WriteString(encrypted)
			}
		}
	}
}

func ensureBuilder(bufs *[]*strings.Builder, idx int) *strings.Builder {
	if idx < 0 {
		return nil
	}

	if idx >= len(*bufs) {
		extra := idx + 1 - len(*bufs)
		*bufs = slices.Grow(*bufs, extra)
		*bufs = (*bufs)[:idx+1]
	}

	if (*bufs)[idx] == nil {
		b := builderPool.Get().(*strings.Builder)
		b.Reset()
		(*bufs)[idx] = b
	}

	return (*bufs)[idx]
}

func releaseBuilders(bufs *[]*strings.Builder) {
	for i, b := range *bufs {
		if b == nil {
			continue
		}
		b.Reset()
		builderPool.Put(b)
		(*bufs)[i] = nil
	}
}

func (r *Response) ensureToolCallSlot(idx int) {
	if idx < len(r.toolCallScratch) {
		return
	}
	r.toolCallScratch = append(r.toolCallScratch, make([][]*xaipb.ToolCall, idx+1-len(r.toolCallScratch))...)
}

func growthSpareToolCalls(existing int) int {
	if existing == 0 {
		return 0
	}

	spare := existing
	if spare < 2 {
		spare = 2
	} else if spare > 16 {
		spare = 16
	}

	return spare
}

func (r *Response) ensureOutput(idx int) *xaipb.CompletionOutput {
	if idx >= len(r.proto.GetOutputs()) {
		needed := idx + 1 - len(r.proto.GetOutputs())
		r.proto.Outputs = append(r.proto.Outputs, make([]*xaipb.CompletionOutput, needed)...)
	}

	out := r.proto.GetOutputs()[idx]
	if out == nil {
		out = &xaipb.CompletionOutput{}
		r.proto.Outputs[idx] = out
	}

	if out.GetMessage() == nil {
		out.Message = &xaipb.CompletionMessage{}
	}

	return out
}

var builderPool = sync.Pool{
	New: func() any {
		return &strings.Builder{}
	},
}
