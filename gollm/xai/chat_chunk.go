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
	"slices"

	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
)

// Chunk wraps GetChatCompletionChunk with helpers.
type Chunk struct {
	proto        *xaipb.GetChatCompletionChunk
	index        *int32
	hasIndex     bool
	indexValue   int32
	contentLen   int
	reasoningLen int
	toolCallsLen int

	contentCache    string
	reasoningCache  string
	toolCallsCache  []*xaipb.ToolCall
	contentCached   bool
	reasoningCached bool
	toolCallsCached bool
}

func newChunk(protoChunk *xaipb.GetChatCompletionChunk, index *int32) *Chunk {
	idxVal, hasIdx := deref(index), index != nil
	contentLen, reasoningLen, toolCallsLen := computeChunkStats(protoChunk.GetOutputs(), hasIdx, idxVal)
	return &Chunk{
		proto:        protoChunk,
		index:        index,
		hasIndex:     hasIdx,
		indexValue:   idxVal,
		contentLen:   contentLen,
		reasoningLen: reasoningLen,
		toolCallsLen: toolCallsLen,
	}
}

// Content concatenates chunk content for the tracked index (or all when multi-output).
func (c *Chunk) Content() string {
	if c.contentCached {
		return c.contentCache
	}
	if c.contentLen == 0 {
		c.contentCached = true
		return ""
	}

	buf := make([]byte, c.contentLen)
	pos := 0
	for out := range slices.Values(c.proto.GetOutputs()) {
		delta := out.GetDelta()
		if delta.GetRole() != xaipb.MessageRole_ROLE_ASSISTANT {
			continue
		}
		if c.hasIndex && out.GetIndex() != c.indexValue {
			continue
		}
		part := delta.GetContent()
		if part == "" {
			continue
		}
		pos += copy(buf[pos:], part)
	}

	c.contentCache = string(buf[:pos])
	c.contentCached = true
	return c.contentCache
}

// ReasoningContent concatenates reasoning content for tracked outputs.
func (c *Chunk) ReasoningContent() string {
	if c.reasoningCached {
		return c.reasoningCache
	}
	if c.reasoningLen == 0 {
		c.reasoningCached = true
		return ""
	}

	buf := make([]byte, c.reasoningLen)
	pos := 0
	for out := range slices.Values(c.proto.GetOutputs()) {
		delta := out.GetDelta()
		if delta.GetRole() != xaipb.MessageRole_ROLE_ASSISTANT {
			continue
		}
		if c.hasIndex && out.GetIndex() != c.indexValue {
			continue
		}
		part := delta.GetReasoningContent()
		if part == "" {
			continue
		}
		pos += copy(buf[pos:], part)
	}

	c.reasoningCache = string(buf[:pos])
	c.reasoningCached = true
	return c.reasoningCache
}

// ToolCalls returns tool calls for this chunk.
func (c *Chunk) ToolCalls() []*xaipb.ToolCall {
	if c.toolCallsCached {
		return c.toolCallsCache
	}
	if c.toolCallsLen == 0 {
		c.toolCallsCached = true
		return nil
	}

	calls := make([]*xaipb.ToolCall, 0, c.toolCallsLen)
	for out := range slices.Values(c.proto.GetOutputs()) {
		delta := out.GetDelta()
		if delta.GetRole() != xaipb.MessageRole_ROLE_ASSISTANT {
			continue
		}
		if c.hasIndex && out.GetIndex() != c.indexValue {
			continue
		}
		if toolCalls := delta.GetToolCalls(); len(toolCalls) > 0 {
			calls = append(calls, toolCalls...)
		}
	}

	c.toolCallsCache = calls
	c.toolCallsCached = true
	return calls
}

func computeChunkStats(chunks []*xaipb.CompletionOutputChunk, hasIdx bool, idxVal int32) (contentTotal, reasoningTotal, toolCallsTotal int) {
	for out := range slices.Values(chunks) {
		delta := out.GetDelta()
		if delta.GetRole() != xaipb.MessageRole_ROLE_ASSISTANT {
			continue
		}
		if hasIdx && out.GetIndex() != idxVal {
			continue
		}
		contentTotal += len(delta.GetContent())
		reasoningTotal += len(delta.GetReasoningContent())
		toolCallsTotal += len(delta.GetToolCalls())
	}

	return contentTotal, reasoningTotal, toolCallsTotal
}
