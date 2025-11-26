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
	"context"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/bytedance/sonic"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/protobuf/proto"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

const (
	defaultDeferredTimeout  = 10 * time.Minute
	defaultDeferredInterval = 100 * time.Millisecond
)

// ChatClient handles chat operations.
type ChatClient struct {
	chat xaipb.ChatClient
}

// Create initializes a new chat session for the specified model.
func (c *ChatClient) Create(model string, opts ...ChatOption) *ChatSession {
	req := &xaipb.GetCompletionsRequest{
		Model: model,
	}
	session := &ChatSession{
		stub:    c.chat,
		request: req,
	}
	for _, opt := range opts {
		opt(req, session)
	}

	return session
}

// GetStoredCompletion retrieves a stored response using the response ID.
func (c *ChatClient) GetStoredCompletion(ctx context.Context, responseID string) (*xaipb.GetChatCompletionResponse, error) {
	req := &xaipb.GetStoredCompletionRequest{
		ResponseId: responseID,
	}
	return c.chat.GetStoredCompletion(ctx, req)
}

// DeleteStoredCompletion deletes a stored response using the response ID.
func (c *ChatClient) DeleteStoredCompletion(ctx context.Context, responseID string) error {
	req := &xaipb.DeleteStoredCompletionRequest{
		ResponseId: responseID,
	}
	_, err := c.chat.DeleteStoredCompletion(ctx, req)
	return err
}

// StartDeferredCompletion starts sampling of the model and immediately returns a response containing a request id.
func (c *ChatClient) StartDeferredCompletion(ctx context.Context, req *xaipb.GetCompletionsRequest) (*xaipb.StartDeferredResponse, error) {
	return c.chat.StartDeferredCompletion(ctx, req)
}

// GetDeferredCompletion gets the result of a deferred completion.
func (c *ChatClient) GetDeferredCompletion(ctx context.Context, requestID string) (*xaipb.GetDeferredCompletionResponse, error) {
	req := &xaipb.GetDeferredRequest{
		RequestId: requestID,
	}
	return c.chat.GetDeferredCompletion(ctx, req)
}

// ParseInto is a generic convenience for structured outputs into type T.
func ParseInto[T any](ctx context.Context, s *ChatSession) (*Response, *T, error) {
	var out T
	resp, err := s.Parse(ctx, &out)
	return resp, &out, err
}

func (s *ChatSession) sampleN(ctx context.Context, n int32) ([]*Response, error) {
	if len(s.request.GetMessages()) == 0 {
		return nil, errors.New("chat request requires at least one message")
	}

	req := proto.Clone(s.request).(*xaipb.GetCompletionsRequest)
	req.N = ptr(n)
	resp, err := s.invokeCompletion(ctx, req)
	if err != nil {
		return nil, err
	}

	if n == 1 {
		return []*Response{resp}, nil
	}

	out := make([]*Response, n)
	for i := range n {
		out[i] = newResponse(resp.proto, &i)
	}

	return out, nil
}

func (s *ChatSession) streamN(ctx context.Context, n int32) (*ChatStream, error) {
	if len(s.request.GetMessages()) == 0 {
		return nil, errors.New("chat request requires at least one message")
	}
	req := proto.Clone(s.request).(*xaipb.GetCompletionsRequest)
	req.N = ptr(n)

	stream, err := s.stub.GetCompletionChunk(ctx, req)
	if err != nil {
		return nil, err
	}

	resp := &xaipb.GetChatCompletionResponse{}
	if n > 1 {
		resp.Outputs = make([]*xaipb.CompletionOutput, n)
	}

	return &ChatStream{
		stream:   stream,
		response: newResponse(resp, intPtrIf(n == 1)),
	}, nil
}

func (s *ChatSession) deferN(ctx context.Context, n int32, timeout, interval time.Duration) ([]*Response, error) {
	if len(s.request.GetMessages()) == 0 {
		return nil, errors.New("chat request requires at least one message")
	}
	req := proto.Clone(s.request).(*xaipb.GetCompletionsRequest)
	req.N = ptr(n)

	if timeout <= 0 {
		timeout = defaultDeferredTimeout
	}
	if interval <= 0 {
		interval = defaultDeferredInterval
	}

	startResp, err := s.stub.StartDeferredCompletion(ctx, req)
	if err != nil {
		return nil, WrapError(err)
	}

	deadline := time.Now().Add(timeout)
	for {
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("deferred request timed out after %s", timeout)
		}

		res, err := s.stub.GetDeferredCompletion(ctx, &xaipb.GetDeferredRequest{
			RequestId: startResp.GetRequestId(),
		})
		if err != nil {
			return nil, WrapError(err)
		}

		switch res.GetStatus() {
		case xaipb.DeferredStatus_DONE:
			return splitResponses(res.GetResponse(), n), nil
		case xaipb.DeferredStatus_EXPIRED:
			return nil, fmt.Errorf("deferred request expired")
		case xaipb.DeferredStatus_PENDING:
			time.Sleep(interval)
		default:
			return nil, fmt.Errorf("unknown deferred status %v", res.GetStatus())
		}
	}
}

func (s *ChatSession) invokeCompletion(ctx context.Context, req *xaipb.GetCompletionsRequest) (*Response, error) {
	resp, err := s.stub.GetCompletion(ctx, req)
	if err != nil {
		return nil, WrapError(err)
	}

	index := int32(0)
	if usesServerSideTools(req.GetTools()) {
		index = -1
	}
	idxPtr := (*int32)(nil)
	if index >= 0 {
		idxPtr = &index
	}
	idxPtr = autoDetectMultiOutput(idxPtr, resp.GetOutputs())

	return newResponse(resp, idxPtr), nil
}

//nolint:gocognit,cyclop // TODO(zchee): fix nolint
func (s *ChatSession) makeSpanRequestAttributes() []attribute.KeyValue {
	msgs := s.request.GetMessages()
	attrs := make([]attribute.KeyValue, 0, 18+len(msgs)*4)

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
		attrs = append(attrs, attribute.String("gen_ai.output.type", strings.ToLower(strings.TrimPrefix(rf.GetFormatType().String(), "FORMAT_TYPE_"))))
	}
	if re := s.request.ReasoningEffort; re != nil {
		attrs = append(attrs, attribute.String("gen_ai.request.reasoning_effort", strings.ToLower(strings.TrimPrefix(re.String(), "EFFORT_"))))
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
		role := strings.ToLower(strings.TrimPrefix(msg.GetRole().String(), "ROLE_"))
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

		if tcs := msg.GetToolCalls(); len(tcs) > 0 {
			if b, err := sonic.ConfigFastest.Marshal(tcs); err == nil {
				attrs = append(attrs, attribute.String(prefix+".tool_calls", string(b)))
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

	attrs := make([]attribute.KeyValue, 0, 12+len(responses)*5)
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
		finishReasons[i] = resp.FinishReason()

		prefix := "gen_ai.completion." + strconv.Itoa(i)
		role := strings.ToLower(strings.TrimPrefix(resp.Role(), "ROLE_"))
		attrs = append(attrs,
			attribute.String(prefix+".role", role),
			attribute.String(prefix+".content", resp.Content()),
		)

		if rc := resp.ReasoningContent(); rc != "" {
			attrs = append(attrs, attribute.String(prefix+".reasoning_content", rc))
		}

		if tcs := resp.ToolCalls(); len(tcs) > 0 {
			if b, err := json.Marshal(tcs); err == nil {
				attrs = append(attrs, attribute.String(prefix+".tool_calls", string(b)))
			}
		}
	}
	attrs = append(attrs, attribute.StringSlice("gen_ai.response.finish_reasons", finishReasons))

	return attrs
}

// ChatStream wraps the streaming completion response.
type ChatStream struct {
	stream             xaipb.Chat_GetCompletionChunkClient
	response           *Response
	ctx                context.Context
	span               trace.Span
	firstChunkReceived bool
}

// Close closes the underlying stream and ends the span if present.
// It is safe to call multiple times.
func (s *ChatStream) Close() error {
	var err error
	if s.stream != nil {
		err = s.stream.CloseSend()
		s.stream = nil
	}

	if s.span != nil {
		s.span.End()
		s.span = nil
	}

	return err
}

// Recv returns the next chunk and the aggregated response.
//
// Recv implements [grpc.ServerStreamingClient[xaipb.GetChatCompletionChunk]].
func (s *ChatStream) Recv() (*Response, *Chunk, error) {
	chunk, err := s.stream.Recv()
	if err != nil { //nolint:nestif // TODO(zchee): fix nolint
		if s.span != nil {
			if !errors.Is(err, io.EOF) {
				s.span.RecordError(err)
				return s.response, nil, err
			}

			if usage := s.response.Usage(); usage != nil {
				s.span.SetAttributes(
					attribute.Int("gen_ai.usage.input_tokens", int(usage.GetPromptTokens())),
					attribute.Int("gen_ai.usage.output_tokens", int(usage.GetCompletionTokens())),
					attribute.Int("gen_ai.usage.total_tokens", int(usage.GetTotalTokens())),
				)
			}

			s.span.SetAttributes(
				attribute.String("gen_ai.response.id", s.response.proto.GetId()),
				attribute.String("gen_ai.response.model", s.response.proto.GetModel()),
				attribute.String("gen_ai.response.finish_reasons", s.response.FinishReason()),
			)

			s.span.End()
		}

		return s.response, nil, err
	}

	if !s.firstChunkReceived && s.span != nil {
		s.span.SetAttributes(attribute.String("gen_ai.completion.start_time", time.Now().UTC().Format(time.RFC3339)))
		s.firstChunkReceived = true
	}

	s.response.index = autoDetectMultiOutputChunks(s.response.index, chunk.GetOutputs())
	s.response.processChunk(chunk)

	return s.response, newChunk(chunk, s.response.index), nil
}

// Response wraps GetChatCompletionResponse with convenience accessors.
type Response struct {
	proto             *xaipb.GetChatCompletionResponse
	index             *int32
	contentBuffers    []*strings.Builder
	reasoningBuffers  []*strings.Builder
	encryptedBuffers  []*strings.Builder
	buffersAreInProto bool
}

func newResponse(protoResp *xaipb.GetChatCompletionResponse, index *int32) *Response {
	return &Response{
		proto:             protoResp,
		index:             index,
		contentBuffers:    nil,
		reasoningBuffers:  nil,
		encryptedBuffers:  nil,
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
	return sonic.ConfigFastest.Unmarshal([]byte(r.Content()), out)
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
	calls := make([]*xaipb.ToolCall, 0, len(r.proto.GetOutputs()))
	for out := range slices.Values(r.proto.GetOutputs()) {
		if out.GetMessage().GetRole() == xaipb.MessageRole_ROLE_ASSISTANT {
			calls = append(calls, out.GetMessage().GetToolCalls()...)
		}
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

//nolint:gocognit // TODO(zchee): fix nolint.
func (r *Response) processChunk(chunk *xaipb.GetChatCompletionChunk) {
	r.proto.Usage = chunk.GetUsage()
	r.proto.Created = chunk.GetCreated()
	r.proto.Id = chunk.GetId()
	r.proto.Model = chunk.GetModel()
	r.proto.SystemFingerprint = chunk.GetSystemFingerprint()

	if citations := chunk.GetCitations(); len(citations) > 0 {
		r.proto.Citations = append(slices.Grow(r.proto.Citations, len(citations)), citations...)
	}

	for _, c := range chunk.GetOutputs() {
		idx := int(c.GetIndex())
		delta := c.GetDelta()
		target := r.ensureOutput(idx)
		msg := target.GetMessage()
		target.Index = c.GetIndex()
		msg.Role = delta.GetRole()
		if calls := delta.GetToolCalls(); len(calls) > 0 {
			msg.ToolCalls = slices.Grow(msg.GetToolCalls(), len(calls))
			msg.ToolCalls = append(msg.ToolCalls, calls...)
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

// Chunk wraps GetChatCompletionChunk with helpers.
type Chunk struct {
	proto *xaipb.GetChatCompletionChunk
	index *int32
}

func newChunk(protoChunk *xaipb.GetChatCompletionChunk, index *int32) *Chunk {
	return &Chunk{
		proto: protoChunk,
		index: index,
	}
}

// Content concatenates chunk content for the tracked index (or all when multi-output).
func (c *Chunk) Content() string {
	return concatChunkText(c.proto.GetOutputs(), c.index, func(delta *xaipb.Delta) string {
		return delta.GetContent()
	})
}

// ReasoningContent concatenates reasoning content for tracked outputs.
func (c *Chunk) ReasoningContent() string {
	return concatChunkText(c.proto.GetOutputs(), c.index, func(delta *xaipb.Delta) string {
		return delta.GetReasoningContent()
	})
}

// ToolCalls returns tool calls for this chunk.
func (c *Chunk) ToolCalls() []*xaipb.ToolCall {
	idx, hasIdx := deref(c.index), c.index != nil
	var calls []*xaipb.ToolCall
	for out := range slices.Values(c.proto.GetOutputs()) {
		delta := out.GetDelta()
		if delta.GetRole() != xaipb.MessageRole_ROLE_ASSISTANT {
			continue
		}
		if hasIdx && out.GetIndex() != idx {
			continue
		}
		if toolCalls := delta.GetToolCalls(); len(toolCalls) > 0 {
			calls = append(slices.Grow(calls, len(toolCalls)), toolCalls...)
		}
	}

	return calls
}

// Convenience builders for messages and content.

// User creates a user message with text or content parts.
func User(parts ...any) *xaipb.Message {
	return newMessage(xaipb.MessageRole_ROLE_USER, parts...)
}

// System creates a system message.
func System(parts ...any) *xaipb.Message {
	return newMessage(xaipb.MessageRole_ROLE_SYSTEM, parts...)
}

// Assistant creates an assistant message.
func Assistant(parts ...any) *xaipb.Message {
	return newMessage(xaipb.MessageRole_ROLE_ASSISTANT, parts...)
}

// ToolResult creates a tool result message.
func ToolResult(result string) *xaipb.Message {
	return newMessage(xaipb.MessageRole_ROLE_TOOL, result)
}

func newMessage(role xaipb.MessageRole, parts ...any) *xaipb.Message {
	contents := make([]*xaipb.Content, 0, len(parts))
	for _, part := range parts {
		switch v := part.(type) {
		case string:
			contents = append(contents, TextContent(v))
		case *xaipb.Content:
			contents = append(contents, v)
		default:
			panic("unsupported content type")
		}
	}

	return &xaipb.Message{
		Role:    role,
		Content: contents,
	}
}

func buildMessageFromCompletion(out *xaipb.CompletionOutput) *xaipb.Message {
	var reasoning *string
	if out.GetMessage().GetReasoningContent() != "" {
		rc := out.GetMessage().GetReasoningContent()
		reasoning = &rc
	}

	return &xaipb.Message{
		Role: out.GetMessage().GetRole(),
		Content: []*xaipb.Content{
			TextContent(out.GetMessage().GetContent()),
		},
		ReasoningContent: reasoning,
		EncryptedContent: out.GetMessage().GetEncryptedContent(),
		ToolCalls:        out.GetMessage().GetToolCalls(),
	}
}

// TextContent wraps plain text into a Content message.
func TextContent(text string) *xaipb.Content {
	return &xaipb.Content{
		Content: &xaipb.Content_Text{
			Text: text,
		},
	}
}

// FileContentWithName references an uploaded file and provides a display name.
func FileContentWithName(fileID, name string) *xaipb.Content {
	_ = name // name not supported in current proto; kept for parity but ignored
	return FileContent(fileID)
}

// ImageContent creates an image content entry with optional detail.
func ImageContent(url string, detail xaipb.ImageDetail) *xaipb.Content {
	return &xaipb.Content{
		Content: &xaipb.Content_ImageUrl{
			ImageUrl: &xaipb.ImageUrlContent{
				ImageUrl: url,
				Detail:   detail,
			},
		},
	}
}

// FileContent references an uploaded file (id only).
func FileContent(fileID string) *xaipb.Content {
	return &xaipb.Content{
		Content: &xaipb.Content_File{
			File: &xaipb.FileContent{
				FileId: fileID,
			},
		},
	}
}

func usesServerSideTools(tools []*xaipb.Tool) bool {
	for _, t := range tools {
		switch t.GetTool().(type) {
		case *xaipb.Tool_Function:
			continue
		default:
			return true
		}
	}

	return false
}

func autoDetectMultiOutput(index *int32, outputs []*xaipb.CompletionOutput) *int32 {
	if index != nil {
		maxIdx := deref(index)
		for _, out := range outputs {
			if out.GetIndex() > maxIdx {
				return nil
			}
		}
	}

	return index
}

func autoDetectMultiOutputChunks(index *int32, outputs []*xaipb.CompletionOutputChunk) *int32 {
	if index != nil {
		maxIdx := deref(index)
		for _, out := range outputs {
			if out.GetIndex() > maxIdx {
				return nil
			}
		}
	}

	return index
}

func intPtrIf(condition bool) *int32 {
	if !condition {
		return nil
	}

	return ptr(int32(0))
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

func concatChunkText(chunks []*xaipb.CompletionOutputChunk, idx *int32, pick func(*xaipb.Delta) string) string {
	idxVal, hasIdx := deref(idx), idx != nil
	total := 0
	for out := range slices.Values(chunks) {
		delta := out.GetDelta()
		if delta.GetRole() != xaipb.MessageRole_ROLE_ASSISTANT {
			continue
		}
		if hasIdx && out.GetIndex() != idxVal {
			continue
		}
		part := pick(delta)
		if part == "" {
			continue
		}
		total += len(part)
	}
	if total == 0 {
		return ""
	}
	buf := make([]byte, total)
	pos := 0
	for out := range slices.Values(chunks) {
		delta := out.GetDelta()
		if delta.GetRole() != xaipb.MessageRole_ROLE_ASSISTANT {
			continue
		}
		if hasIdx && out.GetIndex() != idxVal {
			continue
		}
		part := pick(delta)
		if part == "" {
			continue
		}
		pos += copy(buf[pos:], part)
	}
	return string(buf)
}

func splitResponses(resp *xaipb.GetChatCompletionResponse, n int32) []*Response {
	responses := make([]*Response, n)
	for i := range n {
		responses[i] = newResponse(resp, &i)
	}

	return responses
}
