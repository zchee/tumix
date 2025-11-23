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
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/protobuf/proto"

	pb "github.com/zchee/tumix/model/xai/pb/xai/api/v1"
)

const (
	defaultDeferredTimeout  = 10 * time.Minute
	defaultDeferredInterval = 100 * time.Millisecond
)

// ChatClient handles chat operations.
type ChatClient struct {
	stub pb.ChatClient
}

// ChatOption customizes a chat request before execution.
type ChatOption func(*pb.GetCompletionsRequest, *ChatSession)

// Create initializes a new chat session for the specified model.
func (c *ChatClient) Create(model string, opts ...ChatOption) *ChatSession {
	req := &pb.GetCompletionsRequest{Model: model}
	session := &ChatSession{stub: c.stub, request: req}
	for _, opt := range opts {
		opt(req, session)
	}
	return session
}

// WithConversationID stores an optional conversation identifier (client-side only).
func WithConversationID(id string) ChatOption {
	return func(_ *pb.GetCompletionsRequest, s *ChatSession) {
		s.conversationID = id
	}
}

// WithMessages sets initial messages.
func WithMessages(msgs ...*pb.Message) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) {
		req.Messages = append(req.Messages, msgs...)
	}
}

// WithUser sets the user identifier for the request.
func WithUser(user string) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) { req.User = user }
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(max int) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) {
		v := int32(max)
		req.MaxTokens = &v
	}
}

// WithSeed sets the random seed for deterministic generation.
func WithSeed(seed int) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) {
		v := int32(seed)
		req.Seed = &v
	}
}

// WithStop sets the stop sequences.
func WithStop(stop ...string) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) { req.Stop = append(req.Stop, stop...) }
}

// WithTemperature sets the sampling temperature.
func WithTemperature(t float32) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) {
		v := t
		req.Temperature = &v
	}
}

// WithTopP sets the nucleus sampling probability.
func WithTopP(p float32) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) {
		v := p
		req.TopP = &v
	}
}

// WithLogprobs enables log probabilities return.
func WithLogprobs(enabled bool) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) { req.Logprobs = enabled }
}

// WithTopLogprobs sets the number of top log probabilities to return.
func WithTopLogprobs(v int) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) {
		vv := int32(v)
		req.TopLogprobs = &vv
	}
}

// WithTools sets the tools available to the model.
func WithTools(tools ...*pb.Tool) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) { req.Tools = append(req.Tools, tools...) }
}

// WithToolChoice sets the tool choice strategy.
func WithToolChoice(choice *pb.ToolChoice) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) { req.ToolChoice = choice }
}

// WithParallelToolCalls enables or disables parallel tool calls.
func WithParallelToolCalls(enabled bool) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) {
		v := enabled
		req.ParallelToolCalls = &v
	}
}

// WithResponseFormat sets the desired response format (e.g. JSON).
func WithResponseFormat(format *pb.ResponseFormat) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) { req.ResponseFormat = format }
}

// WithFrequencyPenalty sets the frequency penalty.
func WithFrequencyPenalty(v float32) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) {
		vv := v
		req.FrequencyPenalty = &vv
	}
}

// WithPresencePenalty sets the presence penalty.
func WithPresencePenalty(v float32) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) {
		vv := v
		req.PresencePenalty = &vv
	}
}

// WithReasoningEffort sets the reasoning effort level.
func WithReasoningEffort(effort pb.ReasoningEffort) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) { req.ReasoningEffort = &effort }
}

// WithSearchParameters sets the search parameters for the request.
func WithSearchParameters(params *pb.SearchParameters) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) { req.SearchParameters = params }
}

// WithSearch configures search using the helper struct.
func WithSearch(params SearchParameters) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) { req.SearchParameters = params.Proto() }
}

// WithStoreMessages enables message storage.
func WithStoreMessages(store bool) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) { req.StoreMessages = store }
}

// WithPreviousResponse sets the previous response ID for context.
func WithPreviousResponse(id string) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) { req.PreviousResponseId = &id }
}

// WithEncryptedContent enables encrypted content in the response.
func WithEncryptedContent(enabled bool) ChatOption {
	return func(req *pb.GetCompletionsRequest, _ *ChatSession) { req.UseEncryptedContent = enabled }
}

// ChatSession represents an active chat session.
type ChatSession struct {
	stub           pb.ChatClient
	request        *pb.GetCompletionsRequest
	conversationID string
}

// Append adds a message or response to the chat session.
func (s *ChatSession) Append(message any) *ChatSession {
	switch msg := message.(type) {
	case *pb.Message:
		s.request.Messages = append(s.request.Messages, msg)
	case *Response:
		if msg.index == nil {
			for _, out := range msg.proto.Outputs {
				s.request.Messages = append(s.request.Messages, buildMessageFromCompletion(out))
			}
		} else if out := msg.output(); out != nil {
			s.request.Messages = append(s.request.Messages, buildMessageFromCompletion(out))
		}
	default:
		panic("append accepts *pb.Message or *Response")
	}
	return s
}

// Messages returns the current conversation history.
func (s *ChatSession) Messages() []*pb.Message { return s.request.Messages }

// Sample sends the chat request and returns the first response.
func (s *ChatSession) Sample(ctx context.Context) (*Response, error) {
	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.sample %s", s.request.Model),
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(s.makeSpanRequestAttributes()...),
	)
	defer span.End()

	responses, err := s.sampleN(ctx, 1)
	if err != nil {
		span.RecordError(err)
		return nil, err
	}
	span.SetAttributes(s.makeSpanResponseAttributes(responses)...)
	return responses[0], nil
}

// SampleBatch requests n responses in a single call.
func (s *ChatSession) SampleBatch(ctx context.Context, n int) ([]*Response, error) {
	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.sample_batch %s", s.request.Model),
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(s.makeSpanRequestAttributes()...),
	)
	defer span.End()

	responses, err := s.sampleN(ctx, n)
	if err != nil {
		span.RecordError(err)
		return nil, err
	}
	span.SetAttributes(s.makeSpanResponseAttributes(responses)...)
	return responses, nil
}

// Stream returns a streaming iterator for a single response.
func (s *ChatSession) Stream(ctx context.Context) (*ChatStream, error) {
	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.stream %s", s.request.Model),
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(s.makeSpanRequestAttributes()...),
	)
	// Span ended by ChatStream.Close or implied lifecycle handling?
	// Usually streaming spans end when stream closes.
	// Here we attach span to ChatStream so user can end it or we hook into Close.
	// For now, we just start it and let the user manage context or rely on garbage collection (not ideal).
	// Better: ChatStream should hold the span and End() it when stream is exhausted or error occurs.

	stream, err := s.streamN(ctx, 1)
	if err != nil {
		span.RecordError(err)
		span.End()
		return nil, err
	}
	stream.span = span
	return stream, nil
}

// StreamBatch returns a streaming iterator for multiple responses.
func (s *ChatSession) StreamBatch(ctx context.Context, n int) (*ChatStream, error) {
	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.stream_batch %s", s.request.Model),
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(s.makeSpanRequestAttributes()...),
	)

	stream, err := s.streamN(ctx, n)
	if err != nil {
		span.RecordError(err)
		span.End()
		return nil, err
	}
	stream.span = span
	return stream, nil
}

// Defer executes the request using deferred polling.
func (s *ChatSession) Defer(ctx context.Context, timeout, interval time.Duration) (*Response, error) {
	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.defer %s", s.request.Model),
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(s.makeSpanRequestAttributes()...),
	)
	defer span.End()

	responses, err := s.deferN(ctx, 1, timeout, interval)
	if err != nil {
		span.RecordError(err)
		return nil, err
	}
	span.SetAttributes(s.makeSpanResponseAttributes(responses)...)
	return responses[0], nil
}

// DeferBatch executes the request using deferred polling and returns n responses.
func (s *ChatSession) DeferBatch(ctx context.Context, n int, timeout, interval time.Duration) ([]*Response, error) {
	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.defer_batch %s", s.request.Model),
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(s.makeSpanRequestAttributes()...),
	)
	defer span.End()

	responses, err := s.deferN(ctx, n, timeout, interval)
	if err != nil {
		span.RecordError(err)
		return nil, err
	}
	span.SetAttributes(s.makeSpanResponseAttributes(responses)...)
	return responses, nil
}

// Parse sets response_format to a JSON schema derived from the provided sample value and decodes into it.
// Pass a pointer to a struct value to populate it.
func (s *ChatSession) Parse(ctx context.Context, out any) (*Response, error) {
	reflector := &jsonschema.Reflector{}
	schema := reflector.Reflect(out)
	schemaBytes, err := json.Marshal(schema)
	if err != nil {
		return nil, err
	}
	schemaStr := string(schemaBytes)
	req := proto.Clone(s.request).(*pb.GetCompletionsRequest)
	req.ResponseFormat = &pb.ResponseFormat{
		FormatType: pb.FormatType_FORMAT_TYPE_JSON_SCHEMA,
		Schema:     &schemaStr,
	}

	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.parse %s", s.request.Model),
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(s.makeSpanRequestAttributes()...),
	)
	defer span.End()

	resp, err := s.invokeCompletion(ctx, req)
	if err != nil {
		span.RecordError(err)
		return nil, err
	}
	span.SetAttributes(s.makeSpanResponseAttributes([]*Response{resp})...)

	if err := json.Unmarshal([]byte(resp.Content()), out); err != nil {
		return resp, err
	}
	return resp, nil
}

func (s *ChatSession) sampleN(ctx context.Context, n int) ([]*Response, error) {
	if len(s.request.Messages) == 0 {
		return nil, errors.New("chat request requires at least one message")
	}
	req := proto.Clone(s.request).(*pb.GetCompletionsRequest)
	setN(req, n)
	resp, err := s.invokeCompletion(ctx, req)
	if err != nil {
		return nil, err
	}
	if n == 1 {
		return []*Response{resp}, nil
	}
	out := make([]*Response, n)
	for i := range n {
		idx := i
		out[i] = newResponse(resp.proto, &idx)
	}
	return out, nil
}

func (s *ChatSession) streamN(ctx context.Context, n int) (*ChatStream, error) {
	if len(s.request.Messages) == 0 {
		return nil, errors.New("chat request requires at least one message")
	}
	req := proto.Clone(s.request).(*pb.GetCompletionsRequest)
	setN(req, n)

	stream, err := s.stub.GetCompletionChunk(ctx, req)
	if err != nil {
		return nil, err
	}

	resp := &pb.GetChatCompletionResponse{}
	if n > 1 {
		resp.Outputs = make([]*pb.CompletionOutput, n)
	}
	return &ChatStream{stream: stream, response: newResponse(resp, intPtrIf(n == 1))}, nil
}

func (s *ChatSession) deferN(ctx context.Context, n int, timeout, interval time.Duration) ([]*Response, error) {
	if len(s.request.Messages) == 0 {
		return nil, errors.New("chat request requires at least one message")
	}
	req := proto.Clone(s.request).(*pb.GetCompletionsRequest)
	setN(req, n)
	if timeout <= 0 {
		timeout = defaultDeferredTimeout
	}
	if interval <= 0 {
		interval = defaultDeferredInterval
	}
	startResp, err := s.stub.StartDeferredCompletion(ctx, req)
	if err != nil {
		return nil, err
	}
	deadline := time.Now().Add(timeout)
	for {
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("deferred request timed out after %s", timeout)
		}
		res, err := s.stub.GetDeferredCompletion(ctx, &pb.GetDeferredRequest{RequestId: startResp.RequestId})
		if err != nil {
			return nil, err
		}
		switch res.Status {
		case pb.DeferredStatus_DONE:
			return splitResponses(res.Response, n), nil
		case pb.DeferredStatus_EXPIRED:
			return nil, fmt.Errorf("deferred request expired")
		case pb.DeferredStatus_PENDING:
			time.Sleep(interval)
		default:
			return nil, fmt.Errorf("unknown deferred status %v", res.Status)
		}
	}
}

func (s *ChatSession) invokeCompletion(ctx context.Context, req *pb.GetCompletionsRequest) (*Response, error) {
	resp, err := s.stub.GetCompletion(ctx, req)
	if err != nil {
		return nil, err
	}
	index := 0
	if usesServerSideTools(req.Tools) {
		index = -1
	}
	idxPtr := (*int)(nil)
	if index >= 0 {
		idxPtr = &index
	}
	idxPtr = autoDetectMultiOutput(idxPtr, resp.Outputs)
	return newResponse(resp, idxPtr), nil
}

func (s *ChatSession) makeSpanRequestAttributes() []attribute.KeyValue {
	attrs := []attribute.KeyValue{
		attribute.String("gen_ai.operation.name", "chat"),
		attribute.String("gen_ai.system", "xai"),
		attribute.String("gen_ai.output.type", "text"),
		attribute.String("gen_ai.request.model", s.request.Model),
		attribute.Int("server.port", 443),
	}

	// Optional fields
	attrs = append(attrs,
		attribute.Float64("gen_ai.request.frequency_penalty", float64(valueOrZero32(s.request.FrequencyPenalty))),
		attribute.Float64("gen_ai.request.presence_penalty", float64(valueOrZero32(s.request.PresencePenalty))),
		attribute.Float64("gen_ai.request.temperature", float64(valueOrZero32(s.request.Temperature))),
		attribute.Bool("gen_ai.request.parallel_tool_calls", valueOrZeroBool(s.request.ParallelToolCalls)),
		attribute.Bool("gen_ai.request.store_messages", s.request.StoreMessages),
		attribute.Bool("gen_ai.request.use_encrypted_content", s.request.UseEncryptedContent),
		attribute.Bool("gen_ai.request.logprobs", s.request.Logprobs),
	)

	if s.request.TopP != nil {
		attrs = append(attrs, attribute.Float64("gen_ai.request.top_p", float64(*s.request.TopP)))
	}
	if s.request.N != nil {
		attrs = append(attrs, attribute.Int("gen_ai.request.choice.count", int(*s.request.N)))
	}
	if s.request.Seed != nil {
		attrs = append(attrs, attribute.Int("gen_ai.request.seed", int(*s.request.Seed)))
	}
	if s.request.MaxTokens != nil {
		attrs = append(attrs, attribute.Int("gen_ai.request.max_tokens", int(*s.request.MaxTokens)))
	}
	if s.request.TopLogprobs != nil {
		attrs = append(attrs, attribute.Int("gen_ai.request.top_logprobs", int(*s.request.TopLogprobs)))
	}

	if s.conversationID != "" {
		attrs = append(attrs, attribute.String("gen_ai.conversation.id", s.conversationID))
	}
	if len(s.request.Stop) > 0 {
		attrs = append(attrs, attribute.StringSlice("gen_ai.request.stop_sequences", s.request.Stop))
	}
	if s.request.ResponseFormat != nil {
		attrs = append(attrs, attribute.String("gen_ai.output.type", strings.ToLower(strings.TrimPrefix(s.request.ResponseFormat.FormatType.String(), "FORMAT_TYPE_"))))
	}
	if s.request.ReasoningEffort != nil {
		attrs = append(attrs, attribute.String("gen_ai.request.reasoning_effort", strings.ToLower(strings.TrimPrefix(s.request.ReasoningEffort.String(), "EFFORT_"))))
	}
	if s.request.User != "" {
		attrs = append(attrs, attribute.String("user_id", s.request.User))
	}
	if s.request.PreviousResponseId != nil {
		attrs = append(attrs, attribute.String("gen_ai.request.previous_response_id", *s.request.PreviousResponseId))
	}

	// Prompt attributes
	for i, msg := range s.request.Messages {
		prefix := fmt.Sprintf("gen_ai.prompt.%d", i)
		role := strings.ToLower(strings.TrimPrefix(msg.Role.String(), "ROLE_"))
		attrs = append(attrs, attribute.String(prefix+".role", role))

		var contentStr strings.Builder
		for _, c := range msg.Content {
			if txt := c.GetText(); txt != "" {
				contentStr.WriteString(txt)
			}
		}
		attrs = append(attrs, attribute.String(prefix+".content", contentStr.String()))

		if len(msg.ToolCalls) > 0 {
			// Serialize tool calls mostly for debug, simplified here
			// Python does full JSON serialization
			if b, err := json.Marshal(msg.ToolCalls); err == nil {
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

	attrs := []attribute.KeyValue{
		attribute.String("gen_ai.response.id", first.proto.Id),
		attribute.String("gen_ai.response.model", first.proto.Model),
		attribute.String("gen_ai.response.system_fingerprint", first.proto.SystemFingerprint),
	}

	if usage != nil {
		attrs = append(attrs,
			attribute.Int("gen_ai.usage.input_tokens", int(usage.PromptTokens)),
			attribute.Int("gen_ai.usage.output_tokens", int(usage.CompletionTokens)),
			attribute.Int("gen_ai.usage.total_tokens", int(usage.TotalTokens)),
			attribute.Int("gen_ai.usage.reasoning_tokens", int(usage.ReasoningTokens)),
		)
	}

	var finishReasons []string
	for i, resp := range responses {
		finishReasons = append(finishReasons, resp.FinishReason())

		prefix := fmt.Sprintf("gen_ai.completion.%d", i)
		attrs = append(attrs, attribute.String(prefix+".role", strings.ToLower(strings.TrimPrefix(resp.Role(), "ROLE_"))))
		attrs = append(attrs, attribute.String(prefix+".content", resp.Content()))
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

func valueOrZero32[T ~float32](p *T) T {
	if p == nil {
		return 0
	}
	return *p
}

func valueOrZeroBool(p *bool) bool {
	if p == nil {
		return false
	}
	return *p
}

// ChatStream wraps the streaming completion response.
type ChatStream struct {
	stream             pb.Chat_GetCompletionChunkClient
	response           *Response
	span               trace.Span
	firstChunkReceived bool
}

// Recv returns the next chunk and the aggregated response.
func (s *ChatStream) Recv() (*Response, *Chunk, error) {
	chunk, err := s.stream.Recv()
	if err != nil {
		if s.span != nil {
			if err.Error() != "EOF" { // Use io.EOF check ideally, but string check for now or io package
				s.span.RecordError(err)
			} else {
				// Stream finished successfully
				if s.span != nil {
					// We can set response attributes here if we accumulated the full response in s.response
					// But makeSpanResponseAttributes needs access to session or just use manual logic
					// Here we just set what we have in s.response
					// Note: s.response is accumulating content.
					// Replicating makeSpanResponseAttributes logic for single response:

					usage := s.response.Usage()
					if usage != nil {
						s.span.SetAttributes(
							attribute.Int("gen_ai.usage.input_tokens", int(usage.PromptTokens)),
							attribute.Int("gen_ai.usage.output_tokens", int(usage.CompletionTokens)),
							attribute.Int("gen_ai.usage.total_tokens", int(usage.TotalTokens)),
						)
					}
					s.span.SetAttributes(
						attribute.String("gen_ai.response.id", s.response.proto.Id),
						attribute.String("gen_ai.response.model", s.response.proto.Model),
						attribute.String("gen_ai.response.finish_reasons", s.response.FinishReason()),
					)
				}
			}
			s.span.End()
		}
		return s.response, nil, err
	}

	if !s.firstChunkReceived && s.span != nil {
		s.span.SetAttributes(attribute.String("gen_ai.completion.start_time", time.Now().UTC().Format(time.RFC3339)))
		s.firstChunkReceived = true
	}

	s.response.index = autoDetectMultiOutputChunks(s.response.index, chunk.Outputs)
	s.response.processChunk(chunk)
	return s.response, newChunk(chunk, s.response.index), nil
}

// Response wraps GetChatCompletionResponse with convenience accessors.
type Response struct {
	proto             *pb.GetChatCompletionResponse
	index             *int
	contentBuffers    map[int]*strings.Builder
	reasoningBuffers  map[int]*strings.Builder
	encryptedBuffers  map[int]*strings.Builder
	buffersAreInProto bool
}

func newResponse(protoResp *pb.GetChatCompletionResponse, index *int) *Response {
	return &Response{
		proto:             protoResp,
		index:             index,
		contentBuffers:    map[int]*strings.Builder{},
		reasoningBuffers:  map[int]*strings.Builder{},
		encryptedBuffers:  map[int]*strings.Builder{},
		buffersAreInProto: true,
	}
}

// Proto returns the underlying protobuf message (materializing buffered chunks).
func (r *Response) Proto() *pb.GetChatCompletionResponse {
	r.flushBuffers()
	return r.proto
}

// Content returns the content string for the selected output(s).
func (r *Response) Content() string {
	r.flushBuffers()
	if out := r.output(); out != nil {
		return out.Message.Content
	}
	return ""
}

// ReasoningContent returns any reasoning trace text.
func (r *Response) ReasoningContent() string {
	r.flushBuffers()
	if out := r.output(); out != nil {
		return out.Message.ReasoningContent
	}
	return ""
}

// EncryptedContent returns encrypted reasoning content when present.
func (r *Response) EncryptedContent() string {
	r.flushBuffers()
	if out := r.output(); out != nil {
		return out.Message.EncryptedContent
	}
	return ""
}

// Role returns the assistant role string.
func (r *Response) Role() string {
	r.flushBuffers()
	if out := r.output(); out != nil {
		return pb.MessageRole_name[int32(out.Message.Role)]
	}
	return ""
}

// ToolCalls returns tool calls from all assistant outputs.
func (r *Response) ToolCalls() []*pb.ToolCall {
	r.flushBuffers()
	var calls []*pb.ToolCall
	for _, out := range r.proto.Outputs {
		if out.Message.Role == pb.MessageRole_ROLE_ASSISTANT {
			calls = append(calls, out.Message.ToolCalls...)
		}
	}
	return calls
}

// FinishReason returns the finish reason string.
func (r *Response) FinishReason() string {
	if out := r.outputNoFlush(); out != nil {
		return pb.FinishReason_name[int32(out.FinishReason)]
	}
	return ""
}

// Usage returns token usage.
func (r *Response) Usage() *pb.SamplingUsage { return r.proto.Usage }

// Citations returns any citations returned by the model.
func (r *Response) Citations() []string { return r.proto.Citations }

// SystemFingerprint returns system fingerprint.
func (r *Response) SystemFingerprint() string { return r.proto.SystemFingerprint }

func (r *Response) output() *pb.CompletionOutput {
	r.flushBuffers()
	return r.outputNoFlush()
}

func (r *Response) outputNoFlush() *pb.CompletionOutput {
	var outputs []*pb.CompletionOutput
	for _, out := range r.proto.Outputs {
		if out.Message.Role == pb.MessageRole_ROLE_ASSISTANT && (r.index == nil || int32(out.Index) == int32(valueOrZero(r.index))) {
			outputs = append(outputs, out)
		}
	}
	if len(outputs) == 0 {
		return nil
	}
	return outputs[len(outputs)-1]
}

func (r *Response) flushBuffers() {
	if r.buffersAreInProto {
		return
	}
	for idx, b := range r.contentBuffers {
		if idx < len(r.proto.Outputs) {
			r.proto.Outputs[idx].Message.Content = b.String()
		}
	}
	for idx, b := range r.reasoningBuffers {
		if idx < len(r.proto.Outputs) {
			r.proto.Outputs[idx].Message.ReasoningContent = b.String()
		}
	}
	for idx, b := range r.encryptedBuffers {
		if idx < len(r.proto.Outputs) {
			r.proto.Outputs[idx].Message.EncryptedContent = b.String()
		}
	}
	r.buffersAreInProto = true
}

func (r *Response) processChunk(chunk *pb.GetChatCompletionChunk) {
	r.proto.Usage = chunk.Usage
	r.proto.Created = chunk.Created
	r.proto.Id = chunk.Id
	r.proto.Model = chunk.Model
	r.proto.SystemFingerprint = chunk.SystemFingerprint
	r.proto.Citations = append(r.proto.Citations, chunk.Citations...)

	maxIndex := 0
	for _, out := range chunk.Outputs {
		if int(out.Index) > maxIndex {
			maxIndex = int(out.Index)
		}
	}
	for len(r.proto.Outputs) <= maxIndex {
		r.proto.Outputs = append(r.proto.Outputs, &pb.CompletionOutput{})
	}

	for _, c := range chunk.Outputs {
		target := r.proto.Outputs[c.Index]
		target.Index = c.Index
		if target.Message == nil {
			target.Message = &pb.CompletionMessage{}
		}
		target.Message.Role = c.Delta.Role
		target.Message.ToolCalls = append(target.Message.ToolCalls, c.Delta.ToolCalls...)
		target.FinishReason = c.FinishReason

		if c.Delta.Content != "" {
			appendToBuilder(r.contentBuffers, int(c.Index), c.Delta.Content)
			r.buffersAreInProto = false
		}
		if c.Delta.ReasoningContent != "" {
			appendToBuilder(r.reasoningBuffers, int(c.Index), c.Delta.ReasoningContent)
			r.buffersAreInProto = false
		}
		if c.Delta.EncryptedContent != "" {
			appendToBuilder(r.encryptedBuffers, int(c.Index), c.Delta.EncryptedContent)
			r.buffersAreInProto = false
		}
	}
}

// Chunk wraps GetChatCompletionChunk with helpers.
type Chunk struct {
	proto *pb.GetChatCompletionChunk
	index *int
}

func newChunk(protoChunk *pb.GetChatCompletionChunk, index *int) *Chunk {
	return &Chunk{proto: protoChunk, index: index}
}

// Content concatenates chunk content for the tracked index (or all when multi-output).
func (c *Chunk) Content() string {
	var b strings.Builder
	for _, out := range c.outputs() {
		b.WriteString(out.Delta.Content)
	}
	return b.String()
}

// ReasoningContent concatenates reasoning content for tracked outputs.
func (c *Chunk) ReasoningContent() string {
	var b strings.Builder
	for _, out := range c.outputs() {
		b.WriteString(out.Delta.ReasoningContent)
	}
	return b.String()
}

// ToolCalls returns tool calls for this chunk.
func (c *Chunk) ToolCalls() []*pb.ToolCall {
	var calls []*pb.ToolCall
	for _, out := range c.outputs() {
		calls = append(calls, out.Delta.ToolCalls...)
	}
	return calls
}

// Outputs returns the raw chunk outputs filtered by index.
func (c *Chunk) outputs() []*pb.CompletionOutputChunk {
	var outs []*pb.CompletionOutputChunk
	for _, out := range c.proto.Outputs {
		if out.Delta.Role == pb.MessageRole_ROLE_ASSISTANT && (c.index == nil || out.Index == int32(*c.index)) {
			outs = append(outs, out)
		}
	}
	return outs
}

// Convenience builders for messages and content.

// User creates a user message with text or content parts.
func User(parts ...any) *pb.Message { return newMessage(pb.MessageRole_ROLE_USER, parts...) }

// System creates a system message.
func System(parts ...any) *pb.Message { return newMessage(pb.MessageRole_ROLE_SYSTEM, parts...) }

// Assistant creates an assistant message.
func Assistant(parts ...any) *pb.Message { return newMessage(pb.MessageRole_ROLE_ASSISTANT, parts...) }

// ToolResult creates a tool result message.
func ToolResult(result string) *pb.Message { return newMessage(pb.MessageRole_ROLE_TOOL, result) }

func newMessage(role pb.MessageRole, parts ...any) *pb.Message {
	contents := make([]*pb.Content, 0, len(parts))
	for _, part := range parts {
		switch v := part.(type) {
		case string:
			contents = append(contents, TextContent(v))
		case *pb.Content:
			contents = append(contents, v)
		default:
			panic("unsupported content type")
		}
	}
	return &pb.Message{Role: role, Content: contents}
}

func buildMessageFromCompletion(out *pb.CompletionOutput) *pb.Message {
	var reasoning *string
	if out.Message.ReasoningContent != "" {
		rc := out.Message.ReasoningContent
		reasoning = &rc
	}
	return &pb.Message{
		Role:             out.Message.Role,
		Content:          []*pb.Content{TextContent(out.Message.Content)},
		ReasoningContent: reasoning,
		EncryptedContent: out.Message.EncryptedContent,
		ToolCalls:        out.Message.ToolCalls,
	}
}

// TextContent wraps plain text into a Content message.
func TextContent(text string) *pb.Content { return &pb.Content{Content: &pb.Content_Text{Text: text}} }

// ImageContent creates an image content entry with optional detail.
func ImageContent(url string, detail pb.ImageDetail) *pb.Content {
	return &pb.Content{Content: &pb.Content_ImageUrl{
		ImageUrl: &pb.ImageUrlContent{ImageUrl: url, Detail: detail},
	}}
}

// FileContent references an uploaded file.
func FileContent(fileID string) *pb.Content {
	return &pb.Content{Content: &pb.Content_File{File: &pb.FileContent{FileId: fileID}}}
}

func usesServerSideTools(tools []*pb.Tool) bool {
	for _, t := range tools {
		switch t.Tool.(type) {
		case *pb.Tool_Function:
			continue
		default:
			return true
		}
	}
	return false
}

func autoDetectMultiOutput(index *int, outputs []*pb.CompletionOutput) *int {
	if index != nil {
		maxIdx := int32(valueOrZero(index))
		for _, out := range outputs {
			if out.Index > maxIdx {
				return nil
			}
		}
	}
	return index
}

func autoDetectMultiOutputChunks(index *int, outputs []*pb.CompletionOutputChunk) *int {
	if index != nil {
		maxIdx := valueOrZero(index)
		for _, out := range outputs {
			if int(out.Index) > maxIdx {
				return nil
			}
		}
	}
	return index
}

func setN(req *pb.GetCompletionsRequest, n int) {
	val := int32(n)
	req.N = &val
}

func valueOrZero(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

func intPtrIf(condition bool) *int {
	if !condition {
		return nil
	}
	zero := 0
	return &zero
}

func appendToBuilder(m map[int]*strings.Builder, idx int, s string) {
	b, ok := m[idx]
	if !ok {
		b = &strings.Builder{}
		m[idx] = b
	}
	b.WriteString(s)
}

func splitResponses(resp *pb.GetChatCompletionResponse, n int) []*Response {
	responses := make([]*Response, n)
	for i := 0; i < n; i++ {
		idx := i
		responses[i] = newResponse(resp, &idx)
	}
	return responses
}
