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
	"strings"
	"time"

	"github.com/invopop/jsonschema"
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

// ChatOption customizes a chat request before execution.
type ChatOption func(*xaipb.GetCompletionsRequest, *ChatSession)

// WithConversationID stores an optional conversation identifier (client-side only).
func WithConversationID(id string) ChatOption {
	return func(_ *xaipb.GetCompletionsRequest, s *ChatSession) {
		s.conversationID = id
	}
}

// WithMessages sets initial messages.
func WithMessages(msgs ...*xaipb.Message) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		req.Messages = append(req.Messages, msgs...)
	}
}

// WithUser sets the user identifier for the request.
func WithUser(user string) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) { req.User = user }
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(max int) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		v := int32(max)
		req.MaxTokens = &v
	}
}

// WithSeed sets the random seed for deterministic generation.
func WithSeed(seed int) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		v := int32(seed)
		req.Seed = &v
	}
}

// WithStop sets the stop sequences.
func WithStop(stop ...string) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) { req.Stop = append(req.Stop, stop...) }
}

// WithTemperature sets the sampling temperature.
func WithTemperature(t float32) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		v := t
		req.Temperature = &v
	}
}

// WithTopP sets the nucleus sampling probability.
func WithTopP(p float32) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		v := p
		req.TopP = &v
	}
}

// WithLogprobs enables log probabilities return.
func WithLogprobs(enabled bool) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) { req.Logprobs = enabled }
}

// WithTopLogprobs sets the number of top log probabilities to return.
func WithTopLogprobs(v int) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		vv := int32(v)
		req.TopLogprobs = &vv
	}
}

// WithTools sets the tools available to the model.
func WithTools(tools ...*xaipb.Tool) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) { req.Tools = append(req.Tools, tools...) }
}

// WithToolChoice sets the tool choice strategy.
func WithToolChoice(choice *xaipb.ToolChoice) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) { req.ToolChoice = choice }
}

// WithParallelToolCalls enables or disables parallel tool calls.
func WithParallelToolCalls(enabled bool) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		v := enabled
		req.ParallelToolCalls = &v
	}
}

// WithResponseFormat sets the desired response format (e.g. JSON).
func WithResponseFormat(format *xaipb.ResponseFormat) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) { req.ResponseFormat = format }
}

// WithJSONSchema sets a JSON schema string for structured outputs.
func WithJSONSchema(schema string) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		req.ResponseFormat = &xaipb.ResponseFormat{
			FormatType: xaipb.FormatType_FORMAT_TYPE_JSON_SCHEMA,
			Schema:     &schema,
		}
	}
}

// WithJSONStruct derives a JSON Schema from the generic type T (pointer recommended) for structured outputs.
func WithJSONStruct[T any]() ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		var zero T
		refl := &jsonschema.Reflector{}
		schema := refl.Reflect(zero)
		if schema == nil {
			return
		}
		b, err := json.Marshal(schema)
		if err != nil {
			return
		}
		s := string(b)
		req.ResponseFormat = &xaipb.ResponseFormat{
			FormatType: xaipb.FormatType_FORMAT_TYPE_JSON_SCHEMA,
			Schema:     &s,
		}
	}
}

// WithFrequencyPenalty sets the frequency penalty.
func WithFrequencyPenalty(v float32) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		vv := v
		req.FrequencyPenalty = &vv
	}
}

// WithPresencePenalty sets the presence penalty.
func WithPresencePenalty(v float32) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		vv := v
		req.PresencePenalty = &vv
	}
}

// WithReasoningEffort sets the reasoning effort level.
func WithReasoningEffort(effort xaipb.ReasoningEffort) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) { req.ReasoningEffort = &effort }
}

// WithSearchParameters sets the search parameters for the request.
func WithSearchParameters(params *xaipb.SearchParameters) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) { req.SearchParameters = params }
}

// WithSearch configures search using the helper struct.
func WithSearch(params SearchParameters) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) { req.SearchParameters = params.Proto() }
}

// WithStoreMessages enables message storage.
func WithStoreMessages(store bool) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) { req.StoreMessages = store }
}

// WithPreviousResponse sets the previous response ID for context.
func WithPreviousResponse(id string) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) { req.PreviousResponseId = &id }
}

// WithEncryptedContent enables encrypted content in the response.
func WithEncryptedContent(enabled bool) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) { req.UseEncryptedContent = enabled }
}

// ChatSession represents an active chat session.
type ChatSession struct {
	stub           xaipb.ChatClient
	request        *xaipb.GetCompletionsRequest
	conversationID string
}

// Append adds a message or response to the chat session.
func (s *ChatSession) Append(message any) *ChatSession {
	switch msg := message.(type) {
	case *xaipb.Message:
		s.request.Messages = append(s.request.Messages, msg)
	case *Response:
		if msg.index == nil {
			for _, out := range msg.proto.GetOutputs() {
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

// AppendToolResultJSON appends a tool result message with JSON payload (string or marshaled value).
// toolCallID is optional; if empty, it is omitted because the proto does not carry it.
func (s *ChatSession) AppendToolResultJSON(toolCallID string, result any) *ChatSession {
	var payload string
	switch v := result.(type) {
	case string:
		payload = v
	default:
		b, err := json.Marshal(v)
		if err != nil {
			panic(err)
		}
		payload = string(b)
	}
	msg := &xaipb.Message{
		Role:    xaipb.MessageRole_ROLE_TOOL,
		Content: []*xaipb.Content{TextContent(payload)},
	}
	// toolCallID currently not represented in proto; retained parameter for forward compatibility
	_ = toolCallID
	return s.Append(msg)
}

// Messages returns the current conversation history.
func (s *ChatSession) Messages() []*xaipb.Message { return s.request.GetMessages() }

// Sample sends the chat request and returns the first response.
func (s *ChatSession) Sample(ctx context.Context) (*Response, error) {
	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.sample %s", s.request.GetModel()),
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
	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.sample_batch %s", s.request.GetModel()),
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
	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.stream %s", s.request.GetModel()),
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
	stream.ctx = ctx
	return stream, nil
}

// StreamBatch returns a streaming iterator for multiple responses.
func (s *ChatSession) StreamBatch(ctx context.Context, n int) (*ChatStream, error) {
	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.stream_batch %s", s.request.GetModel()),
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
	stream.ctx = ctx
	return stream, nil
}

// Defer executes the request using deferred polling.
func (s *ChatSession) Defer(ctx context.Context, timeout, interval time.Duration) (*Response, error) {
	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.defer %s", s.request.GetModel()),
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
	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.defer_batch %s", s.request.GetModel()),
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
	if s.request.GetResponseFormat() != nil && s.request.GetResponseFormat().GetFormatType() == xaipb.FormatType_FORMAT_TYPE_JSON_SCHEMA {
		// allow caller to override schema via options; don't overwrite
		return s.parseWithRequest(ctx, out, proto.Clone(s.request).(*xaipb.GetCompletionsRequest))
	}
	schemaStr := string(schemaBytes)
	req := proto.Clone(s.request).(*xaipb.GetCompletionsRequest)
	req.ResponseFormat = &xaipb.ResponseFormat{
		FormatType: xaipb.FormatType_FORMAT_TYPE_JSON_SCHEMA,
		Schema:     &schemaStr,
	}
	return s.parseWithRequest(ctx, out, req)
}

// parseWithRequest executes a parse with the provided request.
func (s *ChatSession) parseWithRequest(ctx context.Context, out any, req *xaipb.GetCompletionsRequest) (*Response, error) {
	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.parse %s", s.request.GetModel()),
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

// ParseInto is a generic convenience for structured outputs into type T.
func ParseInto[T any](ctx context.Context, s *ChatSession) (*Response, *T, error) {
	var out T
	resp, err := s.Parse(ctx, &out)
	return resp, &out, err
}

func (s *ChatSession) sampleN(ctx context.Context, n int) ([]*Response, error) {
	if len(s.request.GetMessages()) == 0 {
		return nil, errors.New("chat request requires at least one message")
	}
	req := proto.Clone(s.request).(*xaipb.GetCompletionsRequest)
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
	if len(s.request.GetMessages()) == 0 {
		return nil, errors.New("chat request requires at least one message")
	}
	req := proto.Clone(s.request).(*xaipb.GetCompletionsRequest)
	setN(req, n)

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

func (s *ChatSession) deferN(ctx context.Context, n int, timeout, interval time.Duration) ([]*Response, error) {
	if len(s.request.GetMessages()) == 0 {
		return nil, errors.New("chat request requires at least one message")
	}
	req := proto.Clone(s.request).(*xaipb.GetCompletionsRequest)
	setN(req, n)
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
	index := 0
	if usesServerSideTools(req.GetTools()) {
		index = -1
	}
	idxPtr := (*int)(nil)
	if index >= 0 {
		idxPtr = &index
	}
	idxPtr = autoDetectMultiOutput(idxPtr, resp.GetOutputs())
	return newResponse(resp, idxPtr), nil
}

func (s *ChatSession) makeSpanRequestAttributes() []attribute.KeyValue {
	attrs := make([]attribute.KeyValue, 0, 16+len(s.request.GetMessages())*3)
	attrs = append(attrs,
		attribute.String("gen_ai.operation.name", "chat"),
		attribute.String("gen_ai.system", "xai"),
		attribute.String("gen_ai.output.type", "text"),
		attribute.String("gen_ai.request.model", s.request.GetModel()),
		attribute.Int("server.port", 443),
	)

	// Optional fields
	attrs = append(attrs,
		attribute.Float64("gen_ai.request.frequency_penalty", float64(valueOrZeroFloat32(s.request.FrequencyPenalty))),
		attribute.Float64("gen_ai.request.presence_penalty", float64(valueOrZeroFloat32(s.request.PresencePenalty))),
		attribute.Float64("gen_ai.request.temperature", float64(valueOrZeroFloat32(s.request.Temperature))),
		attribute.Bool("gen_ai.request.parallel_tool_calls", valueOrZeroBool(s.request.ParallelToolCalls)),
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
	if len(s.request.GetStop()) > 0 {
		attrs = append(attrs, attribute.StringSlice("gen_ai.request.stop_sequences", s.request.GetStop()))
	}
	if s.request.GetResponseFormat() != nil {
		attrs = append(attrs, attribute.String("gen_ai.output.type", strings.ToLower(strings.TrimPrefix(s.request.GetResponseFormat().GetFormatType().String(), "FORMAT_TYPE_"))))
	}
	if s.request.ReasoningEffort != nil {
		attrs = append(attrs, attribute.String("gen_ai.request.reasoning_effort", strings.ToLower(strings.TrimPrefix(s.request.GetReasoningEffort().String(), "EFFORT_"))))
	}
	if s.request.GetUser() != "" {
		attrs = append(attrs, attribute.String("user_id", s.request.GetUser()))
	}
	if s.request.PreviousResponseId != nil {
		attrs = append(attrs, attribute.String("gen_ai.request.previous_response_id", s.request.GetPreviousResponseId()))
	}

	// Prompt attributes
	for i, msg := range s.request.GetMessages() {
		prefix := fmt.Sprintf("gen_ai.prompt.%d", i)
		role := strings.ToLower(strings.TrimPrefix(msg.GetRole().String(), "ROLE_"))
		attrs = append(attrs, attribute.String(prefix+".role", role))

		var contentStr strings.Builder
		for _, c := range msg.GetContent() {
			if txt := c.GetText(); txt != "" {
				contentStr.WriteString(txt)
			}
		}
		attrs = append(attrs, attribute.String(prefix+".content", contentStr.String()))

		if len(msg.GetToolCalls()) > 0 {
			// Serialize tool calls mostly for debug, simplified here
			// Python does full JSON serialization
			if b, err := json.Marshal(msg.GetToolCalls()); err == nil {
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

	attrs := make([]attribute.KeyValue, 0, 12+len(responses)*4)
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

func valueOrZeroFloat32[T ~float32](p *T) T {
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
	stream             xaipb.Chat_GetCompletionChunkClient
	response           *Response
	span               trace.Span
	firstChunkReceived bool
	ctx                context.Context
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
	if err != nil {
		if s.span != nil {
			if !errors.Is(err, io.EOF) {
				s.span.RecordError(err)
			} else {
				usage := s.response.Usage()
				if usage != nil {
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
			}
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
	index             *int
	contentBuffers    []*strings.Builder
	reasoningBuffers  []*strings.Builder
	encryptedBuffers  []*strings.Builder
	buffersAreInProto bool
}

func newResponse(protoResp *xaipb.GetChatCompletionResponse, index *int) *Response {
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
func (r *Response) Usage() *xaipb.SamplingUsage { return r.proto.GetUsage() }

// Citations returns any citations returned by the model.
func (r *Response) Citations() []string { return r.proto.GetCitations() }

// SystemFingerprint returns system fingerprint.
func (r *Response) SystemFingerprint() string { return r.proto.GetSystemFingerprint() }

func (r *Response) output() *xaipb.CompletionOutput {
	r.flushBuffers()
	return r.outputNoFlush()
}

func (r *Response) outputNoFlush() *xaipb.CompletionOutput {
	var outputs []*xaipb.CompletionOutput
	for _, out := range r.proto.GetOutputs() {
		if out == nil || out.GetMessage() == nil {
			continue
		}
		if out.GetMessage().GetRole() == xaipb.MessageRole_ROLE_ASSISTANT && (r.index == nil || out.GetIndex() == int32(valueOrZero(r.index))) {
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
}

func (r *Response) processChunk(chunk *xaipb.GetChatCompletionChunk) {
	r.proto.Usage = chunk.GetUsage()
	r.proto.Created = chunk.GetCreated()
	r.proto.Id = chunk.GetId()
	r.proto.Model = chunk.GetModel()
	r.proto.SystemFingerprint = chunk.GetSystemFingerprint()
	r.proto.Citations = append(r.proto.Citations, chunk.GetCitations()...)

	maxIndex := 0
	hasContent := false
	hasReasoning := false
	hasEncrypted := false
	for _, out := range chunk.GetOutputs() {
		if idx := int(out.GetIndex()); idx > maxIndex {
			maxIndex = idx
		}
		hasContent = hasContent || out.GetDelta().GetContent() != ""
		hasReasoning = hasReasoning || out.GetDelta().GetReasoningContent() != ""
		hasEncrypted = hasEncrypted || out.GetDelta().GetEncryptedContent() != ""
	}
	if needed := maxIndex + 1 - len(r.proto.GetOutputs()); needed > 0 {
		start := len(r.proto.GetOutputs())
		r.proto.Outputs = append(r.proto.Outputs, make([]*xaipb.CompletionOutput, needed)...)
		for i := start; i < len(r.proto.GetOutputs()); i++ {
			r.proto.Outputs[i] = nil
		}
	}

	if size := maxIndex + 1; size > 0 {
		if hasContent {
			growBuilders(&r.contentBuffers, size)
		}
		if hasReasoning {
			growBuilders(&r.reasoningBuffers, size)
		}
		if hasEncrypted {
			growBuilders(&r.encryptedBuffers, size)
		}
	}

	for _, c := range chunk.GetOutputs() {
		target := r.proto.GetOutputs()[c.GetIndex()]
		if target == nil {
			target = &xaipb.CompletionOutput{}
			r.proto.Outputs[c.GetIndex()] = target
		}
		target.Index = c.GetIndex()
		if target.GetMessage() == nil {
			target.Message = &xaipb.CompletionMessage{}
		}
		target.Message.Role = c.GetDelta().GetRole()
		target.Message.ToolCalls = append(target.Message.ToolCalls, c.GetDelta().GetToolCalls()...)
		target.FinishReason = c.GetFinishReason()

		if c.GetDelta().GetContent() != "" {
			ensureBuilder(&r.contentBuffers, int(c.GetIndex())).WriteString(c.GetDelta().GetContent())
			r.buffersAreInProto = false
		}
		if c.GetDelta().GetReasoningContent() != "" {
			ensureBuilder(&r.reasoningBuffers, int(c.GetIndex())).WriteString(c.GetDelta().GetReasoningContent())
			r.buffersAreInProto = false
		}
		if c.GetDelta().GetEncryptedContent() != "" {
			ensureBuilder(&r.encryptedBuffers, int(c.GetIndex())).WriteString(c.GetDelta().GetEncryptedContent())
			r.buffersAreInProto = false
		}
	}
}

// Chunk wraps GetChatCompletionChunk with helpers.
type Chunk struct {
	proto *xaipb.GetChatCompletionChunk
	index *int
}

func newChunk(protoChunk *xaipb.GetChatCompletionChunk, index *int) *Chunk {
	return &Chunk{
		proto: protoChunk,
		index: index,
	}
}

// Content concatenates chunk content for the tracked index (or all when multi-output).
func (c *Chunk) Content() string {
	var b strings.Builder
	for _, out := range c.outputs() {
		b.WriteString(out.GetDelta().GetContent())
	}
	return b.String()
}

// ReasoningContent concatenates reasoning content for tracked outputs.
func (c *Chunk) ReasoningContent() string {
	var b strings.Builder
	for _, out := range c.outputs() {
		b.WriteString(out.GetDelta().GetReasoningContent())
	}
	return b.String()
}

// ToolCalls returns tool calls for this chunk.
func (c *Chunk) ToolCalls() []*xaipb.ToolCall {
	var calls []*xaipb.ToolCall
	for _, out := range c.outputs() {
		calls = append(calls, out.GetDelta().GetToolCalls()...)
	}
	return calls
}

// Outputs returns the raw chunk outputs filtered by index.
func (c *Chunk) outputs() []*xaipb.CompletionOutputChunk {
	var outs []*xaipb.CompletionOutputChunk
	for _, out := range c.proto.GetOutputs() {
		if out.GetDelta().GetRole() == xaipb.MessageRole_ROLE_ASSISTANT && (c.index == nil || out.GetIndex() == int32(*c.index)) {
			outs = append(outs, out)
		}
	}
	return outs
}

// Convenience builders for messages and content.

// User creates a user message with text or content parts.
func User(parts ...any) *xaipb.Message { return newMessage(xaipb.MessageRole_ROLE_USER, parts...) }

// System creates a system message.
func System(parts ...any) *xaipb.Message { return newMessage(xaipb.MessageRole_ROLE_SYSTEM, parts...) }

// Assistant creates an assistant message.
func Assistant(parts ...any) *xaipb.Message {
	return newMessage(xaipb.MessageRole_ROLE_ASSISTANT, parts...)
}

// ToolResult creates a tool result message.
func ToolResult(result string) *xaipb.Message { return newMessage(xaipb.MessageRole_ROLE_TOOL, result) }

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
	return &xaipb.Content{Content: &xaipb.Content_ImageUrl{
		ImageUrl: &xaipb.ImageUrlContent{
			ImageUrl: url,
			Detail:   detail,
		},
	}}
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

func autoDetectMultiOutput(index *int, outputs []*xaipb.CompletionOutput) *int {
	if index != nil {
		maxIdx := int32(valueOrZero(index))
		for _, out := range outputs {
			if out.GetIndex() > maxIdx {
				return nil
			}
		}
	}
	return index
}

func autoDetectMultiOutputChunks(index *int, outputs []*xaipb.CompletionOutputChunk) *int {
	if index != nil {
		maxIdx := valueOrZero(index)
		for _, out := range outputs {
			if int(out.GetIndex()) > maxIdx {
				return nil
			}
		}
	}
	return index
}

func setN(req *xaipb.GetCompletionsRequest, n int) {
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
		(*bufs)[idx] = &strings.Builder{}
	}
	return (*bufs)[idx]
}

func growBuilders(bufs *[]*strings.Builder, size int) {
	if size <= len(*bufs) {
		return
	}
	extra := size - len(*bufs)
	*bufs = slices.Grow(*bufs, extra)
	*bufs = (*bufs)[:size]
}

func splitResponses(resp *xaipb.GetChatCompletionResponse, n int) []*Response {
	responses := make([]*Response, n)
	for i := range n {
		idx := i
		responses[i] = newResponse(resp, &idx)
	}
	return responses
}
