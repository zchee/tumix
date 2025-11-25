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
	"errors"
	"fmt"
	"reflect"
	"sync"
	"time"

	"github.com/bytedance/sonic"
	"github.com/invopop/jsonschema"
	xaipb "github.com/zchee/tumix/model/xai/api/v1"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/protobuf/proto"
)

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
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		req.User = user
	}
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(maxToken int32) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		req.MaxTokens = &maxToken
	}
}

// WithSeed sets the random seed for deterministic generation.
func WithSeed(seed int32) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		req.Seed = &seed
	}
}

// WithStop sets the stop sequences.
func WithStop(stop ...string) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		req.Stop = append(req.Stop, stop...)
	}
}

// WithTemperature sets the sampling temperature.
func WithTemperature(t float32) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		req.Temperature = &t
	}
}

// WithTopP sets the nucleus sampling probability.
func WithTopP(p float32) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		req.TopP = &p
	}
}

// WithLogprobs enables log probabilities return.
func WithLogprobs(enabled bool) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) { req.Logprobs = enabled }
}

// WithTopLogprobs sets the number of top log probabilities to return.
func WithTopLogprobs(v int32) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		req.TopLogprobs = &v
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
		req.ParallelToolCalls = &enabled
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
		typ := reflect.TypeOf((*T)(nil)).Elem()
		b, err := schemaBytesForType(typ)
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
		req.FrequencyPenalty = &v
	}
}

// WithPresencePenalty sets the presence penalty.
func WithPresencePenalty(v float32) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		req.PresencePenalty = &v
	}
}

// WithReasoningEffort sets the reasoning effort level.
func WithReasoningEffort(effort xaipb.ReasoningEffort) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		req.ReasoningEffort = &effort
	}
}

// WithSearchParameters sets the search parameters for the request.
func WithSearchParameters(params *xaipb.SearchParameters) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		req.SearchParameters = params
	}
}

// WithSearch configures search using the helper struct.
func WithSearch(params SearchParameters) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		req.SearchParameters = params.Proto()
	}
}

// WithStoreMessages enables message storage.
func WithStoreMessages(store bool) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		req.StoreMessages = store
	}
}

// WithPreviousResponse sets the previous response ID for context.
func WithPreviousResponse(id string) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		req.PreviousResponseId = &id
	}
}

// WithEncryptedContent enables encrypted content in the response.
func WithEncryptedContent(enabled bool) ChatOption {
	return func(req *xaipb.GetCompletionsRequest, _ *ChatSession) {
		req.UseEncryptedContent = enabled
	}
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
		switch {
		case msg.index == nil:
			for _, out := range msg.proto.GetOutputs() {
				s.request.Messages = append(s.request.Messages, buildMessageFromCompletion(out))
			}

		case msg.output() != nil:
			s.request.Messages = append(s.request.Messages, buildMessageFromCompletion(msg.output()))
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
		b, err := sonic.ConfigFastest.Marshal(v)
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
func (s *ChatSession) Messages() []*xaipb.Message {
	return s.request.GetMessages()
}

// Sample sends the chat request and returns the first response.
func (s *ChatSession) Sample(ctx context.Context) (*Response, error) {
	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.sample %s", s.request.GetModel()),
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(s.makeSpanRequestAttributes()...),
	)
	defer span.End()

	responses, err := s.sampleN(ctx, 1)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		span.RecordError(err)
		return nil, err
	}

	span.SetAttributes(s.makeSpanResponseAttributes(responses)...)
	span.SetStatus(codes.Ok, "")

	return responses[0], nil
}

// SampleBatch requests n responses in a single call.
func (s *ChatSession) SampleBatch(ctx context.Context, n int32) ([]*Response, error) {
	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.sample_batch %s", s.request.GetModel()),
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(s.makeSpanRequestAttributes()...),
	)
	defer span.End()

	responses, err := s.sampleN(ctx, n)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		span.RecordError(err)
		return nil, err
	}

	span.SetAttributes(s.makeSpanResponseAttributes(responses)...)
	span.SetStatus(codes.Ok, "")

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
		span.SetStatus(codes.Error, err.Error())
		span.RecordError(err)
		span.End()
		return nil, err
	}

	stream.span = span
	stream.ctx = ctx

	return stream, nil
}

// StreamBatch returns a streaming iterator for multiple responses.
func (s *ChatSession) StreamBatch(ctx context.Context, n int32) (*ChatStream, error) {
	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.stream_batch %s", s.request.GetModel()),
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(s.makeSpanRequestAttributes()...),
	)

	stream, err := s.streamN(ctx, n)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
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
		span.SetStatus(codes.Error, err.Error())
		span.RecordError(err)
		return nil, err
	}

	span.SetAttributes(s.makeSpanResponseAttributes(responses)...)

	return responses[0], nil
}

// DeferBatch executes the request using deferred polling and returns n responses.
func (s *ChatSession) DeferBatch(ctx context.Context, n int32, timeout, interval time.Duration) ([]*Response, error) {
	ctx, span := tracer.Start(ctx, fmt.Sprintf("chat.defer_batch %s", s.request.GetModel()),
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(s.makeSpanRequestAttributes()...),
	)
	defer span.End()

	responses, err := s.deferN(ctx, n, timeout, interval)
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		span.RecordError(err)
		return nil, err
	}

	span.SetAttributes(s.makeSpanResponseAttributes(responses)...)

	return responses, nil
}

// Parse sets response_format to a JSON schema derived from the provided sample value and decodes into it.
// Pass a pointer to a struct value to populate it.
func (s *ChatSession) Parse(ctx context.Context, out any) (*Response, error) {
	if s.request.GetResponseFormat() != nil && s.request.GetResponseFormat().GetFormatType() == xaipb.FormatType_FORMAT_TYPE_JSON_SCHEMA {
		// allow caller to override schema via options; don't overwrite
		return s.parseWithRequest(ctx, out, proto.Clone(s.request).(*xaipb.GetCompletionsRequest))
	}

	schemaBytes, err := schemaBytesForValue(out)
	if err != nil {
		return nil, err
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
		span.SetStatus(codes.Error, err.Error())
		span.RecordError(err)
		return nil, err
	}
	span.SetAttributes(s.makeSpanResponseAttributes([]*Response{resp})...)

	if err := sonic.ConfigFastest.Unmarshal([]byte(resp.Content()), out); err != nil {
		span.SetStatus(codes.Error, err.Error())
		span.RecordError(err)
		return resp, err
	}

	return resp, nil
}

var jsonSchemaCache sync.Map

func schemaBytesForValue(v any) ([]byte, error) {
	if v == nil {
		return nil, errors.New("schema value must be non-nil")
	}
	return schemaBytesForType(reflect.TypeOf(v))
}

func schemaBytesForType(t reflect.Type) ([]byte, error) {
	if t == nil {
		return nil, errors.New("schema type is nil")
	}
	if cached, ok := jsonSchemaCache.Load(t); ok {
		return cached.([]byte), nil
	}

	refl := &jsonschema.Reflector{}
	zero := reflect.New(t).Elem().Interface()
	schema := refl.Reflect(zero)
	if schema == nil {
		return nil, errors.New("schema reflection returned nil")
	}

	b, err := sonic.ConfigFastest.Marshal(schema)
	if err != nil {
		return nil, err
	}
	jsonSchemaCache.Store(t, b)
	return b, nil
}
