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
	"io"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

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
