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
	"time"

	"google.golang.org/protobuf/proto"

	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
)

const (
	defaultDeferredTimeout  = 10 * time.Minute
	defaultDeferredInterval = 100 * time.Millisecond
)

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

	stream, err := s.chat.GetCompletionChunk(ctx, req)
	if err != nil {
		return nil, err
	}

	resp := &xaipb.GetChatCompletionResponse{}
	if n > 1 {
		resp.Outputs = make([]*xaipb.CompletionOutput, n)
	}

	intPtrIf := func(condition bool) *int32 {
		if !condition {
			return nil
		}
		return ptr(int32(0))
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

	startResp, err := s.chat.StartDeferredCompletion(ctx, req)
	if err != nil {
		return nil, WrapError(err)
	}

	deadline := time.Now().Add(timeout)
	for {
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("deferred request timed out after %s", timeout)
		}

		res, err := s.chat.GetDeferredCompletion(ctx, &xaipb.GetDeferredRequest{
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
	resp, err := s.chat.GetCompletion(ctx, req)
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

func splitResponses(resp *xaipb.GetChatCompletionResponse, n int32) []*Response {
	responses := make([]*Response, n)
	for i := range n {
		responses[i] = newResponse(resp, &i)
	}

	return responses
}
