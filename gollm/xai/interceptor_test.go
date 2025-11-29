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
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

func TestTimeoutStreamInterceptor_NoImplicitCancel(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	interceptor := TimeoutStreamInterceptor(10 * time.Millisecond)

	streamDesc := &grpc.StreamDesc{ServerStreams: true}
	streamerCalled := false

	stream, err := interceptor(ctx, streamDesc, nil, "/xai.Chat/GetCompletionChunk",
		func(ctx context.Context, desc *grpc.StreamDesc, _ *grpc.ClientConn, _ string, _ ...grpc.CallOption) (grpc.ClientStream, error) {
			streamerCalled = true
			if desc != streamDesc {
				t.Fatalf("stream desc mismatch")
			}
			if _, ok := ctx.Deadline(); !ok {
				t.Fatalf("expected deadline set by interceptor")
			}
			if err := ctx.Err(); err != nil {
				t.Fatalf("context already canceled: %v", err)
			}
			return &noopClientStream{ctx: ctx}, nil
		})
	if err != nil {
		t.Fatalf("interceptor returned error: %v", err)
	}
	if !streamerCalled {
		t.Fatalf("streamer was not invoked")
	}
	if err := stream.Context().Err(); err != nil {
		t.Fatalf("stream context canceled prematurely: %v", err)
	}

	if err := stream.RecvMsg(nil); err != nil {
		t.Fatalf("RecvMsg error: %v", err)
	}
	if err := stream.Context().Err(); err != nil {
		t.Fatalf("context canceled after RecvMsg nil: %v", err)
	}

	if err := stream.RecvMsg(nil); err == nil {
		t.Fatalf("expected EOF on second RecvMsg")
	}
	if stream.Context().Err() == nil {
		t.Fatalf("context should be canceled after terminal RecvMsg")
	}
}

type noopClientStream struct {
	ctx       context.Context
	recvCalls int
}

//nolint:nilnil
func (n *noopClientStream) Header() (metadata.MD, error) { return nil, nil }

func (n *noopClientStream) Trailer() metadata.MD { return nil }

func (n *noopClientStream) CloseSend() error { return nil }

func (n *noopClientStream) Context() context.Context {
	if n.ctx != nil {
		return n.ctx
	}
	return context.Background()
}

func (n *noopClientStream) SendMsg(any) error { return nil }

func (n *noopClientStream) RecvMsg(any) error {
	n.recvCalls++
	if n.recvCalls >= 2 {
		return context.Canceled // sentinel to simulate terminal error
	}
	return nil
}
