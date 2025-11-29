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
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

func AuthUnaryInterceptor(token string, md map[string]string) grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, callOpts ...grpc.CallOption) error {
		ctx = attachMetadata(ctx, token, md)
		return invoker(ctx, method, req, reply, cc, callOpts...)
	}
}

func AuthStreamInterceptor(token string, md map[string]string) grpc.StreamClientInterceptor {
	return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, callOpts ...grpc.CallOption) (grpc.ClientStream, error) {
		ctx = attachMetadata(ctx, token, md)
		return streamer(ctx, desc, cc, method, callOpts...)
	}
}

func TimeoutUnaryInterceptor(timeout time.Duration) grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, callOpts ...grpc.CallOption) error {
		if _, ok := ctx.Deadline(); !ok && timeout > 0 {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, timeout)
			defer cancel()
		}
		return invoker(ctx, method, req, reply, cc, callOpts...)
	}
}

func TimeoutStreamInterceptor(timeout time.Duration) grpc.StreamClientInterceptor {
	return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, callOpts ...grpc.CallOption) (grpc.ClientStream, error) {
		if timeout <= 0 {
			return streamer(ctx, desc, cc, method, callOpts...)
		}
		if _, ok := ctx.Deadline(); ok {
			return streamer(ctx, desc, cc, method, callOpts...)
		}

		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)

		stream, err := streamer(ctx, desc, cc, method, callOpts...)
		if err != nil {
			cancel()
			return stream, err
		}

		return &cancelOnCloseClientStream{
			ClientStream: stream,
			cancel:       cancel,
		}, nil
	}
}

type cancelOnCloseClientStream struct {
	grpc.ClientStream
	cancel func()
	once   sync.Once
}

func (c *cancelOnCloseClientStream) CloseSend() error {
	err := c.ClientStream.CloseSend()
	c.once.Do(func() {
		if c.cancel != nil {
			c.cancel()
		}
	})
	return err
}

func (c *cancelOnCloseClientStream) RecvMsg(m any) error {
	err := c.ClientStream.RecvMsg(m)
	if err != nil {
		c.once.Do(func() {
			if c.cancel != nil {
				c.cancel()
			}
		})
	}
	return err
}

func attachMetadata(ctx context.Context, token string, static map[string]string) context.Context {
	pairs := []string{"authorization", "Bearer " + token}
	for k, v := range static {
		pairs = append(pairs, k, v)
	}
	return metadata.AppendToOutgoingContext(ctx, pairs...)
}
