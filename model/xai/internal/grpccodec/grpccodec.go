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

// Package grpccodec provides a gRPC codec that uses vtproto for serialization.
package grpccodec

import (
	"unsafe"

	"google.golang.org/grpc/encoding"
	grpcmem "google.golang.org/grpc/mem"

	// Guarantee that the built-in proto is called registered before this one
	// so that it can be replaced.
	_ "google.golang.org/grpc/encoding/proto"

	"github.com/zchee/tumix/model/xai/internal/grpccodec/mem"
)

func init() {
	encoding.RegisterCodecV2(&Codec{
		fallback: encoding.GetCodecV2("proto"),
	})
}

type vtprotoMessage interface {
	MarshalToSizedBufferVT(data []byte) (int, error)
	UnmarshalVT(data []byte) error
	SizeVT() int
}

// Name is the name registered for the proto compressor.
const Name = "proto"

// Codec represents a gRPC [encoding.CodecV2] that uses vtproto for messages that
// implement the vtprotoMessage interface, and falls back to the built-in proto.
type Codec struct {
	fallback encoding.CodecV2
}

var _ encoding.CodecV2 = (*Codec)(nil)

// Name implements [encoding.CodecV2].
func (Codec) Name() string { return Name }

// Marshal implements [encoding.CodecV2].
func (c *Codec) Marshal(v any) (grpcmem.BufferSlice, error) {
	if m, ok := v.(vtprotoMessage); ok {
		size := m.SizeVT()
		if mem.IsBelowBufferPoolingThreshold(size) {
			buf := make([]byte, size)
			if _, err := m.MarshalToSizedBufferVT(buf); err != nil {
				return nil, err
			}
			buffer := mem.SliceBuffer(buf)
			gbuffer := (*grpcmem.SliceBuffer)(unsafe.Pointer(&buffer))
			return grpcmem.BufferSlice{gbuffer}, nil
		}

		pool := mem.DefaultBufferPool()
		buf := pool.Get(size)
		if _, err := m.MarshalToSizedBufferVT((*buf)[:size]); err != nil {
			pool.Put(buf)
			return nil, err
		}

		buffer := mem.NewBuffer(buf, pool)
		gbuffer := *(*grpcmem.Buffer)(unsafe.Pointer(&buffer))
		return grpcmem.BufferSlice{gbuffer}, nil
	}

	return c.fallback.Marshal(v)
}

// Unmarshal implements [encoding.CodecV2].
func (c *Codec) Unmarshal(data grpcmem.BufferSlice, v any) error {
	data2 := *(*mem.BufferSlice)(unsafe.Pointer(&data))

	if m, ok := v.(vtprotoMessage); ok {
		buf := data2.MaterializeToBuffer(mem.DefaultBufferPool())
		defer buf.Free()
		return m.UnmarshalVT(buf.ReadOnlyData())
	}

	return c.fallback.Unmarshal(data, v)
}
