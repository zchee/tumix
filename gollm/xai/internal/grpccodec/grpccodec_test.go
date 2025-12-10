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

package grpccodec

import (
	"bytes"
	"errors"
	"fmt"
	"testing"
	"unsafe"

	"google.golang.org/grpc/encoding"
	grpcmem "google.golang.org/grpc/mem"

	"github.com/zchee/tumix/gollm/xai/internal/grpccodec/mem"
)

func TestCodecName(t *testing.T) {
	t.Helper()

	if got := (&Codec{}).Name(); got != Name {
		t.Fatalf("Name() = %q, want %q", got, Name)
	}
}

func TestCodecMarshalVTProtoSmallPayload(t *testing.T) {
	payload := []byte("small vtproto message")
	if !mem.IsBelowBufferPoolingThreshold(len(payload)) {
		t.Fatalf("payload size %d exceeds pooling threshold", len(payload))
	}

	msg := &vtprotoStub{payload: payload}
	codec := Codec{fallback: &stubCodec{}}

	got, err := codec.Marshal(msg)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}

	if msg.marshalCalls != 1 {
		t.Fatalf("MarshalToSizedBufferVT called %d times, want 1", msg.marshalCalls)
	}

	if len(got) != 1 {
		t.Fatalf("Marshal() returned %d buffers, want 1", len(got))
	}

	if data := got.Materialize(); !bytes.Equal(data, payload) {
		t.Fatalf("Marshal() data = %v, want %v", data, payload)
	}

	got.Free()
}

func TestCodecMarshalVTProtoPooledBuffer(t *testing.T) {
	size := pooledSizeAboveThreshold()
	payload := bytes.Repeat([]byte{0xAB}, size)

	msg := &vtprotoStub{payload: payload}
	codec := Codec{fallback: &stubCodec{}}

	got, err := codec.Marshal(msg)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}

	if msg.marshalCalls != 1 {
		t.Fatalf("MarshalToSizedBufferVT called %d times, want 1", msg.marshalCalls)
	}

	if len(got) != 1 {
		t.Fatalf("Marshal() returned %d buffers, want 1", len(got))
	}

	if data := got.Materialize(); !bytes.Equal(data, payload) {
		t.Fatalf("Marshal() data mismatch")
	}

	if _, ok := got[0].(grpcmem.SliceBuffer); ok {
		t.Fatalf("Marshal() returned mem.SliceBuffer; want pooled buffer")
	}

	got.Free()
}

func TestCodecMarshalVTProtoError(t *testing.T) {
	t.Parallel()

	errWant := errors.New("marshal failure")
	msg := &vtprotoStub{
		payload:    bytes.Repeat([]byte{0xCD}, pooledSizeAboveThreshold()),
		marshalErr: errWant,
	}
	codec := Codec{fallback: &stubCodec{}}

	got, err := codec.Marshal(msg)
	if !errors.Is(err, errWant) {
		t.Fatalf("Marshal() error = %v, want %v", err, errWant)
	}

	if got != nil {
		t.Fatalf("Marshal() buffers = %v, want nil", got)
	}

	if msg.marshalCalls != 1 {
		t.Fatalf("MarshalToSizedBufferVT called %d times, want 1", msg.marshalCalls)
	}
}

func TestCodecUnmarshalVTProto(t *testing.T) {
	payload := []byte("unmarshal vtproto payload")
	msg := &vtprotoStub{}
	codec := Codec{
		fallback: &stubCodec{},
	}

	buffer := mem.SliceBuffer(payload)
	gbuffer := (*grpcmem.SliceBuffer)(unsafe.Pointer(&buffer))
	data := grpcmem.BufferSlice{gbuffer}

	if err := codec.Unmarshal(data, msg); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}

	if msg.unmarshalCalls != 1 {
		t.Fatalf("UnmarshalVT called %d times, want 1", msg.unmarshalCalls)
	}

	if !bytes.Equal(msg.unmarshalGot, payload) {
		t.Fatalf("UnmarshalVT received %v, want %v", msg.unmarshalGot, payload)
	}
}

func TestCodecMarshalFallback(t *testing.T) {
	buffer := mem.SliceBuffer([]byte("fallback data"))
	gbuffer := (*grpcmem.SliceBuffer)(unsafe.Pointer(&buffer))
	want := grpcmem.BufferSlice{gbuffer}
	fallback := &stubCodec{
		returned: want,
	}
	codec := Codec{
		fallback: fallback,
	}

	got, err := codec.Marshal(struct{ A int }{A: 1})
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}

	if fallback.marshalCalls != 1 {
		t.Fatalf("fallback Marshal called %d times, want 1", fallback.marshalCalls)
	}

	if len(got) != len(want) {
		t.Fatalf("Marshal() returned %d buffers, want %d", len(got), len(want))
	}

	if data := got.Materialize(); !bytes.Equal(data, want.Materialize()) {
		t.Fatalf("Marshal() data mismatch")
	}
}

func TestCodecUnmarshalFallback(t *testing.T) {
	fallback := &stubCodec{}
	codec := Codec{fallback: fallback}

	buffer := mem.SliceBuffer([]byte("fallback unmarshal payload"))
	gbuffer := (*grpcmem.SliceBuffer)(unsafe.Pointer(&buffer))
	data := grpcmem.BufferSlice{gbuffer}
	target := &struct{ A int }{}

	if err := codec.Unmarshal(data, target); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}

	if fallback.unmarshalCalls != 1 {
		t.Fatalf("fallback Unmarshal called %d times, want 1", fallback.unmarshalCalls)
	}

	if fallback.lastUnmarshalValue != target {
		t.Fatalf("fallback Unmarshal target mismatch: got %v, want %v", fallback.lastUnmarshalValue, target)
	}

	if gotData := fallback.lastUnmarshalData.Materialize(); !bytes.Equal(gotData, data.Materialize()) {
		t.Fatalf("fallback Unmarshal data mismatch")
	}
}

func pooledSizeAboveThreshold() int {
	size := 1
	for mem.IsBelowBufferPoolingThreshold(size) {
		size *= 2
	}
	return size
}

type vtprotoStub struct {
	payload        []byte
	marshalErr     error
	unmarshalErr   error
	marshalCalls   int
	unmarshalCalls int
	unmarshalGot   []byte
}

func (m *vtprotoStub) MarshalToSizedBufferVT(dst []byte) (int, error) {
	m.marshalCalls++

	if m.marshalErr != nil {
		return 0, m.marshalErr
	}

	if len(dst) < len(m.payload) {
		return 0, fmt.Errorf("dst too small: got %d want %d", len(dst), len(m.payload))
	}

	copy(dst, m.payload)
	return len(m.payload), nil
}

func (m *vtprotoStub) UnmarshalVT(data []byte) error {
	m.unmarshalCalls++

	if m.unmarshalErr != nil {
		return m.unmarshalErr
	}

	m.unmarshalGot = append([]byte(nil), data...)
	return nil
}

func (m *vtprotoStub) SizeVT() int {
	return len(m.payload)
}

type stubCodec struct {
	marshalCalls       int
	unmarshalCalls     int
	marshalErr         error
	unmarshalErr       error
	returned           grpcmem.BufferSlice
	lastMarshalValue   any
	lastUnmarshalData  grpcmem.BufferSlice
	lastUnmarshalValue any
}

var _ encoding.CodecV2 = (*stubCodec)(nil)

func (c *stubCodec) Name() string { return "stub" }

func (c *stubCodec) Marshal(v any) (grpcmem.BufferSlice, error) {
	c.marshalCalls++
	c.lastMarshalValue = v

	if c.marshalErr != nil {
		return nil, c.marshalErr
	}

	return c.returned, nil
}

func (c *stubCodec) Unmarshal(data grpcmem.BufferSlice, v any) error {
	c.unmarshalCalls++
	c.lastUnmarshalData = data
	c.lastUnmarshalValue = v

	if c.unmarshalErr != nil {
		return c.unmarshalErr
	}

	return nil
}
