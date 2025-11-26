/*
 *
 * Copyright 2025 tumix authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package mem

import (
	"errors"
	"io"
	"testing"
)

var (
	materializeSink []byte
	readerCountSink int
)

func makeBufferSlice(pool BufferPool, segments, segmentSize int) BufferSlice {
	buffers := make(BufferSlice, 0, segments)
	for i := range segments {
		buf := pool.Get(segmentSize)
		(*buf)[0] = byte(i)
		buffers = append(buffers, NewBuffer(buf, pool))
	}

	return buffers
}

func BenchmarkBufferSliceMaterialize(b *testing.B) {
	pool := DefaultBufferPool()

	tests := []struct {
		name        string
		segments    int
		segmentSize int
	}{
		{name: "single_segment", segments: 1, segmentSize: 8 << 10},
		{name: "multi_segment", segments: 4, segmentSize: 8 << 10},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			slice := makeBufferSlice(pool, tt.segments, tt.segmentSize)
			b.Cleanup(slice.Free)

			b.ReportAllocs()
			b.SetBytes(int64(slice.Len()))

			for b.Loop() {
				materializeSink = slice.Materialize()
			}
		})
	}
}

func BenchmarkBufferSliceMaterializeToBuffer(b *testing.B) {
	pool := DefaultBufferPool()
	slice := makeBufferSlice(pool, 4, 8<<10)

	b.Cleanup(slice.Free)

	b.ReportAllocs()
	b.SetBytes(int64(slice.Len()))

	for b.Loop() {
		buf := slice.MaterializeToBuffer(pool)
		readerCountSink += buf.Len()
		buf.Free()
	}
}

func BenchmarkBufferSliceReader(b *testing.B) {
	pool := DefaultBufferPool()
	slice := makeBufferSlice(pool, 4, 8<<10)
	reader := slice.Reader()

	b.Cleanup(slice.Free)
	b.Cleanup(func() {
		_ = reader.Close()
	})

	scratch := make([]byte, 1<<10)

	b.ReportAllocs()
	b.SetBytes(int64(slice.Len()))

	for b.Loop() {
		reader.Reset(slice)
		for {
			n, err := reader.Read(scratch)
			readerCountSink += n
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				b.Fatalf("unexpected read error: %v", err)
			}
		}
	}
}
