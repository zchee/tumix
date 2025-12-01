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
	"bytes"
	"testing"
)

var (
	bufferSink Buffer
	lenSink    int
)

func BenchmarkNewBuffer(b *testing.B) {
	pool := DefaultBufferPool()

	tests := []struct {
		name string
		size int
	}{
		{name: "4KB", size: 4 << 10},
		{name: "256KB", size: 256 << 10},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(tt.size))

			for b.Loop() {
				buf := pool.Get(tt.size)
				bufferSink = NewBuffer(buf, pool)
				bufferSink.Free()
			}
		})
	}
}

func BenchmarkCopy(b *testing.B) {
	pool := DefaultBufferPool()

	sources := []struct {
		name string
		data []byte
	}{
		{name: "small_below_threshold", data: bytes.Repeat([]byte("a"), 512)},
		{name: "pooled", data: bytes.Repeat([]byte("b"), 8<<10)},
	}

	for _, src := range sources {
		b.Run(src.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(src.data)))

			for b.Loop() {
				bufferSink = Copy(src.data, pool)
				lenSink += bufferSink.Len()
				bufferSink.Free()
			}
		})
	}
}

func BenchmarkSplitUnsafe(b *testing.B) {
	pool := DefaultBufferPool()
	const size = 8 << 10

	b.ReportAllocs()
	b.SetBytes(size)

	for b.Loop() {
		buf := pool.Get(size)
		buffer := NewBuffer(buf, pool)
		left, right := SplitUnsafe(buffer, size/2)
		lenSink += left.Len()
		right.Free()
		left.Free()
	}
}
