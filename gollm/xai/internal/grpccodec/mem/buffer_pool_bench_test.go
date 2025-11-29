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

import "testing"

func benchmarkTieredBufferPool(b *testing.B, size int, parallel bool) {
	b.Helper()

	pool := NewTieredBufferPool(256, 4<<10, 16<<10, 32<<10, 1<<20)

	b.SetBytes(int64(size))
	b.ReportAllocs()

	if parallel {
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				buf := pool.Get(size)
				(*buf)[0] = byte(size)
				pool.Put(buf)
			}
		})
		return
	}

	for b.Loop() {
		buf := pool.Get(size)
		(*buf)[0] = byte(size)
		pool.Put(buf)
	}
}

func BenchmarkTieredBufferPool(b *testing.B) {
	tests := []struct {
		name     string
		size     int
		parallel bool
	}{
		{name: "256B", size: 256},
		{name: "4KB", size: 4 << 10},
		{name: "32KB", size: 32 << 10},
		{name: "2MBFallback", size: 2 << 20},
		{name: "4KBParallel", size: 4 << 10, parallel: true},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkTieredBufferPool(b, tt.size, tt.parallel)
		})
	}
}

func BenchmarkSimpleBufferPool(b *testing.B) {
	const size = 8 << 10
	var pool simpleBufferPool

	b.SetBytes(size)
	b.ReportAllocs()

	for b.Loop() {
		buf := pool.Get(size)
		(*buf)[0] = 1
		pool.Put(buf)
	}
}
