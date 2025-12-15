/*
 *
 * Copyright 2023 gRPC authors.
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

package mem_test

import (
	"bytes"
	"testing"
	"unsafe"

	"github.com/zchee/tumix/gollm/xai/internal/grpccodec/mem"
)

func TestBufferPool(t *testing.T) {
	poolSizes := []int{4, 8, 16, 32}
	pools := []mem.BufferPool{
		mem.NopBufferPool{},
		mem.NewTieredBufferPool(poolSizes...),
	}

	testSizes := append([]int{1}, poolSizes...)
	testSizes = append(testSizes, 64)

	for _, p := range pools {
		for _, l := range testSizes {
			bs := p.Get(l)
			if len(*bs) != l {
				t.Fatalf("Get(%d) returned buffer of length %d, want %d", l, len(*bs), l)
			}

			p.Put(bs)
		}
	}
}

func TestBufferPoolClears(t *testing.T) {
	const poolSize = 4
	pool := mem.NewTieredBufferPool(poolSize)
	tests := []struct {
		name       string
		bufferSize int
	}{
		{
			name:       "sized_buffer_pool",
			bufferSize: poolSize,
		},
		{
			name:       "simple_buffer_pool",
			bufferSize: poolSize + 1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			for {
				buf1 := pool.Get(tc.bufferSize)
				copy(*buf1, "1234")
				pool.Put(buf1)

				buf2 := pool.Get(tc.bufferSize)
				if unsafe.SliceData(*buf1) != unsafe.SliceData(*buf2) {
					pool.Put(buf2)
					// This test is only relevant if a buffer is reused, otherwise try again. This
					// can happen if a GC pause happens between putting the buffer back in the pool
					// and getting a new one.
					continue
				}

				if !bytes.Equal(*buf1, make([]byte, tc.bufferSize)) {
					t.Fatalf("buffer not cleared")
				}
				break
			}
		})
	}
}

func TestBufferPoolIgnoresShortBuffers(t *testing.T) {
	pool := mem.NewTieredBufferPool(10, 20)
	buf := pool.Get(1)
	if cap(*buf) != 10 {
		t.Fatalf("Get(1) returned buffer with capacity: %d, want 10", cap(*buf))
	}

	// Insert a short buffer into the pool, which is currently empty.
	short := make([]byte, 1)
	pool.Put(&short)
	// Then immediately request a buffer that would be pulled from the pool where the
	// short buffer would have been returned. If the short buffer is pulled from the
	// pool, it could cause a panic.
	pool.Get(10)
}

func TestTieredBufferPoolClearsIntermediateSizes(t *testing.T) {
	const (
		smallTier = 32 << 10
		largeTier = 1 << 20
		reqSize   = 40 << 10
	)

	pool := mem.NewTieredBufferPool(smallTier, largeTier)

	for {
		buf1 := pool.Get(reqSize)
		for i := range *buf1 {
			(*buf1)[i] = 0xFF
		}
		pool.Put(buf1)

		buf2 := pool.Get(reqSize)
		if unsafe.SliceData(*buf1) != unsafe.SliceData(*buf2) {
			pool.Put(buf2)
			// Try again until the pool reuses the same buffer.
			continue
		}

		if !bytes.Equal(*buf2, make([]byte, reqSize)) {
			t.Fatalf("buffer not cleared for intermediate size request")
		}
		pool.Put(buf2)
		break
	}
}

func TestDefaultBufferPoolHasIntermediateTiers(t *testing.T) {
	pool := mem.DefaultBufferPool()

	const reqSize = 40 << 10 // between 32KB and 64KB tiers
	buf := pool.Get(reqSize)
	if gotCap := cap(*buf); gotCap != 64<<10 {
		t.Fatalf("DefaultBufferPool.Get(%d) cap=%d, want %d", reqSize, gotCap, 64<<10)
	}
	pool.Put(buf)
}
