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

package httputil

import (
	"sync"
	"testing"
	"time"

	"github.com/zchee/tumix/telemetry/httptelemetry"
)

func TestNewClient(t *testing.T) {
	t.Parallel()

	c := NewClient(5 * time.Second)
	if c.Timeout != 5*time.Second {
		t.Fatalf("Timeout = %v, want 5s", c.Timeout)
	}
	if _, ok := c.Transport.(*httptelemetry.Transport); !ok {
		t.Fatalf("Transport type = %T, want *Transport", c.Transport)
	}
}

func TestDefaultTraceEnabled(t *testing.T) {
	tests := map[string]struct {
		env  string
		want bool
	}{
		"unset defaults false": {
			env:  "",
			want: false,
		},
		"true": {
			env:  "true",
			want: true,
		},
		"1": {
			env:  "1",
			want: true,
		},
		"false": {
			env:  "false",
			want: false,
		},
		"junk": {
			env:  "nope",
			want: false,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			t.Setenv("TUMIX_HTTP_TRACE", tc.env)

			traceOnce = sync.Once{} // reset cached value
			if got := DefaultTraceEnabled(); got != tc.want {
				t.Fatalf("DefaultTraceEnabled() = %v, want %v (env=%q)", got, tc.want, tc.env)
			}
		})
	}
}
