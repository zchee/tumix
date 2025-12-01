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
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/zchee/tumix/telemetry/httptelemetry"
)

var (
	pooledBaseOnce sync.Once
	pooledBase     http.RoundTripper

	tracedTransport   *httptelemetry.Transport
	untracedTransport *httptelemetry.Transport
	transportsOnce    sync.Once
)

func baseTransport() http.RoundTripper {
	pooledBaseOnce.Do(func() {
		if dt, ok := http.DefaultTransport.(*http.Transport); ok {
			clone := dt.Clone()
			clone.MaxIdleConns = 256
			clone.MaxIdleConnsPerHost = 64
			clone.IdleConnTimeout = 90 * time.Second
			pooledBase = clone
		} else {
			pooledBase = http.DefaultTransport
		}
	})
	return pooledBase
}

func defaultTransports() {
	transportsOnce.Do(func() {
		base := baseTransport()
		tracedTransport = httptelemetry.NewTransportWithTrace(base, true)
		untracedTransport = httptelemetry.NewTransportWithTrace(base, false)
	})
}

// NewClient creates a new HTTP client with a specified timeout and tracing controlled by environment defaults.
func NewClient(timeout time.Duration) *http.Client {
	return NewClientWithTracing(timeout, DefaultTraceEnabled())
}

// NewClientWithTracing creates a new HTTP client with optional OpenTelemetry tracing.
func NewClientWithTracing(timeout time.Duration, traceEnabled bool) *http.Client {
	defaultTransports()

	transport := tracedTransport
	if !traceEnabled {
		transport = untracedTransport
	}

	return &http.Client{
		Timeout:   timeout,
		Transport: transport,
	}
}

var (
	traceOnce sync.Once
	traceEnv  bool
)

// DefaultTraceEnabled reads TUMIX_HTTP_TRACE environment variable ("1", "true") to decide tracing default.
// Falls back to false when unset or invalid.
func DefaultTraceEnabled() bool {
	traceOnce.Do(func() {
		raw := os.Getenv("TUMIX_HTTP_TRACE")
		if raw == "" {
			traceEnv = false
			return
		}
		val, err := strconv.ParseBool(raw)
		traceEnv = err == nil && val
	})
	return traceEnv
}
