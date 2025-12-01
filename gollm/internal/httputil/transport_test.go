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
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

type stubRoundTripper struct {
	calls int
	resp  *http.Response
	err   error
	last  *http.Request
}

func (s *stubRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	s.calls++
	s.last = req
	return s.resp, s.err
}

func TestTransportRoundTripSuccess(t *testing.T) {
	t.Parallel()

	rr := httptest.NewRecorder()
	rr.WriteHeader(http.StatusNoContent)

	stub := &stubRoundTripper{resp: rr.Result()}
	tr := NewTransport(stub)

	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "https://example.com/foo?bar=baz", http.NoBody)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := tr.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip error: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusNoContent)
	}
	if stub.calls != 1 {
		t.Fatalf("base round trips = %d, want 1", stub.calls)
	}
}

func TestTransportRoundTripError(t *testing.T) {
	t.Parallel()

	stub := &stubRoundTripper{err: errors.New("boom")}
	tr := NewTransport(stub)

	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "https://example.org", http.NoBody)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := tr.RoundTrip(req)
	if err == nil || err.Error() == "boom" {
		t.Fatalf("error = %v, want wrapped RoundTrip failed", err)
	}
	if resp != nil {
		defer resp.Body.Close()
	}

	if stub.calls != 1 {
		t.Fatalf("base round trips = %d, want 1", stub.calls)
	}
}

func TestTransportTracingToggle(t *testing.T) {
	tests := map[string]struct {
		traceEnabled  bool
		wantSpanCount int
	}{
		"trace enabled": {
			traceEnabled:  true,
			wantSpanCount: 1,
		},
		"trace disabled": {
			traceEnabled:  false,
			wantSpanCount: 0,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			exporter := &recordingExporter{}
			tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sdktrace.NewSimpleSpanProcessor(exporter)))

			prev := otel.GetTracerProvider()
			otel.SetTracerProvider(tp)
			t.Cleanup(func() {
				otel.SetTracerProvider(prev)
				_ = tp.Shutdown(t.Context())
			})

			stub := &stubRoundTripper{
				resp: newResponse(http.StatusAccepted),
			}
			defer stub.resp.Body.Close()
			tr := NewTransportWithTrace(stub, tc.traceEnabled)

			req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "https://example.com/foo?bar=baz", http.NoBody)
			if err != nil {
				t.Fatal(err)
			}

			resp, err := tr.RoundTrip(req)
			if err != nil {
				t.Fatalf("RoundTrip error: %v", err)
			}
			if err := resp.Body.Close(); err != nil {
				t.Fatalf("failed to close response body: %v", err)
			}

			if stub.calls != 1 {
				t.Fatalf("base round trips = %d, want 1", stub.calls)
			}

			if got := exporter.SpanCount(); got != tc.wantSpanCount {
				t.Fatalf("ended spans = %d, want %d", got, tc.wantSpanCount)
			}
		})
	}
}

type recordingExporter struct {
	mu    sync.Mutex
	spans []sdktrace.ReadOnlySpan
}

func (e *recordingExporter) ExportSpans(_ context.Context, spans []sdktrace.ReadOnlySpan) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.spans = append(e.spans, spans...)

	return nil
}

func (e *recordingExporter) Shutdown(context.Context) error { return nil }

func (e *recordingExporter) SpanCount() int {
	e.mu.Lock()
	defer e.mu.Unlock()

	return len(e.spans)
}

func newResponse(statusCode int) *http.Response {
	rr := httptest.NewRecorder()
	rr.WriteHeader(statusCode)

	return rr.Result()
}
