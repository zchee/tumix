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
	"fmt"
	"net/http"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/propagation"
	semconv "go.opentelemetry.io/otel/semconv/v1.37.0"
	"go.opentelemetry.io/otel/semconv/v1.37.0/httpconv"
	"go.opentelemetry.io/otel/trace"
)

var _ httpconv.RequestMethodAttr

const instrumentationName = "github.com/zchee/tumix/gollm/internal/httputil"

// Transport implements [http.RoundTripper] with OpenTelemetry tracing.
type Transport struct {
	Base   http.RoundTripper
	Tracer trace.Tracer
	trace  bool
}

// NewTransport creates a new Transport. If base is nil, http.DefaultTransport is used.
func NewTransport(base http.RoundTripper) *Transport {
	return NewTransportWithTrace(base, true)
}

// NewTransportWithTrace creates a new Transport with optional OpenTelemetry tracing.
//
// If base is nil, http.DefaultTransport is used.
func NewTransportWithTrace(base http.RoundTripper, traceEnabled bool) *Transport {
	if base == nil {
		base = http.DefaultTransport.(*http.Transport).Clone()
	}
	var tracer trace.Tracer
	if traceEnabled {
		tracer = otel.Tracer(instrumentationName)
	}

	opts := []otelhttp.Option{
		otelhttp.WithMessageEvents(otelhttp.ReadEvents, otelhttp.WriteEvents),
		otelhttp.WithTracerProvider(otel.GetTracerProvider()),
		otelhttp.WithMeterProvider(otel.GetMeterProvider()),
		otelhttp.WithServerName("tumix"),
	}
	base = otelhttp.NewTransport(base, opts...)

	return &Transport{
		Base:   base,
		Tracer: tracer,
		trace:  traceEnabled,
	}
}

// RoundTrip implements [http.RoundTripper].
func (t *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	if !t.trace {
		return t.Base.RoundTrip(req) //nolint:wrapcheck
	}

	ctx := req.Context()
	method := req.Method
	if method == "" {
		method = http.MethodGet
	}

	attrs := []attribute.KeyValue{
		attribute.String("http.request.method", method),
		semconv.URLFull(req.URL.String()),
		semconv.URLFull(req.URL.String()),
		semconv.URLScheme(req.URL.Scheme),
		semconv.URLPath(req.URL.Path),
		attribute.String("server.address", req.URL.Hostname()),
	}

	if req.URL.RawQuery != "" {
		attrs = append(attrs, attribute.String("url.query", req.URL.RawQuery))
	}
	if req.URL.Port() != "" {
		attrs = append(attrs, attribute.String("server.port", req.URL.Port()))
	}

	var span trace.Span
	ctx, span = t.Tracer.Start(ctx, "HTTP "+method,
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(attrs...),
	)
	defer span.End()

	// Inject trace context into the request headers
	otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(req.Header))

	resp, err := t.Base.RoundTrip(req.WithContext(ctx))
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		return nil, fmt.Errorf("RoundTrip failed: %w", err)
	}

	span.SetAttributes(attribute.Int("http.response.status_code", resp.StatusCode))
	if resp.StatusCode >= 400 {
		span.SetStatus(codes.Error, "HTTP error")
	} else {
		span.SetStatus(codes.Ok, "")
	}

	return resp, nil
}
