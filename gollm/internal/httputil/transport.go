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

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
)

const instrumentationName = "github.com/zchee/tumix/gollm/internal/httputil"

// Transport implements [http.RoundTripper] with OpenTelemetry tracing.
type Transport struct {
	Base   http.RoundTripper
	Tracer trace.Tracer
}

// NewTransport creates a new Transport. If base is nil, http.DefaultTransport is used.
func NewTransport(base http.RoundTripper) *Transport {
	if base == nil {
		base = http.DefaultTransport
	}
	return &Transport{
		Base:   base,
		Tracer: otel.Tracer(instrumentationName),
	}
}

// RoundTrip implements [http.RoundTripper].
func (t *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	ctx := req.Context()
	method := req.Method
	if method == "" {
		method = http.MethodGet
	}

	attrs := []attribute.KeyValue{
		attribute.String("http.request.method", method),
		attribute.String("url.full", req.URL.String()),
		attribute.String("url.scheme", req.URL.Scheme),
		attribute.String("url.path", req.URL.Path),
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
		return nil, err
	}

	span.SetAttributes(attribute.Int("http.response.status_code", resp.StatusCode))
	if resp.StatusCode >= 400 {
		span.SetStatus(codes.Error, "HTTP error")
	} else {
		span.SetStatus(codes.Ok, "")
	}

	return resp, nil
}