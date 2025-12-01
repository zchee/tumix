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
)

// Transport implements [http.RoundTripper] with optional OpenTelemetry tracing.
type Transport struct {
	Base http.RoundTripper
	rt   http.RoundTripper
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

	rt := base
	if traceEnabled {
		opts := []otelhttp.Option{
			otelhttp.WithMessageEvents(otelhttp.ReadEvents, otelhttp.WriteEvents),
			otelhttp.WithTracerProvider(otel.GetTracerProvider()),
			otelhttp.WithMeterProvider(otel.GetMeterProvider()),
			otelhttp.WithServerName("tumix"),
		}
		rt = otelhttp.NewTransport(base, opts...)
	}

	return &Transport{
		Base: base,
		rt:   rt,
	}
}

// RoundTrip implements [http.RoundTripper].
func (t *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.rt.RoundTrip(req)
	if err != nil {
		return nil, fmt.Errorf("RoundTrip failed: %w", err)
	}

	return resp, nil
}
