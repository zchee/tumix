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

package xai

import (
	"context"
	"fmt"
	"runtime"
	"strings"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	"go.opentelemetry.io/otel/sdk/trace"
)

// OTLPTransport selects HTTP or gRPC exporter.
type OTLPTransport string

const (
	OTLPHTTP OTLPTransport = "http"
	OTLPGRPC OTLPTransport = "grpc"
)

// OTLPConfig mirrors the Python telemetry setup knobs.
type OTLPConfig struct {
	Endpoint    string
	Headers     map[string]string
	Insecure    bool // default secure (false)
	Transport   OTLPTransport
	Resource    map[string]string
	Compression string
}

// InitOTLP configures a global OTLP trace exporter using the provided config.
// Defaults: transport http, secure by default.
func InitOTLP(ctx context.Context, cfg OTLPConfig) (*trace.TracerProvider, error) {
	if cfg.Transport == "" {
		cfg.Transport = OTLPHTTP
	}
	if cfg.Endpoint == "" {
		return nil, fmt.Errorf("OTLP endpoint is required")
	}
	exporter, err := newOTLPExporter(ctx, cfg)
	if err != nil {
		return nil, err
	}
	resAttrs := defaultResource()
	for k, v := range cfg.Resource {
		resAttrs = append(resAttrs, attribute.String(k, v))
	}
	res, _ := resource.Merge(resource.Empty(), resource.NewSchemaless(resAttrs...))
	tp := trace.NewTracerProvider(
		trace.WithBatcher(exporter),
		trace.WithResource(res),
	)
	otel.SetTracerProvider(tp)
	return tp, nil
}

func newOTLPExporter(ctx context.Context, cfg OTLPConfig) (trace.SpanExporter, error) {
	switch cfg.Transport {
	case OTLPGRPC:
		opts := []otlptracegrpc.Option{otlptracegrpc.WithEndpoint(cfg.Endpoint)}
		if cfg.Insecure {
			opts = append(opts, otlptracegrpc.WithInsecure())
		}
		switch strings.ToLower(cfg.Compression) {
		case "gzip":
			opts = append(opts, otlptracegrpc.WithCompressor("gzip"))
		}
		if len(cfg.Headers) > 0 {
			opts = append(opts, otlptracegrpc.WithHeaders(cfg.Headers))
		}
		return otlptracegrpc.New(ctx, opts...)
	case OTLPHTTP:
		fallthrough
	default:
		opts := []otlptracehttp.Option{otlptracehttp.WithEndpoint(cfg.Endpoint)}
		if cfg.Insecure {
			opts = append(opts, otlptracehttp.WithInsecure())
		}
		switch strings.ToLower(cfg.Compression) {
		case "gzip":
			opts = append(opts, otlptracehttp.WithCompression(otlptracehttp.GzipCompression))
		}
		if len(cfg.Headers) > 0 {
			opts = append(opts, otlptracehttp.WithHeaders(cfg.Headers))
		}
		return otlptracehttp.New(ctx, opts...)
	}
}

func defaultResource() []attribute.KeyValue {
	return []attribute.KeyValue{
		attribute.String("service.name", "xai-sdk-go"),
		attribute.String("service.version", SDKVersion()),
		attribute.String("telemetry.sdk.language", "go"),
		attribute.String("runtime.version", runtime.Version()),
	}
}

var tracer = otel.Tracer("github.com/zchee/tumix/model/xai")
