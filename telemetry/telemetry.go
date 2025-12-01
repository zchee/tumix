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

package telemetry

import (
	"go.opentelemetry.io/otel/codes"
	semconv "go.opentelemetry.io/otel/semconv/v1.37.0"
	"go.opentelemetry.io/otel/trace"
)

// End ends the span with the appropriate status based on the error.
func End(span trace.Span, err error, options ...trace.SpanEndOption) {
	// Set Ok code by default.
	code, description := codes.Ok, ""

	if err != nil {
		code, description = codes.Error, err.Error()
		span.SetAttributes(
			semconv.ErrorMessage(err.Error()),
			semconv.ErrorType(err),
		)
	}

	span.SetStatus(code, description)
	span.End(options...)
}
