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
	"errors"
	"slices"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Error provides structured access to gRPC status errors returned by xAI services.
type Error struct {
	Code    codes.Code
	Message string
	Details []any
}

// Error implements the error interface.
func (e *Error) Error() string {
	return e.Message
}

// GRPCStatus builds a [status.Status] from the structured error for interoperability.
func (e *Error) GRPCStatus() *status.Status {
	st := status.New(e.Code, e.Message)
	return st
}

// Unwrap enables [errors.Is] and [errors.As] to reach the underlying status.
func (e *Error) Unwrap() error {
	return status.New(e.Code, e.Message).Err()
}

// ParseError converts a gRPC status error into [Error] if possible.
func ParseError(err error) (*Error, bool) {
	// fast path
	if err == nil {
		return nil, false
	}

	st, ok := status.FromError(err)
	if !ok {
		return nil, false
	}

	return &Error{
		Code:    st.Code(),
		Message: st.Message(),
		Details: st.Details(),
	}, true
}

// WrapError returns an [Error] if err is a gRPC status error; otherwise returns err unchanged.
func WrapError(err error) error {
	if xe, ok := ParseError(err); ok {
		return xe
	}
	return err
}

var retryableCodes = []codes.Code{
	codes.Unavailable,
	codes.DeadlineExceeded,
}

// IsRetryable reports whether an error is retryable (currently [codes.Unavailable] or [codes.DeadlineExceeded]).
//
// This mirrors the retryable set used in the client service config.
func IsRetryable(err error) bool {
	if xe, ok := ParseError(err); ok {
		return slices.Contains(retryableCodes, xe.Code)
	}
	return false
}

// AsError is a helper for [errors.As].
func AsError(err error) (*Error, bool) {
	if target := new(Error); errors.As(err, &target) {
		return target, true
	}

	return nil, false
}
