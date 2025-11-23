// Copyright 2025 The tumix Authors.
//
// SPDX-License-Identifier: Apache-2.0

package xai

import (
	"errors"

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

// GRPCStatus builds a status.Status from the structured error for interoperability.
func (e *Error) GRPCStatus() *status.Status {
	st := status.New(e.Code, e.Message)
	return st
}

// ParseError converts a gRPC status error into XAIError if possible.
func ParseError(err error) (*Error, bool) {
	if err == nil {
		return nil, false
	}
	st, ok := status.FromError(err)
	if !ok {
		return nil, false
	}
	return &Error{Code: st.Code(), Message: st.Message(), Details: st.Details()}, true
}

// WrapError returns an XAIError if err is a gRPC status error; otherwise returns err unchanged.
func WrapError(err error) error {
	if xe, ok := ParseError(err); ok {
		return xe
	}
	return err
}

// IsRetryable reports whether an error is retryable (currently UNAVAILABLE or DEADLINE_EXCEEDED).
// This mirrors the retryable set used in the client service config.
func IsRetryable(err error) bool {
	if xe, ok := ParseError(err); ok {
		switch xe.Code {
		case codes.Unavailable, codes.DeadlineExceeded:
			return true
		}
	}
	return false
}

// Unwrap enables errors.Is/As to reach the underlying status.
func (e *Error) Unwrap() error {
	return status.New(e.Code, e.Message).Err()
}

// AsXAIError is a helper for errors.As.
func AsXAIError(err error) (*Error, bool) {
	var target *Error
	if errors.As(err, &target) {
		return target, true
	}
	return nil, false
}
