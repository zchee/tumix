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
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestErrorImplements(t *testing.T) {
	e := &Error{Code: codes.InvalidArgument, Message: "bad"}
	if got := e.Error(); got != "bad" {
		t.Fatalf("Error() = %q", got)
	}
	st := e.GRPCStatus()
	if st.Code() != codes.InvalidArgument || st.Message() != "bad" {
		t.Fatalf("GRPCStatus mismatch: %+v", st)
	}
	if !errors.Is(e, status.Error(codes.InvalidArgument, "bad")) {
		t.Fatalf("Unwrap should make errors.Is succeed")
	}
}

func TestParseAndWrapError(t *testing.T) {
	statusErr := status.Error(codes.Unavailable, "try again")
	t.Run("ParseError", func(t *testing.T) {
		tests := []struct {
			name string
			err  error
			want *Error
			ok   bool
		}{
			{name: "nil", err: nil, want: nil, ok: false},
			{name: "non-status", err: errors.New("plain"), want: nil, ok: false},
			{name: "status", err: statusErr, want: &Error{Code: codes.Unavailable, Message: "try again"}, ok: true},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				got, ok := ParseError(tt.err)
				if ok != tt.ok {
					t.Fatalf("ParseError(%v) ok = %v, want %v", tt.err, ok, tt.ok)
				}
				if tt.want == nil {
					if got != nil {
						t.Fatalf("ParseError(%v) = %+v, want nil", tt.err, got)
					}
					return
				}
				if got == nil {
					t.Fatalf("ParseError(%v) = nil, want %+v", tt.err, tt.want)
				}
				if got.Code != tt.want.Code || got.Message != tt.want.Message {
					t.Fatalf("ParseError(%v) = %+v, want %+v", tt.err, got, tt.want)
				}
			})
		}
	})

	t.Run("WrapError", func(t *testing.T) {
		plain := errors.New("plain")

		wrapped := WrapError(statusErr)
		xe := new(Error)
		if !errors.As(wrapped, &xe) {
			t.Fatalf("WrapError(statusErr) type = %T, want *Error", wrapped)
		}
		if xe.Code != codes.Unavailable || xe.Message != "try again" {
			t.Fatalf("WrapError(statusErr) = %+v, want Code=%v Message=%q", xe, codes.Unavailable, "try again")
		}
		if !errors.Is(wrapped, statusErr) {
			t.Fatalf("WrapError(statusErr) should remain comparable to original status error")
		}

		if got := WrapError(plain); !errors.Is(got, plain) {
			t.Fatalf("WrapError(non-status) = %v, want original %v", got, plain)
		}
		if err := WrapError(nil); err != nil {
			t.Fatalf("WrapError(nil) = %v, want nil", err)
		}

		again := WrapError(wrapped)
		xe2 := new(Error)
		if !errors.As(wrapped, &xe2) {
			t.Fatalf("WrapError(*Error) type = %T, want *Error", again)
		}
		if xe2.Code != xe.Code || xe2.Message != xe.Message {
			t.Fatalf("WrapError(*Error) = %+v, want %+v", xe2, xe)
		}
		if !errors.Is(again, statusErr) {
			t.Fatalf("WrapError(*Error) should still compare equal to original status")
		}
	})
}

func TestIsRetryable(t *testing.T) {
	retryable := status.Error(codes.DeadlineExceeded, "timeout")
	if !IsRetryable(retryable) {
		t.Fatalf("DeadlineExceeded should be retryable")
	}
	if IsRetryable(errors.New("no status")) {
		t.Fatalf("non-status error should not be retryable")
	}
}

func TestAsError(t *testing.T) {
	orig := &Error{Code: codes.PermissionDenied, Message: "nope"}
	wrapped := WrapError(status.Error(codes.PermissionDenied, "nope"))
	if err, ok := AsError(wrapped); err != nil && !ok {
		t.Fatalf("AsError should unwrap wrapped status error")
	}
	if got, ok := AsError(orig); !ok || got != orig {
		t.Fatalf("AsError should return original *Error")
	}
}
