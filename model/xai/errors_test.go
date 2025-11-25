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
	gerr := status.Error(codes.Unavailable, "try again")
	parsed, ok := ParseError(gerr)
	if !ok || parsed.Code != codes.Unavailable || parsed.Message != "try again" {
		t.Fatalf("ParseError failed: ok=%v parsed=%+v", ok, parsed)
	}
	if errors.Is(WrapError(gerr), gerr) {
		t.Fatalf("WrapError should convert grpc status errors")
	}
	nonStatus := errors.New("plain")
	if got := WrapError(nonStatus); !errors.Is(got, nonStatus) {
		t.Fatalf("WrapError should return original non-status error")
	}
	if err := WrapError(nil); err != nil {
		t.Fatalf("WrapError nil should stay nil")
	}
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
	if _, ok := AsError(wrapped); !ok {
		t.Fatalf("AsError should unwrap wrapped status error")
	}
	if got, ok := AsError(orig); !ok || got != orig {
		t.Fatalf("AsError should return original *Error")
	}
}
