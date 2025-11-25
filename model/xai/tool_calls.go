// Copyright 2025 The tumix Authors.
//
// SPDX-License-Identifier: Apache-2.0

package xai

import (
	json "encoding/json"
	"errors"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

// ToolCallArguments unmarshals a tool call's arguments into the provided destination.
// Returns an error if the tool call has no function payload or JSON is invalid.
func ToolCallArguments(tc *xaipb.ToolCall, out any) error {
	if tc == nil {
		return errors.New("tool call is nil")
	}
	fn := tc.GetFunction()
	if fn == nil {
		return errors.New("tool call does not contain a function")
	}
	if fn.GetArguments() == "" {
		return errors.New("tool call arguments empty")
	}
	return json.Unmarshal([]byte(fn.GetArguments()), out)
}

// ToolCallJSON returns the raw JSON arguments string (empty if absent).
func ToolCallJSON(tc *xaipb.ToolCall) string {
	if tc == nil || tc.GetFunction() == nil {
		return ""
	}
	return tc.GetFunction().GetArguments()
}
