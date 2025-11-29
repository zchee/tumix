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
	json "encoding/json/v2"
	"errors"

	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
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
