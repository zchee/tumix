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

package adapter

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestToolCallAccumulator(t *testing.T) {
	t.Parallel()

	acc := NewToolCallAccumulator()

	tc0 := acc.Ensure(1, "id-0")
	tc0.Name = "f0"
	tc0.ArgsBuilder().WriteString(`{"a":1`)

	// same index merges
	tc0b := acc.Ensure(1, "id-0")
	tc0b.ArgsBuilder().WriteString(`,"b":2}`)

	tc1 := acc.Ensure(0, "")
	tc1.Name = "f1"
	tc1.ArgsBuilder().WriteString(`{"x":true}`)

	parts := acc.Parts()
	if got, want := len(parts), 2; got != want {
		t.Fatalf("parts len = %d, want %d", got, want)
	}
	if diff := cmp.Diff("f1", parts[0].FunctionCall.Name); diff != "" {
		t.Fatalf("order diff: %s", diff)
	}
	if parts[1].FunctionCall.Args["b"] != float64(2) {
		t.Fatalf("merged args = %+v", parts[1].FunctionCall.Args)
	}
}

func TestParseArgs(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		raw string
		key string
	}{
		"valid_json": {raw: `{"k":"v"}`, key: "k"},
		"malformed":  {raw: `not-json`, key: "raw"},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			args := ParseArgs(tt.raw)
			if _, ok := args[tt.key]; !ok {
				t.Fatalf("key %q missing in %+v", tt.key, args)
			}
		})
	}
}
