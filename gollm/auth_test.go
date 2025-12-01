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

package gollm

import (
	"testing"
)

func TestAuthMethodValues(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		method AuthMethod
		want   string
	}{
		"api key":   {method: AuthMethodAPIKey("k1"), want: "k1"},
		"api token": {method: AuthMethodAPIToken("t1"), want: "t1"},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if got := tt.method.value(); got != tt.want {
				t.Fatalf("value() = %q, want %q", got, tt.want)
			}
		})
	}
}
