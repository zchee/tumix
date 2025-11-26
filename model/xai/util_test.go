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
	"testing"
)

func TestPtrAndDeref(t *testing.T) {
	v := 42
	if got := ptr(v); got == nil || *got != v {
		t.Fatalf("ptr(%d) = %v", v, got)
	}

	if got := deref(ptr("ok")); got != "ok" {
		t.Fatalf("deref pointer returned %q", got)
	}
	if got := deref[string](nil); got != "" {
		t.Fatalf("deref nil returned %q", got)
	}
}
