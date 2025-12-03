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

package main

import "testing"

func TestEnvOrDefault(t *testing.T) {
	const key = "TUMIX_ENV_OR_DEFAULT"
	t.Setenv(key, "value")
	if got := envOrDefault(key, "fallback"); got != "value" {
		t.Fatalf("envOrDefault() = %q, want %q", got, "value")
	}

	t.Setenv(key, "")
	if got := envOrDefault(key, "fallback"); got != "fallback" {
		t.Fatalf("envOrDefault() = %q, want %q", got, "fallback")
	}
}

func TestParseUintEnv(t *testing.T) {
	const key = "TUMIX_UINT"
	if got := parseUintEnv(key, 7); got != 7 {
		t.Fatalf("parseUintEnv() fallback = %d, want 7", got)
	}

	t.Setenv(key, "9")
	if got := parseUintEnv(key, 7); got != 9 {
		t.Fatalf("parseUintEnv() = %d, want 9", got)
	}
}

func TestParseFloatEnv(t *testing.T) {
	const key = "TUMIX_FLOAT"
	if got := parseFloatEnv(key, 0.25); got != 0.25 {
		t.Fatalf("parseFloatEnv() fallback = %f, want 0.25", got)
	}

	t.Setenv(key, "0.75")
	if got := parseFloatEnv(key, 0.25); got != 0.75 {
		t.Fatalf("parseFloatEnv() = %f, want 0.75", got)
	}
}

func TestParseBoolEnv(t *testing.T) {
	const key = "TUMIX_BOOL"
	if got := parseBoolEnv(key); got {
		t.Fatalf("parseBoolEnv() default = %v, want false", got)
	}

	t.Setenv(key, "true")
	if got := parseBoolEnv(key); !got {
		t.Fatalf("parseBoolEnv() = %v, want true", got)
	}

	t.Setenv(key, "0")
	if got := parseBoolEnv(key); got {
		t.Fatalf("parseBoolEnv() = %v, want false", got)
	}
}

func TestEstimateTokensFromChars(t *testing.T) {
	if got := estimateTokensFromChars(0); got != 0 {
		t.Fatalf("estimateTokensFromChars(0)=%d", got)
	}
	if got := estimateTokensFromChars(4); got != 1 {
		t.Fatalf("estimateTokensFromChars(4)=%d", got)
	}
	if got := estimateTokensFromChars(9); got != 3 {
		t.Fatalf("estimateTokensFromChars(9)=%d", got)
	}
}

func TestLoadPricingInvalidPath(t *testing.T) {
	loadPricing() // should not panic when env unset
	if _, ok := prices["gemini-2.5-flash"]; !ok {
		t.Fatalf("default pricing missing")
	}
}
