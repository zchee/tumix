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

import (
	"context"
	"testing"

	"google.golang.org/genai"
)

func assertParseEnv[T comparable](t *testing.T, key, raw string, fallback, want T) {
	t.Helper()
	t.Setenv(key, raw)
	got := parseEnv(key, fallback)
	if got != want {
		t.Fatalf("parseEnv[%T](%q, %v) with env %q=%q = %v; want %v", fallback, key, fallback, key, raw, got, want)
	}
}

func TestParseEnvSuccess(t *testing.T) {
	t.Run("bool", func(t *testing.T) {
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_BOOL", "true", false, true)
	})
	t.Run("int", func(t *testing.T) {
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_INT", "42", int(7), int(42))
	})
	t.Run("int64", func(t *testing.T) {
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_INT64", "42", int64(7), int64(42))
	})
	t.Run("uint", func(t *testing.T) {
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_UINT", "42", uint(7), uint(42))
	})
	t.Run("float64", func(t *testing.T) {
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_FLOAT64", "1.5", float64(0), float64(1.5))
	})
	t.Run("string", func(t *testing.T) {
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_STRING", "hello", "fallback", "hello")
	})
	t.Run("aliases", func(t *testing.T) {
		type (
			myBool    bool
			myFloat32 float32
			myInt     int
			myString  string
			myUint    uint
		)
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_MYBOOL", "true", myBool(false), myBool(true))
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_MYFLOAT32", "1.5", myFloat32(0), myFloat32(1.5))
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_MYINT", "9", myInt(1), myInt(9))
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_MYSTRING", "v", myString("fallback"), myString("v"))
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_MYUINT", "9", myUint(1), myUint(9))
	})
}

func TestParseEnvReturnsFallbackOnEmptyValue(t *testing.T) {
	t.Run("empty-string treated as unset", func(t *testing.T) {
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_EMPTY", "", int(7), int(7))
	})
}

func TestParseEnvReturnsFallbackOnInvalidValue(t *testing.T) {
	t.Run("invalid int", func(t *testing.T) {
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_INVALID_INT", "not-an-int", int(7), int(7))
	})
	t.Run("invalid uint", func(t *testing.T) {
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_INVALID_UINT", "-1", uint(7), uint(7))
	})
	t.Run("invalid float64", func(t *testing.T) {
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_INVALID_FLOAT64", "nope", float64(1.5), float64(1.5))
	})
	t.Run("invalid bool", func(t *testing.T) {
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_INVALID_BOOL", "notabool", true, true)
	})
}

func TestParseEnvReturnsFallbackOnOutOfRangeValue(t *testing.T) {
	t.Run("int overflow", func(t *testing.T) {
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_OVERFLOW_INT", "9223372036854775808", int(7), int(7))
	})
	t.Run("uint overflow", func(t *testing.T) {
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_OVERFLOW_UINT", "18446744073709551616", uint(7), uint(7))
	})
	t.Run("int8 overflow", func(t *testing.T) {
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_OVERFLOW_INT8", "128", int8(7), int8(7))
	})
	t.Run("uint8 overflow", func(t *testing.T) {
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_OVERFLOW_UINT8", "256", uint8(7), uint8(7))
	})
	t.Run("float64 overflow", func(t *testing.T) {
		assertParseEnv(t, "TUMIX_TEST_PARSEENV_OVERFLOW_FLOAT64", "1e309", float64(1.5), float64(1.5))
	})
}

func TestParseEnvReturnsFallbackOnUnsupportedType(t *testing.T) {
	type sample struct {
		A int
	}
	assertParseEnv(t, "TUMIX_TEST_PARSEENV_UNSUPPORTED", "anything", sample{A: 1}, sample{A: 1})
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
	loadPricing(t.Context()) // should not panic when env unset
	if _, ok := prices["gemini-2.5-flash"]; !ok {
		t.Fatalf("default pricing missing")
	}
}

func TestEnforcePromptTokensWithCounter(t *testing.T) {
	cfg := config{Prompt: "hello world", MaxPromptTokens: 20, ModelName: "m"}
	callCount := 0
	counter := func(ctx context.Context, model string, contents []*genai.Content, config *genai.CountTokensConfig) (*genai.CountTokensResponse, error) {
		callCount++
		return &genai.CountTokensResponse{TotalTokens: 5}, nil
	}
	if err := enforcePromptTokensWithCounter(t.Context(), &cfg, counter); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if callCount != 1 {
		t.Fatalf("expected counter called once, got %d", callCount)
	}
}

func TestEnforcePromptTokensWithCounterRejects(t *testing.T) {
	cfg := config{Prompt: "hello world", MaxPromptTokens: 4, ModelName: "m"}
	counter := func(ctx context.Context, model string, contents []*genai.Content, config *genai.CountTokensConfig) (*genai.CountTokensResponse, error) {
		return &genai.CountTokensResponse{TotalTokens: 10}, nil
	}
	if err := enforcePromptTokensWithCounter(t.Context(), &cfg, counter); err == nil {
		t.Fatalf("expected error when tokens exceed limit")
	}
}

func TestCapRoundsByBudget(t *testing.T) {
	cfg := config{
		ModelName:       "gemini-2.5-flash",
		MaxCostUSD:      0.01,
		MaxRounds:       6,
		MinRounds:       2,
		MaxTokens:       128,
		MaxPromptTokens: 200,
		Prompt:          "hello",
	}
	roundCap := capRoundsByBudget(&cfg, 15)
	if roundCap < 1 {
		t.Fatalf("cap must be >=1, got %d", roundCap)
	}
	if roundCap > cfg.MaxRounds {
		t.Fatalf("cap should not exceed max_rounds: %d > %d", roundCap, cfg.MaxRounds)
	}
}
