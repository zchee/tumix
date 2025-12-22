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
	json "encoding/json/v2"
	"flag"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
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

func TestBuildGenConfig(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		cfg      config
		wantNil  bool
		wantTopK float32
	}{
		"nil_when_all_defaults": {
			cfg: config{
				Temperature: -1,
				TopP:        -1,
				TopK:        0,
				MaxTokens:   0,
				Seed:        0,
			},
			wantNil: true,
		},
		"populated": {
			cfg: config{
				Temperature: 0.7,
				TopP:        0.9,
				TopK:        12,
				MaxTokens:   128,
				Seed:        42,
			},
			wantNil:  false,
			wantTopK: 12,
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			got := buildGenConfig(&tt.cfg)
			if tt.wantNil && got != nil {
				t.Fatalf("expected nil config")
			}
			if !tt.wantNil {
				if got == nil {
					t.Fatalf("expected config, got nil")
				}
				if got.TopK == nil || *got.TopK != tt.wantTopK {
					t.Fatalf("TopK = %v, want %v", got.TopK, tt.wantTopK)
				}
			}
		})
	}
}

func TestParseConfigSuccessAndEnvFallback(t *testing.T) {
	resetFlags := func(args []string) func() {
		origArgs := os.Args
		origFlag := flag.CommandLine
		flag.CommandLine = flag.NewFlagSet(args[0], flag.ContinueOnError)
		flag.CommandLine.SetOutput(os.Stdout)
		os.Args = args
		return func() {
			os.Args = origArgs
			flag.CommandLine = origFlag
		}
	}

	t.Run("success_with_flags", func(t *testing.T) {
		restore := resetFlags([]string{"cmd", "-api_key=key", "-backend=gemini", "-model=gemini-2.5-flash", "hello world"})
		defer restore()

		t.Setenv("TUMIX_MAX_COST_USD", "0.02")
		cfg, err := parseConfig()
		if err != nil {
			t.Fatalf("parseConfig error = %v", err)
		}
		if cfg.Prompt != "hello world" {
			t.Fatalf("Prompt = %q", cfg.Prompt)
		}
		if cfg.APIKey != "key" {
			t.Fatalf("APIKey = %q, want key", cfg.APIKey)
		}
		if cfg.MaxCostUSD != 0.02 {
			t.Fatalf("MaxCostUSD = %f, want 0.02", cfg.MaxCostUSD)
		}
	})

	t.Run("env_api_key_used_when_flag_missing", func(t *testing.T) {
		restore := resetFlags([]string{"cmd", "-backend=gemini", "hi"})
		defer restore()

		t.Setenv("GOOGLE_API_KEY", "env-key")
		cfg, err := parseConfig()
		if err != nil {
			t.Fatalf("parseConfig error = %v", err)
		}
		if cfg.APIKey != "env-key" {
			t.Fatalf("APIKey = %q, want env-key", cfg.APIKey)
		}
	})
}

func TestParseConfigErrors(t *testing.T) {
	resetFlags := func(args []string) func() {
		origArgs := os.Args
		origFlag := flag.CommandLine
		flag.CommandLine = flag.NewFlagSet(args[0], flag.ContinueOnError)
		flag.CommandLine.SetOutput(os.Stdout)
		os.Args = args
		return func() {
			os.Args = origArgs
			flag.CommandLine = origFlag
		}
	}

	tests := map[string]struct {
		args []string
		env  map[string]string
	}{
		"missing_prompt": {
			args: []string{"cmd", "-api_key=k"},
		},
		"invalid_backend": {
			args: []string{"cmd", "-api_key=k", "-backend=bad", "hello"},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			restore := resetFlags(tt.args)
			defer restore()
			for k, v := range tt.env {
				t.Setenv(k, v)
			}
			if _, err := parseConfig(); err == nil {
				t.Fatalf("expected error for case %s", name)
			}
		})
	}
}

func TestLoadPricingOverride(t *testing.T) {
	tmp := t.TempDir()
	file := filepath.Join(tmp, "pricing.json")
	custom := map[string]any{
		"custom-model": map[string]any{
			"in_per_kt":  0.123,
			"out_per_kt": 0.321,
		},
	}
	data, err := json.Marshal(custom)
	if err != nil {
		t.Fatalf("marshal pricing: %v", err)
	}

	if err := os.WriteFile(file, data, 0o600); err != nil {
		t.Fatalf("write pricing: %v", err)
	}

	t.Setenv("TUMIX_PRICING_FILE", file)
	loadPricing(t.Context())

	p, ok := prices["custom-model"]
	if !ok {
		t.Fatalf("custom pricing not loaded")
	}
	if p.inUSDPerKT != 0.123 || p.outUSDPerKT != 0.321 {
		t.Fatalf("pricing = %+v", p)
	}
}

func TestEnforcePromptTokensWithCounterNilResponse(t *testing.T) {
	t.Parallel()

	cfg := config{Prompt: "hello world", MaxPromptTokens: 5, ModelName: "m"}
	counter := func(ctx context.Context, model string, contents []*genai.Content, config *genai.CountTokensConfig) (*genai.CountTokensResponse, error) {
		return nil, nil //nolint:nilnil
	}
	if err := enforcePromptTokensWithCounter(t.Context(), &cfg, counter); err == nil {
		t.Fatalf("expected error on nil response")
	}
}

func TestRecordUsageUpdatesCounters(t *testing.T) {
	t.Parallel()

	// Reset expvar counters to zero for deterministic assertions.
	expRequests.Set(0)
	expInputTokens.Set(0)
	expOutputTokens.Set(0)
	expCostUSD.Set(0)
	if err := initMetrics(); err != nil {
		t.Fatalf("initMetrics: %v", err)
	}

	event := &session.Event{}
	event.LLMResponse = model.LLMResponse{
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     10,
			CandidatesTokenCount: 5,
		},
	}
	in, out := recordUsage(t.Context(), event)
	if in != 10 || out != 5 {
		t.Fatalf("recordUsage returned %d,%d want 10,5", in, out)
	}
	if expRequests.Value() != 1 || expInputTokens.Value() != 10 || expOutputTokens.Value() != 5 {
		t.Fatalf("expvars not updated: req=%d in=%d out=%d", expRequests.Value(), expInputTokens.Value(), expOutputTokens.Value())
	}
}

func TestBuildRunOutput(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		cfg          config
		author       string
		text         string
		inputTokens  int64
		outputTokens int64
		want         runOutput
	}{
		"success: fields copied": {
			cfg: config{
				SessionID:   "session-1",
				ModelName:   "gemini-2.5-flash",
				MaxRounds:   3,
				Temperature: 0.7,
				TopP:        0.9,
				TopK:        4,
				MaxTokens:   256,
				Seed:        42,
			},
			author:       "agent",
			text:         "answer",
			inputTokens:  12,
			outputTokens: 34,
			want: runOutput{
				SessionID:    "session-1",
				Author:       "agent",
				Text:         "answer",
				InputTokens:  12,
				OutputTokens: 34,
				Config: runOutputConfig{
					Model:       "gemini-2.5-flash",
					MaxRounds:   3,
					Temperature: 0.7,
					TopP:        0.9,
					TopK:        4,
					MaxTokens:   256,
					Seed:        42,
				},
			},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			got := buildRunOutput(&tt.cfg, tt.author, tt.text, tt.inputTokens, tt.outputTokens)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Fatalf("run output mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestParseConfigAllowsNoPromptInBatchOrA2A(t *testing.T) {
	resetFlags := func(args []string) func() {
		origArgs := os.Args
		origFlag := flag.CommandLine
		flag.CommandLine = flag.NewFlagSet(args[0], flag.ContinueOnError)
		flag.CommandLine.SetOutput(os.Stdout)
		os.Args = args
		return func() {
			os.Args = origArgs
			flag.CommandLine = origFlag
		}
	}

	tests := map[string]struct {
		args []string
		env  map[string]string
	}{
		"batch: no prompt required": {
			args: []string{"cmd", "-api_key=key", "-batch_file=/tmp/prompts.txt"},
		},
		"bench: no prompt required": {
			args: []string{"cmd", "-api_key=key", "-bench_local=2"},
		},
		"a2a: no prompt required": {
			args: []string{"cmd", "-api_key=key", "-a2a_addr=:8081"},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			restore := resetFlags(tt.args)
			defer restore()
			for k, v := range tt.env {
				t.Setenv(k, v)
			}
			if _, err := parseConfig(); err != nil {
				t.Fatalf("parseConfig error = %v", err)
			}
		})
	}
}
