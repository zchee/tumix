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

package log

import (
	"bytes"
	"context"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"testing"
)

func TestLogAddSource(t *testing.T) {
	wantFile, wantFunc := "logger_test.go", "TestLogAddSource"

	tests := map[string]struct {
		call func(context.Context)
	}{
		"Log": {
			call: func(ctx context.Context) {
				Log(ctx, slog.LevelInfo, "hello")
			},
		},
		"Info": {
			call: func(ctx context.Context) {
				Info(ctx, "hello")
			},
		},
		"Warn": {
			call: func(ctx context.Context) {
				Warn(ctx, "hello")
			},
		},
		"Error": {
			call: func(ctx context.Context) {
				Error(ctx, "hello", fmt.Errorf("fail"))
			},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			var buf bytes.Buffer
			handler := slog.NewJSONHandler(&buf, &slog.HandlerOptions{AddSource: true})

			tt.call(WithLogger(t.Context(), slog.New(handler)))

			var written map[string]any
			if err := json.Unmarshal(buf.Bytes(), &written); err != nil {
				t.Fatalf("json.Unmarshal() error = %v, for %q", err, buf.String())
			}

			source := written["source"].(map[string]any)
			if file := source["file"].(string); !strings.HasSuffix(file, wantFile) {
				t.Fatalf("logged source path %q, want file %q", source, wantFile)
			}
			if funcName := source["function"].(string); !strings.Contains(funcName, wantFunc) {
				t.Fatalf("logged source function %q, want containing %q", source, wantFunc)
			}
		})
	}
}

func BenchmarkLogCallerCapture(b *testing.B) {
	cases := map[string]struct {
		capture bool
	}{
		"capture:on": {
			capture: true,
		},
		"capture:off": {
			capture: false,
		},
	}

	for name, tc := range cases {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			SetCaptureCaller(tc.capture)
			b.Cleanup(func() { SetCaptureCaller(true) })

			logger := slog.New(slog.DiscardHandler)
			ctx := b.Context()
			ctx = WithLogger(ctx, logger)

			for b.Loop() {
				Info(ctx, "hello", "k", 1)
			}
		})
	}
}

func TestFromContextAndCaptureToggle(t *testing.T) {
	cases := map[string]struct {
		capture bool
		useCtx  bool
	}{
		"capture:on_with_context_logger": {
			capture: true,
			useCtx:  true,
		},
		"capture:off_with_context_logger": {
			capture: false,
			useCtx:  true,
		},
		"default_logger_fallback": {
			capture: true,
			useCtx:  false,
		},
	}

	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			SetCaptureCaller(tt.capture)
			t.Cleanup(func() { SetCaptureCaller(true) })

			rec := &recordingHandler{}
			logger := slog.New(rec)

			var ctx context.Context
			if tt.useCtx {
				ctx = WithLogger(t.Context(), logger)
			} else {
				ctx = t.Context()
				orig := slog.Default()
				slog.SetDefault(logger)
				t.Cleanup(func() { slog.SetDefault(orig) })
			}

			Info(ctx, "hello")

			if len(rec.records) != 1 {
				t.Fatalf("records = %d, want 1", len(rec.records))
			}
			gotPC := rec.records[0].PC
			if tt.capture && gotPC == 0 {
				t.Fatalf("capture enabled but pc=0")
			}
			if !tt.capture && gotPC != 0 {
				t.Fatalf("capture disabled but pc=%d", gotPC)
			}
		})
	}
}

func TestErrorAddsErrorAttribute(t *testing.T) {
	t.Parallel()

	rec := &recordingHandler{}
	logger := slog.New(rec)
	ctx := WithLogger(t.Context(), logger)

	err := fmt.Errorf("boom")
	Error(ctx, "failed", err, "k", 1)

	if len(rec.records) != 1 {
		t.Fatalf("records = %d, want 1", len(rec.records))
	}
	attrs := attrsToMap(rec.records[0])
	gotErr, ok := attrs["error"].(error)
	if !ok || !errors.Is(gotErr, err) {
		t.Fatalf("error attr = %v, want %v", attrs["error"], err)
	}
	if got := fmt.Sprint(attrs["k"]); got != "1" {
		t.Fatalf("k attr = %v, want 1", attrs["k"])
	}
}

type recordingHandler struct {
	mu      sync.Mutex
	records []slog.Record
}

func (r *recordingHandler) Enabled(context.Context, slog.Level) bool { return true }

func (r *recordingHandler) Handle(_ context.Context, rec slog.Record) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.records = append(r.records, rec.Clone())
	return nil
}

func (r *recordingHandler) WithAttrs([]slog.Attr) slog.Handler { return r }

func (r *recordingHandler) WithGroup(string) slog.Handler { return r }

func attrsToMap(rec slog.Record) map[string]any {
	out := make(map[string]any)
	rec.Attrs(func(a slog.Attr) bool {
		out[a.Key] = a.Value.Any()
		return true
	})
	return out
}
