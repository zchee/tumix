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

// Package log provides a way to attach a [slog.Logger] with request-specific attributes to
// a [context.Context]. This logger can then be retrieved via the [context.Context] or
// used indirectly through package-level logging function calls.
//
// The tumix implementations must use this package when outputting logs, rather than using other loggers or [slog.Logger] directly.
package log

import (
	"context"
	"log/slog"
	"runtime"
	"slices"
	"sync/atomic"
	"time"
)

// loggerKey is the type used for the [context.Context] key for storing the logger.
type loggerKey struct{}

// WithLogger returns a new [context.Context] that carries the provided [*slog.Logger].
func WithLogger(ctx context.Context, logger *slog.Logger) context.Context {
	return context.WithValue(ctx, loggerKey{}, logger)
}

var captureCaller atomic.Bool

func init() {
	captureCaller.Store(true)
}

// SetCaptureCaller toggles whether log records include call-site information.
//
// Disabling caller capture avoids the runtime.Callers overhead in hot logging paths.
func SetCaptureCaller(enabled bool) {
	captureCaller.Store(enabled)
}

// FromContext returns the [*slog.Logger] associated with the provided [context.Context] or [slog.Default] if no context-scoped logger is available.
func FromContext(ctx context.Context) *slog.Logger {
	if logger, ok := ctx.Value(loggerKey{}).(*slog.Logger); ok {
		return logger
	}
	return slog.Default()
}

// Log invokes [slog.Logger.Log] associated with the provided [context.Context], or [slog.Default] if no context-scoped logger is available.
func Log(ctx context.Context, level slog.Level, msg string, args ...any) {
	doLog(ctx, level, msg, args...)
}

// Info invokes [slog.Logger.InfoContext] associated with the provided [context.Context], or [slog.Default] if no context-scoped logger is available.
func Info(ctx context.Context, msg string, args ...any) {
	doLog(ctx, slog.LevelInfo, msg, args...)
}

// Warn invokes [slog.Logger.WarnContext] associated with the provided [context.Context], or [slog.Default] if no context-scoped logger is available.
func Warn(ctx context.Context, msg string, args ...any) {
	doLog(ctx, slog.LevelWarn, msg, args...)
}

// Error invokes [slog.Logger.ErrorContext] associated with the provided [context.Context], or [slog.Default] if no context-scoped logger is available.
func Error(ctx context.Context, msg string, err error, args ...any) {
	doLog(ctx, slog.LevelError, msg, slices.Concat([]any{"error", err}, args)...)
}

// If we use [slog.Logger.Log] directly in our log package methods, these methods are logged as the call site.
func doLog(ctx context.Context, level slog.Level, msg string, args ...any) {
	if logger := FromContext(ctx); logger.Enabled(ctx, level) {
		var pc uintptr
		if captureCaller.Load() {
			var pcs [1]uintptr
			// skip [runtime.Callers], [doLog], and caller
			runtime.Callers(3, pcs[:])
			pc = pcs[0]
		}

		record := slog.NewRecord(time.Now(), level, msg, pc)
		record.Add(args...)
		logger.Handler().Handle(ctx, record) //nolint:errcheck
	}
}
