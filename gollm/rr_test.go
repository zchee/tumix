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
	"net"
	"net/http"
	"slices"
	"testing"
)

// startStubHTTP spins up a minimal HTTP server that responds with payload when the request path matches any allowed path.
// The caller is responsible for selecting unique addresses per test.
func startStubHTTP(t *testing.T, addr string, allowed []string, payload string) func() {
	t.Helper()

	var lc net.ListenConfig
	ln, err := lc.Listen(t.Context(), "tcp", addr)
	if err != nil {
		t.Fatalf("listen %s: %v", addr, err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		ok := slices.Contains(allowed, r.URL.Path)
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(payload))
	})

	srv := &http.Server{Handler: mux}
	go func() {
		if err := srv.Serve(ln); err != nil {
			t.Error(err)
		}
	}()

	return func() {
		if err := srv.Shutdown(t.Context()); err != nil {
			t.Fatal(err)
		}
		if err := ln.Close(); err != nil {
			t.Fatal(err)
		}
		if t.Failed() {
			t.FailNow()
		}
	}
}
