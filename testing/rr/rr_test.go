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

package rr

import (
	"errors"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-replayers/httpreplay"
)

func TestNewHTTPClientRecordCreatesReplay(t *testing.T) {
	t.Parallel()

	const httpAddr = "127.0.0.1:28090"
	replayPath := filepath.Join("testdata", t.Name()+".replay")
	restore, recording := withRecordMode(t, replayPath)
	t.Cleanup(restore)

	client, cleanup, _ := NewHTTPClient(t, func(r *httpreplay.Recorder) {})

	mux := http.NewServeMux()
	mux.HandleFunc("/ping", func(w http.ResponseWriter, _ *http.Request) {
		if _, err := w.Write([]byte("pong")); err != nil {
			t.Fatal(err)
		}
	})
	srv := http.Server{Handler: mux}
	if recording {
		ln := mustListen(t, httpAddr)
		t.Cleanup(func() {
			if err := srv.Shutdown(t.Context()); err != nil {
				t.Fatal(err)
			}
		})
		go func() {
			if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
				t.Error(err)
			}
		}()
	}

	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "http://"+httpAddr+"/ping", http.NoBody)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("GET /ping: %v", err)
	}
	_ = resp.Body.Close()

	secondReq, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "http://"+httpAddr+"/ping", http.NoBody)
	if err != nil {
		t.Fatalf("build second request: %v", err)
	}
	secondReq.Header.Set("User-Agent", "test-ua-different")
	resp2, err := client.Do(secondReq)
	if err != nil {
		t.Fatalf("GET /ping second: %v", err)
	}
	_ = resp2.Body.Close()

	cleanup()

	if !recording {
		return
	}
	if _, statErr := os.Stat(replayPath); statErr != nil {
		t.Fatalf("expected replay file %s: %v", replayPath, statErr)
	}

	if t.Failed() {
		t.FailNow()
	}
}

func TestNewHTTPClientReplayUsesGoldenFile(t *testing.T) {
	const httpAddr = "127.0.0.1:28090"

	orig := *Record
	*Record = false
	t.Cleanup(func() { *Record = orig })

	replayPath := filepath.Join("testdata", t.Name()+".replay")
	src := filepath.Join("testdata", "TestNewHTTPClientRecordCreatesReplay.replay")
	copyReplay(t, src, replayPath)

	client, cleanup, _ := NewHTTPClient(t, func(r *httpreplay.Recorder) {})
	t.Cleanup(cleanup)

	// request 1: default headers (matches first recorded interaction)
	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "http://"+httpAddr+"/ping", http.NoBody)
	if err != nil {
		t.Fatalf("build replay request: %v", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("replay GET /ping: %v", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if got := strings.TrimSpace(string(body)); got != "pong" {
		t.Fatalf("replay body = %q, want pong", got)
	}

	// request 2: different UA (matches second recorded interaction; headers are ignored)
	req, err = http.NewRequestWithContext(t.Context(), http.MethodGet, "http://"+httpAddr+"/ping", http.NoBody)
	if err != nil {
		t.Fatalf("build replay request #2: %v", err)
	}
	req.Header.Set("User-Agent", "test-ua-different")
	resp2, err := client.Do(req)
	if err != nil {
		t.Fatalf("replay GET /ping #2: %v", err)
	}
	defer resp2.Body.Close()
	body2, err := io.ReadAll(resp2.Body)
	if err != nil {
		t.Fatalf("read body #2: %v", err)
	}
	if got := strings.TrimSpace(string(body2)); got != "pong" {
		t.Fatalf("replay body #2 = %q, want pong", got)
	}
}

func TestWithRecordMode(t *testing.T) {
	tmp := t.TempDir()
	existing := filepath.Join(tmp, "has.replay")
	if err := os.WriteFile(existing, []byte("dummy"), 0o644); err != nil {
		t.Fatalf("write dummy: %v", err)
	}

	t.Run("existing file leaves record false", func(t *testing.T) {
		orig := *Record
		*Record = false
		restore, recording := withRecordMode(t, existing)
		defer restore()
		if recording {
			t.Fatalf("recording = true, want false")
		}
		if *Record != false {
			t.Fatalf("Record flag mutated to %v", *Record)
		}
		if *Record != orig && orig {
			t.Fatalf("did not restore correctly")
		}
	})

	t.Run("missing file sets record true", func(t *testing.T) {
		path := filepath.Join(tmp, "missing.replay")
		orig := *Record
		*Record = false
		restore, recording := withRecordMode(t, path)
		defer restore()
		if !recording {
			t.Fatalf("recording = false, want true")
		}
		if *Record != true {
			t.Fatalf("Record flag not set")
		}
		restore()
		if *Record != orig {
			t.Fatalf("Record not restored")
		}
	})
}

func copyReplay(t *testing.T, src, dst string) {
	t.Helper()

	data, err := os.ReadFile(src)
	if err != nil {
		t.Fatalf("read %s: %v", src, err)
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		t.Fatalf("mkdir testdata: %v", err)
	}
	if err := os.WriteFile(dst, data, 0o644); err != nil {
		t.Fatalf("write %s: %v", dst, err)
	}
}

func withRecordMode(t *testing.T, replay string) (func(), bool) {
	t.Helper()

	orig := *Record
	recording := orig
	if _, err := os.Stat(replay); err != nil && errors.Is(err, os.ErrNotExist) {
		recording = true
	}
	*Record = recording
	return func() { *Record = orig }, recording
}

func mustListen(t *testing.T, addr string) net.Listener {
	t.Helper()

	var lc net.ListenConfig
	ln, err := lc.Listen(t.Context(), "tcp", addr)
	if err != nil {
		t.Fatalf("listen %s: %v", addr, err)
	}
	return ln
}
