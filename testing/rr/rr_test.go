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
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-replayers/httpreplay"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/emptypb"
)

type pingServer struct{}

func (pingServer) Ping(context.Context, *emptypb.Empty) (*emptypb.Empty, error) {
	return &emptypb.Empty{}, nil
}

type pingService interface {
	Ping(ctx context.Context, req *emptypb.Empty) (*emptypb.Empty, error)
}

func registerPingService(s *grpc.Server) {
	s.RegisterService(&grpc.ServiceDesc{
		ServiceName: "test.Ping",
		HandlerType: (*pingService)(nil),
		Methods: []grpc.MethodDesc{
			{
				MethodName: "Ping",
				Handler: func(srv any, ctx context.Context, dec func(any) error, interceptor grpc.UnaryServerInterceptor) (any, error) {
					in := &emptypb.Empty{}
					if err := dec(in); err != nil {
						return nil, err
					}
					if interceptor == nil {
						return srv.(pingServer).Ping(ctx, in)
					}
					info := &grpc.UnaryServerInfo{
						Server:     srv,
						FullMethod: "/test.Ping/Ping",
					}
					handler := func(ctx context.Context, req any) (any, error) {
						return srv.(pingServer).Ping(ctx, req.(*emptypb.Empty))
					}
					return interceptor(ctx, in, info, handler)
				},
			},
		},
		Streams:  []grpc.StreamDesc{},
		Metadata: "test.proto",
	}, pingServer{})
}

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
			if err := srv.Serve(ln); err != nil {
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
	*Record = true
	t.Cleanup(func() { *Record = orig })

	replayPath := filepath.Join("testdata", t.Name()+".replay")
	_ = os.Remove(replayPath)

	client, cleanup, _ := NewHTTPClient(t, func(r *httpreplay.Recorder) {})

	mux := http.NewServeMux()
	mux.HandleFunc("/ping", func(w http.ResponseWriter, _ *http.Request) {
		if _, err := w.Write([]byte("pong")); err != nil {
			t.Fatal(err)
		}
	})
	srv := http.Server{Handler: mux}
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

	// record
	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "http://"+httpAddr+"/ping", http.NoBody)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("record GET /ping: %v", err)
	}
	_ = resp.Body.Close()

	req, err = http.NewRequestWithContext(t.Context(), http.MethodGet, "http://"+httpAddr+"/ping", http.NoBody)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	req.Header.Set("User-Agent", "test-ua-different")

	resp, err = client.Do(req)
	if err != nil {
		t.Fatalf("record GET /ping: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if got := strings.TrimSpace(string(body)); got != "pong" {
		t.Fatalf("body = %q, want pong", got)
	}

	// replay
	cleanup()
	*Record = false
	replayClient, replayCleanup, _ := NewHTTPClient(t, func(r *httpreplay.Recorder) {})
	t.Cleanup(replayCleanup)

	replayReq, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "http://"+httpAddr+"/ping", http.NoBody)
	if err != nil {
		t.Fatalf("build replay request: %v", err)
	}
	replayReq.Header.Set("User-Agent", "test-ua-different")
	replayResp, err := replayClient.Do(replayReq)
	if err != nil {
		t.Fatalf("replay GET /ping: %v", err)
	}
	defer replayResp.Body.Close()
	replayBody, err := io.ReadAll(replayResp.Body)
	if err != nil {
		t.Fatalf("read replay body: %v", err)
	}
	if got := strings.TrimSpace(string(replayBody)); got != "pong" {
		t.Fatalf("replay body = %q, want pong", got)
	}
}

func TestNewInsecureGRPCConnRecordCreatesReplay(t *testing.T) {
	const grpcAddr = "127.0.0.1:28091"
	replayPath := filepath.Join("testdata", t.Name()+".replay")
	restore, recording := withRecordMode(t, replayPath)
	t.Cleanup(restore)

	if recording {
		ln := mustListen(t, grpcAddr)
		s := grpc.NewServer()
		registerPingService(s)
		go func() {
			if err := s.Serve(ln); err != nil {
				t.Error(err)
			}
		}()
		t.Cleanup(s.Stop)
	}

	conn, cleanup := NewInsecureGRPCConn(t, "rr-test", grpcAddr)

	ctx := t.Context()
	if err := conn.Invoke(ctx, "/test.Ping/Ping", &emptypb.Empty{}, &emptypb.Empty{}); err != nil {
		t.Fatalf("Invoke Ping: %v", err)
	}

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

func TestNewInsecureGRPCConnReplayUsesGoldenFile(t *testing.T) {
	const grpcAddr = "127.0.0.1:28091"

	orig := *Record
	*Record = true
	t.Cleanup(func() { *Record = orig })

	replayPath := filepath.Join("testdata", t.Name()+".replay")
	_ = os.Remove(replayPath)

	ln := mustListen(t, grpcAddr)
	s := grpc.NewServer()
	registerPingService(s)
	go func() {
		if err := s.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			t.Error(err)
		}
	}()
	t.Cleanup(s.Stop)

	conn, cleanup := NewInsecureGRPCConn(t, "rr-test", grpcAddr)
	if err := conn.Invoke(t.Context(), "/test.Ping/Ping", &emptypb.Empty{}, &emptypb.Empty{}); err != nil {
		t.Fatalf("record Invoke Ping: %v", err)
	}
	cleanup()

	// replay without server
	s.Stop()
	*Record = false

	replayConn, replayCleanup := NewInsecureGRPCConn(t, "rr-test", grpcAddr)
	defer replayCleanup()

	ctx := t.Context()
	if err := replayConn.Invoke(ctx, "/test.Ping/Ping", &emptypb.Empty{}, &emptypb.Empty{}); err != nil {
		t.Fatalf("Invoke Ping on replay: %v", err)
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
