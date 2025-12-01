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

// Package rr provides the recording and replaying testing for HTTP and gRPC.
package rr

import (
	"context"
	"flag"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	gcp_credentials "cloud.google.com/go/auth/credentials"
	"cloud.google.com/go/auth/oauth2adapt"
	"github.com/google/go-replayers/grpcreplay"
	"github.com/google/go-replayers/httpreplay"
	"google.golang.org/grpc"
	grpc_credentials "google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	grpc_oauth "google.golang.org/grpc/credentials/oauth"

	"github.com/zchee/tumix/internal/version"
	"github.com/zchee/tumix/log"
)

var ignoreHeaders = []string{
	"Accept-Encoding",
	"User-Agent",
	"X-Stainless-Arch",
	"X-Stainless-Os",
	"X-Stainless-Retry-Count",
	"X-Stainless-Runtime-Version",
}

// Record is true iff the tests are being run in "record" mode.
var Record = flag.Bool("record", false, "whether to run tests against SaaS resources and record the interactions")

// Recorder is an alias of [httpreplay.Recorder].
type Recorder = httpreplay.Recorder

func init() {
	httpreplay.DebugHeaders()
}

// NewHTTPClient creates a new [*http.Client] for testing.
//
// This client's activity is being either recorded to files (when [*httpreplay.Record] is set) or replayed from files.
// The recorder will save its data to a file named "testdata/<test name>.replay".
//
// The rf is a modifier function that will be invoked with the address of the [httpreplay.Recorder] object
// used to obtain the client; this function can mutate the recorder to add service-specific header filters,
// for example.
//
// An initState is returned for tests that need a state to have deterministic results,
// for example, a seed to generate random sequences.
//
//nolint:unparam // initState is optional for callers; retained for determinism hooks.
func NewHTTPClient(t *testing.T, rf func(r *Recorder)) (c *http.Client, cleanup func(), initState int64) {
	t.Helper()

	golden := filepath.Join("testdata", strings.ReplaceAll(t.Name(), "/", "_")+".replay")
	if err := os.MkdirAll(filepath.Dir(golden), 0o755); err != nil {
		t.Fatal(err)
	}

	if *Record {
		t.Logf("Recording into golden file: %s", golden)

		state := time.Now()
		b, err := state.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		recoder, err := httpreplay.NewRecorderWithOpts(golden,
			httpreplay.RecorderInitial(b),
		)
		if err != nil {
			t.Fatal(err)
		}
		for _, h := range ignoreHeaders {
			recoder.RemoveRequestHeaders(h)
		}
		rf(recoder)
		cleanup = func() {
			if err := recoder.Close(); err != nil {
				t.Fatal(err)
			}
		}

		return recoder.Client(), cleanup, state.UnixNano()
	}

	t.Logf("Replaying from golden file %s", golden)
	replay, err := httpreplay.NewReplayer(golden)
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range ignoreHeaders {
		replay.IgnoreHeader(h)
	}

	recState := new(time.Time)
	if err := recState.UnmarshalBinary(replay.Initial()); err != nil {
		t.Fatal(err)
	}

	cleanup = func() {
		if err := replay.Close(); err != nil {
			t.Fatal(err)
		}
	}

	return replay.Client(), cleanup, recState.UnixNano()
}

// NewGRPCConn creates a new connection for testing against gRPC.
//
// If the test is in --record mode, the client will call out to GCP, and the
// results are recorded in a replay file.
//
// Otherwise, the session reads a replay file and runs the test as a replay,
// which never makes an outgoing RPC and uses fake credentials.
func NewGRPCConn(t *testing.T, apiName, endPoint string, opts ...grpc.DialOption) (conn *grpc.ClientConn, cleanup func()) {
	t.Helper()

	return NewGRPCConnWithCreds(t, apiName, endPoint, grpc_credentials.NewClientTLSFromCert(nil, ""), opts...)
}

// NewGRPCConnWithCreds mirrors [NewGRPCConn] but allows callers to override transport credentials,
// enabling plaintext replay against local fixtures.
func NewGRPCConnWithCreds(t *testing.T, apiName, endPoint string, creds grpc_credentials.TransportCredentials, opts ...grpc.DialOption) (conn *grpc.ClientConn, cleanup func()) {
	t.Helper()

	golden := filepath.Join("testdata", strings.ReplaceAll(t.Name(), "/", "_")+".replay")
	if err := os.MkdirAll(filepath.Dir(golden), 0o755); err != nil {
		t.Fatal(err)
	}

	if *Record {
		t.Logf("Recording into golden file: %s", golden)

		recoder, err := grpcreplay.NewRecorder(golden, &grpcreplay.RecorderOptions{
			Text: true,
		})
		if err != nil {
			t.Fatal(err)
		}

		if creds == nil {
			creds = grpc_credentials.NewClientTLSFromCert(nil, "")
		}

		dopts := append(recoder.DialOptions(),
			grpc.WithUserAgent(version.UserAgent(apiName)),
			grpc.WithTransportCredentials(creds),
		)
		// overwrite dopts to opts args
		opts = append(dopts, opts...)

		conn, err = grpc.NewClient(endPoint, opts...)
		if err != nil {
			_ = recoder.Close() // force close
			t.Fatal(err)
		}
		cleanup = func() {
			if err := recoder.Close(); err != nil {
				t.Errorf("unable to close recorder: %v", err)
			}
		}

		return conn, cleanup
	}

	t.Logf("Replaying from golden file %s", golden)
	replayer, err := grpcreplay.NewReplayer(golden, nil)
	if err != nil {
		t.Fatal(err)
	}

	conn, err = replayer.Connection()
	if err != nil {
		t.Fatal(err)
	}
	cleanup = func() {
		if err := replayer.Close(); err != nil {
			t.Errorf("unable to close recorder: %v", err)
		}
	}

	return conn, cleanup
}

// NewInsecureGRPCConn creates a plaintext gRPC replay connection using the shared recorder logic.
func NewInsecureGRPCConn(t *testing.T, apiName, endPoint string, opts ...grpc.DialOption) (conn *grpc.ClientConn, cleanup func()) {
	t.Helper()

	return NewGRPCConnWithCreds(t, apiName, endPoint, insecure.NewCredentials(), opts...)
}

// NewGCPGRPCConn creates a new [*grpc.ClientConn] for testing against Google Cloud.
//
// TODO(zchee): Use [google.FindDefaultCredentials] instead of [gcp_credentials.DetectDefault]?
func NewGCPGRPCConn(ctx context.Context, t *testing.T, apiName, endPoint string, opts ...grpc.DialOption) (conn *grpc.ClientConn, done func()) {
	t.Helper()

	// Add GCP credentials for real RPCs
	gcpOpts := &gcp_credentials.DetectOptions{
		Scopes: []string{"https://www.googleapis.com/auth/cloud-platform"},
		Logger: log.FromContext(ctx),
	}
	adcCreds, err := gcp_credentials.DetectDefault(gcpOpts)
	if err != nil {
		t.Fatal(err)
	}

	opts = append(opts,
		grpc.WithPerRPCCredentials(grpc_oauth.TokenSource{
			TokenSource: oauth2adapt.TokenSourceFromTokenProvider(adcCreds),
		}),
	)

	return NewGRPCConn(t, apiName, endPoint, opts...)
}

// Recording reports whether the rr is recoding mode.
func Recording() bool {
	return *Record
}

// Replaying reports whether the rr is replaying mode.
func Replaying() bool {
	return !*Record
}
