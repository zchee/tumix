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
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"cloud.google.com/go/auth/credentials"
	"cloud.google.com/go/auth/oauth2adapt"
	"github.com/google/go-replayers/grpcreplay"
	"github.com/google/go-replayers/httpreplay"
	"google.golang.org/grpc"
	grpc_credentials "google.golang.org/grpc/credentials"
	grpc_oauth "google.golang.org/grpc/credentials/oauth"

	"github.com/zchee/tumix/internal/version"
)

// NewRecordHTTPClient creates a new [*http.Client] for testing.
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
func NewRecordHTTPClient(_ context.Context, t *testing.T, rf func(r *httpreplay.Recorder)) (c *http.Client, cleanup func(), initState int64) {
	t.Helper()

	golden := filepath.Join("testdata", t.Name()+".replay")
	if err := os.MkdirAll(filepath.Dir(golden), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Logf("Recording into golden file: %s", golden)

	httpreplay.DebugHeaders()

	init := time.Now()
	b, err := init.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	recoder, err := httpreplay.NewRecorderWithOpts(golden,
		httpreplay.RecorderInitial(b),
		httpreplay.RecorderPort(0),
	)
	if err != nil {
		t.Fatal(err)
	}

	rf(recoder)

	cleanup = func() {
		if err := recoder.Close(); err != nil {
			t.Fatal(err)
		}
	}

	return recoder.Client(), cleanup, init.UnixNano()
}

// NewGRPCConn creates a new connection for testing against gRPC.
//
// If the test is in --record mode, the client will call out to GCP, and the
// results are recorded in a replay file.
//
// Otherwise, the session reads a replay file and runs the test as a replay,
// which never makes an outgoing RPC and uses fake credentials.
func NewGRPCConn(_ context.Context, t *testing.T, apiName, endPoint string, opts ...grpc.DialOption) (conn *grpc.ClientConn, cleanup func()) {
	t.Helper()

	golden := filepath.Join("testdata", t.Name()+".replay")
	if err := os.MkdirAll(filepath.Dir(golden), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Logf("Recording into golden file: %s", golden)

	// TODO(zchee): needs it?
	init := time.Now()
	b, err := init.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	rr, err := grpcreplay.NewRecorder(golden, &grpcreplay.RecorderOptions{
		Initial: b,
	})
	if err != nil {
		t.Fatal(err)
	}

	dopts := append(rr.DialOptions(),
		grpc.WithUserAgent(version.UserAgent(apiName)),
		grpc.WithTransportCredentials(grpc_credentials.NewClientTLSFromCert(nil, "")),
	)
	// overwrite dopts to opts args
	opts = append(dopts, opts...)

	conn, err = grpc.NewClient(endPoint, opts...)
	if err != nil {
		_ = rr.Close() // force close
		t.Fatal(err)
	}

	cleanup = func() {
		if err := rr.Close(); err != nil {
			t.Errorf("unable to close recorder: %v", err)
		}
	}

	return conn, cleanup
}

// NewGCPGRPCConn creates a new [*grpc.ClientConn] for testing against Google Cloud.
//
// TODO(zchee): Use [google.FindDefaultCredentials] instead of [credentials.DetectDefault] ?
func NewGCPGRPCConn(ctx context.Context, t *testing.T, apiName, endPoint string, opts ...grpc.DialOption) (conn *grpc.ClientConn, done func()) {
	t.Helper()

	// Add GCP credentials for real RPCs
	gcpOpts := &credentials.DetectOptions{
		Scopes: []string{"https://www.googleapis.com/auth/cloud-platform"},
	}
	adcCreds, err := credentials.DetectDefault(gcpOpts)
	if err != nil {
		t.Fatal(err)
	}

	opts = append(opts,
		grpc.WithPerRPCCredentials(grpc_oauth.TokenSource{
			TokenSource: oauth2adapt.TokenSourceFromTokenProvider(adcCreds),
		}),
	)

	return NewGRPCConn(ctx, t, apiName, endPoint, opts...)
}
