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
	"fmt"
	"iter"
	"net"
	"net/http"
	"slices"
	"strings"
	"testing"

	"google.golang.org/adk/model"
)

// TextResponse holds the concatenated text from a response stream,
// separated into partial and final parts.
type TextResponse struct {
	// PartialText is the full text concatenated from all partial (streaming) responses.
	PartialText string
	// FinalText is the full text concatenated from all final (non-partial) responses.
	FinalText string
}

// readResponse transforms a sequence into a TextResponse, concatenating the text value of the response parts
// depending on the readPartial value it will only concatenate the text of partial events or the text of non partial events
func readResponse(s iter.Seq2[*model.LLMResponse, error]) (TextResponse, error) {
	var partialBuilder, finalBuilder strings.Builder
	var result TextResponse

	for resp, err := range s {
		if err != nil {
			// Return what we have so far, along with the error.
			result.PartialText = partialBuilder.String()
			result.FinalText = finalBuilder.String()
			return result, err
		}
		if resp.Content == nil || len(resp.Content.Parts) == 0 {
			return result, fmt.Errorf("encountered an empty response: %v", resp)
		}

		text := resp.Content.Parts[0].Text
		if resp.Partial {
			partialBuilder.WriteString(text)
		} else {
			finalBuilder.WriteString(text)
		}
	}

	result.PartialText = partialBuilder.String()
	result.FinalText = finalBuilder.String()
	return result, nil
}

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
		if _, err := w.Write([]byte(payload)); err != nil {
			t.Fatal(err)
		}
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
