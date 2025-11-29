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
	"testing"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestEnsureUserContent(t *testing.T) {
	t.Run("adds_default_when_empty", func(t *testing.T) {
		req := &model.LLMRequest{}

		ensureUserContent(req)

		if len(req.Contents) != 1 {
			t.Fatalf("ensureUserContent added %d contents, want 1", len(req.Contents))
		}
		if req.Contents[0].Role != genai.RoleUser {
			t.Fatalf("role = %q, want %q", req.Contents[0].Role, genai.RoleUser)
		}
		if got, want := req.Contents[0].Parts[0].Text, "Handle the requests as specified in the System Instruction."; got != want {
			t.Fatalf("default text = %q, want %q", got, want)
		}
	})

	t.Run("appends_when_last_not_user", func(t *testing.T) {
		req := &model.LLMRequest{
			Contents: []*genai.Content{genai.NewContentFromText("system guidance", "system")},
		}

		ensureUserContent(req)

		if got, want := len(req.Contents), 2; got != want {
			t.Fatalf("len(contents) = %d, want %d", got, want)
		}
		if req.Contents[1].Role != genai.RoleUser {
			t.Fatalf("role = %q, want %q", req.Contents[1].Role, genai.RoleUser)
		}
	})

	t.Run("no_change_when_last_user", func(t *testing.T) {
		req := &model.LLMRequest{
			Contents: []*genai.Content{genai.NewContentFromText("hello", genai.RoleUser)},
		}

		ensureUserContent(req)

		if got, want := len(req.Contents), 1; got != want {
			t.Fatalf("len(contents) = %d, want %d", got, want)
		}
	})
}
