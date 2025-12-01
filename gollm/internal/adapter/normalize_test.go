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

package adapter

import (
	"net/http"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestNormalizeRequest(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		req             *model.LLMRequest
		userAgent       string
		wantUserAgent   string
		wantContentsLen int
		wantLastRole    string
		wantFirstText   string
	}{
		"fills_missing_fields": {
			req:             &model.LLMRequest{},
			userAgent:       "ua/1.0",
			wantUserAgent:   "ua/1.0",
			wantContentsLen: 1,
			wantLastRole:    genai.RoleUser,
			wantFirstText:   defaultUserPrompt,
		},
		"overrides_existing_user_agent": {
			req: &model.LLMRequest{
				Contents: []*genai.Content{genai.NewContentFromText("hello", genai.RoleUser)},
				Config: &genai.GenerateContentConfig{
					HTTPOptions: &genai.HTTPOptions{
						Headers: http.Header{"User-Agent": []string{"old-agent"}},
					},
				},
			},
			userAgent:       "new-agent",
			wantUserAgent:   "new-agent",
			wantContentsLen: 1,
			wantLastRole:    genai.RoleUser,
			wantFirstText:   "hello",
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			cfg := NormalizeRequest(tt.req, tt.userAgent)

			if cfg == nil || cfg.HTTPOptions == nil || cfg.HTTPOptions.Headers == nil {
				t.Fatalf("config/http options/headers not initialized: %+v", cfg)
			}
			if got := cfg.HTTPOptions.Headers.Get("User-Agent"); got != tt.wantUserAgent {
				t.Fatalf("user-agent = %q, want %q", got, tt.wantUserAgent)
			}
			if got := len(tt.req.Contents); got != tt.wantContentsLen {
				t.Fatalf("contents len = %d, want %d", got, tt.wantContentsLen)
			}
			last := tt.req.Contents[len(tt.req.Contents)-1]
			if last.Role != tt.wantLastRole {
				t.Fatalf("last role = %q, want %q", last.Role, tt.wantLastRole)
			}
			if text := last.Parts[0].Text; text != tt.wantFirstText {
				t.Fatalf("last text = %q, want %q", text, tt.wantFirstText)
			}
		})
	}
}

func TestModelName(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		defaultName string
		req         *model.LLMRequest
		want        string
	}{
		"uses_request_model": {
			defaultName: "default",
			req:         &model.LLMRequest{Model: " custom-model "},
			want:        "custom-model",
		},
		"falls_back_to_default": {
			defaultName: "default",
			req:         &model.LLMRequest{},
			want:        "default",
		},
		"nil_request": {
			defaultName: "default",
			req:         nil,
			want:        "default",
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			got := ModelName(tt.defaultName, tt.req)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Fatalf("ModelName diff (-want +got):\n%s", diff)
			}
			if tt.req != nil && strings.TrimSpace(tt.req.Model) == "" && got != tt.defaultName {
				t.Fatalf("expected fallback to default when model empty")
			}
		})
	}
}
