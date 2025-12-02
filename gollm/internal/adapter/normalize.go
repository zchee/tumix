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

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

const (
	defaultUserPrompt  = "Handle the requests as specified in the System Instruction."
	continueUserPrompt = "Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed."
)

// NormalizeRequest ensures request config, HTTP headers, and user turn presence are set, then applies the provided user-agent.
func NormalizeRequest(req *model.LLMRequest, userAgent string) *genai.GenerateContentConfig {
	req.Contents = EnsureUserContent(req.Contents)
	cfg := ensureConfig(req)

	if ua := strings.TrimSpace(userAgent); ua != "" {
		cfg.HTTPOptions.Headers.Set("User-Agent", ua)
	}

	return cfg
}

// EnsureUserContent appends a user turn if the content list is empty or does not end with a user role.
func EnsureUserContent(contents []*genai.Content) []*genai.Content {
	if len(contents) == 0 {
		contents = append(contents, genai.NewContentFromText(defaultUserPrompt, genai.RoleUser))
		return contents
	}

	if last := contents[len(contents)-1]; last == nil || last.Role != genai.RoleUser {
		contents = append(contents, genai.NewContentFromText(continueUserPrompt, genai.RoleUser))
	}
	return contents
}

// ModelName returns the trimmed request model if set, otherwise the provided default.
func ModelName(defaultName string, req *model.LLMRequest) string {
	if req != nil {
		if name := strings.TrimSpace(req.Model); name != "" {
			return name
		}
	}

	return defaultName
}

func ensureConfig(req *model.LLMRequest) *genai.GenerateContentConfig {
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if req.Config.HTTPOptions == nil {
		req.Config.HTTPOptions = &genai.HTTPOptions{}
	}
	if req.Config.HTTPOptions.Headers == nil {
		req.Config.HTTPOptions.Headers = make(http.Header)
	}

	return req.Config
}
