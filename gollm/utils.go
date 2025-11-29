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
	"google.golang.org/adk/model"

	"github.com/zchee/tumix/gollm/internal/adapter"
)

// ensureUserContent aligns with ADK behavior of ending with a user turn.
//
// It appends a default user message if the request content is empty or
// if the last message is not from the user.
func ensureUserContent(req *model.LLMRequest) {
	adapter.EnsureUserContent(req)
}

// resolveModelName returns the model name to use for the request.
//
// It prefers the model name in the request if provided, otherwise falls back to the default name.
func resolveModelName(req *model.LLMRequest, defaultName string) string {
	return adapter.ModelName(defaultName, req)
}
