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
	"fmt"
	"strings"

	"google.golang.org/genai"
)

// toolID returns a unique identifier for a tool call.
//
// If id is empty, it generates one based on the content and part indices.
func toolID(id string, contentIdx, partIdx int) string {
	if strings.TrimSpace(id) != "" {
		return id
	}
	return fmt.Sprintf("tool_%d_%d", contentIdx, partIdx)
}

func joinTextParts(parts []*genai.Part) string {
	var sb strings.Builder
	for _, p := range parts {
		if p == nil {
			continue
		}
		sb.WriteString(p.Text)
	}
	return sb.String()
}
