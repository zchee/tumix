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

package agent

import (
	"fmt"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// NewAutoAgents creates n lightweight auto-designed agents.
// They emulate the paper's LLM-designed variants by varying tool emphasis.
func NewAutoAgents(llm model.LLM, genCfg *genai.GenerateContentConfig, n int) ([]agent.Agent, error) {
	if n <= 0 {
		return nil, nil
	}
	agents := make([]agent.Agent, 0, n)
	emphases := []string{"textual reasoning", "code execution", "web search"}
	for i := range n {
		name := fmt.Sprintf("Auto-%d", i+1)
		emphasis := emphases[i%len(emphases)]
		cfg := llmagent.Config{
			Name:                  name,
			Description:           fmt.Sprintf("Auto-designed agent focusing on %s.", emphasis),
			Model:                 llm,
			GenerateContentConfig: cloneGenConfig(genCfg),
			Instruction: fmt.Sprintf(`You are an auto-designed agent. Primary focus: %s.
Use chain-of-thought, then pick a single best action: plain answer, one python block, or one <search>…</search> query.
Do not mix code and search in the same turn. Respond with final answer inside «< and »>.`, emphasis),
		}
		applySharedContext(&cfg)
		a, err := llmagent.New(cfg)
		if err != nil {
			return nil, fmt.Errorf("build %s: %w", name, err)
		}
		agents = append(agents, a)
	}
	return agents, nil
}
