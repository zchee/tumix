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
	"context"
	"iter"
	"testing"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

type stubLLM struct{}

var _ model.LLM = (*stubLLM)(nil)

// Name implements [model.LLM].
func (s *stubLLM) Name() string { return "stub" }

// GenerateContent implements [model.LLM].
func (s *stubLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {}
}

func TestNewAutoAgents(t *testing.T) {
	llm := &stubLLM{}
	genCfg := &genai.GenerateContentConfig{}

	agents, err := NewAutoAgents(llm, genCfg, 3)
	if err != nil {
		t.Fatalf("NewAutoAgents error: %v", err)
	}
	if len(agents) != 3 {
		t.Fatalf("expected 3 agents, got %d", len(agents))
	}

	names := map[string]bool{}
	for _, a := range agents {
		if names[a.Name()] {
			t.Fatalf("duplicate agent name %s", a.Name())
		}
		names[a.Name()] = true
	}
}

func TestNewAutoAgentsZero(t *testing.T) {
	llm := &stubLLM{}
	agents, err := NewAutoAgents(llm, nil, 0)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if len(agents) != 0 {
		t.Fatalf("expected zero agents, got %d", len(agents))
	}
}

func TestNewAutoAgentsRealConfig(t *testing.T) {
	agents, err := NewAutoAgents(&stubLLM{}, nil, 1)
	if err != nil {
		t.Fatalf("NewAutoAgents real config: %v", err)
	}
	if len(agents) != 1 {
		t.Fatalf("expected 1 agent, got %d", len(agents))
	}
}
