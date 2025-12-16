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
	"errors"
	"iter"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	adkagent "google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"

	"github.com/zchee/tumix/agent/agenttest"
)

func TestSetStateAndGetState(t *testing.T) {
	t.Parallel()

	sentinelErr := errors.New("sentinel")

	tests := map[string]struct {
		state          session.State
		op             func(ctx adkagent.InvocationContext) (any, error)
		wantValue      any
		wantErrMsg     string
		wantErr        error
		wantMissingKey string
	}{
		"success: setState then getState": {
			state: agenttest.NewInMemoryState(map[string]any{}),
			op: func(ctx adkagent.InvocationContext) (any, error) {
				if err := setState(ctx, "k", "v"); err != nil {
					return nil, err
				}
				return getState(ctx, "k")
			},
			wantValue: "v",
		},
		"error: getState missing key wraps ErrStateKeyNotExist": {
			state: agenttest.NewInMemoryState(map[string]any{}),
			op: func(ctx adkagent.InvocationContext) (any, error) {
				return getState(ctx, "missing")
			},
			wantErrMsg:     "state missing not found",
			wantErr:        session.ErrStateKeyNotExist,
			wantMissingKey: "missing",
		},
		"error: getState wraps non-ErrStateKeyNotExist": {
			state: func() *agenttest.InMemoryState {
				s := agenttest.NewInMemoryState(map[string]any{})
				s.GetErr = sentinelErr
				return s
			}(),
			op: func(ctx adkagent.InvocationContext) (any, error) {
				return getState(ctx, "k")
			},
			wantErrMsg: "get state k",
			wantErr:    sentinelErr,
		},
		"error: setState wraps Set failure": {
			state: func() *agenttest.InMemoryState {
				s := agenttest.NewInMemoryState(map[string]any{})
				s.SetErr = sentinelErr
				return s
			}(),
			op: func(ctx adkagent.InvocationContext) (any, error) {
				return nil, setState(ctx, "k", "v")
			},
			wantErrMsg: "set state k",
			wantErr:    sentinelErr,
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			sess := agenttest.NewInMemorySession("s", "app", "u", tt.state, &agenttest.InMemoryEvents{}, time.Time{})
			ctx := agenttest.NewSessionInvocationContext(t.Context(), sess)

			got, err := tt.op(ctx)
			if tt.wantErrMsg != "" {
				if err == nil {
					t.Fatalf("op err = nil, want substring %q", tt.wantErrMsg)
				}
				if !strings.Contains(err.Error(), tt.wantErrMsg) {
					t.Fatalf("op err = %q, want substring %q", err.Error(), tt.wantErrMsg)
				}
				if tt.wantErr != nil && !errors.Is(err, tt.wantErr) {
					t.Fatalf("errors.Is(err, %v) = false, err=%v", tt.wantErr, err)
				}
				if tt.wantMissingKey != "" {
					_, getErr := tt.state.Get(tt.wantMissingKey)
					if !errors.Is(getErr, session.ErrStateKeyNotExist) {
						t.Fatalf("state unexpectedly contains %q after failing op: %v", tt.wantMissingKey, getErr)
					}
				}
				return
			}

			if err != nil {
				t.Fatalf("op err = %v, want nil", err)
			}
			if diff := cmp.Diff(tt.wantValue, got); diff != "" {
				t.Fatalf("op value mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestNewFinalizeToolRun(t *testing.T) {
	t.Parallel()

	type runnableTool interface {
		Run(ctx tool.Context, args any) (map[string]any, error)
	}

	tests := map[string]struct {
		args         map[string]any
		wantResult   map[string]any
		wantAnswer   string
		wantConf     float64
		wantEscalate bool
		wantErrMsg   string
	}{
		"success: stores trimmed answer and confidence and escalates": {
			args: map[string]any{
				"answer":     "  final  ",
				"confidence": 0.9,
				"stop":       true,
			},
			wantResult: map[string]any{
				"stored": true,
				"stop":   true,
			},
			wantAnswer:   "final",
			wantConf:     0.9,
			wantEscalate: true,
		},
		"success: stop=false does not escalate": {
			args: map[string]any{
				"answer":     "final",
				"confidence": 0.9,
				"stop":       false,
			},
			wantResult: map[string]any{
				"stored": true,
			},
			wantAnswer:   "final",
			wantConf:     0.9,
			wantEscalate: false,
		},
		"error: answer is required": {
			args: map[string]any{
				"answer":     "   ",
				"confidence": 0.9,
				"stop":       true,
			},
			wantErrMsg: "answer is required",
		},
		"error: confidence out of range": {
			args: map[string]any{
				"answer":     "final",
				"confidence": 1.1,
				"stop":       true,
			},
			wantErrMsg: "confidence must be between 0 and 1",
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			toolDef, err := newFinalizeTool()
			if err != nil {
				t.Fatalf("newFinalizeTool() err = %v, want nil", err)
			}

			runner, ok := toolDef.(runnableTool)
			if !ok {
				t.Fatalf("newFinalizeTool() returned %T, missing Run(tool.Context, any) method", toolDef)
			}

			state := agenttest.NewInMemoryState(map[string]any{})
			actions := &session.EventActions{
				StateDelta: make(map[string]any),
			}
			ctx := agenttest.NewToolContext(t.Context(), state, actions, "call-1", nil)

			result, runErr := runner.Run(ctx, tt.args)
			if tt.wantErrMsg != "" {
				if runErr == nil {
					t.Fatalf("Run err = nil, want substring %q", tt.wantErrMsg)
				}
				if !strings.Contains(runErr.Error(), tt.wantErrMsg) {
					t.Fatalf("Run err = %q, want substring %q", runErr.Error(), tt.wantErrMsg)
				}
				if _, err := state.Get(stateKeyAnswer); !errors.Is(err, session.ErrStateKeyNotExist) {
					t.Fatalf("unexpected answer state set on error: %v", err)
				}
				if _, err := state.Get(stateKeyConfidence); !errors.Is(err, session.ErrStateKeyNotExist) {
					t.Fatalf("unexpected confidence state set on error: %v", err)
				}
				return
			}

			if runErr != nil {
				t.Fatalf("Run err = %v, want nil", runErr)
			}
			if diff := cmp.Diff(tt.wantResult, result); diff != "" {
				t.Fatalf("Run result mismatch (-want +got):\n%s", diff)
			}

			gotAnswer, err := state.Get(stateKeyAnswer)
			if err != nil {
				t.Fatalf("state.Get(%q) err = %v, want nil", stateKeyAnswer, err)
			}
			answer, ok := gotAnswer.(string)
			if !ok {
				t.Fatalf("state.Get(%q) = %T, want string", stateKeyAnswer, gotAnswer)
			}
			if diff := cmp.Diff(tt.wantAnswer, answer); diff != "" {
				t.Fatalf("state answer mismatch (-want +got):\n%s", diff)
			}

			gotConf, err := state.Get(stateKeyConfidence)
			if err != nil {
				t.Fatalf("state.Get(%q) err = %v, want nil", stateKeyConfidence, err)
			}
			confidence, ok := gotConf.(float64)
			if !ok {
				t.Fatalf("state.Get(%q) = %T, want float64", stateKeyConfidence, gotConf)
			}
			if diff := cmp.Diff(tt.wantConf, confidence, cmpopts.EquateApprox(0, 1e-12)); diff != "" {
				t.Fatalf("state confidence mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.wantEscalate, ctx.Actions().Escalate); diff != "" {
				t.Fatalf("actions.Escalate mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAgentConstructors(t *testing.T) {
	t.Parallel()

	llm := &stubLLM{}
	genCfg := &genai.GenerateContentConfig{}

	tests := map[string]struct {
		build    func() (adkagent.Agent, error)
		wantName string
	}{
		"NewBaseAgent": {
			build:    func() (adkagent.Agent, error) { return NewBaseAgent(llm, genCfg) },
			wantName: "base",
		},
		"NewCoTAgent": {
			build:    func() (adkagent.Agent, error) { return NewCoTAgent(llm, genCfg) },
			wantName: "cot",
		},
		"NewCoTCodeAgent": {
			build:    func() (adkagent.Agent, error) { return NewCoTCodeAgent(llm, genCfg) },
			wantName: "cot-code",
		},
		"NewSearchAgent": {
			build:    func() (adkagent.Agent, error) { return NewSearchAgent(llm, genCfg) },
			wantName: "search",
		},
		"NewCodeAgent": {
			build:    func() (adkagent.Agent, error) { return NewCodeAgent(llm, genCfg) },
			wantName: "code",
		},
		"NewCodePlusAgent": {
			build:    func() (adkagent.Agent, error) { return NewCodePlusAgent(llm, genCfg) },
			wantName: "code-plus",
		},
		"NewDualToolGSAgent": {
			build:    func() (adkagent.Agent, error) { return NewDualToolGSAgent(llm, genCfg) },
			wantName: "dual-tool-google-search",
		},
		"NewDualToolLLMAgent": {
			build:    func() (adkagent.Agent, error) { return NewDualToolLLMAgent(llm, genCfg) },
			wantName: "dual-tool-llm-search",
		},
		"NewDualToolComAgent": {
			build:    func() (adkagent.Agent, error) { return NewDualToolComAgent(llm, genCfg) },
			wantName: "dual-tool-combine-search",
		},
		"NewGuidedGSAgent": {
			build:    func() (adkagent.Agent, error) { return NewGuidedGSAgent(llm, genCfg) },
			wantName: "guided-google-search",
		},
		"NewGuidedLLMAgent": {
			build:    func() (adkagent.Agent, error) { return NewGuidedLLMAgent(llm, genCfg) },
			wantName: "guided-llm-search",
		},
		"NewGuidedComAgent": {
			build:    func() (adkagent.Agent, error) { return NewGuidedComAgent(llm, genCfg) },
			wantName: "guided-combine-search",
		},
		"NewGuidedPlusGSAgent": {
			build:    func() (adkagent.Agent, error) { return NewGuidedPlusGSAgent(llm, genCfg) },
			wantName: "guided-plus-google-search",
		},
		"NewGuidedPlusLLMAgent": {
			build:    func() (adkagent.Agent, error) { return NewGuidedPlusLLMAgent(llm, genCfg) },
			wantName: "guided-plus-llm-search",
		},
		"NewGuidedPlusComAgent": {
			build:    func() (adkagent.Agent, error) { return NewGuidedPlusComAgent(llm, genCfg) },
			wantName: "guided-plus-combine-Search",
		},
		"NewJudgeAgent": {
			build:    func() (adkagent.Agent, error) { return NewJudgeAgent(llm, genCfg) },
			wantName: "LLM-as-Judge",
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			a, err := tt.build()
			if err != nil {
				t.Fatalf("%s() err = %v, want nil", name, err)
			}
			if a == nil {
				t.Fatalf("%s() agent = nil", name)
			}
			if diff := cmp.Diff(tt.wantName, a.Name()); diff != "" {
				t.Fatalf("%s() agent.Name mismatch (-want +got):\n%s", name, diff)
			}
			if a.Description() == "" {
				t.Fatalf("%s() agent.Description empty", name)
			}
		})
	}
}

func TestNewRefinementAgent(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		subAgents []adkagent.Agent
		wantName  string
		wantSubs  []string
	}{
		"success: wraps subagents in parallel workflow": {
			subAgents: []adkagent.Agent{
				mustAgent(adkagent.New(adkagent.Config{
					Name:        "a",
					Description: "a",
					Run: func(adkagent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(func(*session.Event, error) bool) {}
					},
				})),
				mustAgent(adkagent.New(adkagent.Config{
					Name:        "b",
					Description: "b",
					Run: func(adkagent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(func(*session.Event, error) bool) {}
					},
				})),
			},
			wantName: "Refinement",
			wantSubs: []string{"candidates"},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			a, err := NewRefinementAgent(tt.subAgents...)
			if err != nil {
				t.Fatalf("NewRefinementAgent() err = %v, want nil", err)
			}
			if diff := cmp.Diff(tt.wantName, a.Name()); diff != "" {
				t.Fatalf("agent.Name mismatch (-want +got):\n%s", diff)
			}

			gotSubs := make([]string, 0, len(a.SubAgents()))
			for _, sub := range a.SubAgents() {
				gotSubs = append(gotSubs, sub.Name())
			}
			if diff := cmp.Diff(tt.wantSubs, gotSubs); diff != "" {
				t.Fatalf("agent.SubAgents mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestNewRoundAgent(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		subAgents []adkagent.Agent
		wantName  string
		wantSubs  []string
	}{
		"success: sequential workflow includes subagents": {
			subAgents: []adkagent.Agent{
				stubCandidate("A"),
				stubCandidate("B"),
			},
			wantName: "round",
			wantSubs: []string{"A", "B"},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			a, err := NewRoundAgent(tt.subAgents...)
			if err != nil {
				t.Fatalf("NewRoundAgent() err = %v, want nil", err)
			}

			if diff := cmp.Diff(tt.wantName, a.Name()); diff != "" {
				t.Fatalf("agent.Name mismatch (-want +got):\n%s", diff)
			}

			gotSubs := make([]string, 0, len(a.SubAgents()))
			for _, sub := range a.SubAgents() {
				gotSubs = append(gotSubs, sub.Name())
			}
			if diff := cmp.Diff(tt.wantSubs, gotSubs); diff != "" {
				t.Fatalf("agent.SubAgents mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestNewTumixAgentWrappers(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		build      func() (adkagent.Loader, error)
		wantAgents []string
	}{
		"NewTumixAgent sets root agent name": {
			build: func() (adkagent.Loader, error) {
				return NewTumixAgent([]adkagent.Agent{stubCandidate("A")}, noOpJudge())
			},
			wantAgents: []string{"tumix", "candidates", "judge"},
		},
		"NewTumixAgentWithMaxRounds sets root agent name": {
			build: func() (adkagent.Loader, error) {
				return NewTumixAgentWithMaxRounds([]adkagent.Agent{stubCandidate("A")}, noOpJudge(), 1)
			},
			wantAgents: []string{"tumix", "candidates", "judge"},
		},
		"error: requires at least one candidate": {
			build: func() (adkagent.Loader, error) {
				return NewTumixAgentWithConfig(TumixConfig{
					Candidates: nil,
					Judge:      noOpJudge(),
				})
			},
		},
		"error: requires judge": {
			build: func() (adkagent.Loader, error) {
				return NewTumixAgentWithConfig(TumixConfig{
					Candidates: []adkagent.Agent{stubCandidate("A")},
					Judge:      nil,
				})
			},
		},
		"success: MinRounds clamps to MaxRounds": {
			build: func() (adkagent.Loader, error) {
				return NewTumixAgentWithConfig(TumixConfig{
					Candidates: []adkagent.Agent{stubCandidate("A")},
					Judge:      noOpJudge(),
					MaxRounds:  1,
					MinRounds:  10,
				})
			},
			wantAgents: []string{"tumix", "candidates", "judge"},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			loader, err := tt.build()
			if tt.wantAgents == nil {
				if err == nil {
					t.Fatalf("build err = nil, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("build err = %v, want nil", err)
			}
			if loader == nil {
				t.Fatalf("loader = nil")
			}

			root := loader.RootAgent()
			if root == nil {
				t.Fatalf("loader.RootAgent() = nil")
			}

			gotAgents := []string{root.Name()}
			for _, sub := range root.SubAgents() {
				gotAgents = append(gotAgents, sub.Name())
			}
			if diff := cmp.Diff(tt.wantAgents, gotAgents); diff != "" {
				t.Fatalf("agent tree mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestHelpersEdgeCases(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		content         *genai.Content
		wantFirstText   string
		wantContentText string
	}{
		"nil content": {
			content:         nil,
			wantFirstText:   "",
			wantContentText: "",
		},
		"empty parts": {
			content: &genai.Content{
				Parts: []*genai.Part{},
			},
			wantFirstText:   "",
			wantContentText: "",
		},
		"skips nil part and returns first text": {
			content: &genai.Content{
				Parts: []*genai.Part{
					nil,
					{
						Text: " first ",
					},
					{
						Text: "second",
					},
				},
			},
			wantFirstText:   " first ",
			wantContentText: "first",
		},
		"ignores blank first text part": {
			content: &genai.Content{
				Parts: []*genai.Part{
					{
						Text: "",
					},
					{
						Text: "ok",
					},
				},
			},
			wantFirstText:   "ok",
			wantContentText: "ok",
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			if diff := cmp.Diff(tt.wantFirstText, firstTextFromContent(tt.content)); diff != "" {
				t.Fatalf("firstTextFromContent mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantContentText, firstContentText(tt.content)); diff != "" {
				t.Fatalf("firstContentText mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMajorityVoteTieBreak(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		answers        []candidateAnswer
		wantAnswer     string
		wantConfidence float64
	}{
		"empty: returns empty answer and zero confidence": {
			answers:        nil,
			wantAnswer:     "",
			wantConfidence: 0,
		},
		"tie: picks lexicographically smallest normalized answer": {
			answers: []candidateAnswer{
				{
					Agent: "a",
					Text:  "z",
				},
				{
					Agent: "b",
					Text:  "a",
				},
			},
			wantAnswer:     "a",
			wantConfidence: 0.5,
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			gotAnswer, gotConfidence := majorityVote(tt.answers)
			if diff := cmp.Diff(tt.wantAnswer, gotAnswer); diff != "" {
				t.Fatalf("majorityVote answer mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantConfidence, gotConfidence,
				cmpopts.EquateApprox(0, 1e-12),
			); diff != "" {
				t.Fatalf("majorityVote confidence mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestComputeStatsEmptyInputs(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		answers        []candidateAnswer
		candidateCount int
		want           roundStats
	}{
		"empty answers": {
			answers:        nil,
			candidateCount: 3,
			want:           roundStats{},
		},
		"invalid candidate count": {
			answers: []candidateAnswer{
				{
					Agent: "a",
					Text:  "x",
				},
			},
			candidateCount: 0,
			want:           roundStats{},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			got := computeStats(tt.answers, tt.candidateCount)
			if diff := cmp.Diff(tt.want, got,
				cmp.AllowUnexported(roundStats{}),
				cmpopts.EquateApprox(0, 1e-12),
			); diff != "" {
				t.Fatalf("computeStats mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestRunCandidates(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		candidateAgent adkagent.Agent
		yield          func(*session.Event, error) bool
		wantAnswers    []candidateAnswer
		wantStop       bool
	}{
		"stop: yield aborts iteration": {
			candidateAgent: mustAgent(adkagent.New(adkagent.Config{
				Name:        "candidates",
				Description: "candidate agent",
				Run: func(ctx adkagent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx.InvocationID())
						ev.Author = "A"
						ev.Content = genai.NewContentFromText("foo", genai.RoleModel)
						yield(ev, nil)
					}
				},
			})),
			yield:       func(*session.Event, error) bool { return false },
			wantAnswers: []candidateAnswer{},
			wantStop:    true,
		},
		"success: filters invalid events and trims text": {
			candidateAgent: mustAgent(adkagent.New(adkagent.Config{
				Name:        "candidates",
				Description: "candidate agent",
				Run: func(ctx adkagent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						// event == nil
						if !yield(nil, nil) {
							return
						}

						evNoContent := session.NewEvent(ctx.InvocationID())
						evNoContent.Author = "A"
						// event.Content == nil
						if !yield(evNoContent, nil) {
							return
						}

						evBlank := session.NewEvent(ctx.InvocationID())
						evBlank.Author = "A"
						evBlank.Content = &genai.Content{Parts: []*genai.Part{{Text: ""}}}
						// empty text
						if !yield(evBlank, nil) {
							return
						}

						evErr := session.NewEvent(ctx.InvocationID())
						evErr.Author = "B"
						evErr.Content = genai.NewContentFromText("ignored", genai.RoleModel)
						// err != nil
						if !yield(evErr, errors.New("candidate error")) {
							return
						}

						evOK := session.NewEvent(ctx.InvocationID())
						evOK.Author = "C"
						evOK.Content = genai.NewContentFromText("  ok  ", genai.RoleModel)
						yield(evOK, nil)
					}
				},
			})),
			yield: func(*session.Event, error) bool { return true },
			wantAnswers: []candidateAnswer{
				{Agent: "C", Text: "ok"},
			},
			wantStop: false,
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			orchestrator := &tumixOrchestrator{
				candidateAgent: tt.candidateAgent,
			}
			sess := agenttest.NewInMemorySession("s", "app", "u", agenttest.NewInMemoryState(map[string]any{}), &agenttest.InMemoryEvents{}, time.Time{})
			ctx := agenttest.NewSessionInvocationContext(t.Context(), sess)

			answers, stop := orchestrator.runCandidates(ctx, tt.yield)
			if diff := cmp.Diff(tt.wantStop, stop); diff != "" {
				t.Fatalf("stop mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantAnswers, answers, cmp.AllowUnexported(candidateAnswer{})); diff != "" {
				t.Fatalf("answers mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestRunJudge(t *testing.T) {
	t.Parallel()

	sentinelErr := errors.New("sentinel")

	tests := map[string]struct {
		state        session.State
		judge        adkagent.Agent
		yield        func(*session.Event, error) bool
		wantStop     bool
		wantJudgeAns string
		wantYieldErr string
	}{
		"stop: yield aborts iteration": {
			state:    agenttest.NewInMemoryState(map[string]any{}),
			judge:    noOpJudge(),
			yield:    func(*session.Event, error) bool { return false },
			wantStop: true,
		},
		"stop: escalated event stores normalized judge answer": {
			state: agenttest.NewInMemoryState(map[string]any{}),
			judge: mustAgent(adkagent.New(adkagent.Config{
				Name:        "judge",
				Description: "judge agent",
				Run: func(ctx adkagent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx.InvocationID())
						ev.Author = "judge"
						ev.Actions.Escalate = true
						ev.Content = genai.NewContentFromText("<<< ok >>>", genai.RoleModel)
						yield(ev, nil)
					}
				},
			})),
			yield:        func(*session.Event, error) bool { return true },
			wantStop:     true,
			wantJudgeAns: "ok",
		},
		"stop: escalated event without text does not set judge answer": {
			state: agenttest.NewInMemoryState(map[string]any{}),
			judge: mustAgent(adkagent.New(adkagent.Config{
				Name:        "judge",
				Description: "judge agent",
				Run: func(ctx adkagent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx.InvocationID())
						ev.Author = "judge"
						ev.Actions.Escalate = true
						yield(ev, nil)
					}
				},
			})),
			yield:    func(*session.Event, error) bool { return true },
			wantStop: true,
		},
		"stop: setState failure yields error": {
			state: func() *agenttest.InMemoryState {
				s := agenttest.NewInMemoryState(map[string]any{})
				s.SetErr = sentinelErr
				return s
			}(),
			judge: mustAgent(adkagent.New(adkagent.Config{
				Name:        "judge",
				Description: "judge agent",
				Run: func(ctx adkagent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx.InvocationID())
						ev.Author = "judge"
						ev.Actions.Escalate = true
						ev.Content = genai.NewContentFromText("ok", genai.RoleModel)
						yield(ev, nil)
					}
				},
			})),
			yield:        func(*session.Event, error) bool { return true },
			wantStop:     true,
			wantYieldErr: "set state judge_recommended_answer",
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			orchestrator := &tumixOrchestrator{judge: tt.judge}
			sess := agenttest.NewInMemorySession("s", "app", "u", tt.state, &agenttest.InMemoryEvents{}, time.Time{})
			ctx := agenttest.NewSessionInvocationContext(t.Context(), sess)

			var yieldedErr error
			yield := func(event *session.Event, err error) bool {
				if tt.yield != nil && !tt.yield(event, err) {
					return false
				}
				if err != nil {
					yieldedErr = err
				}
				return true
			}

			stop := orchestrator.runJudge(ctx, yield)
			if diff := cmp.Diff(tt.wantStop, stop); diff != "" {
				t.Fatalf("stop mismatch (-want +got):\n%s", diff)
			}

			if tt.wantYieldErr != "" {
				if yieldedErr == nil {
					t.Fatalf("yieldedErr = nil, want substring %q", tt.wantYieldErr)
				}
				if !strings.Contains(yieldedErr.Error(), tt.wantYieldErr) {
					t.Fatalf("yieldedErr = %q, want substring %q", yieldedErr.Error(), tt.wantYieldErr)
				}
				return
			}

			if tt.wantJudgeAns == "" {
				if _, err := tt.state.Get(stateKeyJudgeAnswer); !errors.Is(err, session.ErrStateKeyNotExist) {
					t.Fatalf("unexpected judge answer state set: %v", err)
				}
				return
			}

			gotJudgeAns, err := tt.state.Get(stateKeyJudgeAnswer)
			if err != nil {
				t.Fatalf("state.Get(%q) err = %v, want nil", stateKeyJudgeAnswer, err)
			}
			judgeAnswer, ok := gotJudgeAns.(string)
			if !ok {
				t.Fatalf("state.Get(%q) = %T, want string", stateKeyJudgeAnswer, gotJudgeAns)
			}
			if diff := cmp.Diff(tt.wantJudgeAns, judgeAnswer); diff != "" {
				t.Fatalf("judge answer mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestEmitFinalFromState(t *testing.T) {
	t.Parallel()

	sentinelErr := errors.New("sentinel")

	tests := map[string]struct {
		state      session.State
		wantText   string
		wantDelta  map[string]any
		wantYield  bool
		wantErrMsg string
	}{
		"success: emits final answer event with state deltas": {
			state: agenttest.NewInMemoryState(map[string]any{
				stateKeyAnswer:     "foo",
				stateKeyConfidence: float64(0.7),
				stateKeyJoined:     "- a: foo",
			}),
			wantText: "Final answer (conf 0.7): foo",
			wantDelta: map[string]any{
				stateKeyAnswer:     "foo",
				stateKeyConfidence: float64(0.7),
				stateKeyJoined:     "- a: foo",
			},
			wantYield: true,
		},
		"success: falls back to judge answer when answer missing": {
			state: agenttest.NewInMemoryState(map[string]any{
				stateKeyJudgeAnswer: "bar",
				stateKeyConfidence:  float64(0.95),
			}),
			wantText: "Final answer (conf 0.95): bar",
			wantDelta: map[string]any{
				stateKeyAnswer:     "bar",
				stateKeyConfidence: float64(0.95),
			},
			wantYield: true,
		},
		"error: getState failure yields error": {
			state: func() *agenttest.InMemoryState {
				s := agenttest.NewInMemoryState(map[string]any{})
				s.GetErr = sentinelErr
				return s
			}(),
			wantYield:  false,
			wantErrMsg: "get state tumix_final_answer",
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			orchestrator := &tumixOrchestrator{}
			sess := agenttest.NewInMemorySession("s", "app", "u", tt.state, &agenttest.InMemoryEvents{}, time.Time{})
			ctx := agenttest.NewSessionInvocationContext(t.Context(), sess)

			var gotEvent *session.Event
			var gotErr error
			yield := func(event *session.Event, err error) bool {
				if err != nil {
					gotErr = err
				}
				if event != nil {
					gotEvent = event
				}
				return true
			}

			orchestrator.emitFinalFromState(ctx, yield)

			if tt.wantErrMsg != "" {
				if gotErr == nil {
					t.Fatalf("gotErr = nil, want substring %q", tt.wantErrMsg)
				}
				if !strings.Contains(gotErr.Error(), tt.wantErrMsg) {
					t.Fatalf("gotErr = %q, want substring %q", gotErr.Error(), tt.wantErrMsg)
				}
				return
			}

			if tt.wantYield && gotEvent == nil {
				t.Fatalf("gotEvent = nil, want event")
			}
			if !tt.wantYield && gotEvent != nil {
				t.Fatalf("gotEvent != nil, want none")
			}
			if gotEvent == nil {
				return
			}

			gotText := firstTextFromContent(gotEvent.Content)
			if diff := cmp.Diff(tt.wantText, gotText); diff != "" {
				t.Fatalf("event text mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff("tumix", gotEvent.Author); diff != "" {
				t.Fatalf("event author mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(tt.wantDelta, gotEvent.Actions.StateDelta); diff != "" {
				t.Fatalf("state delta mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestOrchestratorRunBranches(t *testing.T) {
	t.Parallel()

	sentinelErr := errors.New("sentinel")

	emptyCandidateAgent := mustAgent(adkagent.New(adkagent.Config{
		Name:        "candidates",
		Description: "candidate agent",
		Run: func(adkagent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	}))

	tests := map[string]struct {
		state        session.State
		orchestrator *tumixOrchestrator
		wantErrMsg   string
		wantFinal    string
	}{
		"error: setState(question) failure yields error": {
			state: func() *agenttest.InMemoryState {
				s := agenttest.NewInMemoryState(map[string]any{})
				s.SetErr = sentinelErr
				return s
			}(),
			orchestrator: &tumixOrchestrator{
				candidateAgent: emptyCandidateAgent,
				judge:          noOpJudge(),
				maxRounds:      1,
				minRounds:      1,
			},
			wantErrMsg: "set state question",
		},
		"success: no candidates triggers judge and emits final": {
			state: agenttest.NewInMemoryState(map[string]any{}),
			orchestrator: &tumixOrchestrator{
				candidateAgent: emptyCandidateAgent,
				judge:          stubJudge("done"),
				maxRounds:      1,
				minRounds:      1,
			},
			wantFinal: "Final answer (conf 0.95): done",
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			sess := agenttest.NewInMemorySession("s", "app", "u", tt.state, &agenttest.InMemoryEvents{}, time.Time{})
			ctx := agenttest.NewSessionInvocationContext(t.Context(), sess)

			var gotErr error
			var finalText string
			for event, err := range tt.orchestrator.run(ctx) {
				if err != nil {
					gotErr = err
					continue
				}
				if event == nil || event.Author != "tumix" {
					continue
				}
				finalText = firstTextFromContent(event.Content)
			}

			if tt.wantErrMsg != "" {
				if gotErr == nil {
					t.Fatalf("gotErr = nil, want substring %q", tt.wantErrMsg)
				}
				if !strings.Contains(gotErr.Error(), tt.wantErrMsg) {
					t.Fatalf("gotErr = %q, want substring %q", gotErr.Error(), tt.wantErrMsg)
				}
				return
			}

			if gotErr != nil {
				t.Fatalf("gotErr = %v, want nil", gotErr)
			}
			if diff := cmp.Diff(tt.wantFinal, finalText); diff != "" {
				t.Fatalf("final text mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
