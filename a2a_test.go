// Copyright 2025 The tumix Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//\thttp://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/google/go-cmp/cmp"
	adkagent "google.golang.org/adk/agent"
)

type recordingQueue struct {
	events []a2a.Event
}

func (q *recordingQueue) Read(context.Context) (a2a.Event, error) { // unused in tests
	return nil, errors.New("read not supported")
}

func (q *recordingQueue) Write(_ context.Context, event a2a.Event) error {
	q.events = append(q.events, event)
	return nil
}

func (q *recordingQueue) Close() error {
	return nil
}

func TestExtractA2APrompts(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		msg       *a2a.Message
		prompts   []string
		isBatch   bool
		wantError bool
	}{
		"single_text": {
			msg:     &a2a.Message{Parts: a2a.ContentParts{a2a.TextPart{Text: " hello "}}},
			prompts: []string{"hello"},
		},
		"multi_text": {
			msg:     &a2a.Message{Parts: a2a.ContentParts{a2a.TextPart{Text: "hello"}, a2a.TextPart{Text: "world"}}},
			prompts: []string{"hello\nworld"},
		},
		"data_prompt": {
			msg:     &a2a.Message{Parts: a2a.ContentParts{a2a.DataPart{Data: map[string]any{"prompt": "hi"}}}},
			prompts: []string{"hi"},
		},
		"data_prompts_batch": {
			msg:     &a2a.Message{Parts: a2a.ContentParts{a2a.DataPart{Data: map[string]any{"prompts": []string{"a", "b"}}}}},
			prompts: []string{"a", "b"},
			isBatch: true,
		},
		"mixed_text_and_data": {
			msg:       &a2a.Message{Parts: a2a.ContentParts{a2a.TextPart{Text: "hello"}, a2a.DataPart{Data: map[string]any{"prompt": "hi"}}}},
			wantError: true,
		},
		"empty_prompt": {
			msg:       &a2a.Message{Parts: a2a.ContentParts{a2a.TextPart{Text: " "}}},
			wantError: true,
		},
		"invalid_prompt_type": {
			msg:       &a2a.Message{Parts: a2a.ContentParts{a2a.DataPart{Data: map[string]any{"prompts": []any{1}}}}},
			wantError: true,
		},
	}

	for name, tt := range tests {
		name := name
		tt := tt
		t.Run(name, func(t *testing.T) {
			prompts, isBatch, err := extractA2APrompts(tt.msg)
			if tt.wantError {
				if err == nil {
					t.Fatalf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tt.prompts, prompts); diff != "" {
				t.Fatalf("prompts mismatch (-want +got):\n%s", diff)
			}
			if isBatch != tt.isBatch {
				t.Fatalf("isBatch=%v want %v", isBatch, tt.isBatch)
			}
		})
	}
}

func TestBuildA2AResponseParts(t *testing.T) {
	t.Parallel()

	output := runOutput{Text: "ok"}
	batch := []batchOutput{{Prompt: "p1", Output: runOutput{Text: "a"}}}

	tests := map[string]struct {
		output     runOutput
		batch      []batchOutput
		wantKinds  []string
		wantBatch  bool
		wantResult bool
	}{
		"single": {
			output:     output,
			wantKinds:  []string{"data", "text"},
			wantResult: true,
		},
		"batch": {
			batch:     batch,
			wantKinds: []string{"data", "text"},
			wantBatch: true,
		},
	}

	for name, tt := range tests {
		name := name
		tt := tt
		t.Run(name, func(t *testing.T) {
			parts := buildA2AResponseParts(tt.output, tt.batch)
			gotKinds := make([]string, 0, len(parts))
			foundData := false
			for _, part := range parts {
				switch p := part.(type) {
				case a2a.TextPart:
					gotKinds = append(gotKinds, "text")
					if !tt.wantBatch && p.Text == "" {
						t.Fatalf("expected text for single response")
					}
				case a2a.DataPart:
					gotKinds = append(gotKinds, "data")
					if tt.wantBatch {
						if _, ok := p.Data["results"]; !ok {
							t.Fatalf("expected results in data part")
						}
					} else if tt.wantResult {
						if _, ok := p.Data["result"]; !ok {
							t.Fatalf("expected result in data part")
						}
					}
					foundData = true
				default:
					t.Fatalf("unexpected part type %T", part)
				}
			}
			if !foundData {
				t.Fatalf("expected a data part")
			}
			if diff := cmp.Diff(tt.wantKinds, gotKinds); diff != "" {
				t.Fatalf("part kinds mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestResolveA2AURL(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		cfg       config
		wantURL   string
		wantError bool
	}{
		"explicit_url": {
			cfg:     config{A2AURL: "https://example.com/invoke/"},
			wantURL: "https://example.com/invoke",
		},
		"addr_with_host": {
			cfg:     config{A2AAddr: "127.0.0.1:8081"},
			wantURL: "http://127.0.0.1:8081/invoke",
		},
		"addr_with_port_only": {
			cfg:     config{A2AAddr: ":8081"},
			wantURL: "http://localhost:8081/invoke",
		},
		"missing_addr": {
			cfg:       config{},
			wantError: true,
		},
	}

	for name, tt := range tests {
		name := name
		tt := tt
		t.Run(name, func(t *testing.T) {
			got, err := resolveA2AURL(&tt.cfg)
			if tt.wantError {
				if err == nil {
					t.Fatalf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.wantURL {
				t.Fatalf("url=%q want %q", got, tt.wantURL)
			}
		})
	}
}

func TestBuildA2AAgentCard(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		cfg       config
		wantURL   string
		wantError bool
	}{
		"ok": {
			cfg:     config{A2AAddr: "127.0.0.1:8081"},
			wantURL: "http://127.0.0.1:8081/invoke",
		},
		"missing_addr": {
			cfg:       config{},
			wantError: true,
		},
	}

	for name, tt := range tests {
		name := name
		tt := tt
		t.Run(name, func(t *testing.T) {
			card, err := buildA2AAgentCard(&tt.cfg)
			if tt.wantError {
				if err == nil {
					t.Fatalf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if card.URL != tt.wantURL {
				t.Fatalf("card url=%q want %q", card.URL, tt.wantURL)
			}
			if card.Name == "" || card.ProtocolVersion == "" {
				t.Fatalf("expected name and protocol version to be set")
			}
		})
	}
}

func TestValidatePromptLimits(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		cfg       config
		prompt    string
		wantError bool
	}{
		"ok": {
			cfg:    config{MaxPromptChars: 10, MaxPromptTokens: 10},
			prompt: "hello",
		},
		"chars_exceeded": {
			cfg:       config{MaxPromptChars: 4},
			prompt:    "hello",
			wantError: true,
		},
		"tokens_exceeded": {
			cfg:       config{MaxPromptTokens: 2},
			prompt:    "123456789",
			wantError: true,
		},
	}

	for name, tt := range tests {
		name := name
		tt := tt
		t.Run(name, func(t *testing.T) {
			err := validatePromptLimits(&tt.cfg, tt.prompt)
			if tt.wantError {
				if err == nil {
					t.Fatalf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestReadBatchPrompts(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		lines     string
		want      []string
		wantError bool
	}{
		"ok": {
			lines: "a\n\n  b  \n",
			want:  []string{"a", "b"},
		},
		"missing_file": {
			wantError: true,
		},
	}

	for name, tt := range tests {
		name := name
		tt := tt
		t.Run(name, func(t *testing.T) {
			var path string
			if tt.lines != "" {
				tmp := t.TempDir()
				path = filepath.Join(tmp, "prompts.txt")
				if err := os.WriteFile(path, []byte(tt.lines), 0o600); err != nil {
					t.Fatalf("write file: %v", err)
				}
			} else {
				path = filepath.Join(t.TempDir(), "missing.txt")
			}

			got, err := readBatchPrompts(path)
			if tt.wantError {
				if err == nil {
					t.Fatalf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Fatalf("prompts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestRunBatchPrompts(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		prompts []string
	}{
		"ordered_outputs": {
			prompts: []string{"p1", "p2", "p3"},
		},
	}

	for name, tt := range tests {
		name := name
		tt := tt
		t.Run(name, func(t *testing.T) {
			cfg := &config{Concurrency: 2}
			stub := func(ctx context.Context, cfg *config, _ adkagent.Loader) (runOutput, error) {
				return runOutput{Text: cfg.Prompt}, nil
			}
			outputs, err := runBatchPrompts(t.Context(), cfg, nil, tt.prompts, stub)
			if err != nil {
				t.Fatalf("runBatchPrompts error: %v", err)
			}
			want := make([]batchOutput, 0, len(tt.prompts))
			for _, prompt := range tt.prompts {
				want = append(want, batchOutput{Prompt: prompt, Output: runOutput{Text: prompt}})
			}
			if diff := cmp.Diff(want, outputs); diff != "" {
				t.Fatalf("outputs mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestA2AExecutorExecute(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		msg *a2a.Message
	}{
		"success": {
			msg: &a2a.Message{Parts: a2a.ContentParts{a2a.TextPart{Text: "hello"}}},
		},
	}

	for name, tt := range tests {
		name := name
		tt := tt
		t.Run(name, func(t *testing.T) {
			cfg := &config{Concurrency: 1}
			exec := &a2aExecutor{
				baseCfg: cfg,
				runOnce: func(ctx context.Context, cfg *config, _ adkagent.Loader) (runOutput, error) {
					return runOutput{Text: "ok"}, nil
				},
			}
			queue := &recordingQueue{}
			reqCtx := &a2asrv.RequestContext{
				Message:   tt.msg,
				TaskID:    a2a.TaskID("task-1"),
				ContextID: "context-1",
			}
			if err := exec.Execute(t.Context(), reqCtx, queue); err != nil {
				t.Fatalf("execute error: %v", err)
			}
			if len(queue.events) != 3 {
				t.Fatalf("expected 3 events, got %d", len(queue.events))
			}
			submitted, ok := queue.events[0].(*a2a.TaskStatusUpdateEvent)
			if !ok || submitted.Status.State != a2a.TaskStateSubmitted {
				t.Fatalf("expected submitted status event")
			}
			working, ok := queue.events[1].(*a2a.TaskStatusUpdateEvent)
			if !ok || working.Status.State != a2a.TaskStateWorking {
				t.Fatalf("expected working status event")
			}
			completed, ok := queue.events[2].(*a2a.TaskStatusUpdateEvent)
			if !ok || completed.Status.State != a2a.TaskStateCompleted || !completed.Final {
				t.Fatalf("expected completed final status event")
			}
			if completed.Status.Message == nil || len(completed.Status.Message.Parts) == 0 {
				t.Fatalf("expected status message parts")
			}
		})
	}
}

func TestA2AExecutorCancel(t *testing.T) {
	t.Parallel()

	tests := map[string]struct{}{
		"cancel": {},
	}

	for name := range tests {
		name := name
		t.Run(name, func(t *testing.T) {
			exec := &a2aExecutor{}
			queue := &recordingQueue{}
			reqCtx := &a2asrv.RequestContext{TaskID: a2a.TaskID("task-1"), ContextID: "context-1"}
			if err := exec.Cancel(t.Context(), reqCtx, queue); err != nil {
				t.Fatalf("cancel error: %v", err)
			}
			if len(queue.events) != 1 {
				t.Fatalf("expected 1 event, got %d", len(queue.events))
			}
			event, ok := queue.events[0].(*a2a.TaskStatusUpdateEvent)
			if !ok || event.Status.State != a2a.TaskStateCanceled || !event.Final {
				t.Fatalf("expected canceled final status event")
			}
		})
	}
}
