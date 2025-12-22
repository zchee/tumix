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

package main

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/google/go-cmp/cmp"
	adkagent "google.golang.org/adk/agent"
)

type recordingQueue struct {
	mu     sync.Mutex
	events []a2a.Event
}

func (q *recordingQueue) Write(_ context.Context, event a2a.Event) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.events = append(q.events, event)
	return nil
}

func (q *recordingQueue) Read(_ context.Context) (a2a.Event, error) {
	return nil, errors.New("not implemented")
}

func (q *recordingQueue) Close() error {
	return nil
}

func (q *recordingQueue) snapshot() []a2a.Event {
	q.mu.Lock()
	defer q.mu.Unlock()
	out := make([]a2a.Event, len(q.events))
	copy(out, q.events)
	return out
}

func TestA2AURLFromAddr(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		addr    string
		want    string
		wantErr bool
	}{
		"error: empty": {
			addr:    "",
			wantErr: true,
		},
		"host-port": {
			addr: "127.0.0.1:9090",
			want: "http://127.0.0.1:9090/invoke",
		},
		"port-only": {
			addr: ":8080",
			want: "http://localhost:8080/invoke",
		},
		"http-prefix": {
			addr: "http://example.com/a2a",
			want: "http://example.com/a2a/invoke",
		},
		"https-prefix": {
			addr: "https://example.com",
			want: "https://example.com/invoke",
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := a2aURLFromAddr(tc.addr)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error for addr %q", tc.addr)
				}
				return
			}
			if err != nil {
				t.Fatalf("a2aURLFromAddr error = %v", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Fatalf("url mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestExtractA2APrompts(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		msg        *a2a.Message
		want       []string
		wantBatch  bool
		wantErr    bool
		wantErrMsg string
	}{
		"single: text part": {
			msg:  &a2a.Message{Role: a2a.MessageRoleUser, Parts: a2a.ContentParts{a2a.TextPart{Text: "hello"}}},
			want: []string{"hello"},
		},
		"single: multiple text parts": {
			msg: &a2a.Message{Role: a2a.MessageRoleUser, Parts: a2a.ContentParts{
				a2a.TextPart{Text: "hello"},
				a2a.TextPart{Text: "world"},
			}},
			want: []string{"hello\nworld"},
		},
		"single: data prompt": {
			msg: &a2a.Message{Role: a2a.MessageRoleUser, Parts: a2a.ContentParts{
				a2a.DataPart{Data: map[string]any{"prompt": "from-data"}},
			}},
			want: []string{"from-data"},
		},
		"batch: data prompts": {
			msg: &a2a.Message{Role: a2a.MessageRoleUser, Parts: a2a.ContentParts{
				a2a.DataPart{Data: map[string]any{"prompts": []string{"p1", "p2"}}},
			}},
			want:      []string{"p1", "p2"},
			wantBatch: true,
		},
		"batch: metadata prompts": {
			msg:       &a2a.Message{Role: a2a.MessageRoleUser, Metadata: map[string]any{"prompts": []string{"m1", "m2"}}},
			want:      []string{"m1", "m2"},
			wantBatch: true,
		},
		"error: empty": {
			msg:     &a2a.Message{Role: a2a.MessageRoleUser},
			wantErr: true,
		},
		"error: non-string batch": {
			msg: &a2a.Message{Role: a2a.MessageRoleUser, Parts: a2a.ContentParts{
				a2a.DataPart{Data: map[string]any{"prompts": []any{1}}},
			}},
			wantErr:    true,
			wantErrMsg: "batch prompt must be string",
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			got, batch, err := extractA2APrompts(tc.msg)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error")
				}
				if tc.wantErrMsg != "" && !strings.Contains(err.Error(), tc.wantErrMsg) {
					t.Fatalf("error %q does not contain %q", err.Error(), tc.wantErrMsg)
				}
				return
			}
			if err != nil {
				t.Fatalf("extractA2APrompts error = %v", err)
			}
			if batch != tc.wantBatch {
				t.Fatalf("batch = %v, want %v", batch, tc.wantBatch)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Fatalf("prompts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestBuildA2AResponseParts(t *testing.T) {
	t.Parallel()

	cfg := config{SessionID: "session-1", ModelName: "model", MaxRounds: 2}
	single := buildRunOutput(&cfg, "agent", "answer", 3, 4)
	batch := []batchOutput{
		{Prompt: "p1", Output: buildRunOutput(&cfg, "agent", "a1", 1, 1)},
		{Prompt: "p2", Output: buildRunOutput(&cfg, "agent", "a2", 1, 1)},
	}

	tests := map[string]struct {
		meta       map[string]any
		output     *runOutput
		batch      []batchOutput
		wantText   []string
		wantResult *runOutput
		wantBatch  int
		wantErr    bool
	}{
		"single: default modes": {
			output:     &single,
			wantText:   []string{"answer"},
			wantResult: &single,
		},
		"single: text only": {
			meta:     map[string]any{"accepted_output_modes": []string{"text/plain"}},
			output:   &single,
			wantText: []string{"answer"},
		},
		"single: json only": {
			meta:       map[string]any{"accepted_output_modes": []string{"application/json"}},
			output:     &single,
			wantResult: &single,
		},
		"batch: json only": {
			meta:      map[string]any{"accepted_output_modes": []string{"application/json"}},
			batch:     batch,
			wantBatch: 2,
		},
		"error: unsupported mode": {
			meta:    map[string]any{"accepted_output_modes": []string{"image/png"}},
			output:  &single,
			wantErr: true,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			parts, err := buildA2AResponseParts(tc.meta, tc.output, tc.batch)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("buildA2AResponseParts error = %v", err)
			}

			var (
				gotTexts []string
				gotRun   *runOutput
				gotBatch []batchOutput
			)
			for _, part := range parts {
				switch p := part.(type) {
				case a2a.TextPart:
					gotTexts = append(gotTexts, p.Text)
				case a2a.DataPart:
					if value, ok := p.Data["result"]; ok {
						result, ok := value.(runOutput)
						if !ok {
							t.Fatalf("result type = %T", value)
						}
						gotRun = &result
					}
					if value, ok := p.Data["results"]; ok {
						results, ok := value.([]batchOutput)
						if !ok {
							t.Fatalf("results type = %T", value)
						}
						gotBatch = results
					}
				}
			}

			if diff := cmp.Diff(tc.wantText, gotTexts); diff != "" {
				t.Fatalf("text parts mismatch (-want +got):\n%s", diff)
			}
			if tc.wantResult != nil {
				if gotRun == nil {
					t.Fatalf("expected result data")
				}
				if diff := cmp.Diff(*tc.wantResult, *gotRun); diff != "" {
					t.Fatalf("result mismatch (-want +got):\n%s", diff)
				}
			} else if gotRun != nil {
				t.Fatalf("unexpected result data")
			}
			if tc.wantBatch > 0 {
				if len(gotBatch) != tc.wantBatch {
					t.Fatalf("batch size = %d, want %d", len(gotBatch), tc.wantBatch)
				}
			} else if len(gotBatch) > 0 {
				t.Fatalf("unexpected batch results")
			}
		})
	}
}

func TestReadBatchPrompts(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		lines []string
		want  []string
	}{
		"trims and skips empty": {
			lines: []string{"one", "", " two ", "\t", "three"},
			want:  []string{"one", "two", "three"},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "batch.txt")
			if err := os.WriteFile(path, []byte(strings.Join(tc.lines, "\n")), 0o600); err != nil {
				t.Fatalf("write batch file: %v", err)
			}
			got, err := readBatchPrompts(path)
			if err != nil {
				t.Fatalf("readBatchPrompts error = %v", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Fatalf("prompts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestRunBatchPrompts(t *testing.T) {
	t.Parallel()

	runner := func(ctx context.Context, cfg *config, _ adkagent.Loader) (runOutput, error) {
		if cfg.OutputJSON {
			return runOutput{}, errors.New("outputjson should be disabled")
		}
		return buildRunOutput(cfg, "agent", cfg.Prompt, 1, 1), nil
	}

	tests := map[string]struct {
		prompts []string
		want    []batchOutput
	}{
		"success: preserves order": {
			prompts: []string{"p1", "p2", "p3"},
			want: []batchOutput{
				{Prompt: "p1", Output: buildRunOutput(&config{SessionID: "", ModelName: "model"}, "agent", "p1", 1, 1)},
				{Prompt: "p2", Output: buildRunOutput(&config{SessionID: "", ModelName: "model"}, "agent", "p2", 1, 1)},
				{Prompt: "p3", Output: buildRunOutput(&config{SessionID: "", ModelName: "model"}, "agent", "p3", 1, 1)},
			},
		},
		"empty: returns nil": {
			prompts: nil,
			want:    nil,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			cfg := config{Concurrency: 2, ModelName: "model"}
			got, err := runBatchPrompts(t.Context(), &cfg, nil, tc.prompts, runner)
			if err != nil {
				t.Fatalf("runBatchPrompts error = %v", err)
			}
			if len(tc.prompts) == 0 {
				if got != nil {
					t.Fatalf("expected nil results for empty prompts")
				}
				return
			}
			if len(got) != len(tc.want) {
				t.Fatalf("result length = %d, want %d", len(got), len(tc.want))
			}
			for i := range got {
				if got[i].Prompt != tc.want[i].Prompt {
					t.Fatalf("prompt[%d]=%q want %q", i, got[i].Prompt, tc.want[i].Prompt)
				}
				if got[i].Output.Text != tc.want[i].Output.Text {
					t.Fatalf("text[%d]=%q want %q", i, got[i].Output.Text, tc.want[i].Output.Text)
				}
			}
		})
	}
}

func TestA2AExecutorExecuteSingle(t *testing.T) {
	t.Parallel()

	executor := &a2aExecutor{
		baseConfig: config{ModelName: "model", Concurrency: 1},
		runOnce: func(ctx context.Context, cfg *config, _ adkagent.Loader) (runOutput, error) {
			return buildRunOutput(cfg, "agent", "ok", 1, 1), nil
		},
	}

	msg := &a2a.Message{
		Role:  a2a.MessageRoleUser,
		Parts: a2a.ContentParts{a2a.TextPart{Text: "hi"}},
	}
	reqCtx := &a2asrv.RequestContext{Message: msg, TaskID: "task-1", ContextID: "ctx-1"}
	queue := &recordingQueue{}

	if err := executor.Execute(t.Context(), reqCtx, queue); err != nil {
		t.Fatalf("Execute error = %v", err)
	}

	events := queue.snapshot()
	if len(events) != 3 {
		t.Fatalf("events length = %d, want 3", len(events))
	}

	states := []a2a.TaskState{a2a.TaskStateSubmitted, a2a.TaskStateWorking, a2a.TaskStateCompleted}
	for i, event := range events {
		status, ok := event.(*a2a.TaskStatusUpdateEvent)
		if !ok {
			t.Fatalf("event[%d] type = %T", i, event)
		}
		if status.Status.State != states[i] {
			t.Fatalf("state[%d] = %s, want %s", i, status.Status.State, states[i])
		}
		if i < len(states)-1 && status.Final {
			t.Fatalf("state[%d] should not be final", i)
		}
	}

	final := events[len(events)-1].(*a2a.TaskStatusUpdateEvent)
	if !final.Final {
		t.Fatalf("final event not marked final")
	}
	if final.Status.Message == nil {
		t.Fatalf("final message missing")
	}
	foundText := false
	for _, part := range final.Status.Message.Parts {
		if text, ok := part.(a2a.TextPart); ok && text.Text == "ok" {
			foundText = true
			break
		}
	}
	if !foundText {
		t.Fatalf("final message missing expected text part")
	}
}

func TestA2AExecutorExecuteBatch(t *testing.T) {
	t.Parallel()

	executor := &a2aExecutor{
		baseConfig: config{ModelName: "model", Concurrency: 1},
		runOnce: func(ctx context.Context, cfg *config, _ adkagent.Loader) (runOutput, error) {
			return buildRunOutput(cfg, "agent", cfg.Prompt, 1, 1), nil
		},
	}

	msg := &a2a.Message{
		Role:     a2a.MessageRoleUser,
		Metadata: map[string]any{"accepted_output_modes": []string{"application/json"}},
		Parts: a2a.ContentParts{
			a2a.DataPart{Data: map[string]any{"prompts": []string{"p1", "p2"}}},
		},
	}
	reqCtx := &a2asrv.RequestContext{Message: msg, TaskID: "task-2", ContextID: "ctx-2"}
	queue := &recordingQueue{}

	if err := executor.Execute(t.Context(), reqCtx, queue); err != nil {
		t.Fatalf("Execute error = %v", err)
	}

	events := queue.snapshot()
	if len(events) != 3 {
		t.Fatalf("events length = %d, want 3", len(events))
	}

	final := events[len(events)-1].(*a2a.TaskStatusUpdateEvent)
	if final.Status.Message == nil {
		t.Fatalf("final message missing")
	}

	var got []batchOutput
	for _, part := range final.Status.Message.Parts {
		data, ok := part.(a2a.DataPart)
		if !ok {
			continue
		}
		value, ok := data.Data["results"]
		if !ok {
			continue
		}
		results, ok := value.([]batchOutput)
		if !ok {
			t.Fatalf("results type = %T", value)
		}
		got = results
	}
	if len(got) != 2 {
		t.Fatalf("batch results length = %d, want 2", len(got))
	}
	if got[0].Output.Text != "p1" || got[1].Output.Text != "p2" {
		t.Fatalf("batch outputs mismatch: %#v", got)
	}
}

func TestA2AExecutorCancel(t *testing.T) {
	t.Parallel()

	executor := &a2aExecutor{}
	queue := &recordingQueue{}
	reqCtx := &a2asrv.RequestContext{TaskID: "task-3", ContextID: "ctx-3"}

	if err := executor.Cancel(t.Context(), reqCtx, queue); err != nil {
		t.Fatalf("Cancel error = %v", err)
	}

	events := queue.snapshot()
	if len(events) != 1 {
		t.Fatalf("events length = %d, want 1", len(events))
	}
	status, ok := events[0].(*a2a.TaskStatusUpdateEvent)
	if !ok {
		t.Fatalf("event type = %T", events[0])
	}
	if status.Status.State != a2a.TaskStateCanceled {
		t.Fatalf("state = %s, want %s", status.Status.State, a2a.TaskStateCanceled)
	}
	if !status.Final {
		t.Fatalf("cancel event should be final")
	}
	if status.Status.Message == nil || len(status.Status.Message.Parts) == 0 {
		t.Fatalf("cancel message missing")
	}
	part, ok := status.Status.Message.Parts[0].(a2a.TextPart)
	if !ok {
		t.Fatalf("cancel part type = %T", status.Status.Message.Parts[0])
	}
	if part.Text != "canceled" {
		t.Fatalf("cancel text = %q, want canceled", part.Text)
	}
}

func TestServeA2ARequiresAddr(t *testing.T) {
	t.Parallel()

	cfg := config{}
	if err := serveA2A(t.Context(), &cfg, nil, slog.Default()); err == nil {
		t.Fatalf("expected error on missing a2a_addr")
	}
}
