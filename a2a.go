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
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
	adkagent "google.golang.org/adk/agent"

	"github.com/zchee/tumix/internal/version"
)

type a2aExecutor struct {
	baseCfg *config
	loader  adkagent.Loader
	runOnce runOnceFunc
}

func (e *a2aExecutor) Execute(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
	if reqCtx == nil || reqCtx.Message == nil {
		return a2a.ErrInvalidParams
	}

	if reqCtx.StoredTask == nil {
		event := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateSubmitted, nil)
		if err := queue.Write(ctx, event); err != nil {
			return fmt.Errorf("failed to write state submitted: %w", err)
		}
	}

	working := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateWorking, nil)
	if err := queue.Write(ctx, working); err != nil {
		return fmt.Errorf("failed to write state working: %w", err)
	}

	prompts, isBatch, err := extractA2APrompts(reqCtx.Message)
	if err != nil {
		return writeA2AFailure(ctx, reqCtx, queue, err)
	}
	for _, prompt := range prompts {
		if err := validatePromptLimits(e.baseCfg, prompt); err != nil {
			return writeA2AFailure(ctx, reqCtx, queue, err)
		}
	}

	runFunc := e.runOnce
	if runFunc == nil {
		runFunc = runOnce
	}

	var (
		output       runOutput
		batchOutputs []batchOutput
	)
	if isBatch {
		local := *e.baseCfg
		local.OutputJSON = false
		local.SessionID = ""
		var err error
		batchOutputs, err = runBatchPrompts(ctx, &local, e.loader, prompts, runFunc)
		if err != nil {
			return writeA2AFailure(ctx, reqCtx, queue, err)
		}
	} else {
		local := *e.baseCfg
		local.OutputJSON = false
		local.Prompt = prompts[0]
		if reqCtx.ContextID != "" {
			local.SessionID = reqCtx.ContextID
		} else if reqCtx.TaskID != "" {
			local.SessionID = string(reqCtx.TaskID)
		}
		var err error
		output, err = runFunc(ctx, &local, e.loader)
		if err != nil {
			return writeA2AFailure(ctx, reqCtx, queue, err)
		}
	}

	parts := buildA2AResponseParts(output, batchOutputs)
	msg := a2a.NewMessageForTask(a2a.MessageRoleAgent, reqCtx, parts...)
	completed := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateCompleted, msg)
	completed.Final = true
	if err := queue.Write(ctx, completed); err != nil {
		return fmt.Errorf("failed to write state completed: %w", err)
	}

	return nil
}

func (e *a2aExecutor) Cancel(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
	if reqCtx == nil {
		return a2a.ErrInvalidParams
	}

	msg := a2a.NewMessageForTask(a2a.MessageRoleAgent, reqCtx, a2a.TextPart{
		Text: "canceled",
	})
	event := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateCanceled, msg)
	event.Final = true
	if err := queue.Write(ctx, event); err != nil {
		return fmt.Errorf("failed to write state canceled: %w", err)
	}
	return nil
}

func writeA2AFailure(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue, cause error) error {
	msg := a2a.NewMessageForTask(a2a.MessageRoleAgent, reqCtx, a2a.TextPart{
		Text: cause.Error(),
	})
	event := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateFailed, msg)
	event.Final = true
	if err := queue.Write(ctx, event); err != nil {
		return fmt.Errorf("failed to write state failed: %w", err)
	}
	return nil
}

func extractA2APrompts(msg *a2a.Message) ([]string, bool, error) {
	if msg == nil {
		return nil, false, errors.New("message is required")
	}
	if len(msg.Parts) == 0 {
		return nil, false, errors.New("message parts are required")
	}

	parsePromptSlice := func(value any) ([]string, error) {
		switch v := value.(type) {
		case []string:
			return v, nil
		case []any:
			out := make([]string, len(v))
			for i, item := range v {
				text, ok := item.(string)
				if !ok {
					return nil, fmt.Errorf("prompt %d is not a string", i)
				}
				out[i] = text
			}
			return out, nil
		default:
			return nil, errors.New("prompts must be an array of strings")
		}
	}

	parsePromptValue := func(value any) (string, error) {
		text, ok := value.(string)
		if !ok {
			return "", errors.New("prompt must be a string")
		}
		return text, nil
	}

	var textParts []string
	var dataPrompts []string
	seenBatch := false
	seenPrompt := false

	for _, part := range msg.Parts {
		switch p := part.(type) {
		case a2a.TextPart:
			textParts = append(textParts, p.Text)
		case *a2a.TextPart:
			if p != nil {
				textParts = append(textParts, p.Text)
			}
		case a2a.DataPart:
			prompts, hasBatch, hasPrompt, err := promptsFromData(p.Data, parsePromptSlice, parsePromptValue)
			if err != nil {
				return nil, false, err
			}
			dataPrompts = append(dataPrompts, prompts...)
			seenBatch = seenBatch || hasBatch
			seenPrompt = seenPrompt || hasPrompt
		case *a2a.DataPart:
			if p == nil {
				continue
			}
			prompts, hasBatch, hasPrompt, err := promptsFromData(p.Data, parsePromptSlice, parsePromptValue)
			if err != nil {
				return nil, false, err
			}
			dataPrompts = append(dataPrompts, prompts...)
			seenBatch = seenBatch || hasBatch
			seenPrompt = seenPrompt || hasPrompt
		}
	}

	if len(dataPrompts) > 0 && len(textParts) > 0 {
		return nil, false, errors.New("prompt must be provided as either text parts or data parts")
	}
	if seenBatch && seenPrompt {
		return nil, false, errors.New("prompt must use either prompt or prompts, not both")
	}

	if len(dataPrompts) > 0 {
		prompts := make([]string, 0, len(dataPrompts))
		for _, prompt := range dataPrompts {
			trimmed := strings.TrimSpace(prompt)
			if trimmed == "" {
				return nil, false, errors.New("prompt cannot be empty")
			}
			prompts = append(prompts, trimmed)
		}
		return prompts, seenBatch || len(prompts) > 1, nil
	}

	if len(textParts) == 0 {
		return nil, false, errors.New("prompt is required")
	}

	prompt := strings.TrimSpace(strings.Join(textParts, "\n"))
	if prompt == "" {
		return nil, false, errors.New("prompt cannot be empty")
	}
	return []string{prompt}, false, nil
}

func promptsFromData(data map[string]any, parseSlice func(any) ([]string, error), parseValue func(any) (string, error)) ([]string, bool, bool, error) {
	var (
		prompts   []string
		hasBatch  bool
		hasPrompt bool
	)

	if data == nil {
		return nil, false, false, nil
	}

	if raw, ok := data["prompts"]; ok {
		values, err := parseSlice(raw)
		if err != nil {
			return nil, false, false, err
		}
		prompts = append(prompts, values...)
		hasBatch = true
	}
	if raw, ok := data["prompt"]; ok {
		value, err := parseValue(raw)
		if err != nil {
			return nil, false, false, err
		}
		prompts = append(prompts, value)
		hasPrompt = true
	}

	return prompts, hasBatch, hasPrompt, nil
}

func validatePromptLimits(cfg *config, prompt string) error {
	if cfg.MaxPromptChars > 0 && len(prompt) > cfg.MaxPromptChars {
		return fmt.Errorf("prompt length %d exceeds max_prompt_chars %d", len(prompt), cfg.MaxPromptChars)
	}
	if cfg.MaxPromptTokens > 0 {
		est := estimateTokensFromChars(len(prompt))
		if est > cfg.MaxPromptTokens {
			return fmt.Errorf("prompt token estimate %d exceeds max_prompt_tokens %d", est, cfg.MaxPromptTokens)
		}
	}
	return nil
}

func buildA2AResponseParts(output runOutput, batch []batchOutput) []a2a.Part {
	if len(batch) > 0 {
		parts := []a2a.Part{
			a2a.DataPart{Data: map[string]any{"results": batch}},
		}
		parts = append(parts, a2a.TextPart{
			Text: fmt.Sprintf("batch completed: %d prompts", len(batch)),
		})
		return parts
	}

	parts := []a2a.Part{
		a2a.DataPart{
			Data: map[string]any{"result": output},
		},
	}
	if output.Text != "" {
		parts = append(parts, a2a.TextPart{
			Text: output.Text,
		})
	}
	return parts
}

func resolveA2AURL(cfg *config) (string, error) {
	if cfg.A2AURL != "" {
		return strings.TrimRight(cfg.A2AURL, "/"), nil
	}
	if cfg.A2AAddr == "" {
		return "", errors.New("a2a_addr is required")
	}
	addr := cfg.A2AAddr
	if strings.HasPrefix(addr, ":") {
		addr = "localhost" + addr
	}
	if !strings.Contains(addr, "://") {
		addr = "http://" + addr
	}
	return strings.TrimRight(addr, "/") + "/invoke", nil
}

func buildA2AAgentCard(cfg *config) (*a2a.AgentCard, error) {
	url, err := resolveA2AURL(cfg)
	if err != nil {
		return nil, err
	}

	return &a2a.AgentCard{
		Name:               "tumix",
		Description:        "TUMIX: multi-agent reasoning with tool-use mixture.",
		ProtocolVersion:    "0.3.0",
		PreferredTransport: a2a.TransportProtocolJSONRPC,
		URL:                url,
		AdditionalInterfaces: []a2a.AgentInterface{
			{Transport: a2a.TransportProtocolJSONRPC, URL: url},
		},
		Version:                           version.Version,
		DefaultInputModes:                 []string{"text/plain", "application/json"},
		DefaultOutputModes:                []string{"application/json", "text/plain"},
		SupportsAuthenticatedExtendedCard: false,
		Capabilities: a2a.AgentCapabilities{
			Streaming:              true,
			PushNotifications:      false,
			StateTransitionHistory: false,
		},
		Skills: []a2a.AgentSkill{
			{
				ID:          "tumix-run",
				Name:        "Tumix prompt execution",
				Description: "Runs TUMIX to answer a single prompt.",
				Tags:        []string{"llm", "reasoning", "analysis"},
				Examples: []string{
					"Explain the tradeoffs of rate limiting strategies.",
				},
				InputModes:  []string{"text/plain", "application/json"},
				OutputModes: []string{"application/json", "text/plain"},
			},
			{
				ID:          "tumix-batch",
				Name:        "Tumix batch execution",
				Description: "Runs TUMIX for multiple prompts when a data part includes a prompts array.",
				Tags:        []string{"batch", "llm", "reasoning"},
				Examples: []string{
					"{\"prompts\":[\"Summarize X\", \"Compare Y and Z\"]}",
				},
				InputModes:  []string{"application/json"},
				OutputModes: []string{"application/json"},
			},
		},
	}, nil
}

func serveA2A(ctx context.Context, cfg *config, loader adkagent.Loader, logger *slog.Logger) error {
	if cfg.A2AAddr == "" {
		return errors.New("a2a_addr is required")
	}
	if logger == nil {
		logger = slog.Default()
	}

	card, err := buildA2AAgentCard(cfg)
	if err != nil {
		return err
	}

	executor := &a2aExecutor{
		baseCfg: cfg,
		loader:  loader,
		runOnce: runOnce,
	}
	handler := a2asrv.NewHandler(executor, a2asrv.WithLogger(logger))

	mux := http.NewServeMux()
	mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(card))
	mux.Handle("/invoke", a2asrv.NewJSONRPCHandler(handler))

	server := &http.Server{
		Addr:              cfg.A2AAddr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		errCh <- server.ListenAndServe()
	}()

	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			return err
		}
		return nil
	case err := <-errCh:
		if err == nil || errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	}
}
