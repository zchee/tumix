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
	"github.com/zchee/tumix/log"
)

const (
	a2aInvokePath      = "/invoke"
	a2aProtocolVersion = "0.2.9"
)

type a2aExecutor struct {
	baseConfig config
	loader     adkagent.Loader
	runOnce    runOnceFunc
}

func (e *a2aExecutor) Execute(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
	if reqCtx == nil || reqCtx.Message == nil {
		return errors.New("a2a request requires a message")
	}

	emitStatus := func(state a2a.TaskState, msg *a2a.Message, final bool) error {
		event := a2a.NewStatusUpdateEvent(reqCtx, state, msg)
		event.Final = final
		if err := queue.Write(ctx, event); err != nil {
			return fmt.Errorf("write status %s: %w", state, err)
		}
		return nil
	}
	emitFailure := func(err error) error {
		msg := a2a.NewMessageForTask(a2a.MessageRoleAgent, reqCtx, a2a.TextPart{Text: err.Error()})
		return emitStatus(a2a.TaskStateFailed, msg, true)
	}

	if reqCtx.StoredTask == nil {
		if err := emitStatus(a2a.TaskStateSubmitted, nil, false); err != nil {
			return err
		}
	}
	if err := emitStatus(a2a.TaskStateWorking, nil, false); err != nil {
		return err
	}

	prompts, isBatch, err := extractA2APrompts(reqCtx.Message)
	if err != nil {
		return emitFailure(err)
	}

	local := e.baseConfig
	local.OutputJSON = false
	local.BatchFile = ""
	local.Prompt = ""
	if reqCtx.TaskID != "" {
		local.SessionID = string(reqCtx.TaskID)
	}

	runner := e.runOnce
	if runner == nil {
		runner = runOnce
	}

	var parts []a2a.Part
	if isBatch { //nolint:nestif
		outputs, err := runBatchPrompts(ctx, &local, e.loader, prompts, runner)
		if err != nil {
			return emitFailure(err)
		}
		parts, err = buildA2AResponseParts(reqCtx.Message.Metadata, nil, outputs)
		if err != nil {
			return emitFailure(err)
		}
	} else {
		local.Prompt = prompts[0]
		output, err := runner(ctx, &local, e.loader)
		if err != nil {
			return emitFailure(err)
		}
		parts, err = buildA2AResponseParts(reqCtx.Message.Metadata, &output, nil)
		if err != nil {
			return emitFailure(err)
		}
	}

	msg := a2a.NewMessageForTask(a2a.MessageRoleAgent, reqCtx, parts...)
	if err := emitStatus(a2a.TaskStateCompleted, msg, true); err != nil {
		return err
	}
	return nil
}

func (e *a2aExecutor) Cancel(ctx context.Context, reqCtx *a2asrv.RequestContext, queue eventqueue.Queue) error {
	if reqCtx == nil {
		return errors.New("a2a cancel requires request context")
	}
	msg := a2a.NewMessageForTask(a2a.MessageRoleAgent, reqCtx, a2a.TextPart{Text: "canceled"})
	event := a2a.NewStatusUpdateEvent(reqCtx, a2a.TaskStateCanceled, msg)
	event.Final = true
	if err := queue.Write(ctx, event); err != nil {
		return fmt.Errorf("write status canceled: %w", err)
	}
	return nil
}

func extractA2APrompts(msg *a2a.Message) (texts []string, isBatch bool, err error) {
	if msg == nil {
		return nil, false, errors.New("message is required")
	}

	coerceString := func(value any) (string, bool) {
		str, ok := value.(string)
		if !ok {
			return "", false
		}
		str = strings.TrimSpace(str)
		if str == "" {
			return "", false
		}
		return str, true
	}
	coerceStringSlice := func(value any) ([]string, bool, error) {
		slice, ok := value.([]string)
		if ok {
			out := make([]string, 0, len(slice))
			for _, item := range slice {
				item = strings.TrimSpace(item)
				if item != "" {
					out = append(out, item)
				}
			}
			return out, true, nil
		}
		anySlice, ok := value.([]any)
		if !ok {
			return nil, false, nil
		}
		out := make([]string, 0, len(anySlice))
		for _, item := range anySlice {
			str, ok := item.(string)
			if !ok {
				return nil, true, fmt.Errorf("batch prompt must be string, got %T", item)
			}
			str = strings.TrimSpace(str)
			if str != "" {
				out = append(out, str)
			}
		}
		return out, true, nil
	}

	var textParts []string
	var dataPrompt string
	var batchPrompts []string
	batchSeen := false

	for _, part := range msg.Parts {
		switch p := part.(type) {
		case a2a.TextPart:
			if text := strings.TrimSpace(p.Text); text != "" {
				textParts = append(textParts, text)
			}
		case a2a.DataPart:
			if p.Data == nil {
				continue
			}
			for _, key := range []string{"prompts", "batch_prompts", "batchPrompts"} {
				if value, ok := p.Data[key]; ok {
					prompts, found, err := coerceStringSlice(value)
					if err != nil {
						return nil, false, err
					}
					if found {
						if batchSeen {
							return nil, false, errors.New("multiple batch prompt lists provided")
						}
						batchPrompts = prompts
						batchSeen = true
					}
				}
			}
			if batchSeen {
				continue
			}
			if value, ok := p.Data["prompt"]; ok {
				if str, ok := coerceString(value); ok {
					dataPrompt = str
				}
			}
		}
	}

	if !batchSeen && msg.Metadata != nil { //nolint:nestif
		for _, key := range []string{"prompts", "batch_prompts", "batchPrompts"} {
			if value, ok := msg.Metadata[key]; ok {
				prompts, found, err := coerceStringSlice(value)
				if err != nil {
					return nil, false, err
				}
				if found {
					batchPrompts = prompts
					batchSeen = true
					break
				}
			}
		}
		if !batchSeen && dataPrompt == "" {
			if value, ok := msg.Metadata["prompt"]; ok {
				if str, ok := coerceString(value); ok {
					dataPrompt = str
				}
			}
		}
	}

	if batchSeen {
		if len(batchPrompts) == 0 {
			return nil, false, errors.New("batch prompts are empty")
		}
		return batchPrompts, true, nil
	}

	if dataPrompt != "" {
		textParts = append(textParts, dataPrompt)
	}
	if len(textParts) == 0 {
		return nil, false, errors.New("prompt is required")
	}
	return []string{strings.Join(textParts, "\n")}, false, nil
}

func buildA2AResponseParts(meta map[string]any, output *runOutput, batch []batchOutput) ([]a2a.Part, error) {
	if output == nil && len(batch) == 0 {
		return nil, errors.New("no output to format")
	}

	includeText := true
	includeJSON := true
	if meta != nil { //nolint:nestif
		modes, ok := meta["accepted_output_modes"]
		if !ok {
			modes = meta["acceptedOutputModes"]
		}
		if modes != nil {
			includeText = false
			includeJSON = false
			var values []string
			switch typed := modes.(type) {
			case []string:
				values = typed
			case []any:
				values = make([]string, 0, len(typed))
				for _, item := range typed {
					str, ok := item.(string)
					if !ok {
						return nil, fmt.Errorf("accepted_output_modes must be string values, got %T", item)
					}
					values = append(values, str)
				}
			default:
				return nil, fmt.Errorf("accepted_output_modes must be string list, got %T", modes)
			}
			for _, mode := range values {
				switch strings.ToLower(strings.TrimSpace(mode)) {
				case "text/plain":
					includeText = true
				case "application/json":
					includeJSON = true
				}
			}
			if !includeText && !includeJSON {
				return nil, errors.New("no compatible output modes requested")
			}
		}
	}

	parts := make([]a2a.Part, 0, 2)
	if includeText { //nolint:nestif
		var text string
		if output != nil {
			text = output.Text
		} else {
			var sb strings.Builder
			for i := range batch {
				item := batch[i]
				if i > 0 {
					sb.WriteString("\n")
				}
				sb.WriteString(fmt.Sprintf("[%d] %s", i, item.Output.Text))
			}
			text = sb.String()
		}
		if text != "" {
			parts = append(parts, a2a.TextPart{Text: text})
		}
	}
	if includeJSON {
		if output != nil {
			parts = append(parts, a2a.DataPart{Data: map[string]any{"result": *output}})
		} else {
			parts = append(parts, a2a.DataPart{Data: map[string]any{"results": batch}})
		}
	}

	if len(parts) == 0 {
		return nil, errors.New("no response parts selected")
	}
	return parts, nil
}

func serveA2A(ctx context.Context, cfg *config, loader adkagent.Loader, logger *slog.Logger) error {
	if cfg.A2AAddr == "" {
		return errors.New("a2a_addr is required")
	}

	invokeURL := cfg.A2AURL
	if invokeURL == "" {
		addrURL, err := a2aURLFromAddr(cfg.A2AAddr)
		if err != nil {
			return err
		}
		invokeURL = addrURL
	}

	card := &a2a.AgentCard{
		ProtocolVersion:    a2aProtocolVersion,
		Name:               cfg.AppName,
		Description:        "TUMIX: Multi-Agent Test-Time Scaling with Tool-Use Mixture.",
		URL:                invokeURL,
		PreferredTransport: a2a.TransportProtocolJSONRPC,
		AdditionalInterfaces: []a2a.AgentInterface{
			{
				URL:       invokeURL,
				Transport: a2a.TransportProtocolJSONRPC,
			},
		},
		Version: version.Version,
		Capabilities: a2a.AgentCapabilities{
			Streaming:              true,
			PushNotifications:      false,
			StateTransitionHistory: false,
		},
		DefaultInputModes:  []string{"text/plain", "application/json"},
		DefaultOutputModes: []string{"text/plain", "application/json"},
		Skills: []a2a.AgentSkill{
			{
				ID:          "tumix.run",
				Name:        "Run tumix prompt",
				Description: "Run the tumix multi-agent workflow for a single prompt and return the final answer.",
				Tags:        []string{"tumix", "llm", "reasoning"},
				Examples: []string{
					"Summarize the differences between TCP and UDP.",
				},
				InputModes:  []string{"text/plain", "application/json"},
				OutputModes: []string{"text/plain", "application/json"},
			},
			{
				ID:          "tumix.batch",
				Name:        "Run tumix batch",
				Description: "Run multiple tumix prompts in one request by sending a DataPart with a prompts array.",
				Tags:        []string{"tumix", "batch"},
				Examples: []string{
					"DataPart: {\"prompts\":[\"prompt one\",\"prompt two\"]}",
				},
				InputModes:  []string{"application/json"},
				OutputModes: []string{"application/json"},
			},
		},
	}

	executor := &a2aExecutor{baseConfig: *cfg, loader: loader}
	handler := a2asrv.NewHandler(executor, a2asrv.WithLogger(logger))

	mux := http.NewServeMux()
	mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(card))
	mux.Handle(a2aInvokePath, a2asrv.NewJSONRPCHandler(handler))

	server := &http.Server{
		Addr:              cfg.A2AAddr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	log.Info(ctx, "a2a server listening", "addr", cfg.A2AAddr, "url", invokeURL)
	errCh := make(chan error, 1)
	go func() {
		errCh <- server.ListenAndServe()
	}()

	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			return fmt.Errorf("a2a server shutdown: %w", err)
		}
		return nil
	case err := <-errCh:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return fmt.Errorf("a2a server listen: %w", err)
	}
}

func a2aURLFromAddr(addr string) (string, error) {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		return "", errors.New("a2a_addr is empty")
	}
	if strings.HasPrefix(addr, "http://") || strings.HasPrefix(addr, "https://") {
		return strings.TrimRight(addr, "/") + a2aInvokePath, nil
	}
	if strings.HasPrefix(addr, ":") {
		return "http://localhost" + addr + a2aInvokePath, nil
	}
	return "http://" + strings.TrimRight(addr, "/") + a2aInvokePath, nil
}
