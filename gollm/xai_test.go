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
	"context"
	"net"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"

	"github.com/zchee/tumix/gollm/internal/adapter"
	"github.com/zchee/tumix/gollm/xai"
	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
	"github.com/zchee/tumix/testing/rr"
)

func TestXAILLM_RecordReplay(t *testing.T) {
	t.Skip("fix record logic")
	t.Parallel()

	const xaiReplayAddr = "127.0.0.1:28083"

	var serverCleanup func()
	if *rr.Record {
		var lc net.ListenConfig
		ln, err := lc.Listen(t.Context(), "tcp", xaiReplayAddr)
		if err != nil {
			t.Skipf("unable to listen for xai stub: %v", err)
		}

		grpcServer := grpc.NewServer()
		xaipb.RegisterChatServer(grpcServer, &stubChatServer{
			completionResp: &xaipb.GetChatCompletionResponse{
				Model:             "grok-1",
				SystemFingerprint: "fp-rr",
				Outputs: []*xaipb.CompletionOutput{
					{
						Index:        0,
						FinishReason: xaipb.FinishReason_REASON_STOP,
						Message: &xaipb.CompletionMessage{
							Role:    xaipb.MessageRole_ROLE_ASSISTANT,
							Content: "Paris",
						},
					},
				},
				Usage: &xaipb.SamplingUsage{
					CompletionTokens: 2,
					PromptTokens:     8,
					TotalTokens:      10,
				},
			},
		})

		go func() {
			if err := grpcServer.Serve(ln); err != nil {
				t.Error(err)
			}
		}()
		serverCleanup = func() {
			grpcServer.Stop()
			_ = ln.Close()
		}
	}

	conn, cleanup := rr.NewInsecureGRPCConn(t, "xai", xaiReplayAddr)
	t.Cleanup(cleanup)
	if serverCleanup != nil {
		t.Cleanup(serverCleanup)
	}

	llm, err := NewXAILLM(t.Context(), AuthMethodAPIKey("test-key"), "grok-1",
		xai.WithAPIConn(conn),
		xai.WithAPIHost(xaiReplayAddr),
		xai.WithInsecure(),
	)
	if err != nil {
		t.Fatalf("NewXAILLM() error = %v", err)
	}
	if concrete, ok := llm.(*xaiLLM); ok {
		t.Cleanup(func() {
			_ = concrete.client.Close()
		})
	}

	req := &model.LLMRequest{
		Contents: genai.Text("What is the capital of France?"),
		Config:   &genai.GenerateContentConfig{},
	}

	var got *model.LLMResponse
	for resp, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() unexpected error: %v", err)
		}
		got = resp
	}

	want := &model.LLMResponse{
		Content: genai.NewContentFromText("Paris", genai.RoleModel),
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     8,
			CandidatesTokenCount: 2,
			TotalTokenCount:      10,
		},
		CustomMetadata: map[string]any{
			"xai_finish_reason":      "REASON_STOP",
			"xai_system_fingerprint": "fp-rr",
		},
		FinishReason: genai.FinishReasonStop,
	}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("GenerateContent() diff (-want +got):\n%s", diff)
	}

	if t.Failed() {
		t.FailNow()
	}
}

func TestXAILLM_RecordReplayStream(t *testing.T) {
	t.Skip("fix record logic")
	t.Parallel()

	const xaiReplayAddr = "127.0.0.1:28084"

	var serverCleanup func()
	if *rr.Record {
		var lc net.ListenConfig
		ln, err := lc.Listen(t.Context(), "tcp", xaiReplayAddr)
		if err != nil {
			t.Skipf("unable to listen for xai stream stub: %v", err)
		}

		grpcServer := grpc.NewServer()
		xaipb.RegisterChatServer(grpcServer, &stubChatServer{
			chunks: []*xaipb.GetChatCompletionChunk{
				{
					Outputs: []*xaipb.CompletionOutputChunk{
						{
							Index: 0,
							Delta: &xaipb.Delta{
								Role:    xaipb.MessageRole_ROLE_ASSISTANT,
								Content: "Par",
							},
						},
					},
				},
				{
					Model:             "grok-1",
					SystemFingerprint: "fp-stream",
					Usage: &xaipb.SamplingUsage{
						CompletionTokens: 2,
						PromptTokens:     10,
						TotalTokens:      12,
					},
					Outputs: []*xaipb.CompletionOutputChunk{
						{
							Index:        0,
							FinishReason: xaipb.FinishReason_REASON_STOP,
							Delta: &xaipb.Delta{
								Role:    xaipb.MessageRole_ROLE_ASSISTANT,
								Content: "is",
							},
						},
					},
				},
			},
		})

		go func() {
			if err := grpcServer.Serve(ln); err != nil {
				t.Error(err)
			}
		}()
		serverCleanup = func() {
			grpcServer.Stop()
			_ = ln.Close()
		}
	}

	conn, cleanup := rr.NewInsecureGRPCConn(t, "xai", xaiReplayAddr)
	t.Cleanup(cleanup)
	if serverCleanup != nil {
		t.Cleanup(serverCleanup)
	}

	llm, err := NewXAILLM(t.Context(), AuthMethodAPIKey("test-key"), "grok-4-1-fast-reasoning",
		xai.WithAPIConn(conn),
		xai.WithAPIHost(xaiReplayAddr),
		xai.WithInsecure(),
	)
	if err != nil {
		t.Fatalf("NewXAILLM() error = %v", err)
	}
	if concrete, ok := llm.(*xaiLLM); ok {
		t.Cleanup(func() {
			_ = concrete.client.Close()
		})
	}

	req := &model.LLMRequest{
		Contents: genai.Text("Stream please"),
		Config:   &genai.GenerateContentConfig{},
	}

	var partialTexts []string
	var finalTexts []string
	var finishReasons []genai.FinishReason

	for resp, err := range llm.GenerateContent(t.Context(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent(stream) unexpected error: %v", err)
		}
		if resp == nil || resp.Content == nil || len(resp.Content.Parts) == 0 {
			t.Fatalf("GenerateContent(stream) returned empty response: %+v", resp)
		}

		text := resp.Content.Parts[0].Text
		if resp.Partial {
			partialTexts = append(partialTexts, text)
		} else {
			finalTexts = append(finalTexts, text)
		}
		if resp.FinishReason != genai.FinishReasonUnspecified {
			finishReasons = append(finishReasons, resp.FinishReason)
		}
	}

	if len(partialTexts) == 0 || partialTexts[0] != "Par" {
		t.Fatalf("partial texts = %v, want first \"Par\"", partialTexts)
	}
	if got := partialTexts[len(partialTexts)-1]; got != "Paris" {
		t.Fatalf("last partial = %q, want %q", got, "Paris")
	}
	if diff := cmp.Diff([]string{"Paris"}, finalTexts); diff != "" {
		t.Fatalf("final texts diff (-want +got):\n%s", diff)
	}
	if len(finishReasons) == 0 || finishReasons[len(finishReasons)-1] != genai.FinishReasonStop {
		t.Fatalf("finish reasons = %v, want last FinishReasonStop", finishReasons)
	}

	if t.Failed() {
		t.FailNow()
	}
}

func TestXAILLM_RecordReplayToolCalls(t *testing.T) {
	t.Skip("fix record logic")
	t.Parallel()

	const xaiReplayAddr = "127.0.0.1:28085"

	var serverCleanup func()
	if *rr.Record {
		var lc net.ListenConfig
		ln, err := lc.Listen(t.Context(), "tcp", xaiReplayAddr)
		if err != nil {
			t.Skipf("unable to listen for xai tool stub: %v", err)
		}

		grpcServer := grpc.NewServer()
		xaipb.RegisterChatServer(grpcServer, &stubChatServer{
			chunks: []*xaipb.GetChatCompletionChunk{
				{
					Outputs: []*xaipb.CompletionOutputChunk{
						{
							Index: 0,
							Delta: &xaipb.Delta{
								Role:             xaipb.MessageRole_ROLE_ASSISTANT,
								ReasoningContent: "Thinking ",
								Content:          "Par",
							},
						},
					},
				},
				{
					Outputs: []*xaipb.CompletionOutputChunk{
						{
							Index: 0,
							Delta: &xaipb.Delta{
								Role: xaipb.MessageRole_ROLE_ASSISTANT,
								ToolCalls: []*xaipb.ToolCall{{
									Id: "tc1",
									Tool: &xaipb.ToolCall_Function{Function: &xaipb.FunctionCall{
										Name:      "lookup_city",
										Arguments: `{"city":"Paris"}`,
									}},
								}},
							},
						},
					},
				},
				{
					Model:             "grok-1",
					SystemFingerprint: "fp-tool",
					Usage: &xaipb.SamplingUsage{
						PromptTokens:     3,
						CompletionTokens: 5,
						TotalTokens:      8,
					},
					Outputs: []*xaipb.CompletionOutputChunk{
						{
							Index:        0,
							FinishReason: xaipb.FinishReason_REASON_STOP,
							Delta: &xaipb.Delta{
								Role:    xaipb.MessageRole_ROLE_ASSISTANT,
								Content: "is",
							},
						},
					},
				},
			},
		})

		go func() {
			if err := grpcServer.Serve(ln); err != nil {
				t.Error(err)
			}
		}()
		serverCleanup = func() {
			grpcServer.Stop()
			_ = ln.Close()
		}
	}

	conn, cleanup := rr.NewInsecureGRPCConn(t, "xai", xaiReplayAddr)
	t.Cleanup(cleanup)
	if serverCleanup != nil {
		t.Cleanup(serverCleanup)
	}

	llm, err := NewXAILLM(t.Context(), AuthMethodAPIKey("test-key"), "grok-4-1-fast-reasoning",
		xai.WithAPIConn(conn),
		xai.WithAPIHost(xaiReplayAddr),
		xai.WithInsecure(),
	)
	if err != nil {
		t.Fatalf("NewXAILLM() error = %v", err)
	}
	if concrete, ok := llm.(*xaiLLM); ok {
		t.Cleanup(func() {
			_ = concrete.client.Close()
		})
	}

	req := &model.LLMRequest{
		Contents: genai.Text("Call a tool with reasoning"),
		Config:   &genai.GenerateContentConfig{},
	}

	var (
		thoughts     []string
		texts        []string
		toolCalls    []*genai.FunctionCall
		finishCodes  []genai.FinishReason
		turnComplete bool
	)

	for resp, err := range llm.GenerateContent(t.Context(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent(stream) unexpected error: %v", err)
		}
		if resp == nil || resp.Content == nil || len(resp.Content.Parts) == 0 {
			t.Fatalf("GenerateContent(stream) returned empty response: %+v", resp)
		}

		for _, part := range resp.Content.Parts {
			switch {
			case part.Thought:
				thoughts = append(thoughts, part.Text)
			case part.FunctionCall != nil:
				toolCalls = append(toolCalls, part.FunctionCall)
			case part.Text != "":
				texts = append(texts, part.Text)
			}
		}
		if resp.FinishReason != genai.FinishReasonUnspecified {
			finishCodes = append(finishCodes, resp.FinishReason)
		}
		if resp.TurnComplete {
			turnComplete = true
		}
	}

	if !turnComplete {
		t.Fatalf("no response marked TurnComplete; finish codes = %v", finishCodes)
	}
	if len(thoughts) == 0 || thoughts[0] != "Thinking " {
		t.Fatalf("thoughts = %v, want first \"Thinking \"", thoughts)
	}
	if len(texts) == 0 {
		t.Fatalf("texts empty: %v", texts)
	}
	if lastText := texts[len(texts)-1]; lastText != "Paris" {
		t.Fatalf("final text = %q, want %q", lastText, "Paris")
	}
	if len(toolCalls) == 0 || toolCalls[0].Name != "lookup_city" {
		t.Fatalf("tool calls missing or wrong: %+v", toolCalls)
	}
	if city, ok := toolCalls[0].Args["city"]; !ok || city != "Paris" {
		t.Fatalf("tool call args = %+v, want city=Paris", toolCalls[0].Args)
	}
	if finishCodes[len(finishCodes)-1] != genai.FinishReasonStop {
		t.Fatalf("finish reasons = %v, want last FinishReasonStop", finishCodes)
	}

	if t.Failed() {
		t.FailNow()
	}
}

func TestXAIModel_MaybeAppendUserContent(t *testing.T) {
	t.Parallel()

	t.Run("appends_when_empty", func(t *testing.T) {
		t.Parallel()

		req := &model.LLMRequest{}
		ensureUserContent(req)
		if got := req.Contents[len(req.Contents)-1].Role; got != genai.RoleUser {
			t.Fatalf("last role = %q, want user", got)
		}
	})

	t.Run("appends_when_last_not_user", func(t *testing.T) {
		t.Parallel()

		req := &model.LLMRequest{
			Contents: []*genai.Content{genai.NewContentFromText("assistant output", genai.RoleModel)},
		}
		ensureUserContent(req)
		if got := req.Contents[len(req.Contents)-1].Role; got != genai.RoleUser {
			t.Fatalf("last role = %q, want user", got)
		}
	})
}

func TestGenAI2XAIChatOptions(t *testing.T) {
	t.Parallel()

	temp := float32(0.7)
	topP := float32(0.8)
	maxTokens := int32(32)
	stop := []string{"END"}

	cfg := &genai.GenerateContentConfig{
		Temperature:      &temp,
		TopP:             &topP,
		MaxOutputTokens:  maxTokens,
		StopSequences:    stop,
		ResponseLogprobs: true,
	}

	req := &xaipb.GetCompletionsRequest{}
	session := &xai.ChatSession{}

	opt := adapter.GenAI2XAIChatOptions(cfg)
	if opt == nil {
		t.Fatal("genAI2XAIChatOptions returned nil")
	}

	opt(req, session)

	if req.Temperature == nil || req.GetTemperature() != temp {
		t.Fatalf("temperature = %v, want %v", req.GetTemperature(), temp)
	}
	if req.TopP == nil || req.GetTopP() != topP {
		t.Fatalf("topP = %v, want %v", req.GetTopP(), topP)
	}
	if req.MaxTokens == nil || req.GetMaxTokens() != maxTokens {
		t.Fatalf("maxTokens = %v, want %v", req.GetMaxTokens(), maxTokens)
	}
	if got := req.GetStop(); !cmp.Equal(got, stop) {
		t.Fatalf("stop sequences = %v, want %v", got, stop)
	}
	if !req.GetLogprobs() {
		t.Fatal("logprobs not set")
	}
}

type stubChatServer struct {
	xaipb.UnimplementedChatServer

	completionResp *xaipb.GetChatCompletionResponse
	chunks         []*xaipb.GetChatCompletionChunk
	lastRequest    *xaipb.GetCompletionsRequest
}

func (s *stubChatServer) cloneRequest(req *xaipb.GetCompletionsRequest) *xaipb.GetCompletionsRequest {
	if req == nil {
		return nil
	}
	return proto.Clone(req).(*xaipb.GetCompletionsRequest)
}

func (s *stubChatServer) GetCompletion(ctx context.Context, req *xaipb.GetCompletionsRequest) (*xaipb.GetChatCompletionResponse, error) {
	s.lastRequest = s.cloneRequest(req)
	return proto.Clone(s.completionResp).(*xaipb.GetChatCompletionResponse), nil
}

func (s *stubChatServer) GetCompletionChunk(req *xaipb.GetCompletionsRequest, stream grpc.ServerStreamingServer[xaipb.GetChatCompletionChunk]) error {
	s.lastRequest = s.cloneRequest(req)
	for _, ch := range s.chunks {
		if err := stream.Send(proto.Clone(ch).(*xaipb.GetChatCompletionChunk)); err != nil {
			return err
		}
	}
	return nil
}
