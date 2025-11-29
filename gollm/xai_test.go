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
	json "encoding/json/v2"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"

	"github.com/zchee/tumix/gollm/internal/adapter"
	"github.com/zchee/tumix/gollm/xai"
	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
)

func TestXAIModel_Generate(t *testing.T) {
	t.Parallel()

	temp := float32(0.0)
	req := &model.LLMRequest{
		Contents: genai.Text("What is the capital of France?"),
		Config: &genai.GenerateContentConfig{
			Temperature: &temp,
			HTTPOptions: &genai.HTTPOptions{Headers: make(http.Header)},
		},
	}

	fakeResp := &xaipb.GetChatCompletionResponse{
		Model:             "grok-4-1-fast-reasoning",
		SystemFingerprint: "fp-123",
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
			CompletionTokens:       2,
			PromptTokens:           10,
			TotalTokens:            12,
			CachedPromptTextTokens: 1,
		},
	}

	server := &stubChatServer{completionResp: fakeResp}
	m, cleanup := newTestXAIModel(t, server, "grok-1")
	defer cleanup()

	var got *model.LLMResponse
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	for resp, err := range m.GenerateContent(ctx, req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() unexpected error: %v", err)
		}
		got = resp
	}
	if got == nil {
		t.Fatal("GenerateContent() returned no response")
	}

	want := &model.LLMResponse{
		Content: genai.NewContentFromText("Paris", genai.RoleModel),
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			CachedContentTokenCount: 1,
			CandidatesTokenCount:    2,
			PromptTokenCount:        10,
			TotalTokenCount:         12,
		},
		CustomMetadata: map[string]any{
			"xai_finish_reason":      "REASON_STOP",
			"xai_system_fingerprint": "fp-123",
		},
		FinishReason: genai.FinishReasonStop,
	}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("GenerateContent() diff (-want +got):\n%s", diff)
	}

	if ua := req.Config.HTTPOptions.Headers.Get("User-Agent"); ua != "tumix/test go1.25" {
		t.Fatalf("User-Agent header = %q, want %q", ua, "tumix/test go1.25")
	}

	if server.lastRequest == nil {
		t.Fatal("GetCompletion was not invoked")
	}
	gotText := server.lastRequest.GetMessages()[0].GetContent()[0].GetText()
	if gotText != "What is the capital of France?" {
		t.Fatalf("request message text = %q, want %q", gotText, "What is the capital of France?")
	}
	if gotRole := server.lastRequest.GetMessages()[0].GetRole(); gotRole != xaipb.MessageRole_ROLE_USER {
		t.Fatalf("request message role = %v, want %v", gotRole, xaipb.MessageRole_ROLE_USER)
	}
}

func TestXAIModel_GenerateStream(t *testing.T) {
	t.Parallel()

	temp := float32(0.0)
	req := &model.LLMRequest{
		Contents: genai.Text("Stream please"),
		Config: &genai.GenerateContentConfig{
			Temperature: &temp,
		},
	}

	chunks := []*xaipb.GetChatCompletionChunk{
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
			SystemFingerprint: "fp-123",
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
	}

	server := &stubChatServer{chunks: chunks}
	m, cleanup := newTestXAIModel(t, server, "grok-4-1-fast-reasoning")
	defer cleanup()

	var partialTexts []string
	var finalTexts []string
	var finishReasons []genai.FinishReason

	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	for resp, err := range m.GenerateContent(ctx, req, true) {
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

	if len(partialTexts) < 2 {
		t.Fatalf("partial texts too short: %v", partialTexts)
	}
	if partialTexts[0] != "Par" {
		t.Fatalf("first partial = %q, want %q", partialTexts[0], "Par")
	}
	if got := partialTexts[len(partialTexts)-1]; got != "Paris" {
		t.Fatalf("last partial = %q, want %q", got, "Paris")
	}
	if diff := cmp.Diff([]string{"Paris"}, finalTexts); diff != "" {
		t.Fatalf("final texts diff (-want +got):\n%s", diff)
	}
	if len(finishReasons) == 0 || finishReasons[len(finishReasons)-1] != genai.FinishReasonStop {
		t.Fatalf("finish reasons = %v, want last reason FinishReasonStop", finishReasons)
	}
}

func TestXAIModel_StreamThoughtsAndToolCalls(t *testing.T) {
	t.Parallel()

	req := &model.LLMRequest{
		Contents: genai.Text("Call a tool with reasoning"),
		Config:   &genai.GenerateContentConfig{},
	}

	chunks := []*xaipb.GetChatCompletionChunk{
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
			SystemFingerprint: "fp-xyz",
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
	}

	server := &stubChatServer{chunks: chunks}
	m, cleanup := newTestXAIModel(t, server, "grok-4-1-fast-reasoning")
	defer cleanup()

	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	var (
		thoughts     []string
		texts        []string
		toolCalls    []*genai.FunctionCall
		finishCodes  []genai.FinishReason
		turnComplete bool
	)

	for resp, err := range m.GenerateContent(ctx, req, true) {
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
		t.Fatalf("thoughts = %v, want first thought \"Thinking \"", thoughts)
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

func TestXAIModel_SystemInstructionMapped(t *testing.T) {
	t.Parallel()

	req := &model.LLMRequest{
		Contents: genai.Text("Hello"),
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("stay brief", "system"),
			HTTPOptions:       &genai.HTTPOptions{Headers: make(http.Header)},
		},
	}

	fakeResp := &xaipb.GetChatCompletionResponse{
		Outputs: []*xaipb.CompletionOutput{
			{
				Index:        0,
				FinishReason: xaipb.FinishReason_REASON_STOP,
				Message: &xaipb.CompletionMessage{
					Role:    xaipb.MessageRole_ROLE_ASSISTANT,
					Content: "Hi!",
				},
			},
		},
	}

	server := &stubChatServer{completionResp: fakeResp}
	m, cleanup := newTestXAIModel(t, server, "grok-1")
	defer cleanup()

	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	for range m.GenerateContent(ctx, req, false) {
	}

	msgs := server.lastRequest.GetMessages()
	if len(msgs) < 2 {
		t.Fatalf("messages len = %d, want >=2", len(msgs))
	}

	sys := msgs[0]
	if sys.GetRole() != xaipb.MessageRole_ROLE_SYSTEM {
		t.Fatalf("system role = %v, want %v", sys.GetRole(), xaipb.MessageRole_ROLE_SYSTEM)
	}
	if got := sys.GetContent()[0].GetText(); got != "stay brief" {
		t.Fatalf("system text = %q, want %q", got, "stay brief")
	}
}

func TestXAIModel_FunctionCallConversion(t *testing.T) {
	t.Parallel()

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("call tool", genai.RoleUser),
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "call-1",
							Name: "get_weather",
							Args: map[string]any{"city": "Tokyo"},
						},
					},
				},
			},
		},
		Config: &genai.GenerateContentConfig{},
	}

	fakeResp := &xaipb.GetChatCompletionResponse{
		Outputs: []*xaipb.CompletionOutput{
			{
				Index:        0,
				FinishReason: xaipb.FinishReason_REASON_STOP,
				Message: &xaipb.CompletionMessage{
					Role:    xaipb.MessageRole_ROLE_ASSISTANT,
					Content: "ok",
				},
			},
		},
	}

	server := &stubChatServer{completionResp: fakeResp}
	m, cleanup := newTestXAIModel(t, server, "grok-4-1-fast-reasoning")
	defer cleanup()

	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	for range m.GenerateContent(ctx, req, false) {
	}

	var assistant *xaipb.Message
	for _, msg := range server.lastRequest.GetMessages() {
		if msg.GetRole() == xaipb.MessageRole_ROLE_ASSISTANT && len(msg.GetToolCalls()) > 0 {
			assistant = msg
			break
		}
	}
	if assistant == nil {
		t.Fatalf("assistant message with tool calls not found in %+v", server.lastRequest.GetMessages())
	}

	call := assistant.GetToolCalls()[0].GetFunction()
	if call.GetName() != "get_weather" {
		t.Fatalf("function name = %q, want %q", call.GetName(), "get_weather")
	}
	args := map[string]any{}
	if err := json.Unmarshal([]byte(call.GetArguments()), &args); err != nil {
		t.Fatalf("unmarshal args: %v", err)
	}
	if got := args["city"]; got != "Tokyo" {
		t.Fatalf("args[city] = %v, want %q", got, "Tokyo")
	}
	if assistant.GetToolCalls()[0].GetId() != "call-1" {
		t.Fatalf("tool call id = %q, want call-1", assistant.GetToolCalls()[0].GetId())
	}
}

func TestXAIModel_FunctionResponseConversion(t *testing.T) {
	t.Parallel()

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "tool",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:       "call-2",
							Name:     "get_weather",
							Response: map[string]any{"ok": true},
						},
					},
				},
			},
		},
		Config: &genai.GenerateContentConfig{},
	}

	fakeResp := &xaipb.GetChatCompletionResponse{
		Outputs: []*xaipb.CompletionOutput{
			{
				Index:        0,
				FinishReason: xaipb.FinishReason_REASON_STOP,
				Message: &xaipb.CompletionMessage{
					Role:    xaipb.MessageRole_ROLE_ASSISTANT,
					Content: "ack",
				},
			},
		},
	}

	server := &stubChatServer{completionResp: fakeResp}
	m, cleanup := newTestXAIModel(t, server, "grok-1")
	defer cleanup()

	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()

	for range m.GenerateContent(ctx, req, false) {
	}

	var toolMsg *xaipb.Message
	for _, msg := range server.lastRequest.GetMessages() {
		if msg.GetRole() == xaipb.MessageRole_ROLE_TOOL {
			toolMsg = msg
			break
		}
	}
	if toolMsg == nil {
		t.Fatalf("tool message not found in %+v", server.lastRequest.GetMessages())
	}
	if len(toolMsg.GetContent()) == 0 {
		t.Fatalf("tool message has no content: %+v", toolMsg)
	}

	payload := toolMsg.GetContent()[0].GetText()
	got := map[string]any{}
	if err := json.Unmarshal([]byte(payload), &got); err != nil {
		t.Fatalf("unmarshal tool payload: %v", err)
	}
	if got["name"] != "get_weather" {
		t.Fatalf("payload name = %v, want get_weather", got["name"])
	}
	if got["tool_call_id"] != "call-2" {
		t.Fatalf("payload tool_call_id = %v, want call-2", got["tool_call_id"])
	}
	if resp, ok := got["response"].(map[string]any); !ok || resp["ok"] != true {
		t.Fatalf("payload response = %v, want map[ok:true]", got["response"])
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

func newTestXAIModel(t *testing.T, server *stubChatServer, modelName string) (m *xaiLLM, cleanup func()) {
	t.Helper()

	var lc net.ListenConfig
	lis, err := lc.Listen(t.Context(), "tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	xaipb.RegisterChatServer(grpcServer, server)

	go func() {
		_ = grpcServer.Serve(lis)
	}()

	dialer := &net.Dialer{}
	dialFunc := func(ctx context.Context, _ string) (net.Conn, error) {
		return dialer.DialContext(ctx, "tcp", lis.Addr().String())
	}

	client, err := xai.NewClient("test-key",
		xai.WithAPIHost(lis.Addr().String()),
		xai.WithInsecure(),
		xai.WithTimeout(0),
		xai.WithDialOptions(grpc.WithContextDialer(dialFunc)),
	)
	if err != nil {
		grpcServer.Stop()
		_ = lis.Close()
		t.Fatalf("NewClient: %v", err)
	}

	cleanup = func() {
		_ = client.Close()
		grpcServer.Stop()
		_ = lis.Close()
	}

	return &xaiLLM{
		client:    client,
		name:      modelName,
		userAgent: "tumix/test go1.25",
	}, cleanup
}
