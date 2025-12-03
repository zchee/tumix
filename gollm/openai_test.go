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
	"bytes"
	json "encoding/json/v2"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestOpenAILLM_Generate(t *testing.T) {
	syncResp := mockResponse(
		"resp-sync",
		"The capital of France is Paris.",
		13,
		16,
		"completed",
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" && r.URL.Path != "/v1/responses" {
			w.WriteHeader(http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		raw, err := json.Marshal(syncResp)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		if _, err := w.Write(raw); err != nil {
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	t.Cleanup(server.Close)

	llm, err := NewOpenAILLM(t.Context(), AuthMethodAPIKey("test-key"), "gpt-5-mini",
		option.WithHTTPClient(server.Client()),
		option.WithBaseURL(server.URL),
	)
	if err != nil {
		t.Fatalf("NewOpenAILLM() error = %v", err)
	}

	var got *model.LLMResponse
	for resp, err := range llm.GenerateContent(t.Context(), &model.LLMRequest{
		Contents: genai.Text("What is the capital of France?"),
		Config:   &genai.GenerateContentConfig{},
	}, false) {
		if err != nil {
			t.Fatalf("GenerateContent() unexpected error: %v", err)
		}
		got = resp
	}

	want := &model.LLMResponse{
		Content: genai.NewContentFromText("The capital of France is Paris.", genai.RoleModel),
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     13,
			CandidatesTokenCount: 16,
			TotalTokenCount:      29,
		},
		FinishReason: genai.FinishReasonStop,
		TurnComplete: true,
	}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("GenerateContent() diff (-want +got):\n%s", diff)
	}
}

func TestOpenAILLM_GenerateStream(t *testing.T) {
	streamResp := mockResponse("resp-stream", "Paris", 7, 5, "completed")

	sseBody := bytes.NewBuffer(nil)
	if _, err := sseBody.WriteString("data: {\"type\":\"response.output_text.delta\",\"delta\":\"Par\",\"content_index\":0,\"item_id\":\"\",\"output_index\":0,\"sequence_number\":1,\"logprobs\":[]}\n\n"); err != nil {
		t.Fatalf("WriteString delta1: %v", err)
	}
	if _, err := sseBody.WriteString("data: {\"type\":\"response.output_text.delta\",\"delta\":\"is\",\"content_index\":0,\"item_id\":\"\",\"output_index\":0,\"sequence_number\":2,\"logprobs\":[]}\n\n"); err != nil {
		t.Fatalf("WriteString delta2: %v", err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" && r.URL.Path != "/v1/responses" {
			w.WriteHeader(http.StatusNotFound)
			return
		}

		var body map[string]any
		reqBytes, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read request body: %v", err)
		}
		if err := json.Unmarshal(reqBytes, &body); err != nil {
			t.Fatalf("Decode body: %v", err)
		}
		if stream, ok := body["stream"].(bool); ok && stream {
			w.Header().Set("Content-Type", "text/event-stream")
			if _, err := w.Write(sseBody.Bytes()); err != nil {
				t.Fatalf("write sse body: %v", err)
			}
			return
		}

		w.Header().Set("Content-Type", "application/json")
		raw, err := json.Marshal(streamResp)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		if _, err := w.Write(raw); err != nil {
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	t.Cleanup(server.Close)

	llm, err := NewOpenAILLM(t.Context(), AuthMethodAPIKey("test-key"), "gpt-5-mini",
		option.WithHTTPClient(server.Client()),
		option.WithBaseURL(server.URL),
	)
	if err != nil {
		t.Fatalf("NewOpenAILLM() error = %v", err)
	}

	got, err := readResponse(llm.GenerateContent(t.Context(), &model.LLMRequest{
		Contents: genai.Text("What is the capital of France? One word."),
		Config:   &genai.GenerateContentConfig{},
	}, true))
	if err != nil {
		t.Fatalf("GenerateStream() error = %v", err)
	}

	if diff := cmp.Diff("Paris", got.PartialText); diff != "" {
		t.Errorf("GenerateStream() partial diff (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff("Paris", got.FinalText); diff != "" {
		t.Errorf("GenerateStream() final diff (-want +got):\n%s", diff)
	}
}

func mockResponse(id, text string, promptTokens, completionTokens int64, status string) *responses.Response {
	return &responses.Response{
		ID:        id,
		CreatedAt: 1,
		Error: responses.ResponseError{
			Code:    "",
			Message: "",
		},
		IncompleteDetails: responses.ResponseIncompleteDetails{},
		Instructions:      responses.ResponseInstructionsUnion{OfString: ""},
		Metadata:          shared.Metadata{},
		Model:             shared.ResponsesModel("gpt-5-mini"),
		Object:            constant.ValueOf[constant.Response](),
		Output: []responses.ResponseOutputItemUnion{{
			Type:    "message",
			Role:    constant.ValueOf[constant.Assistant](),
			Content: []responses.ResponseOutputMessageContentUnion{{Type: "output_text", Text: text}},
		}},
		ParallelToolCalls: false,
		Temperature:       0,
		ToolChoice:        responses.ResponseToolChoiceUnion{OfToolChoiceMode: responses.ToolChoiceOptionsAuto},
		Tools:             []responses.ToolUnion{},
		TopP:              1,
		Status:            responses.ResponseStatus(status),
		Text:              responses.ResponseTextConfig{},
		Usage: responses.ResponseUsage{
			InputTokens:  promptTokens,
			OutputTokens: completionTokens,
			TotalTokens:  promptTokens + completionTokens,
			InputTokensDetails: responses.ResponseUsageInputTokensDetails{
				CachedTokens: 0,
			},
			OutputTokensDetails: responses.ResponseUsageOutputTokensDetails{
				ReasoningTokens: 0,
			},
		},
	}
}
