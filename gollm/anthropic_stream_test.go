package gollm

import (
	json "encoding/json/v2"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

func TestAnthropicStreamAggregator_TextAndStop(t *testing.T) {
	t.Parallel()

	agg := newAnthropicStreamAggregator()

	startEvt := anthropic.MessageStreamEventUnion{}
	_ = json.Unmarshal([]byte(`{
	  "type":"message_start",
	  "message":{
	    "id":"m1",
	    "role":"assistant",
	    "content":[{"type":"text","text":""}]
	  }
	}`), &startEvt)

	resp, err := agg.Process(startEvt)
	if err != nil {
		t.Fatalf("Process start: %v", err)
	}

	delta := anthropic.MessageStreamEventUnion{}
	_ = json.Unmarshal([]byte(`{
	  "type": "content_block_delta",
	  "index": 0,
	  "delta": {"type": "text_delta", "text": "Hi"}
	}`), &delta)
	if err != nil {
		t.Fatalf("Process delta: %v", err)
	}
	if len(resp) == 1 {
		if !resp[0].Partial || resp[0].Content.Parts[0].Text != "Hi" {
			t.Fatalf("partial resp = %+v", resp)
		}
	}

	stop := anthropic.MessageStreamEventUnion{}
	_ = json.Unmarshal([]byte(`{"type":"message_stop"}`), &stop)
	resp, err = agg.Process(stop)
	if err != nil {
		t.Fatalf("Process stop: %v", err)
	}
	if len(resp) != 1 {
		t.Fatalf("stop resp len = %d", len(resp))
	}
	final := resp[0]
	if final.Partial {
		t.Fatalf("final should not be partial")
	}
	if final.FinishReason != genai.FinishReasonUnspecified && !final.TurnComplete {
		t.Fatalf("TurnComplete false for finished response")
	}
	if diff := cmp.Diff(genai.RoleModel, final.Content.Role); diff != "" {
		t.Fatalf("role diff: %s", diff)
	}
}
