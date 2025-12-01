package main

import (
	"errors"
	"fmt"
	"io"
	"log"

	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
	"github.com/zchee/tumix/gollm/xai/examples/internal/exampleutil"
)

func main() {
	ctx, cancel := exampleutil.Context()
	defer cancel()

	client, cleanup, err := exampleutil.NewClient()
	if err != nil {
		log.Fatalf("create client: %v", err)
	}
	defer cleanup()

	req := &xaipb.SampleTextRequest{
		Model:  "grok-4",
		Prompt: []string{"Give me a single-sentence fun fact about Mars."},
	}

	resp, err := client.Sampler.SampleText(ctx, req)
	if err != nil {
		log.Fatalf("sample text: %v", err)
	}
	if len(resp.GetChoices()) > 0 {
		fmt.Printf("single response: %s\n", resp.GetChoices()[0].GetText())
	}

	stream, err := client.Sampler.SampleTextStreaming(ctx, req)
	if err != nil {
		log.Fatalf("sample text streaming: %v", err)
	}

	fmt.Println("streaming responses:")
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			log.Fatalf("stream recv: %v", err)
		}
		if len(chunk.GetChoices()) > 0 {
			fmt.Printf("- %s\n", chunk.GetChoices()[0].GetText())
		}
	}
}
