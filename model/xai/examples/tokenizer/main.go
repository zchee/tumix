package main

import (
	"fmt"
	"log"

	"github.com/zchee/tumix/model/xai/examples/internal/exampleutil"
)

func main() {
	ctx, cancel := exampleutil.Context()
	defer cancel()

	client, cleanup, err := exampleutil.NewClient()
	if err != nil {
		log.Fatalf("create client: %v", err)
	}
	defer cleanup()

	text := "Tokenize this short sentence to count tokens."
	resp, err := client.Tokenizer.Tokenize(ctx, text, "grok-4")
	if err != nil {
		log.Fatalf("tokenize: %v", err)
	}

	fmt.Printf("model=%s tokens=%d\n", resp.GetModel(), len(resp.GetTokens()))
	if tokens := resp.GetTokens(); len(tokens) > 0 {
		fmt.Printf("first token ids: %v\n", tokens[:min(len(tokens), 8)])
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
