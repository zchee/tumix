package main

import (
	"fmt"
	"log"

	"github.com/zchee/tumix/model/xai/examples/internal/exampleutil"
)

func main() {
	ctx, cancel := exampleutil.Context()
	defer cancel()

	client, cleanup, err := exampleutil.NewClient(ctx)
	if err != nil {
		log.Fatalf("create client: %v", err)
	}
	defer cleanup()

	inputs := []string{"hello world", "embeddings are vector representations"}
	resp, err := client.Embed.CreateStrings(ctx, "grok-embed", inputs)
	if err != nil {
		log.Fatalf("embed: %v", err)
	}

	fmt.Printf("embeddings generated: %d (model %s)\n", len(resp.GetEmbeddings()), resp.GetModel())
	if len(resp.GetEmbeddings()) > 0 && len(resp.GetEmbeddings()[0].GetEmbeddings()) > 0 {
		vec := resp.GetEmbeddings()[0].GetEmbeddings()[0].GetFloatArray()
		fmt.Printf("first vector length: %d\n", len(vec))
	}
}
