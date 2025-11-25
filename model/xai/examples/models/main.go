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

	lang, err := client.Models.ListLanguageModels(ctx)
	if err != nil {
		log.Fatalf("list language models: %v", err)
	}
	fmt.Printf("language models: %v\n", names(lang.GetModels(), 5))

	embed, err := client.Models.ListEmbeddingModels(ctx)
	if err != nil {
		log.Fatalf("list embedding models: %v", err)
	}
	fmt.Printf("embedding models: %v\n", names(embed.GetModels(), 5))

	img, err := client.Models.ListImageGenerationModels(ctx)
	if err != nil {
		log.Fatalf("list image models: %v", err)
	}
	fmt.Printf("image generation models: %v\n", names(img.GetModels(), 5))
}

func names[T interface{ GetName() string }](models []T, limit int) []string {
	if len(models) == 0 {
		return nil
	}
	if len(models) > limit {
		models = models[:limit]
	}
	out := make([]string, len(models))
	for i, m := range models {
		out[i] = m.GetName()
	}
	return out
}
