package main

import (
	"fmt"
	"log"
	"os"
	"time"

	"github.com/zchee/tumix/model/xai"
	"github.com/zchee/tumix/model/xai/examples/internal/exampleutil"
)

func main() {
	if os.Getenv("XAI_MANAGEMENT_KEY") == "" {
		log.Println("collections example requires XAI_MANAGEMENT_KEY; skipping")
		return
	}

	ctx, cancel := exampleutil.Context()
	defer cancel()

	client, cleanup, err := exampleutil.NewClient(ctx)
	if err != nil {
		log.Fatalf("create client: %v", err)
	}
	defer cleanup()

	name := fmt.Sprintf("example-collection-%d", time.Now().Unix())
	chunkCfg := xai.ChunkConfigTokens(400, 40, "cl100k_base", true, true)
	coll, err := client.Collections.Create(ctx, name, "grok-embed", chunkCfg)
	if err != nil {
		log.Fatalf("create collection: %v", err)
	}
	fmt.Printf("collection created id=%s name=%s\n", coll.GetCollectionId(), coll.GetCollectionName())
	defer client.Collections.Delete(ctx, coll.GetCollectionId())

	data := []byte("Rust is a systems programming language focused on safety and speed.")
	doc, err := client.Collections.UploadDocument(ctx, coll.GetCollectionId(), "rust.txt", data, "text/plain", map[string]string{"topic": "rust"})
	if err != nil {
		log.Fatalf("upload document: %v", err)
	}
	if meta := doc.GetFileMetadata(); meta != nil {
		fmt.Printf("document stored file_id=%s size=%d\n", meta.GetFileId(), meta.GetSizeBytes())
	}

	search, err := client.Collections.Search(ctx, "What is Rust?", []string{coll.GetCollectionId()}, xai.WithSearchLimit(3))
	if err != nil {
		log.Fatalf("search: %v", err)
	}

	for i, match := range search.GetMatches() {
		fmt.Printf("%d) score=%.3f chunk=%q\n", i+1, match.GetScore(), match.GetChunkContent())
	}
}
