package main

import (
	"fmt"
	"log"
	"os"

	"github.com/zchee/tumix/model/xai"
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

	tmp, err := os.CreateTemp("", "xai-example-*.txt")
	if err != nil {
		log.Fatalf("create temp file: %v", err)
	}
	defer os.Remove(tmp.Name())

	content := []byte("Example document for xAI file upload.\nSecond line for good measure.\n")
	if _, err := tmp.Write(content); err != nil {
		log.Fatalf("write temp file: %v", err)
	}
	if err := tmp.Close(); err != nil {
		log.Fatalf("close temp file: %v", err)
	}

	fmt.Printf("Uploading %s (%d bytes)\n", tmp.Name(), len(content))
	uploaded, err := client.Files.Upload(ctx, tmp.Name(), xai.WithProgress(func(done, total int64) {
		if total > 0 {
			fmt.Printf("progress: %d/%d bytes\n", done, total)
		}
	}))
	if err != nil {
		log.Fatalf("upload: %v", err)
	}
	fmt.Printf("File stored with id=%s size=%d\n", uploaded.GetId(), uploaded.GetSize())

	downloaded, err := client.Files.Content(ctx, uploaded.GetId())
	if err != nil {
		log.Fatalf("download content: %v", err)
	}
	fmt.Printf("Downloaded %d bytes. First line: %q\n", len(downloaded), firstLine(string(downloaded)))

	// Clean up to avoid leaving stray uploads around.
	if _, err := client.Files.Delete(ctx, uploaded.GetId()); err != nil {
		log.Fatalf("delete file: %v", err)
	}
	fmt.Println("File deleted.")
}

func firstLine(s string) string {
	for i, r := range s {
		if r == '\n' {
			return s[:i]
		}
	}
	return s
}
