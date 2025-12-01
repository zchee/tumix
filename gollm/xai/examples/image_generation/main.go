package main

import (
	"fmt"
	"log"

	"github.com/zchee/tumix/gollm/xai"
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

	prompt := "A small robot watering a bonsai tree, watercolor style"

	img, err := client.Image.Sample(ctx, prompt, "grok-2-image-1212", xai.WithImageFormat(xai.ImageFormatURL))
	if err != nil {
		log.Fatalf("generate image: %v", err)
	}

	if url, err := img.URL(); err == nil {
		fmt.Printf("image url: %s\n", url)
	}

	data, err := img.Data(ctx)
	if err != nil {
		log.Fatalf("download image data: %v", err)
	}
	fmt.Printf("downloaded image bytes: %d\n", len(data))

	// Request base64 output as an alternative format.
	b64Img, err := client.Image.Sample(ctx, prompt, "grok-2-image-1212", xai.WithImageFormat(xai.ImageFormatBase64))
	if err != nil {
		log.Fatalf("generate base64 image: %v", err)
	}
	b64, err := b64Img.Base64()
	if err != nil {
		log.Fatalf("base64 retrieval: %v", err)
	}
	fmt.Printf("base64 prefix: %.32s...\n", b64)
}
