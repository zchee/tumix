package main

import (
	"fmt"
	"log"

	"github.com/zchee/tumix/model/xai"
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

	basicChat := client.Chat.Create(
		"grok-4-1-fast-reasoning",
		xai.WithMessages(
			xai.System("You are a concise assistant."),
			xai.User("Name one practical productivity tip."),
		),
	)
	fmt.Printf("basicChat: %v\n", basicChat)

	resp, err := basicChat.Sample(ctx)
	if err != nil {
		log.Fatalf("chat sample: %v", err)
	}
	fmt.Printf("Basic reply: %s\n", resp.Content())

	// Streaming follow-up on the same conversation.
	basicChat.Append(xai.User("Summarize that answer in five words."))
	stream, err := basicChat.Stream(ctx)
	if err != nil {
		log.Fatalf("chat stream: %v", err)
	}
	defer stream.Close()

	fmt.Print("Streaming reply: ")
	lastLen := 0
	for resp, err := range stream.Recv() {
		if err != nil {
			log.Fatalf("stream recv: %v", err)
		}
		if resp != nil {
			content := resp.Content()
			if len(content) > lastLen {
				fmt.Print(content[lastLen:])
				lastLen = len(content)
			}
		}
	}

	fmt.Println()
	fmt.Printf("finish reason: %s\n", stream.Response().FinishReason())
}
