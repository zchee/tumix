package main

import (
	"context"
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

	structuredOutput(ctx, client)
	toolCalling(ctx, client)
}

func structuredOutput(ctx context.Context, client *xai.Client) {
	type Todo struct {
		Title string   `json:"title"`
		Steps []string `json:"steps"`
	}

	chat := client.Chat.Create(
		"grok-4",
		xai.WithJSONStruct[Todo](),
		xai.WithMessages(
			xai.User("Return a JSON with a title and three concise steps for learning Go."),
		),
	)

	resp, todo, err := xai.ParseInto[Todo](ctx, chat)
	if err != nil {
		log.Fatalf("structured parse: %v", err)
	}
	fmt.Printf("Structured response (id=%s)\n", resp.Proto().GetId())
	fmt.Printf("Title: %s\n", todo.Title)
	for i, step := range todo.Steps {
		fmt.Printf("%d. %s\n", i+1, step)
	}
}

func toolCalling(ctx context.Context, client *xai.Client) {
	weatherTool := xai.MustTool("get_weather", "Return the current weather for a city", struct {
		City string `json:"city"`
		Unit string `json:"unit"`
	}{})

	chat := client.Chat.Create(
		"grok-4",
		xai.WithTools(weatherTool),
		xai.WithToolChoice(xai.RequiredTool("get_weather")),
		xai.WithMessages(xai.User("What's the weather in Tokyo in Celsius?")),
	)

	resp, err := chat.Sample(ctx)
	if err != nil {
		log.Fatalf("tool call sample: %v", err)
	}

	toolCalls := resp.ToolCalls()
	if len(toolCalls) == 0 {
		fmt.Println("Model did not request any tools; try rerunning.")
		return
	}

	for _, tc := range toolCalls {
		var args struct {
			City string `json:"city"`
			Unit string `json:"unit"`
		}
		if err := xai.ToolCallArguments(tc, &args); err != nil {
			log.Fatalf("decode tool args: %v", err)
		}
		if fn := tc.GetFunction(); fn != nil {
			fmt.Printf("Tool requested: %s %+v\n", fn.GetName(), args)
		}

		// Pretend we executed the tool and append a tool result message.
		chat.AppendToolResultJSON(tc.GetId(), map[string]any{
			"city":    args.City,
			"unit":    args.Unit,
			"summary": "18C, clear skies",
		})
	}

	finalResp, err := chat.Sample(ctx)
	if err != nil {
		log.Fatalf("final chat after tool result: %v", err)
	}
	fmt.Printf("Final reply: %s\n", finalResp.Content())
}
