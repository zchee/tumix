package main

import (
	"fmt"
	"log"

	"github.com/zchee/tumix/model/xai"
	xaipb "github.com/zchee/tumix/model/xai/api/v1"
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

	params := xai.SearchParameters{
		Sources: []*xaipb.Source{
			xai.WebSource("US", nil, nil, true),
			xai.NewsSource("US", nil, true),
		},
		Mode:             xai.SearchModeOn,
		ReturnCitations:  true,
		MaxSearchResults: 5,
	}

	chat := client.Chat.Create(
		"grok-4",
		xai.WithSearch(params),
		xai.WithMessages(xai.User("Give me two bullet updates on recent Starship testing.")),
	)

	resp, err := chat.Sample(ctx)
	if err != nil {
		log.Fatalf("chat with search: %v", err)
	}

	fmt.Printf("Response:\n%s\n", resp.Content())
	if citations := resp.Citations(); len(citations) > 0 {
		fmt.Printf("Citations: %v\n", citations)
	}
}
