package exampleutil

import (
	"context"
	"time"

	"github.com/zchee/tumix/model/xai"
)

const defaultTimeout = 2 * time.Minute

// Context returns a cancellable context with a sensible default timeout
// so examples do not hang indefinitely when network calls stall.
func Context() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), defaultTimeout)
}

// NewClient constructs an xAI client using environment variables for keys
// and returns a cleanup function to close the underlying connections.
func NewClient(ctx context.Context) (*xai.Client, func(), error) {
	client, err := xai.NewClient(ctx, "")
	if err != nil {
		return nil, nil, err
	}
	cleanup := func() { _ = client.Close() }
	return client, cleanup, nil
}
