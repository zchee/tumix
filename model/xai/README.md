# xAI Go Client

## Overview

This package is a Go language port of the [xai-org/xai-sdk-python]. It provides a Go SDK for interacting with xAI's services using gRPC.

## Project Structure

- `api/`: Generated gRPC code from protocol buffers.
- `client.go`: Main `Client` implementation.
- `chat.go`: Chat API client.
- `models.go`: Models API client.
- `files.go`: Files API client.
- `image.go`: Image API client.
- `tokenizer.go`: Tokenizer API client.
- `collections.go`: Collections/document management (management API).
- `tools.go`: Helpers for model/tool calling and search sources.

## Usage

### Installation

```bash
go get github.com/zchee/tumix/model/xai
```

### Authentication

The SDK reads `XAI_API_KEY` by default. The management endpoints (collections) use `XAI_MANAGEMENT_KEY` when present.

You can override hosts, metadata, and timeouts via functional options such as `WithAPIHost`, `WithManagementAPIHost`, `WithTimeout`, and `WithMetadata`.

### Example: Chat

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/zchee/tumix/model/xai"
)

func main() {
    ctx := context.Background()
    client, err := xai.NewClient(ctx, "") // uses XAI_API_KEY from the environment
    if err != nil {
        log.Fatalf("Failed to create client: %v", err)
    }
    defer client.Close()

    chat := client.Chat.Create("grok-4-1-fast-reasoning", xai.WithMessages(xai.System("You are a helpful assistant.")))
    chat.Append(xai.User("Hello!"))

    resp, err := chat.Sample(ctx)
    if err != nil {
        log.Fatalf("Failed to get response: %v", err)
    }

    fmt.Printf("Response: %s\n", resp.Content())
}
```

#### Additional examples:

- **Files**: `client.Files.Upload(ctx, "./doc.pdf")` uploads with chunked streaming; `client.Files.Content` streams bytes back.
- **Images**: `client.Image.Sample(ctx, "a cat in space", "grok-2-image-1212", xai.WithImageFormat(xai.ImageFormatBase64))`.
- **Collections** (requires management key): create/list/update collections and documents via `client.Collections` APIs.
- **Tools/Search**: build server-side tools with `WebSearchTool`, `XSearchTool`, `CodeExecutionTool`, and search sources via helpers in `search.go`.

## Development

### Generating Protos

The protocol buffer definitions were reconstructed from the Python SDK's descriptors.
To regenerate the code, you need `protoc` and the descriptor set (not included in git).

### Building

```bash
go build ./...
```

### Testing

```bash
go test ./...
```

[xai-org/xai-sdk-python]: https://github.com/xai-org/xai-sdk-python
