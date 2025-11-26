# Contributor Guide — model/xai

Go SDK for xAI services using gRPC. Port of [xai-org/xai-sdk-python](https://github.com/xai-org/xai-sdk-python).

## Project Structure

```
model/xai/
├── api/v1/              # Protobuf-generated gRPC types (vtprotobuf)
├── management_api/v1/   # Management API protos (billing)
├── shared/              # Shared proto types (billing, analytics)
├── internal/grpccodec/  # Custom gRPC codec (performance-optimized)
├── examples/            # Usage examples (each has own go.mod)
├── hack/                # Code generation scripts
└── vendor/              # Vendored dependencies
```

Key source files: `client.go` (main entry), `chat.go`, `tools.go`, `tool_calls.go`, `chat_options.go`.

## Build & Test Commands

```bash
# Run tests with race detection
../../tools/bin/gotestsum -f standard-verbose -- -race -count=1 -shuffle=on -cover ./...

# Run benchmarks
/Users/zchee/bin/gobench -timeout (n)h -count (n) -benchtime (n)x -bench=. ./...

# Run linter
../../tools/bin/golangci-lint run -v ./...

# Regenerate protos (requires buf)
cd hack && python3 gen-xai-protos.py
```

## Coding Style

- **License header**: Apache 2.0 required (see `doc.go` for template)
- **Imports**: stdlib → third-party → `github.com/zchee/tumix/model/xai`
- **Formatters**: gofmt, gofumpt (extra-rules), goimports
- **Proto alias**: `xaipb "github.com/zchee/tumix/model/xai/api/v1"`
- **JSON**: Use `encoding/json/v2` for marshal/unmarshal (GOEXPERIMENT=jsonv2 enabled)
- **Line length**: Max 250, recommended 120

## Testing

- Framework: Standard `testing` package
- Naming: `TestFunctionName`, `BenchmarkFunctionName`
- Pattern: Arrange → Act → Assert
- Use `t.Fatalf` for critical failures, `t.Errorf` for non-fatal

```go
func TestResponseProcessChunk(t *testing.T) {
    resp := newResponse(&xaipb.GetChatCompletionResponse{}, &idx)
    chunk := &xaipb.GetChatCompletionChunk{...}
    resp.processChunk(chunk)
    if got := resp.Content(); got != expected {
        t.Fatalf("content mismatch: %q", got)
    }
}
```

## Commit Messages

Format: `model/xai: description` (lowercase, imperative, no period)

Examples from history:
```
model/xai: cache chunk accessors
model/xai: precompute chunk stats
model/xai: optimize encoding/json/v2 marshaling
model/xai/internal/grpccodec: improve CodecV2 performance
```

## Performance Notes

This module emphasizes zero-allocation and high-throughput patterns:
- Pre-size slices and maps when capacity is known
- Reuse buffers via pools (`sync.Pool`, custom buffer pools)
- Use `vtprotobuf` for faster proto marshaling
- Run `go test -bench` before/after changes to validate performance
