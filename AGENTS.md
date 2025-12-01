# Contributor Guide

## Important

- This codebase is in early development. So you can plan for large-scale refactoring.
- MUST READ the following files in the context of the ADK (Agent Development Kit):
  - .agents/llms/adk-go.xml
  - .agents/llms/adk-docs.xml
  - .agents/llms/go-genai.xml

## Project Structure

```
.
├── model/xai/           # xAI API client library (separate Go module)
│   ├── api/v1/          # Protobuf-generated API types
│   ├── examples/        # Usage examples (each with own go.mod)
│   ├── internal/        # Internal packages
│   └── vendor/          # Vendored dependencies
├── tools/               # Development tools (separate go.mod)
│   └── bin/             # Tool binaries (gotestsum, golangci-lint, buf)
├── vendor/              # Root module vendored dependencies
└── .github/workflows/   # CI configuration
```

## Build, Test, and Development Commands

```bash
# Install development tools
GOBIN=$PWD/tools/bin go -C tools install -v -x tool

# Run tests (root module)
./tools/bin/gotestsum -f standard-verbose -- -race -count=1 -shuffle=on -cover -covermode=atomic ./...

# Run tests (model/xai module)
cd model/xai && ../../tools/bin/gotestsum -f standard-verbose -- -race -count=1 -shuffle=on -cover ./...

# Run linter
./tools/bin/golangci-lint run -v ./...

# Run go vet
go vet ./...
```

## Coding Style & Naming Conventions

- **Formatters**: gofmt, gofumpt (extra-rules), goimports
- **Import order**: Standard library → third-party → local (`github.com/zchee/tumix`)
- **License header**: Apache 2.0 required on all source files (see `.golangci.yaml` goheader)
- **Line length**: 250 max, 120 recommended
- **Naming**: Standard Go conventions (CamelCase exports, camelCase unexported)

Run formatting:
```bash
./tools/bin/golangci-lint run --fix ./...
```

## Testing Guidelines

- **Framework**: Standard `testing` package with `gotestsum` runner
- **Naming**: `TestFunctionName`, `BenchmarkFunctionName`
- **Flags**: `-race -count=1 -shuffle=on -cover`
- **Coverage**: Required via `-coverprofile=coverage.out`

Example test structure:
```go
func TestResponseProcessChunk(t *testing.T) {
    // Arrange → Act → Assert pattern
}
```

## Commit & Pull Request Guidelines

**Commit message format**: `scope: description`

- Scope examples: `model/xai`, `bench`, `github/workflows`, `tools`
- Use lowercase, no period at end
- Keep description concise and imperative

Examples from history:
```
model/xai: preallocate span response attributes
bench: add chat session stream aggregation benchmark
github/workflows: add lint job
```

**Pull requests**: Include a "Why" section explaining motivation (see `.github/PULL_REQUEST_TEMPLATE.md`).

## CI Pipeline

Three jobs run on push/PR:
1. **test**: Runs unit tests with race detection and coverage
2. **lint**: Runs golangci-lint
3. **vet**: Runs go vet

All must pass before merge. Coverage uploaded to Codecov.
