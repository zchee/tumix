# Project: xAI (Go Port)

## Overview

This project is a Go language port of the [xai-org/xai-sdk-python](https://github.com/xai-org/xai-sdk-python). It aims to provide a Go SDK for interacting with xAI's services.

## Project Structure

- `go.mod`: Go module definition.
- `README.md`: Project documentation.

## Building and Running

As this is a standard Go project, the following commands are expected to be used once source code is added:

### Prerequisites

- Go 1.25 or later.

### Commands

- **Build:** `go build ./...`
- **Test:** `go test ./...`
- **Download Dependencies:** `go mod download` (or `go mod tidy`)

## Development Conventions

- **Language:** Go (Golang)
- **Style:** Adhere to standard Go formatting (`gofmt`, `gofumpt`, `goimports-rereviser -project-name=github.com/zchee/tumix/gollm/xai -use-cache -cache-fast-skip -format -rm-unused -set-alias ./...`) and idioms.
- **Documentation:** Public exported types and functions should be documented.
