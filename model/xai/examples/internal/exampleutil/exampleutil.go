// Copyright 2025 The tumix Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

// Package exampleutil provides utility functions for xAI examples.
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
func NewClient() (*xai.Client, func(), error) {
	client, err := xai.NewClient("")
	if err != nil {
		return nil, nil, err
	}
	cleanup := func() { _ = client.Close() }
	return client, cleanup, nil
}
