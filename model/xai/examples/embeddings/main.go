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

package main

import (
	"fmt"
	"log"

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

	inputs := []string{"hello world", "embeddings are vector representations"}
	resp, err := client.Embed.CreateStrings(ctx, "grok-embed", inputs)
	if err != nil {
		log.Fatalf("embed: %v", err)
	}

	fmt.Printf("embeddings generated: %d (model %s)\n", len(resp.GetEmbeddings()), resp.GetModel())
	if len(resp.GetEmbeddings()) > 0 && len(resp.GetEmbeddings()[0].GetEmbeddings()) > 0 {
		vec := resp.GetEmbeddings()[0].GetEmbeddings()[0].GetFloatArray()
		fmt.Printf("first vector length: %d\n", len(vec))
	}
}
