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

package agent

import "testing"

func TestComputeStats(t *testing.T) {
	answers := []candidateAnswer{
		{
			Agent: "a",
			Text:  "foo",
		},
		{
			Agent: "b",
			Text:  "foo",
		},
		{
			Agent: "c",
			Text:  "bar",
		},
	}
	stats := computeStats(answers, 5)
	if stats.voteMargin <= 0.0 {
		t.Fatalf("expected positive vote margin, got %f", stats.voteMargin)
	}
	if stats.unique != 2 {
		t.Fatalf("unique want 2, got %d", stats.unique)
	}
	if stats.coverage <= 0 || stats.coverage > 1 {
		t.Fatalf("coverage out of range: %f", stats.coverage)
	}
	if stats.topAnswer != "foo" {
		t.Fatalf("top answer foo, got %s", stats.topAnswer)
	}
}

func TestMajorityVoteNormalizesDelimiters(t *testing.T) {
	ans := []candidateAnswer{
		{
			Agent: "a",
			Text:  "<<<foo>>> ",
		},
		{
			Agent: "b",
			Text:  "foo",
		},
		{
			Agent: "c",
			Text:  "<<<foo >>>",
		},
	}
	answer, conf := majorityVote(ans)
	if answer != "foo" {
		t.Fatalf("expected normalized foo, got %s", answer)
	}
	if conf < 0.66 {
		t.Fatalf("unexpected confidence %f", conf)
	}
}
