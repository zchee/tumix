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

import (
	"context"
	"errors"
	"iter"
	"maps"
	"time"

	adkagent "google.golang.org/adk/agent"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

type mapState struct {
	values map[string]any
	getErr error
	setErr error
}

var _ session.State = (*mapState)(nil)

func newMapState(values map[string]any) *mapState {
	copied := make(map[string]any, len(values))
	maps.Copy(copied, values)
	return &mapState{
		values: copied,
	}
}

func (s *mapState) Get(key string) (any, error) {
	if s.getErr != nil {
		return nil, s.getErr
	}
	val, ok := s.values[key]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return val, nil
}

func (s *mapState) Set(key string, value any) error {
	if s.setErr != nil {
		return s.setErr
	}
	s.values[key] = value
	return nil
}

func (s *mapState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range s.values {
			if !yield(k, v) {
				return
			}
		}
	}
}

type sliceEvents struct {
	events []*session.Event
}

var _ session.Events = (*sliceEvents)(nil)

func (e *sliceEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, event := range e.events {
			if !yield(event) {
				return
			}
		}
	}
}

func (e *sliceEvents) Len() int {
	return len(e.events)
}

func (e *sliceEvents) At(i int) *session.Event {
	return e.events[i]
}

type stubSession struct {
	id             string
	appName        string
	userID         string
	state          session.State
	events         session.Events
	lastUpdateTime time.Time
}

var _ session.Session = (*stubSession)(nil)

func (s *stubSession) ID() string {
	return s.id
}

func (s *stubSession) AppName() string {
	return s.appName
}

func (s *stubSession) UserID() string {
	return s.userID
}

func (s *stubSession) State() session.State {
	return s.state
}

func (s *stubSession) Events() session.Events {
	return s.events
}

func (s *stubSession) LastUpdateTime() time.Time {
	return s.lastUpdateTime
}

type stubInvocationContext struct {
	context.Context
	session session.Session
}

var _ adkagent.InvocationContext = (*stubInvocationContext)(nil)

func (c *stubInvocationContext) Agent() adkagent.Agent {
	return nil
}

func (c *stubInvocationContext) Artifacts() adkagent.Artifacts {
	return nil
}

func (c *stubInvocationContext) Memory() adkagent.Memory {
	return nil
}

func (c *stubInvocationContext) Session() session.Session {
	return c.session
}

func (c *stubInvocationContext) InvocationID() string {
	return "invocation"
}

func (c *stubInvocationContext) Branch() string {
	return "branch"
}

func (c *stubInvocationContext) UserContent() *genai.Content {
	return nil
}

func (c *stubInvocationContext) RunConfig() *adkagent.RunConfig {
	return &adkagent.RunConfig{}
}

func (c *stubInvocationContext) EndInvocation() {}

func (c *stubInvocationContext) Ended() bool { return false }

type stubToolContext struct {
	context.Context

	state         session.State
	actions       *session.EventActions
	functionCall  string
	readonlyState session.ReadonlyState
}

var _ tool.Context = (*stubToolContext)(nil)

func (c *stubToolContext) UserContent() *genai.Content { return nil }

func (c *stubToolContext) InvocationID() string { return "invocation" }

func (c *stubToolContext) AgentName() string { return "agent" }

func (c *stubToolContext) ReadonlyState() session.ReadonlyState {
	if c.readonlyState != nil {
		return c.readonlyState
	}
	return c.state
}

func (c *stubToolContext) UserID() string { return "user" }

func (c *stubToolContext) AppName() string { return "app" }

func (c *stubToolContext) SessionID() string { return "session" }

func (c *stubToolContext) Branch() string { return "branch" }

func (c *stubToolContext) Artifacts() adkagent.Artifacts { return nil }

func (c *stubToolContext) State() session.State { return c.state }

func (c *stubToolContext) FunctionCallID() string { return c.functionCall }

func (c *stubToolContext) Actions() *session.EventActions { return c.actions }

func (c *stubToolContext) SearchMemory(context.Context, string) (*memory.SearchResponse, error) {
	return nil, errors.New("SearchMemory is not supported by stubToolContext")
}
