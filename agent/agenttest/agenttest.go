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

// Package agenttest provides in-memory implementations of agent-related interfaces for testing.
package agenttest

import (
	"context"
	"errors"
	"iter"
	"maps"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

// InMemoryState is a simple in-memory implementation of [session.State] for testing.
type InMemoryState struct {
	values map[string]any
	GetErr error
	SetErr error
}

var _ session.State = (*InMemoryState)(nil)

// NewInMemoryState creates a new [InMemoryState] with the given initial values.
func NewInMemoryState(v map[string]any) *InMemoryState {
	vv := make(map[string]any, len(v))
	maps.Copy(vv, v)
	return &InMemoryState{
		values: vv,
	}
}

// Get implements [session.State].
func (s *InMemoryState) Get(key string) (any, error) {
	if s.GetErr != nil {
		return nil, s.GetErr
	}
	val, ok := s.values[key]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return val, nil
}

// Set implements [session.State].
func (s *InMemoryState) Set(key string, value any) error {
	if s.SetErr != nil {
		return s.SetErr
	}
	s.values[key] = value
	return nil
}

// All implements [session.State].
func (s *InMemoryState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range s.values {
			if !yield(k, v) {
				return
			}
		}
	}
}

// InMemoryEvents is a simple in-memory implementation of [session.Events] for testing.
type InMemoryEvents struct {
	events []*session.Event
}

var _ session.Events = (*InMemoryEvents)(nil)

// All implements [session.Events].
func (e *InMemoryEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, event := range e.events {
			if !yield(event) {
				return
			}
		}
	}
}

// Len implements [session.Events].
func (e *InMemoryEvents) Len() int {
	return len(e.events)
}

// At implements [session.Events].
func (e *InMemoryEvents) At(i int) *session.Event {
	return e.events[i]
}

// InMemorySession is a simple in-memory implementation of [session.Session] for testing.
type InMemorySession struct {
	id             string
	appName        string
	userID         string
	state          session.State
	events         session.Events
	lastUpdateTime time.Time
}

var _ session.Session = (*InMemorySession)(nil)

// NewInMemorySession creates a new [InMemorySession] with the given parameters.
func NewInMemorySession(id, appName, userID string, state session.State, events session.Events, lastUpdateTime time.Time) session.Session {
	return &InMemorySession{
		id:             id,
		appName:        appName,
		userID:         userID,
		state:          state,
		events:         events,
		lastUpdateTime: lastUpdateTime,
	}
}

// ID implements [session.Session].
func (s *InMemorySession) ID() string {
	return s.id
}

// AppName implements [session.Session].
func (s *InMemorySession) AppName() string {
	return s.appName
}

// UserID implements [session.Session].
func (s *InMemorySession) UserID() string {
	return s.userID
}

// State implements [session.Session].
func (s *InMemorySession) State() session.State {
	return s.state
}

// Events implements [session.Session].
func (s *InMemorySession) Events() session.Events {
	return s.events
}

// LastUpdateTime implements [session.Session].
func (s *InMemorySession) LastUpdateTime() time.Time {
	return s.lastUpdateTime
}

// SessionInvocationContext is a simple implementation of [agent.InvocationContext] for testing.
type SessionInvocationContext struct {
	context.Context

	session session.Session
}

var _ agent.InvocationContext = (*SessionInvocationContext)(nil)

// NewSessionInvocationContext creates a new [SessionInvocationContext] with the given session.
func NewSessionInvocationContext(ctx context.Context, sess session.Session) agent.InvocationContext {
	return &SessionInvocationContext{
		Context: ctx,
		session: sess,
	}
}

// Agent implements [agent.InvocationContext].
func (c *SessionInvocationContext) Agent() agent.Agent { return nil }

// Artifacts implements [agent.InvocationContext].
func (c *SessionInvocationContext) Artifacts() agent.Artifacts { return nil }

// Memory implements [agent.InvocationContext].
func (c *SessionInvocationContext) Memory() agent.Memory { return nil }

// Session implements [agent.InvocationContext].
func (c *SessionInvocationContext) Session() session.Session { return c.session }

// InvocationID implements [agent.InvocationContext].
func (c *SessionInvocationContext) InvocationID() string { return "invocation" }

// Branch implements [agent.InvocationContext].
func (c *SessionInvocationContext) Branch() string { return "branch" }

// UserContent implements [agent.InvocationContext].
func (c *SessionInvocationContext) UserContent() *genai.Content { return nil }

// RunConfig implements [agent.InvocationContext].
func (c *SessionInvocationContext) RunConfig() *agent.RunConfig { return &agent.RunConfig{} }

// EndInvocation implements [agent.InvocationContext].
func (c *SessionInvocationContext) EndInvocation() {}

// Ended implements [agent.InvocationContext].
func (c *SessionInvocationContext) Ended() bool { return false }

// ToolContext is a simple implementation of [tool.Context] for testing.
type ToolContext struct {
	context.Context

	state         session.State
	actions       *session.EventActions
	functionCall  string
	readonlyState session.ReadonlyState
}

var _ tool.Context = (*ToolContext)(nil)

// NewToolContext creates a new [ToolContext] with the given parameters.
func NewToolContext(ctx context.Context, state session.State, actions *session.EventActions, funcCall string, readonlyState session.ReadonlyState) tool.Context {
	return &ToolContext{
		Context:       ctx,
		state:         state,
		actions:       actions,
		functionCall:  funcCall,
		readonlyState: readonlyState,
	}
}

// UserContent implements [tool.Context].
func (c *ToolContext) UserContent() *genai.Content { return nil }

// InvocationID implements [tool.Context].
func (c *ToolContext) InvocationID() string { return "invocation" }

// AgentName implements [tool.Context].
func (c *ToolContext) AgentName() string { return "agent" }

// ReadonlyState implements [tool.Context].
func (c *ToolContext) ReadonlyState() session.ReadonlyState {
	if c.readonlyState != nil {
		return c.readonlyState
	}
	return c.state
}

// UserID implements [tool.Context].
func (c *ToolContext) UserID() string { return "user" }

// AppName implements [tool.Context].
func (c *ToolContext) AppName() string { return "app" }

// SessionID implements [tool.Context].
func (c *ToolContext) SessionID() string { return "session" }

// Branch implements [tool.Context].
func (c *ToolContext) Branch() string { return "branch" }

// Artifacts implements [tool.Context].
func (c *ToolContext) Artifacts() agent.Artifacts { return nil }

// State implements [tool.Context].
func (c *ToolContext) State() session.State { return c.state }

// FunctionCallID implements [tool.Context].
func (c *ToolContext) FunctionCallID() string { return c.functionCall }

// Actions implements [tool.Context].
func (c *ToolContext) Actions() *session.EventActions { return c.actions }

// SearchMemory implements [tool.Context].
func (c *ToolContext) SearchMemory(context.Context, string) (*memory.SearchResponse, error) {
	return nil, errors.New("SearchMemory is not supported by ToolContext")
}
