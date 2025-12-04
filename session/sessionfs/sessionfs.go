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

// Package sessionfs provides a lightweight file-backed implementation of
// adk's session.Service. It keeps an in-memory map and snapshots it to disk
// after each mutation. Concurrency is guarded with an in-process mutex only;
// cross-process synchronization is not provided.
package sessionfs

import (
	"context"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"iter"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"google.golang.org/adk/session"
)

// Service returns a file-backed session service rooted at dir.
func Service(dir string) (session.Service, error) {
	if dir == "" {
		return nil, errors.New("sessionfs: dir is required")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("sessionfs: mkdir %s: %w", dir, err)
	}
	lockFile, err := os.OpenFile(filepath.Join(dir, "sessions.lock"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, fmt.Errorf("sessionfs: lock file: %w", err)
	}
	if err := syscall.Flock(int(lockFile.Fd()), syscall.LOCK_EX); err != nil {
		return nil, fmt.Errorf("sessionfs: flock: %w", err)
	}
	fs := &fileService{root: dir, sessions: make(map[string]*persistSession), lockFile: lockFile}
	if err := fs.load(); err != nil {
		return nil, err
	}
	return fs, nil
}

type fileService struct {
	mu       sync.RWMutex
	root     string
	sessions map[string]*persistSession
	lockFile *os.File
}

// persistSession is the on-disk representation.
type persistSession struct {
	AppName   string           `json:"app_name"`
	UserID    string           `json:"user_id"`
	SessionID string           `json:"session_id"`
	State     map[string]any   `json:"state"`
	Events    []*session.Event `json:"events"`
	UpdatedAt time.Time        `json:"updated_at"`
}

// fileSession implements session.Session.
type fileSession struct {
	s  *persistSession
	mu *sync.RWMutex
}

func (f *fileSession) ID() string { return f.s.SessionID }

func (f *fileSession) AppName() string { return f.s.AppName }

func (f *fileSession) UserID() string { return f.s.UserID }

func (f *fileSession) LastUpdateTime() time.Time { return f.s.UpdatedAt }

func (f *fileSession) Events() session.Events { return fileEvents{f} }

func (f *fileSession) State() session.State { return &fileState{mu: f.mu, state: f.s.State} }

type fileEvents struct{ fs *fileSession }

func (e fileEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		e.fs.mu.RLock()
		defer e.fs.mu.RUnlock()
		for _, ev := range e.fs.s.Events {
			if !yield(ev) {
				return
			}
		}
	}
}

func (e fileEvents) Len() int {
	e.fs.mu.RLock()
	defer e.fs.mu.RUnlock()
	return len(e.fs.s.Events)
}

func (e fileEvents) At(i int) *session.Event {
	e.fs.mu.RLock()
	defer e.fs.mu.RUnlock()
	if i < 0 || i >= len(e.fs.s.Events) {
		return nil
	}
	return e.fs.s.Events[i]
}

type fileState struct {
	mu    *sync.RWMutex
	state map[string]any
}

func (s *fileState) Get(key string) (any, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	val, ok := s.state[key]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return val, nil
}

func (s *fileState) Set(key string, value any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state[key] = value
	return nil
}

func (s *fileState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		s.mu.RLock()
		for k, v := range s.state {
			s.mu.RUnlock()
			if !yield(k, v) {
				return
			}
			s.mu.RLock()
		}
		s.mu.RUnlock()
	}
}

func (f *fileService) Create(_ context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	if req.AppName == "" || req.UserID == "" {
		return nil, fmt.Errorf("sessionfs: app_name and user_id are required")
	}
	id := req.SessionID
	if id == "" {
		id = fmt.Sprintf("%d", time.Now().UnixNano())
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	key := f.key(req.AppName, req.UserID, id)
	if _, exists := f.sessions[key]; exists {
		return nil, fmt.Errorf("sessionfs: session %s exists", id)
	}

	st := req.State
	if st == nil {
		st = make(map[string]any)
	}
	ps := &persistSession{
		AppName:   req.AppName,
		UserID:    req.UserID,
		SessionID: id,
		State:     st,
		UpdatedAt: time.Now(),
	}
	f.sessions[key] = ps
	if err := f.saveLocked(); err != nil {
		return nil, err
	}

	return &session.CreateResponse{Session: &fileSession{s: ps, mu: &f.mu}}, nil
}

func (f *fileService) Get(_ context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	if req.AppName == "" || req.UserID == "" || req.SessionID == "" {
		return nil, errors.New("sessionfs: app_name, user_id, session_id required")
	}
	f.mu.RLock()
	ps, ok := f.sessions[f.key(req.AppName, req.UserID, req.SessionID)]
	f.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("sessionfs: session %s not found", req.SessionID)
	}

	return &session.GetResponse{Session: &fileSession{s: ps, mu: &f.mu}}, nil
}

func (f *fileService) List(_ context.Context, req *session.ListRequest) (*session.ListResponse, error) {
	if req.AppName == "" {
		return nil, errors.New("sessionfs: app_name required")
	}
	f.mu.RLock()
	defer f.mu.RUnlock()
	out := make([]session.Session, 0)
	prefix := req.AppName + "/"
	for k, ps := range f.sessions {
		if !strings.HasPrefix(k, prefix) {
			continue
		}
		if req.UserID != "" && ps.UserID != req.UserID {
			continue
		}
		out = append(out, &fileSession{s: ps, mu: &f.mu})
	}
	return &session.ListResponse{Sessions: out}, nil
}

func (f *fileService) Delete(_ context.Context, req *session.DeleteRequest) error {
	if req.AppName == "" || req.UserID == "" || req.SessionID == "" {
		return errors.New("sessionfs: app_name, user_id, session_id required")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.sessions, f.key(req.AppName, req.UserID, req.SessionID))
	return f.saveLocked()
}

func (f *fileService) AppendEvent(_ context.Context, sess session.Session, ev *session.Event) error {
	if sess == nil || ev == nil {
		return errors.New("sessionfs: session and event required")
	}
	if ev.Partial {
		return nil
	}

	fsess, ok := sess.(*fileSession)
	if !ok {
		return fmt.Errorf("sessionfs: unexpected session type %T", sess)
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	key := f.key(fsess.AppName(), fsess.UserID(), fsess.ID())
	ps, ok := f.sessions[key]
	if !ok {
		return fmt.Errorf("sessionfs: session %s missing", fsess.ID())
	}

	trimmed := trimTemp(ev)
	applyState(ps, trimmed)
	ps.Events = append(ps.Events, trimmed)
	ps.UpdatedAt = trimmed.Timestamp

	return f.saveLocked()
}

// --- helpers ---

func (f *fileService) key(app, user, sessionID string) string {
	return filepath.Join(app, user, sessionID)
}

func (f *fileService) load() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	dataPath := filepath.Join(f.root, "sessions.json")
	data, err := os.ReadFile(dataPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return fmt.Errorf("sessionfs: read: %w", err)
	}
	if len(data) == 0 {
		return nil
	}
	if err := json.Unmarshal(data, &f.sessions); err != nil {
		return fmt.Errorf("sessionfs: unmarshal sessions: %w", err)
	}
	return nil
}

func (f *fileService) saveLocked() error {
	dataPath := filepath.Join(f.root, "sessions.json")
	tmp := dataPath + ".tmp"
	data, err := json.Marshal(f.sessions)
	if err != nil {
		return fmt.Errorf("sessionfs: marshal: %w", err)
	}
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return fmt.Errorf("sessionfs: write tmp: %w", err)
	}
	if err := os.Rename(tmp, dataPath); err != nil {
		return fmt.Errorf("sessionfs: rename: %w", err)
	}
	return nil
}

func trimTemp(ev *session.Event) *session.Event {
	if len(ev.Actions.StateDelta) == 0 {
		return ev
	}
	filtered := make(map[string]any)
	for k, v := range ev.Actions.StateDelta {
		if !strings.HasPrefix(k, session.KeyPrefixTemp) {
			filtered[k] = v
		}
	}
	ev.Actions.StateDelta = filtered
	return ev
}

func applyState(ps *persistSession, ev *session.Event) {
	if ev.Actions.StateDelta == nil {
		return
	}
	if ps.State == nil {
		ps.State = make(map[string]any)
	}
	for k, v := range ev.Actions.StateDelta {
		if strings.HasPrefix(k, session.KeyPrefixTemp) {
			continue
		}
		ps.State[k] = v
	}
}

var _ session.Service = (*fileService)(nil)
