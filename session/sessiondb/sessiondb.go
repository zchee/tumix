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

package sessiondb

import (
	context "context"
	"database/sql"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"iter"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"google.golang.org/adk/session"

	_ "modernc.org/sqlite"
)

// Service returns a sqlite-backed [session.Service] stored at file path.
func Service(ctx context.Context, path string) (session.Service, error) {
	if path == "" {
		return nil, errors.New("sessiondb: path required")
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("sessiondb: mkdir: %w", err)
	}

	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("sessiondb: open sqlite: %w", err)
	}

	if _, err := db.ExecContext(ctx, `CREATE TABLE IF NOT EXISTS sessions (
		key TEXT PRIMARY KEY,
		blob BLOB NOT NULL
	)`); err != nil {
		return nil, fmt.Errorf("sessiondb: create table: %w", err)
	}

	return &store{db: db}, nil
}

// store implements [session.Service] using sqlite.
type store struct {
	db *sql.DB
	mu sync.Mutex
}

var _ session.Service = (*store)(nil)

type persistSession struct {
	AppName   string           `json:"app_name"`
	UserID    string           `json:"user_id"`
	SessionID string           `json:"session_id"`
	State     map[string]any   `json:"state"`
	Events    []*session.Event `json:"events"`
	UpdatedAt time.Time        `json:"updated_at"`
}

func (s *store) key(app, user, sessionID string) string {
	return fmt.Sprintf("%s/%s/%s", app, user, sessionID)
}

// Create implements [session.Service].
func (s *store) Create(ctx context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	if req.AppName == "" || req.UserID == "" {
		return nil, errors.New("sessiondb: app_name and user_id required")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if req.SessionID == "" {
		req.SessionID = fmt.Sprintf("%d", time.Now().UnixNano())
	}

	k := s.key(req.AppName, req.UserID, req.SessionID)
	ps := &persistSession{
		AppName:   req.AppName,
		UserID:    req.UserID,
		SessionID: req.SessionID,
		State:     map[string]any{},
		UpdatedAt: time.Now(),
	}
	if req.State != nil {
		ps.State = req.State
	}

	if err := s.put(ctx, k, ps, true); err != nil {
		return nil, err
	}

	return &session.CreateResponse{
		Session: &dbSession{
			s: ps,
		},
	}, nil
}

// Get implements [session.Service].
func (s *store) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	ps, err := s.load(ctx, req.AppName, req.UserID, req.SessionID)
	if err != nil {
		return nil, err
	}

	return &session.GetResponse{
		Session: &dbSession{
			s: ps,
		},
	}, nil
}

// List implements [session.Service].
func (s *store) List(ctx context.Context, req *session.ListRequest) (*session.ListResponse, error) {
	if req.AppName == "" {
		return nil, errors.New("sessiondb: app_name required")
	}

	rows, err := s.db.QueryContext(ctx, `SELECT blob FROM sessions WHERE key LIKE ?`, req.AppName+"/%")
	if err != nil {
		return nil, fmt.Errorf("sessiondb: list query: %w", err)
	}
	defer rows.Close()

	var sessions []session.Session
	for rows.Next() {
		var b []byte
		if err := rows.Scan(&b); err != nil {
			return nil, fmt.Errorf("sessiondb: scan: %w", err)
		}
		ps := &persistSession{}
		if err := json.Unmarshal(b, ps); err != nil {
			return nil, fmt.Errorf("sessiondb: unmarshal session: %w", err)
		}
		if req.UserID != "" && ps.UserID != req.UserID {
			continue
		}
		sessions = append(sessions, &dbSession{
			s: ps,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sessiondb: iterate rows: %w", err)
	}

	return &session.ListResponse{
		Sessions: sessions,
	}, nil
}

// Delete implements [session.Service].
func (s *store) Delete(ctx context.Context, req *session.DeleteRequest) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, err := s.db.ExecContext(ctx, `DELETE FROM sessions WHERE key = ?`, s.key(req.AppName, req.UserID, req.SessionID)); err != nil {
		return fmt.Errorf("sessiondb: delete: %w", err)
	}

	return nil
}

// AppendEvent implements [session.Service].
func (s *store) AppendEvent(ctx context.Context, sess session.Session, ev *session.Event) error {
	if ev == nil || sess == nil {
		return errors.New("sessiondb: event and session required")
	}

	if ev.Partial {
		return nil
	}

	ps, err := s.load(ctx, sess.AppName(), sess.UserID(), sess.ID())
	if err != nil {
		return fmt.Errorf("sessiondb: load session: %w", err)
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

	ps.Events = append(ps.Events, ev)
	ps.UpdatedAt = ev.Timestamp

	return s.put(ctx, s.key(ps.AppName, ps.UserID, ps.SessionID), ps, false)
}

func (s *store) put(ctx context.Context, key string, ps *persistSession, failIfExists bool) error {
	b, err := json.Marshal(ps)
	if err != nil {
		return fmt.Errorf("sessiondb: marshal: %w", err)
	}

	op := `INSERT OR REPLACE`
	if failIfExists {
		op = `INSERT`
	}

	if _, err := s.db.ExecContext(ctx, op+` INTO sessions(key, blob) VALUES(?,?)`, key, b); err != nil {
		return fmt.Errorf("sessiondb: upsert: %w", err)
	}

	return nil
}

func (s *store) load(ctx context.Context, app, user, sid string) (*persistSession, error) {
	if app == "" || user == "" || sid == "" {
		return nil, errors.New("sessiondb: app/user/session required")
	}

	row := s.db.QueryRowContext(ctx, `SELECT blob FROM sessions WHERE key = ?`, s.key(app, user, sid))

	var b []byte
	if err := row.Scan(&b); err != nil {
		return nil, fmt.Errorf("sessiondb: scan row: %w", err)
	}

	ps := &persistSession{}
	if err := json.Unmarshal(b, ps); err != nil {
		return nil, fmt.Errorf("sessiondb: unmarshal session: %w", err)
	}

	return ps, nil
}

// dbSession implements [session.Session] using in-memory struct.
type dbSession struct {
	s *persistSession
}

var _ session.Session = (*dbSession)(nil)

// ID implements [session.Session].
func (d *dbSession) ID() string { return d.s.SessionID }

// AppName implements [session.Session].
func (d *dbSession) AppName() string { return d.s.AppName }

// UserID implements [session.Session].
func (d *dbSession) UserID() string { return d.s.UserID }

// State implements [session.Session].
func (d *dbSession) State() session.State { return sessionfsState{state: d.s.State} }

// Events implements [session.Session].
func (d *dbSession) Events() session.Events { return sessionfsEvents(d.s.Events) }

// LastUpdateTime implements [session.Session].
func (d *dbSession) LastUpdateTime() time.Time { return d.s.UpdatedAt }

// sessionfsEvents reuse lightweight implementations from sessionfs style.
type sessionfsEvents []*session.Event

var _ session.Events = sessionfsEvents{}

// All implements [session.Events].
func (e sessionfsEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, ev := range e {
			if !yield(ev) {
				return
			}
		}
	}
}

// Len implements [session.Events].
func (e sessionfsEvents) Len() int { return len(e) }

// At implements [session.Events].
func (e sessionfsEvents) At(i int) *session.Event {
	if i < 0 || i >= len(e) {
		return nil
	}
	return e[i]
}

// sessionfsState reuse lightweight implementations from sessionfs style.
type sessionfsState struct {
	state map[string]any
}

var _ session.State = (*sessionfsState)(nil)

// Get implements [session.State].
func (s sessionfsState) Get(key string) (any, error) {
	val, ok := s.state[key]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return val, nil
}

// Set implements [session.State].
func (s sessionfsState) Set(key string, value any) error {
	s.state[key] = value
	return nil
}

// All implements [session.State].
func (s sessionfsState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range s.state {
			if !yield(k, v) {
				return
			}
		}
	}
}
