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

	_ "modernc.org/sqlite"

	"google.golang.org/adk/session"
)

// Service returns a sqlite-backed session.Service stored at file path.
func Service(path string) (session.Service, error) {
	if path == "" {
		return nil, errors.New("sessiondb: path required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("sessiondb: mkdir: %w", err)
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	if _, err := db.Exec(`CREATE TABLE IF NOT EXISTS sessions (
		key TEXT PRIMARY KEY,
		blob BLOB NOT NULL
	)`); err != nil {
		return nil, err
	}
	return &store{db: db}, nil
}

type store struct {
	db *sql.DB
	mu sync.Mutex
}

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

func (s *store) Create(_ context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
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
	if err := s.put(k, ps, true); err != nil {
		return nil, err
	}
	return &session.CreateResponse{Session: &dbSession{s: ps}}, nil
}

func (s *store) Get(_ context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	ps, err := s.load(req.AppName, req.UserID, req.SessionID)
	if err != nil {
		return nil, err
	}
	return &session.GetResponse{Session: &dbSession{s: ps}}, nil
}

func (s *store) List(_ context.Context, req *session.ListRequest) (*session.ListResponse, error) {
	if req.AppName == "" {
		return nil, errors.New("sessiondb: app_name required")
	}
	rows, err := s.db.Query(`SELECT blob FROM sessions WHERE key LIKE ?`, req.AppName+"/%")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var sessions []session.Session
	for rows.Next() {
		var b []byte
		if err := rows.Scan(&b); err != nil {
			return nil, err
		}
		ps := &persistSession{}
		if err := json.Unmarshal(b, ps); err != nil {
			return nil, err
		}
		if req.UserID != "" && ps.UserID != req.UserID {
			continue
		}
		sessions = append(sessions, &dbSession{s: ps})
	}
	return &session.ListResponse{Sessions: sessions}, nil
}

func (s *store) Delete(_ context.Context, req *session.DeleteRequest) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.db.Exec(`DELETE FROM sessions WHERE key = ?`, s.key(req.AppName, req.UserID, req.SessionID))
	return err
}

func (s *store) AppendEvent(_ context.Context, sess session.Session, ev *session.Event) error {
	if ev == nil || sess == nil {
		return errors.New("sessiondb: event and session required")
	}
	if ev.Partial {
		return nil
	}
	ps, err := s.load(sess.AppName(), sess.UserID(), sess.ID())
	if err != nil {
		return err
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
	return s.put(s.key(ps.AppName, ps.UserID, ps.SessionID), ps, false)
}

// helpers

func (s *store) put(key string, ps *persistSession, failIfExists bool) error {
	b, err := json.Marshal(ps)
	if err != nil {
		return err
	}
	op := "INSERT OR REPLACE"
	if failIfExists {
		op = "INSERT"
	}
	_, err = s.db.Exec(op+` INTO sessions(key, blob) VALUES(?,?)`, key, b)
	return err
}

func (s *store) load(app, user, sid string) (*persistSession, error) {
	if app == "" || user == "" || sid == "" {
		return nil, errors.New("sessiondb: app/user/session required")
	}
	row := s.db.QueryRow(`SELECT blob FROM sessions WHERE key = ?`, s.key(app, user, sid))
	var b []byte
	if err := row.Scan(&b); err != nil {
		return nil, err
	}
	ps := &persistSession{}
	if err := json.Unmarshal(b, ps); err != nil {
		return nil, err
	}
	return ps, nil
}

// dbSession implements session.Session using in-memory struct.
type dbSession struct{ s *persistSession }

func (d *dbSession) ID() string                { return d.s.SessionID }
func (d *dbSession) AppName() string           { return d.s.AppName }
func (d *dbSession) UserID() string            { return d.s.UserID }
func (d *dbSession) LastUpdateTime() time.Time { return d.s.UpdatedAt }
func (d *dbSession) Events() session.Events    { return sessionfsEvents(d.s.Events) }
func (d *dbSession) State() session.State      { return sessionfsState{state: d.s.State} }

// reuse lightweight implementations from sessionfs style.
type sessionfsEvents []*session.Event

func (e sessionfsEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, ev := range e {
			if !yield(ev) {
				return
			}
		}
	}
}
func (e sessionfsEvents) Len() int { return len(e) }
func (e sessionfsEvents) At(i int) *session.Event {
	if i < 0 || i >= len(e) {
		return nil
	}
	return e[i]
}

type sessionfsState struct{ state map[string]any }

func (s sessionfsState) Get(key string) (any, error) {
	val, ok := s.state[key]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return val, nil
}

func (s sessionfsState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range s.state {
			if !yield(k, v) {
				return
			}
		}
	}
}

func (s sessionfsState) Set(key string, value any) error {
	s.state[key] = value
	return nil
}

var _ session.Service = (*store)(nil)
