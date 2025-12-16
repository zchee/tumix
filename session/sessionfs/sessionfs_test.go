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

package sessionfs

import (
	"errors"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/session"
)

func TestFileServiceLifecycleAndPersistence(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	svc, err := Service(dir)
	if err != nil {
		t.Fatalf("Service() error = %v", err)
	}
	t.Cleanup(func() {
		if fs, ok := svc.(*fileService); ok && fs.lockFile != nil {
			_ = fs.lockFile.Close()
		}
	})

	ctx := t.Context()
	app, user, sid := "app", "user", "sid"
	_, err = svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    user,
		SessionID: sid,
		State:     map[string]any{"init": 1},
	})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	ev := session.NewEvent("inv1")
	ev.Timestamp = time.Now()
	ev.Author = "author"
	ev.Actions.StateDelta = map[string]any{
		"k":                             "v",
		session.KeyPrefixTemp + "tempk": "tempv",
	}

	gotSess, err := svc.Get(ctx, &session.GetRequest{AppName: app, UserID: user, SessionID: sid})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}

	if err := svc.AppendEvent(ctx, gotSess.Session, ev); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}

	// Re-open service to verify on-disk persistence and temp-key trimming.
	if fs, ok := svc.(*fileService); ok && fs.lockFile != nil {
		_ = fs.lockFile.Close()
	}
	svc2, err := Service(dir)
	if err != nil {
		t.Fatalf("Service reload error = %v", err)
	}

	ctx = t.Context()
	got, err := svc2.Get(ctx, &session.GetRequest{AppName: app, UserID: user, SessionID: sid})
	if err != nil {
		t.Fatalf("Get reload error = %v", err)
	}

	stateVal, err := got.Session.State().Get("k")
	if err != nil {
		t.Fatalf("State get error = %v", err)
	}
	if stateVal != "v" {
		t.Fatalf("state value = %v, want v", stateVal)
	}

	if _, err := got.Session.State().Get(session.KeyPrefixTemp + "tempk"); !errors.Is(err, session.ErrStateKeyNotExist) {
		t.Fatalf("temp state key should be filtered, got err = %v", err)
	}
	if got.Session.Events().Len() != 1 {
		t.Fatalf("events len = %d, want 1", got.Session.Events().Len())
	}

	evStored := got.Session.Events().At(0)
	if evStored == nil || len(evStored.Actions.StateDelta) != 1 {
		t.Fatalf("stored event delta = %#v, want single non-temp key", evStored.Actions.StateDelta)
	}
}

func TestFileServiceListDeleteAndErrors(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	svc, err := Service(dir)
	if err != nil {
		t.Fatalf("Service() error = %v", err)
	}
	t.Cleanup(func() {
		if fs, ok := svc.(*fileService); ok && fs.lockFile != nil {
			_ = fs.lockFile.Close()
		}
	})

	ctx := t.Context()
	app := "app"
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: app, UserID: "u1", SessionID: "s1"}); err != nil {
		t.Fatalf("Create u1 s1 error = %v", err)
	}
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: app, UserID: "u2", SessionID: "s2"}); err != nil {
		t.Fatalf("Create u2 s2 error = %v", err)
	}

	listAll, err := svc.List(ctx, &session.ListRequest{AppName: app})
	if err != nil {
		t.Fatalf("List all error = %v", err)
	}
	if len(listAll.Sessions) != 2 {
		t.Fatalf("List all sessions = %d, want 2", len(listAll.Sessions))
	}

	listU1, err := svc.List(ctx, &session.ListRequest{AppName: app, UserID: "u1"})
	if err != nil {
		t.Fatalf("List user filter error = %v", err)
	}
	if len(listU1.Sessions) != 1 || listU1.Sessions[0].ID() != "s1" {
		t.Fatalf("List user filter got %+v", listU1.Sessions)
	}

	if err := svc.Delete(ctx, &session.DeleteRequest{AppName: app, UserID: "u1", SessionID: "s1"}); err != nil {
		t.Fatalf("Delete error = %v", err)
	}

	if _, err := svc.Get(ctx, &session.GetRequest{AppName: app, UserID: "u1", SessionID: "s1"}); err == nil {
		t.Fatalf("expected error after delete, got nil")
	}
}

func TestFileServiceErrorCases(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	// Invalid JSON should fail to load.
	invalidPath := filepath.Join(dir, "sessions.json")
	if err := os.WriteFile(invalidPath, []byte("{not-json"), 0o600); err != nil {
		t.Fatalf("write invalid json: %v", err)
	}
	if _, err := Service(dir); err == nil {
		t.Fatalf("Service should fail on invalid JSON")
	}

	svc, err := Service(filepath.Join(dir, "ok"))
	if err != nil {
		t.Fatalf("Service ok dir error = %v", err)
	}
	t.Cleanup(func() {
		if fs, ok := svc.(*fileService); ok && fs.lockFile != nil {
			_ = fs.lockFile.Close()
		}
	})
	ctx := t.Context()
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create error = %v", err)
	}

	err = svc.AppendEvent(ctx, created.Session, nil)
	if err == nil || err.Error() != "sessionfs: session and event required" {
		t.Fatalf("AppendEvent nil event error = %v", err)
	}

	err = svc.AppendEvent(ctx, nil, session.NewEvent("x"))
	if err == nil || err.Error() != "sessionfs: session and event required" {
		t.Fatalf("AppendEvent nil session error = %v", err)
	}

	err = svc.AppendEvent(ctx, fakeSession{}, session.NewEvent("x"))
	if err == nil || !strings.Contains(err.Error(), "unexpected session type") {
		t.Fatalf("AppendEvent unexpected session should error, got %v", err)
	}

	// Create validation.
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "", UserID: "u"}); err == nil {
		t.Fatalf("Create missing app should error")
	}

	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "a", UserID: ""}); err == nil {
		t.Fatalf("Create missing user should error")
	}

	// All iterator should traverse state map.
	st := created.Session.State()
	if err := st.Set("k", "v"); err != nil {
		t.Fatalf("State.Set error = %v", err)
	}

	want := map[string]any{
		"k": "v",
	}
	collected := maps.Collect(st.All())
	if diff := cmp.Diff(want, collected); diff != "" {
		t.Fatalf("state iteration diff (-want +got): %s", diff)
	}
}

type fakeSession struct{}

var _ session.Session = fakeSession{}

func (fakeSession) ID() string                { return "id" }
func (fakeSession) AppName() string           { return "app" }
func (fakeSession) UserID() string            { return "user" }
func (fakeSession) LastUpdateTime() time.Time { return time.Now() }
func (fakeSession) Events() session.Events    { return nil }
func (fakeSession) State() session.State      { return nil }
