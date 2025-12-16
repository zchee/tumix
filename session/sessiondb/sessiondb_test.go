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
	"errors"
	"maps"
	"path/filepath"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/session"
)

func TestSQLiteSessionLifecycleAndFilters(t *testing.T) {
	t.Parallel()

	dbPath := filepath.Join(t.TempDir(), "sessions.db")
	svc, err := Service(t.Context(), dbPath)
	if err != nil {
		t.Fatalf("Service error: %v", err)
	}

	ctx := t.Context()
	_, err = svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "u1",
		SessionID: "s1",
	})
	if err != nil {
		t.Fatalf("Create u1 s1: %v", err)
	}

	_, err = svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "u2",
		SessionID: "s2",
	})
	if err != nil {
		t.Fatalf("Create u2 s2: %v", err)
	}

	listAll, err := svc.List(ctx, &session.ListRequest{
		AppName: "app",
	})
	if err != nil {
		t.Fatalf("List all: %v", err)
	}
	if len(listAll.Sessions) != 2 {
		t.Fatalf("List all sessions = %d, want 2", len(listAll.Sessions))
	}

	listU1, err := svc.List(ctx, &session.ListRequest{
		AppName: "app",
		UserID:  "u1",
	})
	if err != nil {
		t.Fatalf("List user filter: %v", err)
	}
	if len(listU1.Sessions) != 1 || listU1.Sessions[0].ID() != "s1" {
		t.Fatalf("List user filter got %+v", listU1.Sessions)
	}

	ev := session.NewEvent("inv")
	ev.Author = "a"
	ev.Timestamp = time.Now()
	ev.Actions.StateDelta = map[string]any{
		"k":                             "v",
		session.KeyPrefixTemp + "tempk": "tempv",
	}
	got, err := svc.Get(ctx, &session.GetRequest{
		AppName:   "app",
		UserID:    "u1",
		SessionID: "s1",
	})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}

	if err := svc.AppendEvent(ctx, got.Session, ev); err != nil {
		t.Fatalf("AppendEvent: %v", err)
	}

	// Re-open to verify persistence and temp-key filtering.
	svc2, err := Service(t.Context(), dbPath)
	if err != nil {
		t.Fatalf("Service reload: %v", err)
	}

	got2, err := svc2.Get(ctx, &session.GetRequest{
		AppName:   "app",
		UserID:    "u1",
		SessionID: "s1",
	})
	if err != nil {
		t.Fatalf("Get reload: %v", err)
	}

	val, err := got2.Session.State().Get("k")
	if err != nil || val != "v" {
		t.Fatalf("state after reload = %v err %v", val, err)
	}

	if _, err := got2.Session.State().Get(session.KeyPrefixTemp + "tempk"); !errors.Is(err, session.ErrStateKeyNotExist) {
		t.Fatalf("temp key should be filtered, err=%v", err)
	}

	if got2.Session.Events().Len() != 1 {
		t.Fatalf("events len = %d, want 1", got2.Session.Events().Len())
	}

	if err := svc2.Delete(ctx, &session.DeleteRequest{
		AppName:   "app",
		UserID:    "u1",
		SessionID: "s1",
	}); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	if _, err := svc2.Get(ctx, &session.GetRequest{
		AppName:   "app",
		UserID:    "u1",
		SessionID: "s1",
	}); err == nil {
		t.Fatalf("expected error after delete, got nil")
	}
}

func TestSQLiteSessionErrorPaths(t *testing.T) {
	t.Parallel()

	dbPath := filepath.Join(t.TempDir(), "sessions.db")
	svc, err := Service(t.Context(), dbPath)
	if err != nil {
		t.Fatalf("Service error: %v", err)
	}

	ctx := t.Context()

	if _, err := svc.Create(ctx, &session.CreateRequest{
		AppName: "",
		UserID:  "u",
	}); err == nil {
		t.Fatalf("Create without app should error")
	}

	if _, err := svc.Create(ctx, &session.CreateRequest{
		AppName: "app",
		UserID:  "",
	}); err == nil {
		t.Fatalf("Create without user should error")
	}

	// Duplicate create should fail due to primary key constraint.
	if _, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "u",
		SessionID: "dup",
	}); err != nil {
		t.Fatalf("Create dup first: %v", err)
	}

	if _, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "u",
		SessionID: "dup",
	}); err == nil {
		t.Fatalf("expected duplicate create to fail")
	}

	// Append with nils or partial should be no-op or error.
	ev := session.NewEvent("id")
	ev.Partial = true
	if err := svc.AppendEvent(ctx, nil, ev); err == nil {
		t.Fatalf("AppendEvent nil session should error")
	}

	// partial true should no-op
	if err := svc.AppendEvent(ctx, &dbSession{
		s: &persistSession{
			AppName:   "app",
			UserID:    "u",
			SessionID: "dup",
		},
	}, ev); err != nil {
		t.Fatalf("AppendEvent partial should not error, got %v", err)
	}

	// load should error if fields missing
	store := svc.(*store)
	if _, err := store.load(ctx, "", "u", "dup"); err == nil {
		t.Fatalf("load missing app should error")
	}

	if _, err := store.load(ctx, "app", "", "dup"); err == nil {
		t.Fatalf("load missing user should error")
	}

	if _, err := store.load(ctx, "app", "u", ""); err == nil {
		t.Fatalf("load missing session should error")
	}
}

func TestSessionfsStateReuseInSessiondb(t *testing.T) {
	t.Parallel()

	st := sessionfsState{
		state: map[string]any{
			"a": 1,
		},
	}
	if err := st.Set("b", 2); err != nil {
		t.Fatalf("Set error = %v", err)
	}

	got, err := st.Get("a")
	if err != nil || got != 1 {
		t.Fatalf("Get a = %v err %v", got, err)
	}

	want := map[string]any{
		"a": 1,
		"b": 2,
	}
	collected := maps.Collect(st.All())
	if diff := cmp.Diff(want, collected); diff != "" {
		t.Fatalf("state iteration diff (-want +got): %s", diff)
	}
}
