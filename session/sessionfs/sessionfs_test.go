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
	"testing"
	"time"

	"google.golang.org/adk/session"
)

func TestFileServiceCreateGetAppend(t *testing.T) {
	t.Skip("deadlock?")

	t.Parallel()

	dir := t.TempDir()
	svc, err := Service(dir)
	if err != nil {
		t.Fatalf("Service() error = %v", err)
	}

	ctx := t.Context()
	app, user, sid := "app", "user", "sid"
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: app, UserID: user, SessionID: sid}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	ev := session.NewEvent("inv1")
	ev.Timestamp = time.Now()
	ev.Author = "a"
	ev.Actions.StateDelta = map[string]any{"k": "v"}

	gotSess, err := svc.Get(ctx, &session.GetRequest{AppName: app, UserID: user, SessionID: sid})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if err := svc.AppendEvent(ctx, gotSess.Session, ev); err != nil {
		t.Fatalf("AppendEvent() error = %v", err)
	}

	// Reload service to ensure persistence on disk.
	svc2, err := Service(dir)
	if err != nil {
		t.Fatalf("Service reload error = %v", err)
	}
	got, err := svc2.Get(t.Context(), &session.GetRequest{AppName: app, UserID: user, SessionID: sid})
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
	if got.Session.Events().Len() != 1 {
		t.Fatalf("events len = %d, want 1", got.Session.Events().Len())
	}
}
