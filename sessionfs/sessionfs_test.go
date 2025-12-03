package sessionfs

import (
	"context"
	"testing"
	"time"

	"google.golang.org/adk/session"
)

func TestFileServiceCreateGetAppend(t *testing.T) {
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
	got, err := svc2.Get(context.Background(), &session.GetRequest{AppName: app, UserID: user, SessionID: sid})
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
