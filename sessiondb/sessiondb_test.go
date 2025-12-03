package sessiondb

import (
	"context"
	"testing"

	"google.golang.org/adk/session"
)

func TestSQLiteSessionLifecycle(t *testing.T) {
	t.Parallel()

	dbPath := t.TempDir() + "/sessions.db"
	svc, err := Service(dbPath)
	if err != nil {
		t.Fatalf("Service error: %v", err)
	}

	ctx := context.Background()
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s1"}); err != nil {
		t.Fatalf("Create: %v", err)
	}

	ev := session.NewEvent("inv")
	ev.Author = "a"
	ev.Actions.StateDelta = map[string]any{"k": "v"}

	got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if err := svc.AppendEvent(ctx, got.Session, ev); err != nil {
		t.Fatalf("AppendEvent: %v", err)
	}

	svc2, err := Service(dbPath)
	if err != nil {
		t.Fatalf("Service reload: %v", err)
	}
	got2, err := svc2.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Get reload: %v", err)
	}
	val, err := got2.Session.State().Get("k")
	if err != nil || val != "v" {
		t.Fatalf("state after reload = %v err %v", val, err)
	}
	if got2.Session.Events().Len() != 1 {
		t.Fatalf("events len = %d", got2.Session.Events().Len())
	}
}
