package agent

import (
	"context"
	"fmt"
	"testing"

	"iter"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

func TestTumixStopsWhenJudgeEscalates(t *testing.T) {
	candidates := []agent.Agent{stubCandidate("A"), stubCandidate("B")}
	judge := stubJudge("done")
	loader, err := NewTumixAgentWithConfig(TumixConfig{
		Candidates: candidates,
		Judge:      judge,
		MaxRounds:  3,
		MinRounds:  2,
	})
	if err != nil {
		t.Fatalf("loader: %v", err)
	}

	ctx := context.Background()
	svc := session.InMemoryService()
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
		t.Fatalf("create session: %v", err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: loader.RootAgent(), SessionService: svc})
	if err != nil {
		t.Fatalf("runner: %v", err)
	}

	for event, err := range r.Run(ctx, "u", "s", genai.NewContentFromText("q", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("run err: %v", err)
		}
		if event == nil {
			t.Fatalf("nil event")
		}
	}

	res, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s"})
	if err != nil {
		t.Fatalf("get session: %v", err)
	}
	answer, err := res.Session.State().Get(stateKeyAnswer)
	if err != nil {
		t.Fatalf("state answer: %v", err)
	}
	if answer != "done" {
		t.Fatalf("expected final answer 'done', got %v", answer)
	}
	joined, _ := res.Session.State().Get(stateKeyJoined)
	if joined == "" {
		t.Fatalf("expected joined answers to be recorded")
	}
}

func TestTumixMajorityFallback(t *testing.T) {
	candidates := []agent.Agent{staticCandidate("X", "foo"), staticCandidate("Y", "foo"), staticCandidate("Z", "bar")}
	judge := noOpJudge()
	loader, err := NewTumixAgentWithConfig(TumixConfig{
		Candidates: candidates,
		Judge:      judge,
		MaxRounds:  2,
		MinRounds:  1,
	})
	if err != nil {
		t.Fatalf("loader: %v", err)
	}

	ctx := context.Background()
	svc := session.InMemoryService()
	if _, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s2"}); err != nil {
		t.Fatalf("create session: %v", err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: loader.RootAgent(), SessionService: svc})
	if err != nil {
		t.Fatalf("runner: %v", err)
	}

	for range r.Run(ctx, "u", "s2", genai.NewContentFromText("q2", genai.RoleUser), agent.RunConfig{}) {
		// drain events
	}

	res, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "u", SessionID: "s2"})
	if err != nil {
		t.Fatalf("get session: %v", err)
	}
	answer, err := res.Session.State().Get(stateKeyAnswer)
	if err != nil {
		t.Fatalf("state answer: %v", err)
	}
	if answer != "foo" {
		t.Fatalf("expected majority answer foo, got %v", answer)
	}
}

func stubCandidate(name string) agent.Agent {
	return mustAgent(agent.New(agent.Config{
		Name:        name,
		Description: "stub candidate",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				round, _ := ctx.Session().State().Get(stateKeyRound)
				text := fmt.Sprintf("%s-round-%v", name, round)
				ev := session.NewEvent(ctx.InvocationID())
				ev.LLMResponse = model.LLMResponse{Content: genai.NewContentFromText(text, genai.RoleModel)}
				yield(ev, nil)
			}
		},
	}))
}

func staticCandidate(name, answer string) agent.Agent {
	return mustAgent(agent.New(agent.Config{
		Name:        name,
		Description: "static candidate",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx.InvocationID())
				ev.LLMResponse = model.LLMResponse{Content: genai.NewContentFromText(answer, genai.RoleModel)}
				yield(ev, nil)
			}
		},
	}))
}

func stubJudge(answer string) agent.Agent {
	return mustAgent(agent.New(agent.Config{
		Name:        "judge",
		Description: "stub judge",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				_ = ctx.Session().State().Set(stateKeyAnswer, answer)
				_ = ctx.Session().State().Set(stateKeyConfidence, 0.95)
				ev := session.NewEvent(ctx.InvocationID())
				ev.Author = "judge"
				ev.Actions.Escalate = true
				ev.LLMResponse = model.LLMResponse{Content: genai.NewContentFromText("stop", genai.RoleModel)}
				yield(ev, nil)
			}
		},
	}))
}

func noOpJudge() agent.Agent {
	return mustAgent(agent.New(agent.Config{
		Name:        "judge",
		Description: "noop judge",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx.InvocationID())
				ev.Author = "judge"
				ev.LLMResponse = model.LLMResponse{Content: genai.NewContentFromText("continue", genai.RoleModel)}
				yield(ev, nil)
			}
		},
	}))
}

func mustAgent(a agent.Agent, err error) agent.Agent {
	if err != nil {
		panic(err)
	}
	return a
}
