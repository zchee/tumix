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

// Command tumix implements a [TUMIX: Multi-Agent Test-Time Scaling with Tool-Use Mixture] in Go.
//
// [TUMIX: Multi-Agent Test-Time Scaling with Tool-Use Mixture]: https://arxiv.org/abs/2510.01279
package main

import (
	"bufio"
	"context"
	json "encoding/json/v2"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	tumixagent "github.com/zchee/tumix/agent"
	"github.com/zchee/tumix/internal/version"
	"github.com/zchee/tumix/log"
	"github.com/zchee/tumix/sessiondb"
	"github.com/zchee/tumix/sessionfs"
	"github.com/zchee/tumix/telemetry/httptelemetry"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	"go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.37.0"
	adkagent "google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type config struct {
	AppName        string
	ModelName      string
	APIKey         string
	TraceHTTP      bool
	UserID         string
	SessionID      string
	SessionDir     string
	MaxRounds      uint
	Temperature    float64
	TopP           float64
	TopK           int
	MaxTokens      int
	Seed           int64
	OutputJSON     bool
	DryRun         bool
	LogJSON        bool
	OTLPEndpoint   string
	CallWarn       int
	BatchFile      string
	Concurrency    int
	MaxPromptChars int
	BenchLocal     int
	Prompt         string
}

var (
	prices = map[string]struct {
		inUSDPerKT, outUSDPerKT float64
	}{
		"gemini-2.5-flash": {inUSDPerKT: 0.00015, outUSDPerKT: 0.0006},
		"gemini-2.5-pro":   {inUSDPerKT: 0.00125, outUSDPerKT: 0.0050},
	}
	meter            metric.Meter
	requestCounter   metric.Int64Counter
	inputTokCounter  metric.Int64Counter
	outputTokCounter metric.Int64Counter
	costCounter      metric.Float64Counter
)

func main() {
	cfg, err := parseConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "config error: %v\n", err)
		os.Exit(2)
	}

	var handler slog.Handler = slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo})
	if cfg.LogJSON {
		handler = slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo})
	}
	logger := slog.New(handler)
	ctx, stop := signal.NotifyContext(log.WithLogger(context.Background(), logger), os.Interrupt, syscall.SIGTERM)
	defer stop()

	shutdownTrace, err := initTracing(ctx, cfg)
	if err != nil {
		logger.Warn("tracing disabled", "error", err)
	}
	defer shutdownTrace()
	initMetrics()

	httpClient := newHTTPClient(cfg.TraceHTTP)
	llm, err := buildModel(ctx, cfg, httpClient)
	if err != nil {
		logger.Error("failed to create model", "error", err)
		os.Exit(1)
	}

	genCfg := buildGenConfig(cfg)
	loader, err := buildTumixLoader(llm, genCfg, cfg.MaxRounds)
	if err != nil {
		logger.Error("failed to build tumix agent", "error", err)
		os.Exit(1)
	}

	if cfg.DryRun {
		printConfig(cfg)
		return
	}
	if cfg.BenchLocal > 0 {
		benchLocal(cfg)
		return
	}

	if cfg.BatchFile != "" {
		if err := runBatch(ctx, cfg, loader); err != nil {
			logger.Error("batch run failed", "error", err)
			os.Exit(1)
		}
		return
	}

	if err := runOnce(ctx, cfg, loader); err != nil {
		logger.Error("run failed", "error", err)
		os.Exit(1)
	}
}

func parseConfig() (config, error) {
	cfg := config{
		AppName:        "tumix",
		ModelName:      envOrDefault("TUMIX_MODEL", "gemini-2.5-flash"),
		APIKey:         os.Getenv("GOOGLE_API_KEY"),
		TraceHTTP:      parseBoolEnv("TUMIX_HTTP_TRACE"),
		UserID:         envOrDefault("TUMIX_USER", "user"),
		SessionID:      envOrDefault("TUMIX_SESSION", ""),
		SessionDir:     envOrDefault("TUMIX_SESSION_DIR", ""),
		MaxRounds:      parseUintEnv("TUMIX_MAX_ROUNDS", 3),
		Temperature:    parseFloatEnv("TUMIX_TEMPERATURE", -1),
		TopP:           parseFloatEnv("TUMIX_TOP_P", -1),
		TopK:           int(parseUintEnv("TUMIX_TOP_K", 0)),
		MaxTokens:      int(parseUintEnv("TUMIX_MAX_TOKENS", 0)),
		Seed:           int64(parseUintEnv("TUMIX_SEED", 0)),
		CallWarn:       int(parseUintEnv("TUMIX_CALL_WARN", 300)),
		Concurrency:    int(parseUintEnv("TUMIX_CONCURRENCY", 1)),
		MaxPromptChars: int(parseUintEnv("TUMIX_MAX_PROMPT_CHARS", 8000)),
	}

	flag.StringVar(&cfg.ModelName, "model", cfg.ModelName, "Gemini model to use (default TUMIX_MODEL or gemini-2.5-flash)")
	flag.StringVar(&cfg.APIKey, "api_key", cfg.APIKey, "Gemini API key (GOOGLE_API_KEY)")
	flag.BoolVar(&cfg.TraceHTTP, "http_trace", cfg.TraceHTTP, "Enable HTTP client OpenTelemetry spans")
	flag.StringVar(&cfg.UserID, "user", cfg.UserID, "User ID for the session")
	flag.StringVar(&cfg.SessionID, "session", cfg.SessionID, "Session ID (auto-generated if empty)")
	flag.StringVar(&cfg.SessionDir, "session_dir", cfg.SessionDir, "Directory to persist sessions (optional, uses in-memory if empty)")
	flag.UintVar(&cfg.MaxRounds, "max_rounds", cfg.MaxRounds, "Maximum TUMIX iterations (default 3, overridable via TUMIX_MAX_ROUNDS)")
	flag.Float64Var(&cfg.Temperature, "temperature", cfg.Temperature, "Sampling temperature (set <0 to leave model default; env TUMIX_TEMPERATURE)")
	flag.Float64Var(&cfg.TopP, "top_p", cfg.TopP, "Top-p nucleus sampling (set <0 to leave model default; env TUMIX_TOP_P)")
	flag.IntVar(&cfg.TopK, "top_k", cfg.TopK, "Top-k sampling (0 to leave default; env TUMIX_TOP_K)")
	flag.IntVar(&cfg.MaxTokens, "max_tokens", cfg.MaxTokens, "Max output tokens (0 to leave default; env TUMIX_MAX_TOKENS)")
	flag.Int64Var(&cfg.Seed, "seed", cfg.Seed, "Deterministic seed (0 to leave unset; env TUMIX_SEED)")
	flag.BoolVar(&cfg.OutputJSON, "json", false, "Emit final answer as JSON to stdout")
	flag.BoolVar(&cfg.DryRun, "dry_run", false, "Print resolved config and exit without calling model")
	flag.BoolVar(&cfg.LogJSON, "log_json", false, "Use JSON logging format")
	flag.StringVar(&cfg.OTLPEndpoint, "otlp_endpoint", cfg.OTLPEndpoint, "OTLP endpoint for tracing (empty to disable)")
	flag.IntVar(&cfg.CallWarn, "call_warn", cfg.CallWarn, "Warn if estimated LLM calls exceed this number")
	flag.StringVar(&cfg.BatchFile, "batch_file", cfg.BatchFile, "Optional file with one prompt per line for batch processing")
	flag.IntVar(&cfg.Concurrency, "concurrency", cfg.Concurrency, "Max concurrent prompts when using -batch_file")
	flag.IntVar(&cfg.MaxPromptChars, "max_prompt_chars", cfg.MaxPromptChars, "Fail if user prompt exceeds this many characters")
	flag.IntVar(&cfg.BenchLocal, "bench_local", cfg.BenchLocal, "Run local synthetic benchmark for N iterations and exit")
	flag.Parse()

	cfg.Prompt = strings.TrimSpace(strings.Join(flag.Args(), " "))
	if cfg.Prompt == "" {
		return cfg, errors.New("prompt is required; pass text after flags")
	}
	if cfg.MaxPromptChars > 0 && len(cfg.Prompt) > cfg.MaxPromptChars {
		return cfg, fmt.Errorf("prompt length %d exceeds max_prompt_chars %d", len(cfg.Prompt), cfg.MaxPromptChars)
	}
	if cfg.APIKey == "" {
		return cfg, errors.New("GOOGLE_API_KEY must be set")
	}
	if cfg.SessionID == "" {
		cfg.SessionID = fmt.Sprintf("session-%d", time.Now().UnixNano())
	}
	if cfg.Temperature >= 0 && (cfg.Temperature < 0 || cfg.Temperature > 2) {
		return cfg, errors.New("temperature must be between 0 and 2")
	}
	if cfg.TopP >= 0 && (cfg.TopP < 0 || cfg.TopP > 1) {
		return cfg, errors.New("top_p must be between 0 and 1")
	}
	if cfg.TopK < 0 {
		return cfg, errors.New("top_k cannot be negative")
	}
	if cfg.MaxTokens < 0 {
		return cfg, errors.New("max_tokens cannot be negative")
	}
	if cfg.Concurrency < 1 {
		cfg.Concurrency = 1
	}

	return cfg, nil
}

func newHTTPClient(traceEnabled bool) *http.Client {
	return &http.Client{
		Transport: httptelemetry.NewTransportWithTrace(nil, traceEnabled),
	}
}

func buildModel(ctx context.Context, cfg config, httpClient *http.Client) (model.LLM, error) {
	clientConfig := &genai.ClientConfig{
		APIKey:     cfg.APIKey,
		HTTPClient: httpClient,
		HTTPOptions: genai.HTTPOptions{
			Headers: http.Header{
				"User-Agent": []string{version.UserAgent("genai")},
			},
		},
	}

	return gemini.NewModel(ctx, cfg.ModelName, clientConfig)
}

func buildGenConfig(cfg config) *genai.GenerateContentConfig {
	if cfg.Temperature < 0 && cfg.TopP < 0 {
		if cfg.TopK == 0 && cfg.MaxTokens == 0 && cfg.Seed == 0 {
			return nil
		}
	}
	c := &genai.GenerateContentConfig{}
	if cfg.Temperature >= 0 {
		val := float32(cfg.Temperature)
		c.Temperature = &val
	}
	if cfg.TopP >= 0 {
		val := float32(cfg.TopP)
		c.TopP = &val
	}
	if cfg.TopK > 0 {
		val := float32(cfg.TopK)
		c.TopK = &val
	}
	if cfg.MaxTokens > 0 {
		c.MaxOutputTokens = int32(cfg.MaxTokens)
	}
	if cfg.Seed > 0 {
		val := int32(cfg.Seed)
		c.Seed = &val
	}
	return c
}

func buildTumixLoader(llm model.LLM, genCfg *genai.GenerateContentConfig, maxRounds uint) (adkagent.Loader, error) {
	builders := []func(model.LLM, *genai.GenerateContentConfig) (adkagent.Agent, error){
		tumixagent.NewBaseAgent,
		tumixagent.NewCoTAgent,
		tumixagent.NewCoTCodeAgent,
		tumixagent.NewSearchAgent,
		tumixagent.NewCodeAgent,
		tumixagent.NewCodePlusAgent,
		tumixagent.NewDualToolGSAgent,
		tumixagent.NewDualToolLLMAgent,
		tumixagent.NewDualToolComAgent,
		tumixagent.NewGuidedGSAgent,
		tumixagent.NewGuidedLLMAgent,
		tumixagent.NewGuidedComAgent,
	}

	var candidates []adkagent.Agent
	for _, builder := range builders {
		a, err := builder(llm, genCfg)
		if err != nil {
			return nil, err
		}
		candidates = append(candidates, a)
	}

	refinement, err := tumixagent.NewRefinementAgent(candidates...)
	if err != nil {
		return nil, err
	}
	judge, err := tumixagent.NewJudgeAgent(llm, genCfg)
	if err != nil {
		return nil, err
	}
	round, err := tumixagent.NewRoundAgent(refinement, judge)
	if err != nil {
		return nil, err
	}

	return tumixagent.NewTumixAgentWithMaxRounds([]adkagent.Agent{round}, maxRounds)
}

func runOnce(ctx context.Context, cfg config, loader adkagent.Loader) error {
	sessionService := session.InMemoryService()
	if cfg.SessionDir != "" {
		svc, err := sessionfs.Service(cfg.SessionDir)
		if err != nil {
			return fmt.Errorf("init session store: %w", err)
		}
		sessionService = svc
	} else if dbPath := os.Getenv("TUMIX_SESSION_SQLITE"); dbPath != "" {
		svc, err := sessiondb.Service(dbPath)
		if err != nil {
			return fmt.Errorf("init sqlite store: %w", err)
		}
		sessionService = svc
	}
	if _, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName:   cfg.AppName,
		UserID:    cfg.UserID,
		SessionID: cfg.SessionID,
	}); err != nil {
		return fmt.Errorf("create session: %w", err)
	}

	r, err := runner.New(runner.Config{
		AppName:        cfg.AppName,
		Agent:          loader.RootAgent(),
		SessionService: sessionService,
	})
	if err != nil {
		return fmt.Errorf("runner init: %w", err)
	}

	content := genai.NewContentFromText(cfg.Prompt, genai.RoleUser)
	var finalAuthor, finalText string
	var totalIn, totalOut int64
	for event, err := range r.Run(ctx, cfg.UserID, cfg.SessionID, content, adkagent.RunConfig{}) {
		if err != nil {
			return fmt.Errorf("agent run: %w", err)
		}
		if !cfg.OutputJSON {
			logEvent(ctx, event)
		}
		if text := firstText(event); text != "" {
			finalText = text
			finalAuthor = event.Author
		}
		inTok, outTok := recordUsage(event, cfg.ModelName)
		totalIn += inTok
		totalOut += outTok
	}
	estimateAndWarn(cfg, int(totalIn), int(totalOut))

	if cfg.OutputJSON {
		out := map[string]any{
			"session_id":    cfg.SessionID,
			"author":        finalAuthor,
			"text":          finalText,
			"input_tokens":  totalIn,
			"output_tokens": totalOut,
			"config": map[string]any{
				"model":       cfg.ModelName,
				"max_rounds":  cfg.MaxRounds,
				"temperature": cfg.Temperature,
				"top_p":       cfg.TopP,
				"top_k":       cfg.TopK,
				"max_tokens":  cfg.MaxTokens,
				"seed":        cfg.Seed,
			},
		}
		data, err := json.Marshal(out)
		if err != nil {
			return fmt.Errorf("encode json: %w", err)
		}
		if _, err := fmt.Fprintln(os.Stdout, string(data)); err != nil {
			return fmt.Errorf("write json: %w", err)
		}
	}

	return nil
}

func runBatch(ctx context.Context, cfg config, loader adkagent.Loader) error {
	f, err := os.Open(filepath.Clean(cfg.BatchFile))
	if err != nil {
		return fmt.Errorf("open batch file: %w", err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	var prompts []string
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		prompts = append(prompts, line)
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read batch file: %w", err)
	}

	promptCh := make(chan string)
	errCh := make(chan error, cfg.Concurrency)
	var wg sync.WaitGroup
	for i := 0; i < cfg.Concurrency; i++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for p := range promptCh {
				local := cfg
				local.Prompt = p
				if local.SessionID == "" {
					local.SessionID = fmt.Sprintf("session-%d-%d", time.Now().UnixNano(), worker)
				}
				if err := runOnce(ctx, local, loader); err != nil {
					errCh <- fmt.Errorf("prompt %q: %w", p, err)
					return
				}
			}
		}(i)
	}

	go func() {
		for _, p := range prompts {
			promptCh <- p
		}
		close(promptCh)
	}()

	wg.Wait()
	select {
	case err := <-errCh:
		return err
	default:
		return nil
	}
}

func logEvent(ctx context.Context, event *session.Event) {
	if event == nil || event.LLMResponse.Partial {
		return
	}

	var texts []string
	if event.LLMResponse.Content != nil {
		for _, part := range event.LLMResponse.Content.Parts {
			if part == nil {
				continue
			}
			if part.Text != "" {
				texts = append(texts, part.Text)
			}
		}
	}
	if len(texts) == 0 {
		return
	}

	log.FromContext(ctx).Info("agent response", "author", event.Author, "text", strings.Join(texts, " "))
}

func firstText(event *session.Event) string {
	if event == nil || event.LLMResponse.Content == nil {
		return ""
	}
	for _, part := range event.LLMResponse.Content.Parts {
		if part == nil {
			continue
		}
		if part.Text != "" {
			return part.Text
		}
	}
	return ""
}

func initTracing(ctx context.Context, cfg config) (func(), error) {
	if cfg.OTLPEndpoint == "" {
		return func() {}, nil
	}
	exp, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(cfg.OTLPEndpoint),
		otlptracegrpc.WithDialOption(grpc.WithTransportCredentials(insecure.NewCredentials())),
	)
	if err != nil {
		return func() {}, fmt.Errorf("otlp exporter: %w", err)
	}
	res, err := resource.Merge(resource.Default(), resource.NewSchemaless(semconv.ServiceNameKey.String("tumix")))
	if err != nil {
		return func() {}, fmt.Errorf("resource: %w", err)
	}
	tp := trace.NewTracerProvider(trace.WithBatcher(exp), trace.WithResource(res))
	otel.SetTracerProvider(tp)
	return func() { _ = tp.Shutdown(context.Background()) }, nil
}

func initMetrics() {
	meter = otel.GetMeterProvider().Meter("tumix")
	requestCounter, _ = meter.Int64Counter("tumix.llm_requests")
	inputTokCounter, _ = meter.Int64Counter("tumix.input_tokens")
	outputTokCounter, _ = meter.Int64Counter("tumix.output_tokens")
	costCounter, _ = meter.Float64Counter("tumix.cost_usd")
}

func benchLocal(cfg config) {
	start := time.Now()
	var wg sync.WaitGroup
	prompts := cfg.BenchLocal
	workers := cfg.Concurrency
	if workers < 1 {
		workers = 1
	}
	ch := make(chan int)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range ch {
				_ = strings.Repeat("x", 1024)
			}
		}()
	}
	for i := 0; i < prompts; i++ {
		ch <- i
	}
	close(ch)
	wg.Wait()
	dur := time.Since(start)
	fmt.Fprintf(os.Stdout, "bench_local iters=%d workers=%d duration=%s per_iter=%s\n", prompts, workers, dur, dur/time.Duration(prompts))
}

func estimateAndWarn(cfg config, totalIn, totalOut int) {
	// Upper-bound call count: (candidates + judge) per round.
	agents := 12 + 1 // 12 candidates + judge
	calls := int(cfg.MaxRounds) * agents
	if cfg.CallWarn > 0 && calls > cfg.CallWarn {
		log.FromContext(context.Background()).Warn("estimated LLM calls high", "calls", calls, "threshold", cfg.CallWarn)
	}
	if cost := estimateCost(cfg.ModelName, totalIn, totalOut); cost > 0 {
		costCounter.Add(context.Background(), cost)
		log.FromContext(context.Background()).Info("usage", "input_tokens", totalIn, "output_tokens", totalOut, "cost_usd", cost)
	}
}

func estimateCost(modelName string, inputTokens, outputTokens int) float64 {
	p, ok := prices[modelName]
	if !ok {
		p = prices["gemini-2.5-flash"]
	}
	return (float64(inputTokens)/1000.0)*p.inUSDPerKT + (float64(outputTokens)/1000.0)*p.outUSDPerKT
}

func recordUsage(event *session.Event, modelName string) (int64, int64) {
	if event == nil || event.UsageMetadata == nil {
		return 0, 0
	}
	in := int64(event.UsageMetadata.PromptTokenCount)
	out := int64(event.UsageMetadata.CandidatesTokenCount)
	ctx := context.Background()
	requestCounter.Add(ctx, 1)
	inputTokCounter.Add(ctx, in)
	outputTokCounter.Add(ctx, out)
	return in, out
}

func printConfig(cfg config) {
	out := map[string]any{
		"model":         cfg.ModelName,
		"max_rounds":    cfg.MaxRounds,
		"temperature":   cfg.Temperature,
		"top_p":         cfg.TopP,
		"top_k":         cfg.TopK,
		"max_tokens":    cfg.MaxTokens,
		"seed":          cfg.Seed,
		"session_dir":   cfg.SessionDir,
		"http_trace":    cfg.TraceHTTP,
		"log_json":      cfg.LogJSON,
		"otlp_endpoint": cfg.OTLPEndpoint,
		"batch_file":    cfg.BatchFile,
		"concurrency":   cfg.Concurrency,
	}
	data, _ := json.Marshal(out)
	fmt.Fprintln(os.Stdout, string(data))
}

func envOrDefault(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return fallback
}

func parseUintEnv(key string, fallback uint) uint {
	raw := os.Getenv(key)
	if raw == "" {
		return fallback
	}
	v, err := strconv.ParseUint(raw, 10, 0)
	if err != nil {
		return fallback
	}
	return uint(v)
}

func parseFloatEnv(key string, fallback float64) float64 {
	raw := os.Getenv(key)
	if raw == "" {
		return fallback
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return fallback
	}
	return v
}

func parseBoolEnv(key string) bool {
	v := os.Getenv(key)
	b, err := strconv.ParseBool(v)
	if err != nil {
		return false
	}
	return b
}
