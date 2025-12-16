module github.com/zchee/tumix

go 1.25

replace github.com/zchee/tumix/gollm/xai => ./gollm/xai

// OpenTelemetry
replace (
	go.opentelemetry.io/auto/sdk => go.opentelemetry.io/auto/sdk v1.2.1
	go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp => go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp v0.64.0
	go.opentelemetry.io/otel => go.opentelemetry.io/otel v1.39.0
	go.opentelemetry.io/otel/exporters/otlp/otlptrace => go.opentelemetry.io/otel/exporters/otlp/otlptrace v1.39.0
	go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc => go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc v1.39.0
	go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp => go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp v1.39.0
	go.opentelemetry.io/otel/metric => go.opentelemetry.io/otel/metric v1.39.0
	go.opentelemetry.io/otel/sdk v1.39.0 => go.opentelemetry.io/otel/sdk v1.39.0
	go.opentelemetry.io/otel/trace => go.opentelemetry.io/otel/trace v1.39.0
	go.opentelemetry.io/proto/otlp => go.opentelemetry.io/proto/otlp v1.9.0
)

replace github.com/invopop/jsonschema => github.com/zchee/jsonschema v0.0.0-20251203212453-664582a47f4a

require (
	cloud.google.com/go/auth v0.18.0
	cloud.google.com/go/auth/oauth2adapt v0.2.8
	github.com/Marlliton/slogpretty v0.1.3
	github.com/anthropics/anthropic-sdk-go v1.19.0
	github.com/google/dotprompt/go v0.0.0-20251212201238-92f6ee4b208a
	github.com/google/go-cmp v0.7.0
	github.com/google/go-replayers/grpcreplay v1.3.1-0.20250327185215-2dbb62fbf480 // @main
	github.com/google/go-replayers/httpreplay v1.2.1-0.20250327185215-2dbb62fbf480 // @main
	github.com/invopop/jsonschema v0.13.0
	github.com/openai/openai-go/v3 v3.13.0
	github.com/zchee/tumix/gollm/xai v0.0.0-00010101000000-000000000000
	go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp v0.64.0
	go.opentelemetry.io/otel v1.39.0
	go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc v1.39.0
	go.opentelemetry.io/otel/metric v1.39.0
	go.opentelemetry.io/otel/sdk v1.39.0
	go.opentelemetry.io/otel/trace v1.39.0
	golang.org/x/sys v0.39.0
	google.golang.org/adk v0.2.1-0.20251215152237-9b193f6426b3 // @main
	google.golang.org/genai v1.39.0
	google.golang.org/grpc v1.77.0
	google.golang.org/protobuf v1.36.11
	modernc.org/sqlite v1.40.1
)

require (
	cloud.google.com/go v0.123.0 // indirect
	cloud.google.com/go/compute/metadata v0.9.0 // indirect
	github.com/bahlo/generic-list-go v0.2.0 // indirect
	github.com/buger/jsonparser v1.1.1 // indirect
	github.com/cenkalti/backoff/v5 v5.0.3 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/dustin/go-humanize v1.0.1 // indirect
	github.com/felixge/httpsnoop v1.0.4 // indirect
	github.com/gaudiy/vtprotobuf v0.6.1-0.20251122131602-5bc3a6fc1d03 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/goccy/go-yaml v1.19.0 // indirect
	github.com/google/jsonschema-go v0.3.0 // indirect
	github.com/google/martian/v3 v3.3.3 // indirect
	github.com/google/s2a-go v0.1.9 // indirect
	github.com/google/safehtml v0.1.0 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/googleapis/enterprise-certificate-proxy v0.3.7 // indirect
	github.com/googleapis/gax-go/v2 v2.15.0 // indirect
	github.com/gorilla/websocket v1.5.3 // indirect
	github.com/grpc-ecosystem/grpc-gateway/v2 v2.27.3 // indirect
	github.com/mailru/easyjson v0.9.1 // indirect
	github.com/mattn/go-isatty v0.0.20 // indirect
	github.com/mbleigh/raymond v0.0.0-20250414171441-6b3a58ab9e0a // indirect
	github.com/ncruces/go-strftime v1.0.0 // indirect
	github.com/puzpuzpuz/xsync/v4 v4.2.0 // indirect
	github.com/remyoudompheng/bigfft v0.0.0-20230129092748-24d4a6f8daec // indirect
	github.com/tidwall/gjson v1.18.0 // indirect
	github.com/tidwall/match v1.1.1 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
	github.com/wk8/go-ordered-map/v2 v2.1.8 // indirect
	go.opentelemetry.io/auto/sdk v1.2.1 // indirect
	go.opentelemetry.io/otel/exporters/otlp/otlptrace v1.39.0 // indirect
	go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp v1.39.0 // indirect
	go.opentelemetry.io/proto/otlp v1.9.0 // indirect
	golang.org/x/crypto v0.46.0 // indirect
	golang.org/x/exp v0.0.0-20251209150349-8475f28825e9 // indirect
	golang.org/x/net v0.48.0 // indirect
	golang.org/x/oauth2 v0.34.0 // indirect
	golang.org/x/sync v0.19.0 // indirect
	golang.org/x/text v0.32.0 // indirect
	google.golang.org/genproto/googleapis/api v0.0.0-20251202230838-ff82c1b0f217 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20251202230838-ff82c1b0f217 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
	modernc.org/libc v1.67.1 // indirect
	modernc.org/mathutil v1.7.1 // indirect
	modernc.org/memory v1.11.0 // indirect
	rsc.io/omap v1.2.0 // indirect
	rsc.io/ordered v1.1.1 // indirect
)
