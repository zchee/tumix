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

// Package agent implements 15 pre-designed agents used in TUMIX.
//
// 15 pre-designed agents used in TUMIX:
//   - 1. w/o TTS (Base)
//   - 2. CoT Agent (CoT)
//   - 3. CoT-Code Agent (CoT code)
//   - 4. Search Agent (S)
//   - 5. Code Agent (C)
//   - 6. Code Agent+ (C+)
//   - 7. Dual-Tool Agent (CS gs)
//   - 8. Dual-Tool Agent (CS llm)
//   - 9. Dual-Tool Agent (CS com)
//   - 10. Guided Agent (CSGgs)
//   - 11. Guided Agent (CSGllm)
//   - 12. Guided Agent (CSGcom)
//   - 13. Guided Agent+ (CSG+gs)
//   - 14. Guided Agent+ (CSG+llm)
//   - 15. Guided Agent+ (CSG+com)
package agent

import (
	"errors"
	"fmt"
	"iter"
	"math"
	"sort"
	"strings"

	"github.com/google/dotprompt/go/dotprompt"
	"github.com/invopop/jsonschema"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/parallelagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/genai"
)

func Prompt() *dotprompt.Dotprompt {
	o := &dotprompt.DotpromptOptions{
		DefaultModel: "",
		ModelConfigs: map[string]any{},
		Helpers:      map[string]any{},
		Partials:     map[string]string{},
		Tools:        map[string]dotprompt.ToolDefinition{},
		ToolResolver: func(toolName string) (dotprompt.ToolDefinition, error) {
			return dotprompt.ToolDefinition{}, nil
		},
		Schemas: map[string]*jsonschema.Schema{},
		SchemaResolver: func(schemaName string) (*jsonschema.Schema, error) {
			return nil, fmt.Errorf("schema %q not configured", schemaName)
		},
		PartialResolver: func(partialName string) (string, error) {
			return "", fmt.Errorf("partial %q not configured", partialName)
		},
	}
	dp := dotprompt.NewDotprompt(o)
	return dp
}

func code(s string) string {
	return "`" + s + "`"
}

func cloneGenConfig(cfg *genai.GenerateContentConfig) *genai.GenerateContentConfig {
	if cfg == nil {
		return nil
	}
	copied := *cfg
	return &copied
}

func setState(ctx agent.InvocationContext, key string, value any) error {
	if err := ctx.Session().State().Set(key, value); err != nil {
		return fmt.Errorf("set state %s: %w", key, err)
	}
	return nil
}

func getState(ctx agent.InvocationContext, key string) (any, error) {
	val, err := ctx.Session().State().Get(key)
	if errors.Is(err, session.ErrStateKeyNotExist) {
		return nil, fmt.Errorf("state %s not found: %w", key, err)
	}
	if err != nil {
		return nil, fmt.Errorf("get state %s: %w", key, err)
	}
	return val, nil
}

var sharedContext = `**TUMIX shared context**
- Round: {round_num}
- Question: {question}
- Vote margin (0-1): {vote_margin?}; Unique answers: {unique_answers?}; Coverage: {coverage?}; Entropy: {answer_entropy?}
- Previous answers (may be empty):
{joined_answers?}

Use the shared context to refine your reasoning. Continue producing an explicit answer enclosed in ` + code(`<<<`) + ` and ` + code(`>>>`) + `.`

func applySharedContext(cfg *llmagent.Config) {
	cfg.GlobalInstruction = sharedContext
}

// NewBaseAgent creates a Base Agent that uses direct prompting to solve problems.
//
// This agent is responsible for "1. w/o TTS (Base)".
func NewBaseAgent(llm model.LLM, genCfg *genai.GenerateContentConfig) (agent.Agent, error) {
	cfg := llmagent.Config{
		Name: "base",
		Description: `Direct prompt.
- Short name: {Base}.`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
	}

	applySharedContext(&cfg)

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Base agent: %w", err)
	}

	return a, nil
}

// NewCoTAgent creates a CoT Agent that uses chain-of-thought reasoning to solve problems.
//
// This agent is responsible for "2. CoT Agent (CoT)".
func NewCoTAgent(llm model.LLM, genCfg *genai.GenerateContentConfig) (agent.Agent, error) {
	cfg := llmagent.Config{
		Name: "cot",
		Description: `Chain-of-thought text-only reasoning.
- Short name: {CoT}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `• Analyze the question step by step and try to list all the careful points.
• Then try to acquire the final answer with step by step analysis.
• In the end of your response, directly output the answer to the question.

**Do not output the code for execution.**`,
	}

	applySharedContext(&cfg)

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build CoT agent: %w", err)
	}

	return a, nil
}

// NewCoTCodeAgent creates a CoT Code Agent that uses chain-of-thought reasoning and output code to solve problems.
//
// This agent is responsible for "3. CoT-Code Agent (CoT code)".
func NewCoTCodeAgent(llm model.LLM, genCfg *genai.GenerateContentConfig) (agent.Agent, error) {
	cfg := llmagent.Config{
		Name: "cot-code",
		Description: `Chain-of-thought text-only reasoning and output code.
- Short name: {CoT code}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `You are a helpful AI assistant. Solve tasks using your coding skills.
In the following cases, suggest <language> code (in a <language> coding block) for the user to execute.

* Don’t include multiple code blocks in one response, **only include one** in the response.
* Do not ask users to copy and paste the result. Instead, use the ’print’ function for the output when relevant.

Think the task step by step if you need to. If a plan is not provided, explain your plan first. You can first
output your thinking steps with texts and then the final <language> code.

**Remember in the final code you still need to output each number or choice in the final print!**

Start the <language> block with ` + "```" + `<language>`,
	}

	applySharedContext(&cfg)

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build CoT code agent: %w", err)
	}

	return a, nil
}

// NewSearchAgent creates a Search Agent that uses web search to solve problems.
//
// This agent is responsible for "4. Search Agent (S)".
func NewSearchAgent(llm model.LLM, genCfg *genai.GenerateContentConfig) (agent.Agent, error) {
	cfg := llmagent.Config{
		Name: "search",
		Description: `Uses WebSearch.
- Short name: {S}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `You are a helpful AI assistant. Solve tasks using WebSearch tool.

You can use the Google Search Tool to search the web and get the information.
You can call a search query with the format of ` + code(`<search>your search query</search>`) + `,
e.g., ` + code(`<search>Who is the current president of US?</search>`) + `. The searched results will be
returned between ` + code(`<information> and </information>`) + `. Once the search query is complete, stop the
generation. Then, the search platform will return the searched results.

**Do not output the code for execution.**`,
	}

	applySharedContext(&cfg)

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Search agent: %w", err)
	}

	return a, nil
}

// NewCodeAgent creates a Code Agent that uses code execution to solve problems.
//
// This agent is responsible for "5. Code Agent (C)".
func NewCodeAgent(llm model.LLM, genCfg *genai.GenerateContentConfig) (agent.Agent, error) {
	cfg := llmagent.Config{
		Name: "code",
		Description: `Code-execution strategy for precise computation.
- Short name: {C}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `The User asks a question, and you solve it. You first generate the reasoning and thinking process and
then provide the User with the final answer.

During the thinking process, **you can generate <language> code** for efficient searching, optimization, and computing with the format of starting the <language> block
with ` + "```" + `<language>. **A code query must involve only a single script that uses ‘print’
function for the output.**. Once the code script is complete, stop the generation. Then, the code
interpreter platform will execute the code and return the execution output and error. Once you feel you are
ready for the final answer, directly return the answer with the format ` + code(`<<<answer content>>>`) + ` at the end
of your response. Otherwise, you can continue your reasoning process and possibly generate more code
query to solve the problem.`,
	}

	applySharedContext(&cfg)

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Code agent: %w", err)
	}

	return a, nil
}

// NewCodePlusAgent creates a Code+ Agent that uses code execution with extra human-pre-designed priors to solve problems.
//
// This agent is responsible for "6. Code Agent+ (C+)".
func NewCodePlusAgent(llm model.LLM, genCfg *genai.GenerateContentConfig) (agent.Agent, error) {
	cfg := llmagent.Config{
		Name: "code-plus",
		Description: `Code-execution strategy for precise computation with a hinted version with extra human-pre-designed priors.
- Short name: {C+}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `The User asks a question, and you solve it. You first generate the reasoning and thinking process and
then provide the User with the final answer.

During the thinking process, **you can generate <language> code** for efficient searching, optimization, and computing with the format of starting the <language> block
with ` + "```" + `<language>. **A code query must involve only a single script that uses ‘print’
function for the output.**. Once the code script is complete, stop the generation. Then, the code
interpreter platform will execute the code and return the execution output and error. Once you feel you are
ready for the final answer, directly return the answer with the format ` + code(`<<<answer content>>>`) + ` at the end
of your response. Otherwise, you can continue your reasoning process and possibly generate more code
query to solve the problem.`,
	}

	applySharedContext(&cfg)

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Code+ agent: %w", err)
	}

	return a, nil
}

// NewDualToolGSAgent creates a Dual-Tool Agent that uses both code execution and Google Search API to solve problems.
//
// This agent is responsible for "7. Dual-Tool Agent (CS gs)".
func NewDualToolGSAgent(llm model.LLM, genCfg *genai.GenerateContentConfig) (agent.Agent, error) {
	cfg := llmagent.Config{
		Name: "dual-tool-google-search",
		Description: `Dual-tool strategy combining code execution and Google Search API search.
- Short name: {CSgs}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `The User asks a question, and you solve it. You first generate the reasoning and thinking process and then
provide the User with the final answer.

During the thinking process, _you can generate <language> code_ for efficient searching, optimization, and
computing with the format of starting the <language> block with ` + "```" + `<language>. **A code query must
involve only a single script that uses ‘print’ function for the output.**. Once
the code script is complete, stop the generation. Then, the code interpreter platform will execute the
code and return the execution output and error.

If you lack the related knowledge, you can use the Google Search Tool to search the web and get the
information. You can call a search query with the format of ` + code(`<search>your search query</search>`) + `,
e.g., ` + code(`<search>Who is the current president of US?</search>`) + `. The searched results will be
returned between ` + code(`<information> and </information>`) + `. Once the search query is complete, stop the
generation. Then, the search platform will return the searched results.

If you need to search the web, **do not generate code in the same response. Vice versa**. You can also solve
the question without code and searching, just by your textual reasoning.

Once you feel you are ready for the final answer, directly return the answer with the format ` + code(`<<<answer content>>>`) + `
at the end of your response. Otherwise, you can continue your reasoning process and possibly
generate more code or search queries to solve the problem.`,
	}

	applySharedContext(&cfg)

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Dual-Tool(CSgs) agent: %w", err)
	}

	return a, nil
}

// NewDualToolLLMAgent creates a Dual-Tool Agent that uses both code execution and LLM search function to solve problems.
//
// This agent is responsible for "8. Dual-Tool Agent (CS llm)".
func NewDualToolLLMAgent(llm model.LLM, genCfg *genai.GenerateContentConfig) (agent.Agent, error) {
	cfg := llmagent.Config{
		Name: "dual-tool-llm-search",
		Description: `Dual-tool strategy combining code execution and LLM search function.
- Short name: {CSllm}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `The User asks a question, and you solve it. You first generate the reasoning and thinking process and then
provide the User with the final answer.

During the thinking process, _you can generate <language> code_ for efficient searching, optimization, and
computing with the format of starting the <language> block with ` + "```" + `<language>. **A code query must
involve only a single script that uses ‘print’ function for the output.**. Once
the code script is complete, stop the generation. Then, the code interpreter platform will execute the
code and return the execution output and error.

If you lack the related knowledge, you can use the LLM search function to search the web and get the
information. You can call a search query with the format of ` + code(`<search>your search query</search>`) + `,
e.g., ` + code(`<search>Who is the current president of US?</search>`) + `. The searched results will be
returned between ` + code(`<information> and </information>`) + `. Once the search query is complete, stop the
generation. Then, the search platform will return the searched results.

If you need to search the web, **do not generate code in the same response. Vice versa**. You can also solve
the question without code and searching, just by your textual reasoning.

Once you feel you are ready for the final answer, directly return the answer with the format ` + code(`<<<answer content>>>`) + `
at the end of your response. Otherwise, you can continue your reasoning process and possibly
generate more code or search queries to solve the problem.`,
	}

	applySharedContext(&cfg)

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Dual-Tool(CSllm) agent: %w", err)
	}

	return a, nil
}

// NewDualToolComAgent creates a Dual-Tool Agent that uses both code execution and a combination of Google Search API and LLM search function to solve problems.
//
// This agent is responsible for "9. Dual-Tool Agent (CS com)".
func NewDualToolComAgent(llm model.LLM, genCfg *genai.GenerateContentConfig) (agent.Agent, error) {
	cfg := llmagent.Config{
		Name: "dual-tool-combine-search",
		Description: `Dual-tool strategy combining code execution and combination of Google Search API search and LLM search function.
- Short name: {CScom}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `The User asks a question, and you solve it. You first generate the reasoning and thinking process and then
provide the User with the final answer.

During the thinking process, _you can generate <language> code_ for efficient searching, optimization, and
computing with the format of starting the <language> block with ` + "```" + `<language>. **A code query must
involve only a single script that uses ‘print’ function for the output.**. Once
the code script is complete, stop the generation. Then, the code interpreter platform will execute the
code and return the execution output and error.

If you lack the related knowledge, you can use the Google Search Tool and LLM search function to search the web and get the
information. You can call a search query with the format of ` + code(`<search>your search query</search>`) + `,
e.g., ` + code(`<search>Who is the current president of US?</search>`) + `. The searched results will be
returned between ` + code(`<information> and </information>`) + `. Once the search query is complete, stop the
generation. Then, the search platform will return the searched results.

If you need to search the web, **do not generate code in the same response. Vice versa**. You can also solve
the question without code and searching, just by your textual reasoning.

Once you feel you are ready for the final answer, directly return the answer with the format ` + code(`<<<answer content>>>`) + `
at the end of your response. Otherwise, you can continue your reasoning process and possibly
generate more code or search queries to solve the problem.`,
	}

	applySharedContext(&cfg)

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Dual-Tool(CSllm) agent: %w", err)
	}

	return a, nil
}

// NewGuidedGSAgent creates a Guided Agent that uses both code execution and Google Search API to solve problems.
//
// This agent is responsible for "10. Guided Agent (CSGgs)".
func NewGuidedGSAgent(llm model.LLM, genCfg *genai.GenerateContentConfig) (agent.Agent, error) {
	cfg := llmagent.Config{
		Name: "guided-google-search",
		Description: `Dual-tool strategy combining code execution and Google Search API search.
- Short name: {CSG}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `You are guiding another TaskLLM to solve a task. You will be presented with a task that can be solved using
textual reasoning, coding, and web searching. Sometimes the TaskLLM may need extra help to solve the
task, such as generating code or searching the web. Then must follow the rules below for both query and
return answer:

During the thinking process, _you can generate <language> code_ for efficient searching, optimization, and
computing with the format of starting the <language> block with ` + "```" + `<language>. **A code query must involve
only a single script that uses ’print’ function for the output.** Once the code script
is complete, stop the generation. Then, the code interpreter platform will execute the code and return the
execution output and error.

If you lack the related knowledge, you can use the Google Search Tool to search the web and get the
information. You can call a search query with the format of ` + code(`<search>your search query</search>`) + `,
e.g., ` + code(`<search>Who is the current president of US?</search>`) + `. The searched results will be
returned between ` + code(`<information> and </information>`) + `. Once the search query is complete, stop the
generation. Then, the search platform will return the searched results.

If you need to search the web, **do not generate code in the same response. Vice versa.** You can also solve
the question without code and searching, just by your textual reasoning.

Once you feel you are ready for the final answer, directly return the answer with the format ` + code(`<<<answer content>>>`) + `
at the end of your response. Otherwise, you can continue your reasoning process and possibly
generate more code or search queries to solve the problem.

**Your goal is to determine which method will be most effective for solving the task.** Then you generate
the guidance prompt for the TaskLLM to follow in the next round. The final returned guidance prompt
should be included between ` + code(`<<<`) + ` and ` + code(`>>>`) + `, such as ` + code(`<<<You need to generate more complex code to solve...>>>`) + `.
Now, here is the task:`,
	}

	applySharedContext(&cfg)

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Guided(CSGgs) agent: %w", err)
	}

	return a, nil
}

// NewGuidedLLMAgent creates a Guided Agent that uses both code execution and LLM search function to solve problems.
//
// This agent is responsible for "11. Guided Agent (CSGllm)".
func NewGuidedLLMAgent(llm model.LLM, genCfg *genai.GenerateContentConfig) (agent.Agent, error) {
	cfg := llmagent.Config{
		Name: "guided-llm-search",
		Description: `Dual-tool strategy combining code execution and LLM search function.
- Short name: {CSGllm}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `You are guiding another TaskLLM to solve a task. You will be presented with a task that can be solved using
textual reasoning, coding, and web searching. Sometimes the TaskLLM may need extra help to solve the
task, such as generating code or searching the web. Then must follow the rules below for both query and
return answer:

During the thinking process, _you can generate <language> code_ for efficient searching, optimization, and
computing with the format of starting the <language> block with ` + "```" + `<language>. **A code query must involve
only a single script that uses ’print’ function for the output.** Once the code script
is complete, stop the generation. Then, the code interpreter platform will execute the code and return the
execution output and error.

If you lack the related knowledge, you can use the LLM search function to search the web and get the
information. You can call a search query with the format of ` + code(`<search>your search query</search>`) + `,
e.g., ` + code(`<search>Who is the current president of US?</search>`) + `. The searched results will be
returned between ` + code(`<information> and </information>`) + `. Once the search query is complete, stop the
generation. Then, the search platform will return the searched results.

If you need to search the web, **do not generate code in the same response. Vice versa.** You can also solve
the question without code and searching, just by your textual reasoning.

Once you feel you are ready for the final answer, directly return the answer with the format ` + code(`<<<answer content>>>`) + `
at the end of your response. Otherwise, you can continue your reasoning process and possibly
generate more code or search queries to solve the problem.

**Your goal is to determine which method will be most effective for solving the task.** Then you generate
the guidance prompt for the TaskLLM to follow in the next round. The final returned guidance prompt
should be included between ` + code(`<<<`) + ` and ` + code(`>>>`) + `, such as ` + code(`<<<You need to generate more complex code to solve...>>>`) + `.
Now, here is the task:`,
	}

	applySharedContext(&cfg)

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Guided(CSGllm) agent: %w", err)
	}

	return a, nil
}

// NewGuidedComAgent creates a Guided Agent that uses both code execution and a combination of Google Search API and LLM search function to solve problems.
//
// This agent is responsible for "12. Guided Agent (CSGcom)".
func NewGuidedComAgent(llm model.LLM, genCfg *genai.GenerateContentConfig) (agent.Agent, error) {
	cfg := llmagent.Config{
		Name: "guided-combine-search",
		Description: `Dual-tool strategy combining code execution and combination of Google Search API search and LLM search function.
- Short name: {CSGcom}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `You are guiding another TaskLLM to solve a task. You will be presented with a task that can be solved using
textual reasoning, coding, and web searching. Sometimes the TaskLLM may need extra help to solve the
task, such as generating code or searching the web. Then must follow the rules below for both query and
return answer:

During the thinking process, _you can generate <language> code_ for efficient searching, optimization, and
computing with the format of starting the <language> block with ` + "```" + `<language>. **A code query must involve
only a single script that uses ’print’ function for the output.** Once the code script
is complete, stop the generation. Then, the code interpreter platform will execute the code and return the
execution output and error.

If you lack the related knowledge, you can use the Google Search Tool and LLM search function to search the web and get the
information. You can call a search query with the format of ` + code(`<search>your search query</search>`) + `,
e.g., ` + code(`<search>Who is the current president of US?</search>`) + `. The searched results will be
returned between ` + code(`<information> and </information>`) + `. Once the search query is complete, stop the
generation. Then, the search platform will return the searched results.

If you need to search the web, **do not generate code in the same response. Vice versa.** You can also solve
the question without code and searching, just by your textual reasoning.

Once you feel you are ready for the final answer, directly return the answer with the format ` + code(`<<<answer content>>>`) + `
at the end of your response. Otherwise, you can continue your reasoning process and possibly
generate more code or search queries to solve the problem.

**Your goal is to determine which method will be most effective for solving the task.** Then you generate
the guidance prompt for the TaskLLM to follow in the next round. The final returned guidance prompt
should be included between ` + code(`<<<`) + ` and ` + code(`>>>`) + `, such as ` + code(`<<<You need to generate more complex code to solve...>>>`) + `.
Now, here is the task:`,
	}

	applySharedContext(&cfg)

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Guided(CSGcom) agent: %w", err)
	}

	return a, nil
}

// NewGuidedPlusGSAgent creates a Guided+ Agent that uses both code execution and Google Search API with extra priors.
//
// This agent is responsible for "13. Guided Agent+ (CSG+gs)".
func NewGuidedPlusGSAgent(llm model.LLM, genCfg *genai.GenerateContentConfig) (agent.Agent, error) {
	cfg := llmagent.Config{
		Name: "guided-plus-google-search",
		Description: `Guided dual-tool with stronger priors combining code execution and Google Search API search.
- Short name: {CSG+gs}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `You steer a TaskLLM with explicit, actionable guidance. Use text, code, and Google Search.

Priority rules:
- Prefer code when arithmetic/symbolic reasoning is involved; isolate a single <language> block with print outputs.
- Prefer search when facts/dates/entities are uncertain; emit exactly one ` + code(`<search>query</search>`) + ` per turn.
- Never mix code and search in the same turn.
- Keep guidance concise: 1–3 bullet steps plus a concrete next action.

When ready, output the guidance between ` + code(`<<<`) + ` and ` + code(`>>>`) + `, e.g.,` + code(`<<<Run a short Python script to factor the polynomial, then verify with a quick search.>>>`) + `.`,
	}

	applySharedContext(&cfg)

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Guided+(CSG+gs) agent: %w", err)
	}

	return a, nil
}

// NewGuidedPlusLLMAgent creates a Guided+ Agent that uses both code execution and LLM search with extra priors.
//
// This agent is responsible for "14. Guided Agent+ (CSG+llm)".
func NewGuidedPlusLLMAgent(llm model.LLM, genCfg *genai.GenerateContentConfig) (agent.Agent, error) {
	cfg := llmagent.Config{
		Name: "guided-plus-llm-search",
		Description: `Guided dual-tool with stronger priors combining code execution and LLM search function.
- Short name: {CSG+llm}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `You steer a TaskLLM with explicit, actionable guidance. Use text, code, and LLM search.

Priority rules:
- Prefer code for math/logic; emit one <language> block with print outputs only.
- Prefer search when factual uncertainty is high; emit one <search>query</search>.
- Do not mix code and search in the same turn.
- Keep guidance concise: 1–3 bullet steps plus a concrete next action.

Return guidance between «< and »>.`,
	}

	applySharedContext(&cfg)

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Guided+(CSG+llm) agent: %w", err)
	}

	return a, nil
}

// NewGuidedPlusComAgent creates a Guided+ Agent that uses both code execution and combined search with extra priors.
//
// This agent is responsible for "15. Guided Agent+ (CSG+com)".
func NewGuidedPlusComAgent(llm model.LLM, genCfg *genai.GenerateContentConfig) (agent.Agent, error) {
	cfg := llmagent.Config{
		Name: "guided-plus-combine-Search",
		Description: `Guided dual-tool with stronger priors combining code execution and mixed Google/LLM search.
- Short name: {CSG+com}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `You steer a TaskLLM with explicit, actionable guidance. Use text, code, and either Google Search or LLM search.

Priority rules mirror Guided+gs/llm: choose code for computation, search for facts, never mix both in one turn, keep one action per turn, and provide crisp bullets.

Return guidance between ` + code(`<<<`) + ` and ` + code(`>>>`) + `.`,
	}

	applySharedContext(&cfg)

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Guided+(CSG+com) agent: %w", err)
	}

	return a, nil
}

// NewRefinementAgent creates a Refinement Agent that gathers candidate answers from sub-agents and judges the final answer.
func NewRefinementAgent(subAgents ...agent.Agent) (agent.Agent, error) {
	parallel, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{
			Name:        "candidates",
			Description: "Runs diverse tool-use agents in parallel.",
			SubAgents:   subAgents,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("build candidates workflow: %w", err)
	}

	cfg := llmagent.Config{
		Name:        "Refinement",
		Description: `One TUMIX round: gather candidates then judge.`,
		SubAgents:   []agent.Agent{parallel},
		Instruction: `**Task**: Decide the final answer based on the following answers from other agents.

**Question**:
{question}

**Candidate answers from several methods**:
{joined_answers}

Based on the candidates above, analyze the question step by step and try to list all the careful points. In the
end of your response, directly output the answer to the question with the format ` + code(`«<answer content»>`) + `.`,
	}

	applySharedContext(&cfg)

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Refinement agent: %w", err)
	}

	return a, nil
}

const (
	defaultMinRounds           uint    = 2
	defaultMaxRounds           uint    = 10
	defaultConfidenceThreshold float64 = 0.6

	stateKeyAnswer      = "tumix_final_answer"
	stateKeyConfidence  = "tumix_final_confidence"
	stateKeyQuestion    = "question"
	stateKeyJoined      = "joined_answers"
	stateKeyRound       = "round_num"
	stateKeyVoteMargin  = "vote_margin"
	stateKeyUnique      = "unique_answers"
	stateKeyCoverage    = "coverage"
	stateKeyEntropy     = "answer_entropy"
	stateKeyTopAnswer   = "top_answer"
	stateKeyJudgeAnswer = "judge_recommended_answer"
)

type finalizeArgs struct {
	Answer     string  `json:"answer,omitzero"`
	Confidence float64 `json:"confidence,omitzero"`
	Stop       bool    `json:"stop,omitzero"`
}

type finalizeResult struct {
	Stored bool `json:"stored,omitzero"`
	Stop   bool `json:"stop,omitzero"`
}

func newFinalizeTool() (tool.Tool, error) {
	cfg := functiontool.Config{
		Name:        "finalize",
		Description: "Store the selected answer, confidence, and optionally stop further rounds.",
	}

	t, err := functiontool.New(cfg, func(ctx tool.Context, args finalizeArgs) (finalizeResult, error) {
		answer := strings.TrimSpace(args.Answer)
		switch {
		case answer == "":
			return finalizeResult{}, fmt.Errorf("answer is required")
		case args.Confidence < 0 || args.Confidence > 1:
			return finalizeResult{}, fmt.Errorf("confidence must be between 0 and 1")
		}

		if err := ctx.State().Set(stateKeyAnswer, answer); err != nil {
			return finalizeResult{}, fmt.Errorf("set answer state: %w", err)
		}
		if err := ctx.State().Set(stateKeyConfidence, args.Confidence); err != nil {
			return finalizeResult{}, fmt.Errorf("set confidence state: %w", err)
		}

		if args.Stop {
			ctx.Actions().Escalate = true
		}

		return finalizeResult{
			Stored: true,
			Stop:   args.Stop,
		}, nil
	})
	if err != nil {
		return nil, fmt.Errorf("build finalize tool: %w", err)
	}
	return t, nil
}

// NewJudgeAgent creates a Judge Agent that evaluates candidate answers and decides whether to finalize or continue.
func NewJudgeAgent(llm model.LLM, genCfg *genai.GenerateContentConfig) (agent.Agent, error) {
	finalizeTool, err := newFinalizeTool()
	if err != nil {
		return nil, fmt.Errorf("build finalize tool: %w", err)
	}

	cfg := llmagent.Config{
		Name:                  "LLM-as-Judge",
		Description:           `Ranks candidate agent outputs and signals when to stop.`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Tools:                 []tool.Tool{finalizeTool},
		Instruction: `Task: Decide STOP or CONTINUE; do not solve the problem yourself.

Round {round_num}; vote margin {vote_margin?}; unique answers {unique_answers?}; coverage {coverage?}; entropy {answer_entropy?}.

Stop only when:
- vote margin >= ` + fmt.Sprintf("%.2f", defaultConfidenceThreshold) + ` AND round >= 2; and
- no material differences in reasoning or conclusions.

Otherwise continue.

Question:
{question}

Candidate answers:
{joined_answers}

Instructions:
1. Briefly compare answers; highlight disagreements or uncertainties.
2. Choose the best current answer (copy verbatim); call finalize exactly once with answer, confidence 0-1, stop=true only when conditions met.
3. If not safe to stop, call finalize with stop=false.

End with ` + code(`<<<YES>>>`) + ` when you set stop=true, else ` + code(`<<<NO>>>`) + `.`,
	}

	applySharedContext(&cfg)

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Judge agent: %w", err)
	}

	return a, nil
}

// NewRoundAgent creates a Round Agent that performs one TUMIX round: gathering candidates and judging.
func NewRoundAgent(subAgents ...agent.Agent) (agent.Agent, error) {
	cfg := sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:        "round",
			Description: "One TUMIX round: gather candidates then judge.",
			SubAgents:   subAgents,
		},
	}
	a, err := sequentialagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Round agent: %w", err)
	}

	return a, nil
}

type TumixConfig struct {
	Candidates []agent.Agent
	Judge      agent.Agent
	MaxRounds  uint
	MinRounds  uint
}

// NewTumixAgent creates the TUMIX Agent that performs multi-agent test-time scaling with tool-use mixture.
func NewTumixAgent(candidates []agent.Agent, judge agent.Agent) (agent.Loader, error) {
	return NewTumixAgentWithMaxRounds(candidates, judge, defaultMaxRounds)
}

// NewTumixAgentWithMaxRounds creates the TUMIX Agent with a configurable
// maximum number of iterations.
func NewTumixAgentWithMaxRounds(candidates []agent.Agent, judge agent.Agent, maxRounds uint) (agent.Loader, error) {
	return NewTumixAgentWithConfig(TumixConfig{
		Candidates: candidates,
		Judge:      judge,
		MinRounds:  defaultMinRounds,
		MaxRounds:  maxRounds,
	})
}

// NewTumixAgentWithConfig creates an orchestrated TUMIX loader that
// propagates previous round answers and consults the judge for early stop.
func NewTumixAgentWithConfig(cfg TumixConfig) (agent.Loader, error) {
	if len(cfg.Candidates) == 0 {
		return nil, errors.New("at least one candidate agent is required")
	}
	if cfg.Judge == nil {
		return nil, errors.New("judge agent is required")
	}
	if cfg.MaxRounds == 0 {
		cfg.MaxRounds = defaultMaxRounds
	}
	if cfg.MinRounds == 0 {
		cfg.MinRounds = defaultMinRounds
	}
	if cfg.MinRounds > cfg.MaxRounds {
		cfg.MinRounds = cfg.MaxRounds
	}

	parallel, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{
			Name:        "candidates",
			Description: "Runs diverse tool-use agents in parallel.",
			SubAgents:   cfg.Candidates,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("build candidates workflow: %w", err)
	}

	orchestrator := &tumixOrchestrator{
		candidateAgent: parallel,
		judge:          cfg.Judge,
		maxRounds:      cfg.MaxRounds,
		minRounds:      cfg.MinRounds,
	}

	tumix, err := agent.New(agent.Config{
		Name:        "tumix",
		Description: "TUMIX: Multi-Agent Test-Time Scaling with Tool-Use Mixture.",
		SubAgents:   append([]agent.Agent{parallel}, cfg.Judge),
		Run:         orchestrator.run,
	})
	if err != nil {
		return nil, fmt.Errorf("build tumix agent: %w", err)
	}

	return agent.NewSingleLoader(tumix), nil
}

type tumixOrchestrator struct {
	candidateAgent agent.Agent
	judge          agent.Agent
	maxRounds      uint
	minRounds      uint
	prevTopAnswer  string
	prevVoteMargin float64
}

type candidateAnswer struct {
	Agent string
	Text  string
}

func (t *tumixOrchestrator) run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		question := firstContentText(ctx.UserContent())
		if err := setState(ctx, stateKeyQuestion, question); err != nil {
			yield(nil, err)
			return
		}

		var lastAnswers []candidateAnswer
		for round := uint(1); round <= t.maxRounds; round++ {
			if err := setState(ctx, stateKeyRound, round); err != nil {
				yield(nil, err)
				return
			}
			if err := setState(ctx, stateKeyJoined, joinAnswers(lastAnswers)); err != nil {
				yield(nil, err)
				return
			}

			answers, stop := t.runCandidates(ctx, yield)
			if stop {
				return
			}
			lastAnswers = answers
			if len(lastAnswers) == 0 {
				if round < t.minRounds {
					continue
				}
				if t.runJudge(ctx, yield) {
					t.emitFinalFromState(ctx, yield)
					return
				}
				continue
			}

			if err := setState(ctx, stateKeyJoined, joinAnswers(lastAnswers)); err != nil {
				yield(nil, err)
				return
			}
			stats := computeStats(lastAnswers, len(t.candidateAgent.SubAgents()))
			if err := setState(ctx, stateKeyVoteMargin, stats.voteMargin); err != nil {
				yield(nil, err)
				return
			}
			if err := setState(ctx, stateKeyUnique, stats.unique); err != nil {
				yield(nil, err)
				return
			}
			if err := setState(ctx, stateKeyCoverage, stats.coverage); err != nil {
				yield(nil, err)
				return
			}
			if err := setState(ctx, stateKeyEntropy, stats.answerEntropy); err != nil {
				yield(nil, err)
				return
			}
			if err := setState(ctx, stateKeyTopAnswer, stats.topAnswer); err != nil {
				yield(nil, err)
				return
			}

			if round >= t.minRounds && stats.topAnswer != "" && stats.topAnswer == t.prevTopAnswer && stats.voteMargin >= defaultConfidenceThreshold && t.prevVoteMargin >= defaultConfidenceThreshold {
				if err := setState(ctx, stateKeyAnswer, stats.topAnswer); err != nil {
					yield(nil, err)
					return
				}
				if err := setState(ctx, stateKeyConfidence, stats.voteMargin); err != nil {
					yield(nil, err)
					return
				}
				t.emitFinalFromState(ctx, yield)
				return
			}

			if round >= t.minRounds && stats.topAnswer != "" {
				t.prevTopAnswer = stats.topAnswer
				t.prevVoteMargin = stats.voteMargin
			}

			if round < t.minRounds {
				continue
			}

			if t.runJudge(ctx, yield) {
				t.emitFinalFromState(ctx, yield)
				return
			}
		}

		if len(lastAnswers) > 0 {
			answer, conf := majorityVote(lastAnswers)
			if err := setState(ctx, stateKeyAnswer, answer); err != nil {
				yield(nil, err)
				return
			}
			if err := setState(ctx, stateKeyConfidence, conf); err != nil {
				yield(nil, err)
				return
			}
		}
		if err := setState(ctx, stateKeyJoined, joinAnswers(lastAnswers)); err != nil {
			yield(nil, err)
			return
		}
		t.emitFinalFromState(ctx, yield)
	}
}

func (t *tumixOrchestrator) runCandidates(ctx agent.InvocationContext, yield func(*session.Event, error) bool) ([]candidateAnswer, bool) {
	answers := make([]candidateAnswer, 0, len(t.candidateAgent.SubAgents()))
	for event, err := range t.candidateAgent.Run(ctx) {
		if !yield(event, err) {
			return answers, true
		}
		if err != nil || event == nil || event.Content == nil {
			continue
		}
		text := firstTextFromContent(event.Content)
		if text == "" {
			continue
		}
		answers = append(answers, candidateAnswer{Agent: event.Author, Text: strings.TrimSpace(text)})
	}
	return answers, false
}

func (t *tumixOrchestrator) runJudge(ctx agent.InvocationContext, yield func(*session.Event, error) bool) bool {
	stop := false
	for event, err := range t.judge.Run(ctx) {
		if !yield(event, err) {
			return true
		}
		if err != nil {
			continue
		}
		if event != nil && event.Actions.Escalate {
			stop = true
			if text := firstTextFromContent(event.Content); text != "" {
				if err := setState(ctx, stateKeyJudgeAnswer, normalizeAnswer(text)); err != nil {
					yield(nil, err)
					return true
				}
			}
		}
	}
	return stop
}

func (t *tumixOrchestrator) emitFinalFromState(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
	answerVal, err := getState(ctx, stateKeyAnswer)
	if err != nil && !errors.Is(err, session.ErrStateKeyNotExist) {
		yield(nil, err)
		return
	}
	confVal, err := getState(ctx, stateKeyConfidence)
	if err != nil && !errors.Is(err, session.ErrStateKeyNotExist) {
		yield(nil, err)
		return
	}
	if answerVal == nil {
		judgeVal, jerr := getState(ctx, stateKeyJudgeAnswer)
		if jerr != nil && !errors.Is(jerr, session.ErrStateKeyNotExist) {
			yield(nil, jerr)
			return
		}
		if judgeVal != nil {
			answerVal = judgeVal
		}
	}
	answer := fmt.Sprintf("%v", answerVal)
	conf := fmt.Sprintf("%v", confVal)
	if answer == "" {
		return
	}

	content := genai.NewContentFromText(fmt.Sprintf("Final answer (conf %s): %s", conf, answer), genai.RoleModel)
	event := session.NewEvent(ctx.InvocationID())
	event.Author = "tumix"
	event.Content = content
	if event.Actions.StateDelta == nil {
		event.Actions.StateDelta = make(map[string]any)
	}
	event.Actions.StateDelta[stateKeyAnswer] = answerVal
	if confVal != nil {
		event.Actions.StateDelta[stateKeyConfidence] = confVal
	}
	joinedVal, jerr := getState(ctx, stateKeyJoined)
	if jerr != nil && !errors.Is(jerr, session.ErrStateKeyNotExist) {
		yield(nil, jerr)
		return
	}
	if joinedVal != nil {
		event.Actions.StateDelta[stateKeyJoined] = joinedVal
	}
	yield(event, nil)
}

func joinAnswers(ans []candidateAnswer) string {
	if len(ans) == 0 {
		return ""
	}
	sb := strings.Builder{}
	for i, a := range ans {
		if i > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString(fmt.Sprintf("- %s: %s", a.Agent, strings.TrimSpace(a.Text)))
	}
	return sb.String()
}

func firstTextFromContent(c *genai.Content) string {
	if c == nil {
		return ""
	}
	for _, p := range c.Parts {
		if p == nil {
			continue
		}
		if p.Text != "" {
			return p.Text
		}
	}
	return ""
}

func firstContentText(c *genai.Content) string {
	if c == nil {
		return ""
	}
	for _, p := range c.Parts {
		if p != nil && p.Text != "" {
			return strings.TrimSpace(p.Text)
		}
	}
	return ""
}

func majorityVote(ans []candidateAnswer) (answer string, confidence float64) {
	if len(ans) == 0 {
		return "", 0
	}
	cnts := make(map[string]int)
	for _, a := range ans {
		key := normalizeAnswer(a.Text)
		cnts[key]++
	}
	type kv struct {
		Answer string
		Count  int
	}
	pairs := make([]kv, 0, len(cnts))
	for k, v := range cnts {
		pairs = append(pairs, kv{Answer: k, Count: v})
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].Count == pairs[j].Count {
			return pairs[i].Answer < pairs[j].Answer
		}
		return pairs[i].Count > pairs[j].Count
	})
	best := pairs[0]
	confidence = float64(best.Count) / float64(len(ans))
	return best.Answer, confidence
}

type roundStats struct {
	voteMargin    float64
	unique        int
	coverage      float64
	answerEntropy float64
	topAnswer     string
}

func computeStats(ans []candidateAnswer, candidateCount int) roundStats {
	if len(ans) == 0 || candidateCount <= 0 {
		return roundStats{}
	}

	cnts := make(map[string]int)
	for _, a := range ans {
		key := normalizeAnswer(a.Text)
		cnts[key]++
	}

	unique := len(cnts)
	total := len(ans)
	topCount := 0
	topAnswer := ""
	entropy := 0.0
	for k, v := range cnts {
		if v > topCount || (v == topCount && k < topAnswer) {
			topCount, topAnswer = v, k
		}
		p := float64(v) / float64(total)
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}

	return roundStats{
		voteMargin:    float64(topCount) / float64(total),
		unique:        unique,
		coverage:      float64(total) / float64(candidateCount),
		answerEntropy: entropy,
		topAnswer:     topAnswer,
	}
}

func normalizeAnswer(text string) string {
	trimmed := strings.TrimSpace(text)
	trimmed = strings.TrimPrefix(trimmed, "«<")
	trimmed = strings.TrimSuffix(trimmed, "»>")
	return strings.TrimSpace(trimmed)
}
