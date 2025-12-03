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
	"fmt"
	"strings"

	"github.com/google/dotprompt/go/dotprompt"
	"github.com/invopop/jsonschema"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/loopagent"
	"google.golang.org/adk/agent/workflowagents/parallelagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/model"
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
			return nil, nil
		},
		PartialResolver: func(partialName string) (string, error) {
			return "", nil
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
	copy := *cfg
	return &copy
}

// NewBaseAgent creates a Base Agent that uses direct prompting to solve problems.
//
// This agent is responsible for "1. w/o TTS (Base)".
func NewBaseAgent(llm model.LLM, genCfg *genai.GenerateContentConfig) (agent.Agent, error) {
	cfg := llmagent.Config{
		Name: "Base",
		Description: `Direct prompt.
- Short name: {Base}.`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
	}

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
		Name: "CoT",
		Description: `Chain-of-thought text-only reasoning.
- Short name: {CoT}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `• Analyze the question step by step and try to list all the careful points.
• Then try to acquire the final answer with step by step analysis.
• In the end of your response, directly output the answer to the question.

**Do not output the code for execution.**`,
	}

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
		Name: "CoT code",
		Description: `Chain-of-thought text-only reasoning and output code.
- Short name: {CoT code}`,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `You are a helpful AI assistant. Solve tasks using your coding skills.
In the following cases, suggest python code (in a python coding block) for the user to execute.

* Don’t include multiple code blocks in one response, **only include one** in the response.
* Do not ask users to copy and paste the result. Instead, use the ’print’ function for the output when relevant.

Think the task step by step if you need to. If a plan is not provided, explain your plan first. You can first
output your thinking steps with texts and then the final python code.

**Remember in the final code you still need to output each number or choice in the final print!**

Start the python block with ` + "```" + `python`,
	}

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
		Name: "Search",
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
		Name: "Code",
		Description: `Code-execution strategy for precise computation.
- Short name: {C}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `The User asks a question, and you solve it. You first generate the reasoning and thinking process and
then provide the User with the final answer.

During the thinking process, **you can generate python code** for efficient searching, optimization, and computing with the format of starting the python block
with ` + "```" + `python. **A code query must involve only a single script that uses ‘print’
function for the output.**. Once the code script is complete, stop the generation. Then, the code
interpreter platform will execute the code and return the execution output and error. Once you feel you are
ready for the final answer, directly return the answer with the format <<<answer content>>> at the end
of your response. Otherwise, you can continue your reasoning process and possibly generate more code
query to solve the problem.`,
	}

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
		Name: "Code+",
		Description: `Code-execution strategy for precise computation with a hinted version with extra human-pre-designed priors.
- Short name: {C+}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `The User asks a question, and you solve it. You first generate the reasoning and thinking process and
then provide the User with the final answer.

During the thinking process, **you can generate python code** for efficient searching, optimization, and computing with the format of starting the python block
with ` + "```" + `python. **A code query must involve only a single script that uses ‘print’
function for the output.**. Once the code script is complete, stop the generation. Then, the code
interpreter platform will execute the code and return the execution output and error. Once you feel you are
ready for the final answer, directly return the answer with the format <<<answer content>>> at the end
of your response. Otherwise, you can continue your reasoning process and possibly generate more code
query to solve the problem.`,
	}

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
		Name: "Dual-Tool",
		Description: `Dual-tool strategy combining code execution and Google Search API search.
- Short name: {CSgs}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `The User asks a question, and you solve it. You first generate the reasoning and thinking process and then
provide the User with the final answer.

During the thinking process, _you can generate python code_ for efficient searching, optimization, and
computing with the format of starting the python block with ` + "```" + `python. **A code query must
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

Once you feel you are ready for the final answer, directly return the answer with the format ` + code(`«<answer content»>`) + `
at the end of your response. Otherwise, you can continue your reasoning process and possibly
generate more code or search queries to solve the problem.`,
	}

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
		Name: "Dual-Tool",
		Description: `Dual-tool strategy combining code execution and LLM search function.
- Short name: {CSllm}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `The User asks a question, and you solve it. You first generate the reasoning and thinking process and then
provide the User with the final answer.

During the thinking process, _you can generate python code_ for efficient searching, optimization, and
computing with the format of starting the python block with ` + "```" + `python. **A code query must
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

Once you feel you are ready for the final answer, directly return the answer with the format ` + code(`«<answer content»>`) + `
at the end of your response. Otherwise, you can continue your reasoning process and possibly
generate more code or search queries to solve the problem.`,
	}

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
		Name: "Dual-Tool",
		Description: `Dual-tool strategy combining code execution and combination of Google Search API search and LLM search function.
- Short name: {CScom}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `The User asks a question, and you solve it. You first generate the reasoning and thinking process and then
provide the User with the final answer.

During the thinking process, _you can generate python code_ for efficient searching, optimization, and
computing with the format of starting the python block with ` + "```" + `python. **A code query must
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

Once you feel you are ready for the final answer, directly return the answer with the format ` + code(`«<answer content»>`) + `
at the end of your response. Otherwise, you can continue your reasoning process and possibly
generate more code or search queries to solve the problem.`,
	}

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
		Name: "Guided",
		Description: `Dual-tool strategy combining code execution and Google Search API search.
- Short name: {CSG}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `You are guiding another TaskLLM to solve a task. You will be presented with a task that can be solved using
textual reasoning, coding, and web searching. Sometimes the TaskLLM may need extra help to solve the
task, such as generating code or searching the web. Then must follow the rules below for both query and
return answer:

During the thinking process, _you can generate python code_ for efficient searching, optimization, and
computing with the format of starting the python block with ` + "```" + `python. **A code query must involve
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

Once you feel you are ready for the final answer, directly return the answer with the format ` + code(`«<answer content»>`) + `
at the end of your response. Otherwise, you can continue your reasoning process and possibly
generate more code or search queries to solve the problem.

**Your goal is to determine which method will be most effective for solving the task.** Then you generate
the guidance prompt for the TaskLLM to follow in the next round. The final returned guidance prompt
should be included between ` + code(`«<`) + ` and ` + code(`»>`) + `, such as ` + code(`«<You need to generate more complex code to solve...»>`) + `.
Now, here is the task:`,
	}

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
		Name: "Guided",
		Description: `Dual-tool strategy combining code execution and LLM search function.
- Short name: {CSGllm}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `You are guiding another TaskLLM to solve a task. You will be presented with a task that can be solved using
textual reasoning, coding, and web searching. Sometimes the TaskLLM may need extra help to solve the
task, such as generating code or searching the web. Then must follow the rules below for both query and
return answer:

During the thinking process, _you can generate python code_ for efficient searching, optimization, and
computing with the format of starting the python block with ` + "```" + `python. **A code query must involve
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

Once you feel you are ready for the final answer, directly return the answer with the format ` + code(`«<answer content»>`) + `
at the end of your response. Otherwise, you can continue your reasoning process and possibly
generate more code or search queries to solve the problem.

**Your goal is to determine which method will be most effective for solving the task.** Then you generate
the guidance prompt for the TaskLLM to follow in the next round. The final returned guidance prompt
should be included between ` + code(`«<`) + ` and ` + code(`»>`) + `, such as ` + code(`«<You need to generate more complex code to solve...»>`) + `.
Now, here is the task:`,
	}

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
		Name: "Guided",
		Description: `Dual-tool strategy combining code execution and combination of Google Search API search and LLM search function.
- Short name: {CSGcom}`,
		Model:                 llm,
		GenerateContentConfig: cloneGenConfig(genCfg),
		Instruction: `You are guiding another TaskLLM to solve a task. You will be presented with a task that can be solved using
textual reasoning, coding, and web searching. Sometimes the TaskLLM may need extra help to solve the
task, such as generating code or searching the web. Then must follow the rules below for both query and
return answer:

During the thinking process, _you can generate python code_ for efficient searching, optimization, and
computing with the format of starting the python block with ` + "```" + `python. **A code query must involve
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

Once you feel you are ready for the final answer, directly return the answer with the format ` + code(`«<answer content»>`) + `
at the end of your response. Otherwise, you can continue your reasoning process and possibly
generate more code or search queries to solve the problem.

**Your goal is to determine which method will be most effective for solving the task.** Then you generate
the guidance prompt for the TaskLLM to follow in the next round. The final returned guidance prompt
should be included between ` + code(`«<`) + ` and ` + code(`»>`) + `, such as ` + code(`«<You need to generate more complex code to solve...»>`) + `.
Now, here is the task:`,
	}

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Guided(CSGcom) agent: %w", err)
	}

	return a, nil
}

// NewRefinementAgent creates a Refinement Agent that gathers candidate answers from sub-agents and judges the final answer.
//
// TODO(zchee): support [agent.InvocationContext] for `{question}` and `{joined_answers}`.
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

	a, err := llmagent.New(cfg)
	if err != nil {
		return nil, fmt.Errorf("build Refinement agent: %w", err)
	}

	return a, nil
}

const (
	defaultMaxRounds           uint    = 10
	defaultConfidenceThreshold float64 = 0.6

	stateKeyAnswer     = "tumix_final_answer"
	stateKeyConfidence = "tumix_final_confidence"
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

	return functiontool.New(cfg, func(ctx tool.Context, args finalizeArgs) (finalizeResult, error) {
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
}

// NewJudgeAgent creates a Judge Agent that evaluates candidate answers and decides whether to finalize or continue.
//
// TODO(zchee): support [agent.InvocationContext] for `{round_num}`, `{question}` and `{joined_answers}`.
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
		Instruction: `Task: Carefully assess whether the answers below (enclosed by ` + code(`«< »>`) + `) show clear and strong consensus, or
if another round of reasoning is needed to improve alignment.

**IMPORTANT**: If there are any differences in reasoning, phrasing, emphasis, conclusions, or interpretation of
key details, you should conservatively decide to continue refinement.

The current round number is {round_num}. Note: **Finalizing before round 3 is uncommon and discouraged unless answers are fully aligned in both logic and language.**

**Question**:
{question}

**Candidate answers from different methods**:
{joined_answers}

**Instructions**:
1. Identify any differences in wording, structure, or logic.
2. Be especially cautious about subtle variations in conclusion or emphasis.
3. Err on the side of caution: if there’s any ambiguity or divergence, recommend another round.
4. Pick the best answer, justify briefly, then call the finalize tool exactly once with:
	- answer: the final answer string
	- confidence: 0-1
	- stop: true when confidence >= ` + fmt.Sprintf("%.2f", defaultConfidenceThreshold) + ` else false.

Output your reasoning first, then conclude clearly with ` + code(`«<YES»>`) + ` if the answers are highly consistent and
finalization is safe, or ` + code(`«<NO»>`) + ` if further refinement is needed.`,
	}

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

// NewTumixAgent creates the TUMIX Agent that performs multi-agent test-time scaling with tool-use mixture.
func NewTumixAgent(subAgents ...agent.Agent) (agent.Loader, error) {
	return NewTumixAgentWithMaxRounds(subAgents, defaultMaxRounds)
}

// NewTumixAgentWithMaxRounds creates the TUMIX Agent with a configurable
// maximum number of iterations.
func NewTumixAgentWithMaxRounds(subAgents []agent.Agent, maxRounds uint) (agent.Loader, error) {
	if maxRounds == 0 {
		maxRounds = defaultMaxRounds
	}

	a, err := loopagent.New(loopagent.Config{
		AgentConfig: agent.Config{
			Name:        "tumix",
			Description: "TUMIX: Multi-Agent Test-Time Scaling with Tool-Use Mixture.",
			SubAgents:   subAgents,
		},
		MaxIterations: maxRounds,
	})
	if err != nil {
		return nil, fmt.Errorf("build tumix agent: %w", err)
	}

	return agent.NewSingleLoader(a), nil
}
