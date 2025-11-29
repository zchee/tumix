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

package xai

import (
	json "encoding/json/v2"
	"time"

	"google.golang.org/protobuf/types/known/timestamppb"

	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
)

// Tool builds a function-calling tool definition.
func Tool(name, description string, parameters any) (*xaipb.Tool, error) {
	data, err := json.Marshal(parameters)
	if err != nil {
		return nil, err
	}
	return &xaipb.Tool{Tool: &xaipb.Tool_Function{Function: &xaipb.Function{
		Name:        name,
		Description: description,
		Parameters:  string(data),
	}}}, nil
}

// MustTool is a convenience that panics on error; useful in init paths.
func MustTool(name, description string, parameters any) *xaipb.Tool {
	tool, err := Tool(name, description, parameters)
	if err != nil {
		panic(err)
	}
	return tool
}

// RequiredTool creates a tool choice that forces invocation of the given tool name.
func RequiredTool(name string) *xaipb.ToolChoice {
	return &xaipb.ToolChoice{
		ToolChoice: &xaipb.ToolChoice_FunctionName{
			FunctionName: name,
		},
	}
}

// WebSearchTool defines a server-side web search tool.
func WebSearchTool(excludedDomains, allowedDomains []string, enableImageUnderstanding bool) *xaipb.Tool {
	enable := enableImageUnderstanding
	return &xaipb.Tool{
		Tool: &xaipb.Tool_WebSearch{
			WebSearch: &xaipb.WebSearch{
				ExcludedDomains:          excludedDomains,
				AllowedDomains:           allowedDomains,
				EnableImageUnderstanding: &enable,
			},
		},
	}
}

// XSearchTool defines a server-side X (Twitter) search tool.
func XSearchTool(fromDate, toDate *time.Time, allowedHandles, excludedHandles []string, enableImageUnderstanding, enableVideoUnderstanding bool) *xaipb.Tool {
	toTS := func(t *time.Time) *timestamppb.Timestamp {
		if t == nil {
			return nil
		}
		return timestamppb.New(*t)
	}
	img := enableImageUnderstanding
	video := enableVideoUnderstanding
	return &xaipb.Tool{
		Tool: &xaipb.Tool_XSearch{
			XSearch: &xaipb.XSearch{
				FromDate:                 toTS(fromDate),
				ToDate:                   toTS(toDate),
				AllowedXHandles:          allowedHandles,
				ExcludedXHandles:         excludedHandles,
				EnableImageUnderstanding: &img,
				EnableVideoUnderstanding: &video,
			},
		},
	}
}

// CodeExecutionTool enables server-side code execution.
func CodeExecutionTool() *xaipb.Tool {
	return &xaipb.Tool{
		Tool: &xaipb.Tool_CodeExecution{
			CodeExecution: &xaipb.CodeExecution{},
		},
	}
}

// CollectionsSearchTool allows querying collections from agentic responses.
func CollectionsSearchTool(collectionIDs []string, limit int32) *xaipb.Tool {
	return &xaipb.Tool{
		Tool: &xaipb.Tool_CollectionsSearch{
			CollectionsSearch: &xaipb.CollectionsSearch{
				CollectionIds: collectionIDs,
				Limit:         ptr(limit),
			},
		},
	}
}

// CollectionsSearchToolIDs is a convenience wrapper that accepts variadic IDs.
func CollectionsSearchToolIDs(limit int32, collectionIDs ...string) *xaipb.Tool {
	return CollectionsSearchTool(collectionIDs, limit)
}

// MCPTool connects to a remote MCP server.
func MCPTool(serverURL, serverLabel, serverDescription string, allowedToolNames []string, authorization string, extraHeaders map[string]string) *xaipb.Tool {
	var auth *string
	if authorization != "" {
		auth = &authorization
	}
	return &xaipb.Tool{
		Tool: &xaipb.Tool_Mcp{
			Mcp: &xaipb.MCP{
				ServerUrl:         serverURL,
				ServerLabel:       serverLabel,
				ServerDescription: serverDescription,
				AllowedToolNames:  allowedToolNames,
				Authorization:     auth,
				ExtraHeaders:      extraHeaders,
			},
		},
	}
}

// DocumentSearchTool enables server-side document search.
func DocumentSearchTool(limit int32) *xaipb.Tool {
	return &xaipb.Tool{
		Tool: &xaipb.Tool_DocumentSearch{
			DocumentSearch: &xaipb.DocumentSearch{
				Limit: &limit,
			},
		},
	}
}
