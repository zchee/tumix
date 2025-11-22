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
	"encoding/json"
	"time"

	"google.golang.org/protobuf/types/known/timestamppb"

	pb "github.com/zchee/tumix/model/xai/pb/xai/api/v1"
)

// Tool builds a function-calling tool definition.
func Tool(name, description string, parameters any) (*pb.Tool, error) {
	data, err := json.Marshal(parameters)
	if err != nil {
		return nil, err
	}
	return &pb.Tool{Tool: &pb.Tool_Function{Function: &pb.Function{
		Name:        name,
		Description: description,
		Parameters:  string(data),
	}}}, nil
}

// RequiredTool creates a tool choice that forces invocation of the given tool name.
func RequiredTool(name string) *pb.ToolChoice {
	return &pb.ToolChoice{ToolChoice: &pb.ToolChoice_FunctionName{FunctionName: name}}
}

// WebSearchTool defines a server-side web search tool.
func WebSearchTool(excludedDomains, allowedDomains []string, enableImageUnderstanding bool) *pb.Tool {
	enable := enableImageUnderstanding
	return &pb.Tool{Tool: &pb.Tool_WebSearch{WebSearch: &pb.WebSearch{
		ExcludedDomains:          excludedDomains,
		AllowedDomains:           allowedDomains,
		EnableImageUnderstanding: &enable,
	}}}
}

// XSearchTool defines a server-side X (Twitter) search tool.
func XSearchTool(fromDate, toDate *time.Time, allowedHandles, excludedHandles []string, enableImageUnderstanding, enableVideoUnderstanding bool) *pb.Tool {
	toTS := func(t *time.Time) *timestamppb.Timestamp {
		if t == nil {
			return nil
		}
		return timestamppb.New(*t)
	}
	img := enableImageUnderstanding
	video := enableVideoUnderstanding
	return &pb.Tool{Tool: &pb.Tool_XSearch{XSearch: &pb.XSearch{
		FromDate:                 toTS(fromDate),
		ToDate:                   toTS(toDate),
		AllowedXHandles:          allowedHandles,
		ExcludedXHandles:         excludedHandles,
		EnableImageUnderstanding: &img,
		EnableVideoUnderstanding: &video,
	}}}
}

// CodeExecutionTool enables server-side code execution.
func CodeExecutionTool() *pb.Tool {
	return &pb.Tool{Tool: &pb.Tool_CodeExecution{CodeExecution: &pb.CodeExecution{}}}
}

// CollectionsSearchTool allows querying collections from agentic responses.
func CollectionsSearchTool(collectionIDs []string, limit int32) *pb.Tool {
	return &pb.Tool{Tool: &pb.Tool_CollectionsSearch{CollectionsSearch: &pb.CollectionsSearch{
		CollectionIds: collectionIDs,
		Limit:         &limit,
	}}}
}

// MCPTool connects to a remote MCP server.
func MCPTool(serverURL string, serverLabel, serverDescription string, allowedToolNames []string, authorization string, extraHeaders map[string]string) *pb.Tool {
	var auth *string
	if authorization != "" {
		auth = &authorization
	}
	return &pb.Tool{Tool: &pb.Tool_Mcp{Mcp: &pb.MCP{
		ServerUrl:         serverURL,
		ServerLabel:       serverLabel,
		ServerDescription: serverDescription,
		AllowedToolNames:  allowedToolNames,
		Authorization:     auth,
		ExtraHeaders:      extraHeaders,
	}}}
}
