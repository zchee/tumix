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
	"context"

	"google.golang.org/protobuf/types/known/emptypb"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

// AuthClient provides access to Auth RPCs.
type AuthClient struct {
	auth xaipb.AuthClient
}

// GetAPIKeyInfo returns metadata for the current API key.
func (c *AuthClient) GetAPIKeyInfo(ctx context.Context) (*xaipb.ApiKey, error) {
	return c.auth.GetApiKeyInfo(ctx, &emptypb.Empty{})
}
