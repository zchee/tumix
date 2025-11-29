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

package gollm

// AuthMethod represents the type of authentication method used.
type AuthMethod interface {
	value() string
}

// AuthMethodAPIKey represents API key authentication.
type AuthMethodAPIKey string

var _ AuthMethod = AuthMethodAPIKey("")

// value returns the API key string for the data plane.
func (a AuthMethodAPIKey) value() string { return string(a) }

// AuthMethodAPIToken represents API token authentication.
type AuthMethodAPIToken string

var _ AuthMethod = AuthMethodAPIToken("")

// value returns the API token string for the data plane.
func (a AuthMethodAPIToken) value() string { return string(a) }
