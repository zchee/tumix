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

	managementpb "github.com/zchee/tumix/model/xai/management_api/v1"
	analyticspb "github.com/zchee/tumix/model/xai/shared/analytics"
	billingpb "github.com/zchee/tumix/model/xai/shared/billing"
)

// BillingClient handles billing operations.
type BillingClient struct {
	uisvc managementpb.UISvcClient
}

// SetBillingInfo sets billing information of a team.
func (c *BillingClient) SetBillingInfo(ctx context.Context, teamID string, info *billingpb.BillingInfo) (*managementpb.SetBillingInfoResp, error) {
	req := &managementpb.SetBillingInfoReq{
		TeamId:      teamID,
		BillingInfo: info,
	}
	return c.uisvc.SetBillingInfo(ctx, req)
}

// GetBillingInfo gets billing information of the team with given team ID.
func (c *BillingClient) GetBillingInfo(ctx context.Context, teamID string) (*billingpb.BillingInfo, error) {
	req := &managementpb.GetBillingInfoReq{
		TeamId: teamID,
	}
	resp, err := c.uisvc.GetBillingInfo(ctx, req)
	if err != nil {
		return nil, err
	}
	return resp.BillingInfo, nil
}

// ListPaymentMethods lists payment methods of a team.
func (c *BillingClient) ListPaymentMethods(ctx context.Context, teamID string) (*managementpb.ListPaymentMethodsResp, error) {
	req := &managementpb.ListPaymentMethodsReq{
		TeamId: teamID,
	}
	return c.uisvc.ListPaymentMethods(ctx, req)
}

// SetDefaultPaymentMethod sets default payment method to an existing payment method on file.
func (c *BillingClient) SetDefaultPaymentMethod(ctx context.Context, teamID, paymentMethodID string) (*managementpb.SetDefaultPaymentMethodResp, error) {
	req := &managementpb.SetDefaultPaymentMethodReq{
		TeamId:          teamID,
		PaymentMethodId: paymentMethodID,
	}
	return c.uisvc.SetDefaultPaymentMethod(ctx, req)
}

// GetAmountToPay previews the amount to pay for postpaid usage in the current billing period.
func (c *BillingClient) GetAmountToPay(ctx context.Context, teamID string) (*managementpb.GetAmountToPayResp, error) {
	req := &managementpb.GetAmountToPayReq{
		TeamId: teamID,
	}
	return c.uisvc.GetAmountToPay(ctx, req)
}

// AnalyzeBillingItems gets historical usage of the API over a time period, aggregated by fields.
func (c *BillingClient) AnalyzeBillingItems(ctx context.Context, teamID string, analyticsReq *analyticspb.AnalyticsRequest) (*analyticspb.AnalyticsResponse, error) {
	req := &managementpb.AnalyzeBillingItemsRequest{
		TeamId:           teamID,
		AnalyticsRequest: analyticsReq,
	}
	return c.uisvc.AnalyzeBillingItems(ctx, req)
}

// ListInvoices lists invoices that belong to a team.
func (c *BillingClient) ListInvoices(ctx context.Context, req *managementpb.ListInvoicesReq) (*managementpb.ListInvoicesResp, error) {
	return c.uisvc.ListInvoices(ctx, req)
}

// ListPrepaidBalanceChanges lists the prepaid credit balance and balance changes of a team.
func (c *BillingClient) ListPrepaidBalanceChanges(ctx context.Context, teamID string) (*managementpb.ListPrepaidBalanceChangesResp, error) {
	req := &managementpb.ListPrepaidBalanceChangesReq{
		TeamId: teamID,
	}
	return c.uisvc.ListPrepaidBalanceChanges(ctx, req)
}

// TopUpOrGetExistingPendingChange tops up prepaid credit using the default payment method.
func (c *BillingClient) TopUpOrGetExistingPendingChange(ctx context.Context, teamID string, amount *billingpb.Cent) (*managementpb.TopUpOrGetExistingPendingChangeResp, error) {
	req := &managementpb.TopUpOrGetExistingPendingChangeReq{
		TeamId: teamID,
		Amount: amount,
	}
	return c.uisvc.TopUpOrGetExistingPendingChange(ctx, req)
}

// GetSpendingLimits gets the postpaid monthly spending limits.
func (c *BillingClient) GetSpendingLimits(ctx context.Context, teamID string) (*managementpb.GetSpendingLimitsResp, error) {
	req := &managementpb.GetSpendingLimitsReq{
		TeamId: teamID,
	}
	return c.uisvc.GetSpendingLimits(ctx, req)
}

// SetSoftSpendingLimit sets the postpaid monthly spending limit of a team.
func (c *BillingClient) SetSoftSpendingLimit(ctx context.Context, teamID string, limit *billingpb.Cent) (*managementpb.SetSoftSpendingLimitResp, error) {
	req := &managementpb.SetSoftSpendingLimitReq{
		TeamId:                   teamID,
		DesiredSoftSpendingLimit: limit,
	}
	return c.uisvc.SetSoftSpendingLimit(ctx, req)
}
