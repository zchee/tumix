package main

import (
	"fmt"
	"log"
	"os"

	"github.com/zchee/tumix/model/xai/examples/internal/exampleutil"
)

func main() {
	ctx, cancel := exampleutil.Context()
	defer cancel()

	client, cleanup, err := exampleutil.NewClient()
	if err != nil {
		log.Fatalf("create client: %v", err)
	}
	defer cleanup()

	apiKeyInfo, err := client.Auth.GetAPIKeyInfo(ctx)
	if err != nil {
		log.Fatalf("get api key info: %v", err)
	}
	fmt.Printf("API key id=%s team=%s name=%s blocked=%v\n", apiKeyInfo.GetApiKeyId(), apiKeyInfo.GetTeamId(), apiKeyInfo.GetName(), apiKeyInfo.GetApiKeyBlocked())

	if client.Billing == nil {
		log.Println("billing client not initialised (set XAI_MANAGEMENT_KEY) -> skipping billing calls")
		return
	}

	teamID := os.Getenv("XAI_TEAM_ID")
	if teamID == "" {
		log.Println("set XAI_TEAM_ID to run billing example; skipping")
		return
	}

	billingInfo, err := client.Billing.GetBillingInfo(ctx, teamID)
	if err != nil {
		log.Fatalf("get billing info: %v", err)
	}
	fmt.Printf("billing name=%s email=%s tax=%s:%s\n", billingInfo.GetName(), billingInfo.GetEmail(), billingInfo.GetTaxIdType(), billingInfo.GetTaxNumber())

	amount, err := client.Billing.GetAmountToPay(ctx, teamID)
	if err != nil {
		log.Fatalf("get amount to pay: %v", err)
	}
	if invoice := amount.GetCoreInvoice(); invoice != nil {
		fmt.Printf("current period total (after VAT) %d cents\n", invoice.GetAmountAfterVat())
	}
}
