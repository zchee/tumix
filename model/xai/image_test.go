package xai

import (
	"context"
	"encoding/base64"
	"testing"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

func TestImageOptions(t *testing.T) {
	req := &xaipb.GenerateImageRequest{}

	WithImageUser("user1")(req)
	WithImageFormat(ImageFormatBase64)(req)

	if req.GetUser() != "user1" {
		t.Fatalf("user not set: %s", req.GetUser())
	}
	if req.GetFormat() != xaipb.ImageFormat_IMG_FORMAT_BASE64 {
		t.Fatalf("format not converted: %v", req.GetFormat())
	}
}

func TestImageResponseHelpers(t *testing.T) {
	raw := []byte("pngdata")
	b64 := "data:image/png;base64," + base64.StdEncoding.EncodeToString(raw)
	proto := &xaipb.ImageResponse{
		Images: []*xaipb.GeneratedImage{
			{
				Image:           &xaipb.GeneratedImage_Base64{Base64: b64},
				UpSampledPrompt: "refined",
			},
		},
	}

	resp := &ImageResponse{proto: proto, index: 0}

	if got, err := resp.Base64(); err != nil || got != b64 {
		t.Fatalf("base64 helper failed: got %q err=%v", got, err)
	}

	if got := resp.Prompt(); got != "refined" {
		t.Fatalf("prompt mismatch: %s", got)
	}

	data, err := resp.Data(context.Background())
	if err != nil {
		t.Fatalf("data decode error: %v", err)
	}
	if string(data) != string(raw) {
		t.Fatalf("data mismatch: %q", data)
	}

	// Clear base64 to exercise URL error path.
	proto.Images[0].Image = &xaipb.GeneratedImage_Url{Url: ""}
	if _, err := resp.URL(); err == nil {
		t.Fatalf("expected URL error when url empty")
	}
	if _, err := resp.Base64(); err == nil {
		t.Fatalf("expected base64 error when empty")
	}
}
