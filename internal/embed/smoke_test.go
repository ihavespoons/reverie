package embed

import (
	"context"
	"os"
	"testing"
)

func TestSmokeVoyageRealAPI(t *testing.T) {
	if os.Getenv("REVERIE_SMOKE_TEST") != "1" {
		t.Skip("skipping smoke test: set REVERIE_SMOKE_TEST=1 to run")
	}

	apiKey := os.Getenv("VOYAGE_API_KEY")
	if apiKey == "" {
		t.Skip("skipping smoke test: VOYAGE_API_KEY not set")
	}

	p := NewVoyageProvider(apiKey, "voyage-3", 0)

	vecs, err := p.Embed(context.Background(), []string{"Hello, world!"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if len(vecs) != 1 {
		t.Fatalf("expected 1 vector, got %d", len(vecs))
	}

	if got := len(vecs[0]); got != p.Dimensions() {
		t.Errorf("vector dim = %d, want %d (from Dimensions())", got, p.Dimensions())
	}
}
