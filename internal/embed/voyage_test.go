package embed

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// newTestServer creates an httptest.Server that responds with Voyage-compatible
// JSON. The handler tracks call count and returns embeddings with shuffled
// indices when shuffle is true. dim controls the vector dimensionality.
func newTestServer(t *testing.T, callCount *atomic.Int32, dim int, shuffle bool) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount.Add(1)

		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if got := r.Header.Get("Authorization"); got == "" {
			http.Error(w, "missing auth", http.StatusUnauthorized)
			return
		}

		var req voyageRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		data := make([]voyageEmbedding, len(req.Input))
		for i := range req.Input {
			vec := make([]float32, dim)
			vec[0] = float32(i + 1)

			idx := i
			if shuffle {
				// Reverse the indices to test re-sorting.
				idx = len(req.Input) - 1 - i
			}
			data[idx] = voyageEmbedding{
				Index:     i,
				Embedding: vec,
			}
		}

		resp := voyageResponse{Data: data}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
}

// newVoyageWithServer creates a voyageProvider pointing at the test server.
func newVoyageWithServer(serverURL, apiKey, model string, batchSize int) *voyageProvider {
	p := NewVoyageProvider(apiKey, model, batchSize).(*voyageProvider)
	// Override the client to use the test server transport isn't needed;
	// instead we override the URL by swapping out the internal state.
	// Since voyageAPIURL is a const, we use a helper to build the provider
	// with a custom URL approach: we'll inject a custom roundtripper.
	p.client = &http.Client{
		Transport: &rewriteTransport{
			base:      http.DefaultTransport,
			targetURL: serverURL,
		},
	}
	return p
}

// rewriteTransport rewrites request URLs to point at a test server.
type rewriteTransport struct {
	base      http.RoundTripper
	targetURL string
}

func (rt *rewriteTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.URL.Scheme = "http"
	// Parse the target to get host.
	parsed, err := http.NewRequest("GET", rt.targetURL, nil)
	if err != nil {
		return nil, err
	}
	req.URL.Host = parsed.URL.Host
	req.URL.Path = parsed.URL.Path + req.URL.Path
	return rt.base.RoundTrip(req)
}

func TestVoyageBatching(t *testing.T) {
	var calls atomic.Int32
	srv := newTestServer(t, &calls, 4, false)
	defer srv.Close()

	p := newVoyageWithServer(srv.URL, "test-key", "voyage-3", 64)

	texts := make([]string, 150)
	for i := range texts {
		texts[i] = "text"
	}

	vecs, err := p.Embed(context.Background(), texts)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if got := calls.Load(); got != 3 {
		t.Errorf("expected 3 API calls (64+64+22), got %d", got)
	}
	if len(vecs) != 150 {
		t.Errorf("expected 150 vectors, got %d", len(vecs))
	}
}

func TestVoyageOrdering(t *testing.T) {
	var calls atomic.Int32
	srv := newTestServer(t, &calls, 4, true) // shuffle=true
	defer srv.Close()

	p := newVoyageWithServer(srv.URL, "test-key", "voyage-3", 128)

	texts := []string{"a", "b", "c", "d", "e"}
	vecs, err := p.Embed(context.Background(), texts)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	// The server returns shuffled indices but our code should re-sort them.
	// vec[i][0] should be float32(i+1).
	for i, vec := range vecs {
		expected := float32(i + 1)
		if vec[0] != expected {
			t.Errorf("vecs[%d][0] = %v, want %v", i, vec[0], expected)
		}
	}
}

func TestVoyageEmptyInput(t *testing.T) {
	var calls atomic.Int32
	srv := newTestServer(t, &calls, 4, false)
	defer srv.Close()

	p := newVoyageWithServer(srv.URL, "test-key", "voyage-3", 128)

	vecs, err := p.Embed(context.Background(), nil)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 0 {
		t.Errorf("expected empty result, got %d vectors", len(vecs))
	}
	if got := calls.Load(); got != 0 {
		t.Errorf("expected 0 API calls for empty input, got %d", got)
	}
}

func TestVoyageServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("internal server error: something went wrong"))
	}))
	defer srv.Close()

	p := newVoyageWithServer(srv.URL, "test-key", "voyage-3", 128)

	_, err := p.Embed(context.Background(), []string{"test"})
	if err == nil {
		t.Fatal("expected error for 500 response")
	}

	errStr := err.Error()
	if !contains(errStr, "500") {
		t.Errorf("error should contain status code 500, got: %s", errStr)
	}
	if !contains(errStr, "internal server error") {
		t.Errorf("error should contain body snippet, got: %s", errStr)
	}
}

func TestVoyageCtxCancellation(t *testing.T) {
	// Server that blocks until context is done.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer srv.Close()

	p := newVoyageWithServer(srv.URL, "test-key", "voyage-3", 128)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	_, err := p.Embed(ctx, []string{"test"})
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
	if ctx.Err() == nil {
		t.Fatal("context should be cancelled")
	}
}

func TestVoyageNoAPIKey(t *testing.T) {
	var calls atomic.Int32
	srv := newTestServer(t, &calls, 4, false)
	defer srv.Close()

	// Create provider with empty API key — should not use the server.
	p := NewVoyageProvider("", "voyage-3", 128)

	_, err := p.Embed(context.Background(), []string{"test"})
	if err == nil {
		t.Fatal("expected error with empty API key")
	}

	errStr := err.Error()
	if !contains(errStr, "voyage:") {
		t.Errorf("error should have voyage: prefix, got: %s", errStr)
	}

	if got := calls.Load(); got != 0 {
		t.Errorf("expected 0 API calls with empty key, got %d", got)
	}
}

func TestVoyageDimensions(t *testing.T) {
	tests := []struct {
		model string
		want  int
	}{
		{"voyage-3", 1024},
		{"voyage-code-3", 1024},
		{"unknown-model", 0},
		{"", 1024}, // defaults to voyage-3
	}

	for _, tt := range tests {
		p := NewVoyageProvider("key", tt.model, 0)
		if got := p.Dimensions(); got != tt.want {
			t.Errorf("Dimensions(%q) = %d, want %d", tt.model, got, tt.want)
		}
	}
}

func TestVoyageModel(t *testing.T) {
	p := NewVoyageProvider("key", "voyage-code-3", 0)
	if got := p.Model(); got != "voyage-code-3" {
		t.Errorf("Model() = %q, want %q", got, "voyage-code-3")
	}

	p2 := NewVoyageProvider("key", "", 0)
	if got := p2.Model(); got != "voyage-3" {
		t.Errorf("Model() = %q, want %q", got, "voyage-3")
	}
}

func TestVoyageBatchSizeDefaults(t *testing.T) {
	p := NewVoyageProvider("key", "", 0).(*voyageProvider)
	if p.batchSize != 128 {
		t.Errorf("batchSize with 0 = %d, want 128", p.batchSize)
	}

	p2 := NewVoyageProvider("key", "", 200).(*voyageProvider)
	if p2.batchSize != 128 {
		t.Errorf("batchSize with 200 = %d, want 128", p2.batchSize)
	}

	p3 := NewVoyageProvider("key", "", 50).(*voyageProvider)
	if p3.batchSize != 50 {
		t.Errorf("batchSize with 50 = %d, want 50", p3.batchSize)
	}
}

// contains is a simple substring check helper.
func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchSubstring(s, substr)
}

func searchSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
