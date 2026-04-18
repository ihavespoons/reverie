package embed

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// stubEmbedServer returns a fixed embedding per text, echoing index back.
// It tracks how many requests it's seen and the path used.
type stubEmbedServer struct {
	calls   atomic.Int64
	lastReq openAICompatRequest
	lastKey string
}

func (s *stubEmbedServer) handler(dim int) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.calls.Add(1)
		s.lastKey = r.Header.Get("Authorization")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &s.lastReq)

		data := make([]openAICompatEmbedding, len(s.lastReq.Input))
		for i := range s.lastReq.Input {
			vec := make([]float32, dim)
			vec[0] = float32(i + 1)
			data[i] = openAICompatEmbedding{Index: i, Embedding: vec}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(openAICompatResponse{Data: data})
	})
}

func TestOpenAICompat_BasicEmbed(t *testing.T) {
	stub := &stubEmbedServer{}
	srv := httptest.NewServer(stub.handler(8))
	defer srv.Close()

	p := NewOpenAICompatProvider(srv.URL+"/v1", "", "nomic-embed-text", 0, 8)
	got, err := p.Embed(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("expected 3 embeddings, got %d", len(got))
	}
	for i, vec := range got {
		if len(vec) != 8 {
			t.Errorf("row %d: expected 8 dims, got %d", i, len(vec))
		}
		if vec[0] != float32(i+1) {
			t.Errorf("row %d: expected vec[0]=%d, got %f", i, i+1, vec[0])
		}
	}
	if p.Dimensions() != 8 {
		t.Errorf("expected Dimensions()=8, got %d", p.Dimensions())
	}
	if p.Model() != "nomic-embed-text" {
		t.Errorf("expected Model()='nomic-embed-text', got %q", p.Model())
	}
}

func TestOpenAICompat_BatchChunking(t *testing.T) {
	stub := &stubEmbedServer{}
	srv := httptest.NewServer(stub.handler(4))
	defer srv.Close()

	texts := make([]string, 100)
	for i := range texts {
		texts[i] = "t"
	}
	p := NewOpenAICompatProvider(srv.URL+"/v1", "", "m", 30, 4)
	if _, err := p.Embed(context.Background(), texts); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	// 100 texts / batch=30 → 4 batches (30+30+30+10).
	if got := stub.calls.Load(); got != 4 {
		t.Errorf("expected 4 batch calls, got %d", got)
	}
}

func TestOpenAICompat_EmptyInput(t *testing.T) {
	stub := &stubEmbedServer{}
	srv := httptest.NewServer(stub.handler(4))
	defer srv.Close()

	p := NewOpenAICompatProvider(srv.URL+"/v1", "", "m", 0, 4)
	got, err := p.Embed(context.Background(), nil)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("expected empty, got %d", len(got))
	}
	if stub.calls.Load() != 0 {
		t.Errorf("expected 0 calls, got %d", stub.calls.Load())
	}
}

func TestOpenAICompat_AuthHeader(t *testing.T) {
	stub := &stubEmbedServer{}
	srv := httptest.NewServer(stub.handler(4))
	defer srv.Close()

	// With a key: Authorization header must be set.
	p := NewOpenAICompatProvider(srv.URL+"/v1", "sk-test", "m", 0, 4)
	_, _ = p.Embed(context.Background(), []string{"x"})
	if stub.lastKey != "Bearer sk-test" {
		t.Errorf("expected 'Bearer sk-test', got %q", stub.lastKey)
	}

	// Without a key (Ollama/LM Studio case): no Authorization header.
	stub.lastKey = ""
	p2 := NewOpenAICompatProvider(srv.URL+"/v1", "", "m", 0, 4)
	_, _ = p2.Embed(context.Background(), []string{"x"})
	if stub.lastKey != "" {
		t.Errorf("expected no Authorization header, got %q", stub.lastKey)
	}
}

func TestOpenAICompat_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("kaboom"))
	}))
	defer srv.Close()

	p := NewOpenAICompatProvider(srv.URL+"/v1", "", "m", 0, 4)
	_, err := p.Embed(context.Background(), []string{"x"})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "500") || !strings.Contains(err.Error(), "kaboom") {
		t.Errorf("expected status + body in error, got %v", err)
	}
}

func TestOpenAICompat_ContextCancellation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(500 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"data":[]}`))
	}))
	defer srv.Close()

	p := NewOpenAICompatProvider(srv.URL+"/v1", "", "m", 0, 4)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := p.Embed(ctx, []string{"x"})
	if err == nil {
		t.Fatal("expected error on cancelled context")
	}
	if !errors.Is(err, context.Canceled) && !strings.Contains(err.Error(), "context canceled") {
		t.Errorf("expected context-canceled error, got %v", err)
	}
}

func TestOpenAICompat_TrailingSlashInBaseURL(t *testing.T) {
	stub := &stubEmbedServer{}
	srv := httptest.NewServer(stub.handler(4))
	defer srv.Close()

	// Trailing slash in base URL should be trimmed.
	p := NewOpenAICompatProvider(srv.URL+"/v1/", "", "m", 0, 4)
	if _, err := p.Embed(context.Background(), []string{"x"}); err != nil {
		t.Fatalf("Embed failed: %v", err)
	}
	if stub.calls.Load() != 1 {
		t.Errorf("expected request to succeed despite trailing slash, got %d calls", stub.calls.Load())
	}
}
