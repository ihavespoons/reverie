package embed

import (
	"context"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/diffsec/reverie/internal/db"
)

// fakeProvider is a test EmbeddingProvider that counts Embed calls and returns
// deterministic vectors: for text at index i, vec = [float32(i+1), 0, 0, ...].
type fakeProvider struct {
	model     string
	dim       int
	callCount atomic.Int32
}

func (f *fakeProvider) Embed(_ context.Context, texts []string) ([][]float32, error) {
	f.callCount.Add(1)
	vecs := make([][]float32, len(texts))
	for i := range texts {
		vec := make([]float32, f.dim)
		vec[0] = float32(i + 1)
		vecs[i] = vec
	}
	return vecs, nil
}

func (f *fakeProvider) Dimensions() int { return f.dim }
func (f *fakeProvider) Model() string   { return f.model }

func TestCachePopulatesOnFirstCall(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	d, err := db.Open(dbPath)
	if err != nil {
		t.Fatalf("db.Open: %v", err)
	}
	defer d.Close()

	inner := &fakeProvider{model: "test-model", dim: 4}
	cached := NewCachedProvider(inner, d)

	texts := []string{"hello", "world"}
	vecs, err := cached.Embed(context.Background(), texts)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if inner.callCount.Load() != 1 {
		t.Errorf("expected 1 inner call, got %d", inner.callCount.Load())
	}
	if len(vecs) != 2 {
		t.Fatalf("expected 2 vectors, got %d", len(vecs))
	}
	if vecs[0][0] != 1 {
		t.Errorf("vecs[0][0] = %v, want 1", vecs[0][0])
	}
	if vecs[1][0] != 2 {
		t.Errorf("vecs[1][0] = %v, want 2", vecs[1][0])
	}
}

func TestCacheHitSkipsInner(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	d, err := db.Open(dbPath)
	if err != nil {
		t.Fatalf("db.Open: %v", err)
	}
	defer d.Close()

	inner := &fakeProvider{model: "test-model", dim: 4}
	cached := NewCachedProvider(inner, d)

	texts := []string{"hello", "world"}

	// First call — populates cache.
	_, err = cached.Embed(context.Background(), texts)
	if err != nil {
		t.Fatalf("first Embed: %v", err)
	}

	// Second call — should hit cache entirely.
	vecs, err := cached.Embed(context.Background(), texts)
	if err != nil {
		t.Fatalf("second Embed: %v", err)
	}

	if inner.callCount.Load() != 1 {
		t.Errorf("expected 1 inner call (cache hit on second), got %d", inner.callCount.Load())
	}
	if len(vecs) != 2 {
		t.Fatalf("expected 2 vectors, got %d", len(vecs))
	}
	// Values should still be correct from cache.
	if vecs[0][0] != 1 {
		t.Errorf("cached vecs[0][0] = %v, want 1", vecs[0][0])
	}
	if vecs[1][0] != 2 {
		t.Errorf("cached vecs[1][0] = %v, want 2", vecs[1][0])
	}
}

func TestCacheMixedHitMiss(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	d, err := db.Open(dbPath)
	if err != nil {
		t.Fatalf("db.Open: %v", err)
	}
	defer d.Close()

	inner := &fakeProvider{model: "test-model", dim: 4}
	cached := NewCachedProvider(inner, d)

	// Populate cache with "hello" and "world".
	_, err = cached.Embed(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("first Embed: %v", err)
	}

	// Now request "hello" (cached), "new" (miss), "world" (cached).
	vecs, err := cached.Embed(context.Background(), []string{"hello", "new", "world"})
	if err != nil {
		t.Fatalf("mixed Embed: %v", err)
	}

	// Inner should be called twice: once for initial, once for "new".
	if inner.callCount.Load() != 2 {
		t.Errorf("expected 2 inner calls, got %d", inner.callCount.Load())
	}

	if len(vecs) != 3 {
		t.Fatalf("expected 3 vectors, got %d", len(vecs))
	}

	// "hello" from cache: vec[0] = 1
	if vecs[0][0] != 1 {
		t.Errorf("vecs[0][0] = %v, want 1 (cached hello)", vecs[0][0])
	}
	// "new" from inner: single miss, so index 0 in the miss batch → vec[0] = 1.
	if vecs[1][0] != 1 {
		t.Errorf("vecs[1][0] = %v, want 1 (new text, first in miss batch)", vecs[1][0])
	}
	// "world" from cache: vec[0] = 2
	if vecs[2][0] != 2 {
		t.Errorf("vecs[2][0] = %v, want 2 (cached world)", vecs[2][0])
	}
}

func TestCacheDifferentModelsNoCollision(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	d, err := db.Open(dbPath)
	if err != nil {
		t.Fatalf("db.Open: %v", err)
	}
	defer d.Close()

	innerA := &fakeProvider{model: "model-a", dim: 4}
	innerB := &fakeProvider{model: "model-b", dim: 4}

	cachedA := NewCachedProvider(innerA, d)
	cachedB := NewCachedProvider(innerB, d)

	texts := []string{"same text"}

	// Populate with model-a.
	_, err = cachedA.Embed(context.Background(), texts)
	if err != nil {
		t.Fatalf("model-a Embed: %v", err)
	}

	// Request with model-b — should NOT hit model-a's cache.
	_, err = cachedB.Embed(context.Background(), texts)
	if err != nil {
		t.Fatalf("model-b Embed: %v", err)
	}

	if innerA.callCount.Load() != 1 {
		t.Errorf("model-a inner calls = %d, want 1", innerA.callCount.Load())
	}
	if innerB.callCount.Load() != 1 {
		t.Errorf("model-b inner calls = %d, want 1", innerB.callCount.Load())
	}
}

func TestCacheOrderPreserved(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	d, err := db.Open(dbPath)
	if err != nil {
		t.Fatalf("db.Open: %v", err)
	}
	defer d.Close()

	inner := &fakeProvider{model: "test-model", dim: 4}
	cached := NewCachedProvider(inner, d)

	// Populate "b" and "d".
	_, err = cached.Embed(context.Background(), []string{"b", "d"})
	if err != nil {
		t.Fatalf("populate Embed: %v", err)
	}

	// Request "a" (miss), "b" (hit), "c" (miss), "d" (hit), "e" (miss).
	vecs, err := cached.Embed(context.Background(), []string{"a", "b", "c", "d", "e"})
	if err != nil {
		t.Fatalf("mixed Embed: %v", err)
	}

	if len(vecs) != 5 {
		t.Fatalf("expected 5 vectors, got %d", len(vecs))
	}

	// "b" was cached with vec[0] = 1 (first in its original batch).
	if vecs[1][0] != 1 {
		t.Errorf("vecs[1][0] (cached b) = %v, want 1", vecs[1][0])
	}
	// "d" was cached with vec[0] = 2 (second in its original batch).
	if vecs[3][0] != 2 {
		t.Errorf("vecs[3][0] (cached d) = %v, want 2", vecs[3][0])
	}

	// Misses "a", "c", "e" were sent as a batch of 3.
	// "a" is index 0 in miss batch → vec[0] = 1.
	if vecs[0][0] != 1 {
		t.Errorf("vecs[0][0] (miss a) = %v, want 1", vecs[0][0])
	}
	// "c" is index 1 in miss batch → vec[0] = 2.
	if vecs[2][0] != 2 {
		t.Errorf("vecs[2][0] (miss c) = %v, want 2", vecs[2][0])
	}
	// "e" is index 2 in miss batch → vec[0] = 3.
	if vecs[4][0] != 3 {
		t.Errorf("vecs[4][0] (miss e) = %v, want 3", vecs[4][0])
	}
}

func TestCacheEmptyInput(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	d, err := db.Open(dbPath)
	if err != nil {
		t.Fatalf("db.Open: %v", err)
	}
	defer d.Close()

	inner := &fakeProvider{model: "test-model", dim: 4}
	cached := NewCachedProvider(inner, d)

	vecs, err := cached.Embed(context.Background(), nil)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 0 {
		t.Errorf("expected empty result, got %d", len(vecs))
	}
	if inner.callCount.Load() != 0 {
		t.Errorf("expected 0 inner calls for empty input, got %d", inner.callCount.Load())
	}
}

func TestCacheDelegatesDimensionsAndModel(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	d, err := db.Open(dbPath)
	if err != nil {
		t.Fatalf("db.Open: %v", err)
	}
	defer d.Close()

	inner := &fakeProvider{model: "test-model", dim: 1024}
	cached := NewCachedProvider(inner, d)

	if got := cached.Dimensions(); got != 1024 {
		t.Errorf("Dimensions() = %d, want 1024", got)
	}
	if got := cached.Model(); got != "test-model" {
		t.Errorf("Model() = %q, want %q", got, "test-model")
	}
}

func TestCacheStatsFresh(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	d, err := db.Open(dbPath)
	if err != nil {
		t.Fatalf("db.Open: %v", err)
	}
	defer d.Close()

	inner := &fakeProvider{model: "test-model", dim: 4}
	cached := NewCachedProvider(inner, d)

	stats := cached.Stats()
	if stats.Hits != 0 || stats.Misses != 0 {
		t.Errorf("fresh Stats() = %+v, want {0, 0}", stats)
	}
}

func TestCacheStatsCountsHitsAndMisses(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	d, err := db.Open(dbPath)
	if err != nil {
		t.Fatalf("db.Open: %v", err)
	}
	defer d.Close()

	inner := &fakeProvider{model: "test-model", dim: 4}
	cached := NewCachedProvider(inner, d)

	// Two misses: first call, both texts go to inner.
	if _, err := cached.Embed(context.Background(), []string{"alpha", "beta"}); err != nil {
		t.Fatalf("first Embed: %v", err)
	}
	// One hit: "alpha" is cached from the previous call.
	if _, err := cached.Embed(context.Background(), []string{"alpha"}); err != nil {
		t.Fatalf("second Embed: %v", err)
	}

	stats := cached.Stats()
	if stats.Hits != 1 {
		t.Errorf("Stats().Hits = %d, want 1", stats.Hits)
	}
	if stats.Misses != 2 {
		t.Errorf("Stats().Misses = %d, want 2", stats.Misses)
	}
}
