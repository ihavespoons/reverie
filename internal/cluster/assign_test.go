package cluster

import (
	"context"
	"fmt"
	"testing"

	"personal/reverie/internal/memory"
)

// ---------------------------------------------------------------------------
// fakeStore — in-memory implementation of clusterStore for testing
// ---------------------------------------------------------------------------

type fakeStore struct {
	clusters map[string]memory.ClusterNode

	// Track calls for assertions.
	createCalls          []memory.ClusterNode
	updateCentroidCalls  []centroidUpdate
}

type centroidUpdate struct {
	ID        string
	Centroid  []float32
	ItemCount int
}

func newFakeStore() *fakeStore {
	return &fakeStore{
		clusters: make(map[string]memory.ClusterNode),
	}
}

func (s *fakeStore) ListClusters(_ context.Context) ([]memory.ClusterNode, error) {
	out := make([]memory.ClusterNode, 0, len(s.clusters))
	for _, c := range s.clusters {
		out = append(out, c)
	}
	return out, nil
}

func (s *fakeStore) GetCluster(_ context.Context, id string) (*memory.ClusterNode, error) {
	c, ok := s.clusters[id]
	if !ok {
		return nil, nil
	}
	return &c, nil
}

func (s *fakeStore) CreateCluster(_ context.Context, c memory.ClusterNode) error {
	s.clusters[c.ID] = c
	s.createCalls = append(s.createCalls, c)
	return nil
}

func (s *fakeStore) UpdateClusterCentroid(_ context.Context, id string, centroid []float32, itemCount int) error {
	c, ok := s.clusters[id]
	if !ok {
		return fmt.Errorf("cluster %q not found", id)
	}
	c.Centroid = centroid
	c.ItemCount = itemCount
	s.clusters[id] = c
	s.updateCentroidCalls = append(s.updateCentroidCalls, centroidUpdate{
		ID:        id,
		Centroid:  centroid,
		ItemCount: itemCount,
	})
	return nil
}

// seed adds a cluster to the fake store for test setup.
func (s *fakeStore) seed(c memory.ClusterNode) {
	s.clusters[c.ID] = c
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

func TestAssign_EmptyStore_CreatesFirstCluster(t *testing.T) {
	store := newFakeStore()
	a := NewAssigner(store, 0.60, 0.5, 0.5)

	vec := []float32{1, 0, 0, 0}
	id, isNew, err := a.Assign(context.Background(), vec)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !isNew {
		t.Error("expected isNew=true for empty store")
	}
	if id == "" {
		t.Fatal("expected non-empty cluster ID")
	}

	// Verify CreateCluster was called with expected values.
	if len(store.createCalls) != 1 {
		t.Fatalf("expected 1 CreateCluster call, got %d", len(store.createCalls))
	}
	created := store.createCalls[0]
	if created.ItemCount != 1 {
		t.Errorf("expected itemCount=1, got %d", created.ItemCount)
	}
	if created.Utility != 0.5 {
		t.Errorf("expected utility=0.5, got %f", created.Utility)
	}
	if created.Frequency != 0.5 {
		t.Errorf("expected frequency=0.5, got %f", created.Frequency)
	}
	if len(created.Centroid) != 4 {
		t.Fatalf("expected centroid len 4, got %d", len(created.Centroid))
	}
	for i, want := range vec {
		if created.Centroid[i] != want {
			t.Errorf("centroid[%d] = %f, want %f", i, created.Centroid[i], want)
		}
	}
}

func TestAssign_AboveThreshold_ReturnsExisting(t *testing.T) {
	store := newFakeStore()
	store.seed(memory.ClusterNode{
		ID:        "cluster-1",
		Centroid:  []float32{1, 0, 0, 0},
		ItemCount: 5,
	})
	a := NewAssigner(store, 0.60, 0.5, 0.5)

	// [0.9, 0.1, 0, 0] is very similar to [1, 0, 0, 0].
	// cosine = 0.9 / sqrt(0.82) ~= 0.994
	vec := []float32{0.9, 0.1, 0, 0}
	id, isNew, err := a.Assign(context.Background(), vec)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if isNew {
		t.Error("expected isNew=false for above-threshold similarity")
	}
	if id != "cluster-1" {
		t.Errorf("expected cluster-1, got %s", id)
	}
	if len(store.createCalls) != 0 {
		t.Errorf("expected no CreateCluster calls, got %d", len(store.createCalls))
	}
}

func TestAssign_BelowThreshold_CreatesNew(t *testing.T) {
	store := newFakeStore()
	store.seed(memory.ClusterNode{
		ID:        "cluster-1",
		Centroid:  []float32{1, 0, 0, 0},
		ItemCount: 5,
	})
	a := NewAssigner(store, 0.60, 0.5, 0.5)

	// [0, 1, 0, 0] is orthogonal to [1, 0, 0, 0] -- cosine = 0.
	vec := []float32{0, 1, 0, 0}
	id, isNew, err := a.Assign(context.Background(), vec)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !isNew {
		t.Error("expected isNew=true for below-threshold similarity")
	}
	if id == "cluster-1" {
		t.Error("expected a NEW cluster ID, not cluster-1")
	}
	if len(store.createCalls) != 1 {
		t.Errorf("expected 1 CreateCluster call, got %d", len(store.createCalls))
	}
}

func TestAssign_PicksNearestAmongMultiple(t *testing.T) {
	store := newFakeStore()
	store.seed(memory.ClusterNode{
		ID:        "far",
		Centroid:  []float32{0, 0, 1, 0},
		ItemCount: 3,
	})
	store.seed(memory.ClusterNode{
		ID:        "closest",
		Centroid:  []float32{1, 0, 0, 0},
		ItemCount: 3,
	})
	store.seed(memory.ClusterNode{
		ID:        "medium",
		Centroid:  []float32{0.5, 0.5, 0, 0},
		ItemCount: 3,
	})
	a := NewAssigner(store, 0.60, 0.5, 0.5)

	// [0.95, 0.05, 0, 0] is closest to [1, 0, 0, 0] ("closest").
	vec := []float32{0.95, 0.05, 0, 0}
	id, isNew, err := a.Assign(context.Background(), vec)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if isNew {
		t.Error("expected isNew=false, nearest cluster should match")
	}
	if id != "closest" {
		t.Errorf("expected 'closest', got %q", id)
	}
}

func TestAssign_NilCentroid_Skipped(t *testing.T) {
	store := newFakeStore()
	// Cluster with nil centroid should be skipped.
	store.seed(memory.ClusterNode{
		ID:        "nil-centroid",
		Centroid:  nil,
		ItemCount: 0,
	})
	// Cluster with empty centroid should also be skipped.
	store.seed(memory.ClusterNode{
		ID:        "empty-centroid",
		Centroid:  []float32{},
		ItemCount: 0,
	})
	a := NewAssigner(store, 0.60, 0.5, 0.5)

	vec := []float32{1, 0, 0, 0}
	id, isNew, err := a.Assign(context.Background(), vec)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// All existing clusters had bad centroids, so a new one is created.
	if !isNew {
		t.Error("expected isNew=true when all clusters have nil/empty centroids")
	}
	if id == "nil-centroid" || id == "empty-centroid" {
		t.Errorf("should not return a cluster with nil/empty centroid, got %q", id)
	}
}

func TestAssign_ZeroVec(t *testing.T) {
	// A zero vector produces cosine=0 against any cluster (vecmath.Cosine
	// returns 0 for zero-magnitude vectors). Since 0 < minSim, a new cluster
	// is created. The new cluster's centroid is all zeros. This is a degenerate
	// case -- the caller should not embed to a zero vector in practice -- but
	// it must not crash.
	store := newFakeStore()
	store.seed(memory.ClusterNode{
		ID:        "existing",
		Centroid:  []float32{1, 0, 0, 0},
		ItemCount: 5,
	})
	a := NewAssigner(store, 0.60, 0.5, 0.5)

	vec := []float32{0, 0, 0, 0}
	id, isNew, err := a.Assign(context.Background(), vec)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// cosine(zero, anything) = 0, which is below 0.60, so a new cluster.
	if !isNew {
		t.Error("expected isNew=true for zero vec")
	}
	if id == "existing" {
		t.Error("should not assign zero vec to an existing cluster")
	}
}

func TestAfterInsert_UpdatesRunningMean(t *testing.T) {
	store := newFakeStore()
	store.seed(memory.ClusterNode{
		ID:        "c1",
		Centroid:  []float32{1, 0},
		ItemCount: 1,
	})
	a := NewAssigner(store, 0.60, 0.5, 0.5)

	vec := []float32{0, 1}
	if err := a.AfterInsert(context.Background(), "c1", vec); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(store.updateCentroidCalls) != 1 {
		t.Fatalf("expected 1 UpdateClusterCentroid call, got %d", len(store.updateCentroidCalls))
	}
	upd := store.updateCentroidCalls[0]
	if upd.ItemCount != 2 {
		t.Errorf("expected itemCount=2, got %d", upd.ItemCount)
	}
	if len(upd.Centroid) != 2 {
		t.Fatalf("expected centroid len 2, got %d", len(upd.Centroid))
	}
	// running mean: (1*1 + 0) / 2 = 0.5, (0*1 + 1) / 2 = 0.5
	if !almostEqual(upd.Centroid[0], 0.5, 1e-6) {
		t.Errorf("centroid[0] = %f, want 0.5", upd.Centroid[0])
	}
	if !almostEqual(upd.Centroid[1], 0.5, 1e-6) {
		t.Errorf("centroid[1] = %f, want 0.5", upd.Centroid[1])
	}
}

func TestAfterInsert_UnknownCluster_ReturnsError(t *testing.T) {
	store := newFakeStore()
	a := NewAssigner(store, 0.60, 0.5, 0.5)

	vec := []float32{1, 0}
	err := a.AfterInsert(context.Background(), "nonexistent", vec)
	if err == nil {
		t.Fatal("expected error for unknown cluster ID")
	}
}
