package manager

import (
	"context"
	"math"
	"testing"

	"github.com/diffsec/reverie/internal/memory"
)

// fakeDecayer satisfies the local decayer interface for testing. The manager
// code currently only needs the interface to exist — Reinforce and TickDecay
// don't call decayer methods. We include it for constructor wiring.
type fakeDecayer struct {
	temperature float64
	threshold   float64
}

func (d *fakeDecayer) Retention(c memory.ClusterNode) float64 {
	stability := (c.Utility + c.Frequency + 0.01) * d.temperature
	if stability <= 0 {
		return 0
	}
	return math.Exp(-float64(c.TurnsSince) / stability)
}

func (d *fakeDecayer) GateC(c memory.ClusterNode) bool {
	return d.Retention(c) > d.threshold
}

func (d *fakeDecayer) Temperature() float64 { return d.temperature }
func (d *fakeDecayer) Threshold() float64   { return d.threshold }

func newTestDecayer() *fakeDecayer {
	return &fakeDecayer{temperature: 10.0, threshold: 0.30}
}

// helper to insert a fact into the mem store and return its ID.
func insertFact(t *testing.T, s memory.Store, id, clusterID, content string) string {
	t.Helper()
	ctx := context.Background()
	retID, err := s.InsertFact(ctx, memory.Fact{
		ID:        id,
		ClusterID: clusterID,
		Content:   content,
	})
	if err != nil {
		t.Fatalf("InsertFact(%s): %v", id, err)
	}
	return retID
}

// --- Reinforce tests ---

func TestReinforce_SingleCluster(t *testing.T) {
	ctx := context.Background()
	store := memory.NewMemStore()
	d := newTestDecayer()
	mgr := NewMemoryManager(store, d, 0.10, 0.05)

	// Insert two facts into the default cluster.
	id1 := insertFact(t, store, "f1", "default", "Go uses structural typing")
	id2 := insertFact(t, store, "f2", "default", "Interfaces are implicit")

	// Credit both facts with different scores.
	credits := map[string]float64{
		id1: 0.8,
		id2: 0.3,
	}

	if err := mgr.Reinforce(ctx, credits); err != nil {
		t.Fatalf("Reinforce: %v", err)
	}

	// Verify cluster state. The default cluster starts at U=0, F=0.
	// maxCredit for "default" cluster = 0.8 (from f1).
	// U_new = 0 + 0.10 * (0.8 - 0) = 0.08
	// F_new = 0 + 0.05 * (1 - 0) = 0.05
	cluster, err := store.GetCluster(ctx, "default")
	if err != nil {
		t.Fatalf("GetCluster: %v", err)
	}
	if cluster == nil {
		t.Fatal("expected default cluster to exist")
	}

	const epsilon = 1e-9
	expectedU := 0.08
	expectedF := 0.05
	if math.Abs(cluster.Utility-expectedU) > epsilon {
		t.Errorf("Utility = %f, want %f", cluster.Utility, expectedU)
	}
	if math.Abs(cluster.Frequency-expectedF) > epsilon {
		t.Errorf("Frequency = %f, want %f", cluster.Frequency, expectedF)
	}
}

func TestReinforce_UnknownID(t *testing.T) {
	ctx := context.Background()
	store := memory.NewMemStore()
	d := newTestDecayer()
	mgr := NewMemoryManager(store, d, 0.10, 0.05)

	// Insert a fact so a cluster exists.
	insertFact(t, store, "f1", "default", "real fact")

	// Credit a nonexistent memory.
	credits := map[string]float64{
		"does-not-exist": 0.9,
	}

	if err := mgr.Reinforce(ctx, credits); err != nil {
		t.Fatalf("Reinforce with unknown ID should not error: %v", err)
	}

	// The default cluster should be untouched.
	cluster, err := store.GetCluster(ctx, "default")
	if err != nil {
		t.Fatalf("GetCluster: %v", err)
	}
	if cluster.Utility != 0 {
		t.Errorf("Utility = %f, want 0 (no change)", cluster.Utility)
	}
	if cluster.Frequency != 0 {
		t.Errorf("Frequency = %f, want 0 (no change)", cluster.Frequency)
	}
}

func TestReinforce_EmptyCredits(t *testing.T) {
	ctx := context.Background()
	store := memory.NewMemStore()
	d := newTestDecayer()
	mgr := NewMemoryManager(store, d, 0.10, 0.05)

	// Empty credits should be a no-op.
	if err := mgr.Reinforce(ctx, map[string]float64{}); err != nil {
		t.Fatalf("Reinforce with empty credits should not error: %v", err)
	}

	// Also nil credits.
	if err := mgr.Reinforce(ctx, nil); err != nil {
		t.Fatalf("Reinforce with nil credits should not error: %v", err)
	}
}

func TestReinforce_UtilityConvergence(t *testing.T) {
	ctx := context.Background()
	store := memory.NewMemStore()
	d := newTestDecayer()
	alpha := 0.10
	mgr := NewMemoryManager(store, d, alpha, 0.05)

	insertFact(t, store, "f1", "default", "convergence test fact")

	// Reinforce 100 times with a constant credit of 0.9.
	// EMA property: U converges to the credit value.
	for i := 0; i < 100; i++ {
		if err := mgr.Reinforce(ctx, map[string]float64{"f1": 0.9}); err != nil {
			t.Fatalf("Reinforce iteration %d: %v", i, err)
		}
	}

	cluster, err := store.GetCluster(ctx, "default")
	if err != nil {
		t.Fatalf("GetCluster: %v", err)
	}

	// After 100 iterations with alpha=0.10, utility should be very close to 0.9.
	// Exact: U_n = 0.9 * (1 - (1-0.10)^100) = 0.9 * (1 - 0.9^100).
	// 0.9^100 ~ 2.66e-5, so U ~ 0.89997.
	if math.Abs(cluster.Utility-0.9) > 0.01 {
		t.Errorf("Utility = %f, want ~0.9 (EMA convergence)", cluster.Utility)
	}
}

func TestReinforce_FrequencySaturation(t *testing.T) {
	ctx := context.Background()
	store := memory.NewMemStore()
	d := newTestDecayer()
	mgr := NewMemoryManager(store, d, 0.10, 0.05)

	insertFact(t, store, "f1", "default", "frequency test fact")

	// Reinforce 50 times. Frequency should approach 1.0 asymptotically.
	for i := 0; i < 50; i++ {
		if err := mgr.Reinforce(ctx, map[string]float64{"f1": 0.5}); err != nil {
			t.Fatalf("Reinforce iteration %d: %v", i, err)
		}
	}

	cluster, err := store.GetCluster(ctx, "default")
	if err != nil {
		t.Fatalf("GetCluster: %v", err)
	}

	// F_n = 1 - (1 - beta)^n = 1 - 0.95^50 ~ 1 - 0.0769 = 0.923
	if cluster.Frequency > 1.0 {
		t.Errorf("Frequency = %f, must never exceed 1.0", cluster.Frequency)
	}
	if cluster.Frequency < 0.90 {
		t.Errorf("Frequency = %f, expected > 0.90 after 50 reinforcements", cluster.Frequency)
	}
}

// --- TickDecay tests ---

func TestTickDecay_BumpsAllZeroesAccessed(t *testing.T) {
	ctx := context.Background()
	store := memory.NewMemStore()
	d := newTestDecayer()
	mgr := NewMemoryManager(store, d, 0.10, 0.05)

	// Create two clusters by inserting facts into each.
	insertFact(t, store, "fA", "clusterA", "fact for A")
	insertFact(t, store, "fB", "clusterB", "fact for B")

	// TickDecay with clusterA as accessed.
	if err := mgr.TickDecay(ctx, []string{"clusterA"}); err != nil {
		t.Fatalf("TickDecay: %v", err)
	}

	cA, err := store.GetCluster(ctx, "clusterA")
	if err != nil {
		t.Fatalf("GetCluster(A): %v", err)
	}
	cB, err := store.GetCluster(ctx, "clusterB")
	if err != nil {
		t.Fatalf("GetCluster(B): %v", err)
	}

	// A was accessed -> turns_since reset to 0 after increment.
	if cA.TurnsSince != 0 {
		t.Errorf("clusterA.TurnsSince = %d, want 0 (accessed)", cA.TurnsSince)
	}
	// B was not accessed -> turns_since incremented from 0 to 1.
	if cB.TurnsSince != 1 {
		t.Errorf("clusterB.TurnsSince = %d, want 1 (not accessed)", cB.TurnsSince)
	}
}

func TestTickDecay_EmptyAccessed(t *testing.T) {
	ctx := context.Background()
	store := memory.NewMemStore()
	d := newTestDecayer()
	mgr := NewMemoryManager(store, d, 0.10, 0.05)

	// Create two clusters.
	insertFact(t, store, "fA", "clusterA", "fact for A")
	insertFact(t, store, "fB", "clusterB", "fact for B")

	// TickDecay with empty accessed list.
	if err := mgr.TickDecay(ctx, []string{}); err != nil {
		t.Fatalf("TickDecay: %v", err)
	}

	cA, err := store.GetCluster(ctx, "clusterA")
	if err != nil {
		t.Fatalf("GetCluster(A): %v", err)
	}
	cB, err := store.GetCluster(ctx, "clusterB")
	if err != nil {
		t.Fatalf("GetCluster(B): %v", err)
	}

	// Both should be bumped from 0 to 1.
	if cA.TurnsSince != 1 {
		t.Errorf("clusterA.TurnsSince = %d, want 1 (all bumped)", cA.TurnsSince)
	}
	if cB.TurnsSince != 1 {
		t.Errorf("clusterB.TurnsSince = %d, want 1 (all bumped)", cB.TurnsSince)
	}
}

// upsertEntity is a helper that adds an entity to the store, using an
// orthogonal one-hot vector at `slot` so dedupe-by-similarity cannot
// collapse two distinct calls onto the same row.
func upsertEntity(t *testing.T, s memory.Store, name string, slot int) string {
	t.Helper()
	ctx := context.Background()
	emb := make([]float32, 8)
	emb[slot%len(emb)] = 1.0
	id, _, sim, err := s.UpsertEntity(ctx, name, "file", emb)
	if err != nil {
		t.Fatalf("UpsertEntity(%s): %v", name, err)
	}
	if sim {
		t.Fatalf("UpsertEntity(%s): unexpected similarity collision (slot=%d)", name, slot)
	}
	return id
}

// getEntity is a helper that looks up an entity by id.
func getEntity(t *testing.T, s memory.Store, id string) memory.Entity {
	t.Helper()
	ent, err := s.GetEntity(context.Background(), id)
	if err != nil {
		t.Fatalf("GetEntity(%s): %v", id, err)
	}
	return ent
}

// TestManagerTickDecayWithEntities asserts that the entity-aware tick
// path bumps every entity exactly once per call. Two ticks with an empty
// entity access set leave both entities at turns_since=2 and retention<1.
func TestManagerTickDecayWithEntities(t *testing.T) {
	ctx := context.Background()
	store := memory.NewMemStore()
	d := newTestDecayer()
	mgr := NewMemoryManager(store, d, 0.10, 0.05)

	e1 := upsertEntity(t, store, "alpha", 0)
	e2 := upsertEntity(t, store, "beta", 1)

	for i := 0; i < 2; i++ {
		if err := mgr.TickDecayWithEntities(ctx, nil, nil); err != nil {
			t.Fatalf("tick %d: %v", i, err)
		}
	}

	for _, id := range []string{e1, e2} {
		ent := getEntity(t, store, id)
		if ent.TurnsSince != 2 {
			t.Errorf("entity %s: turns_since=%d, want 2", id, ent.TurnsSince)
		}
		if ent.Retention >= 1.0 {
			t.Errorf("entity %s: retention=%v, want < 1.0", id, ent.Retention)
		}
	}
}

// TestManagerTickDecayWithEntities_PartialAccess asserts the access set
// applies per-entity: the named entity resets while the other entity
// continues to age. Also confirms TickDecay (legacy wrapper) still bumps
// entities at all (otherwise this test would equal turns_since=1 for the
// untouched entity).
func TestManagerTickDecayWithEntities_PartialAccess(t *testing.T) {
	ctx := context.Background()
	store := memory.NewMemStore()
	d := newTestDecayer()
	mgr := NewMemoryManager(store, d, 0.10, 0.05)

	hot := upsertEntity(t, store, "hot", 0)
	cold := upsertEntity(t, store, "cold", 1)

	// First tick: both age by 1.
	if err := mgr.TickDecayWithEntities(ctx, nil, nil); err != nil {
		t.Fatalf("tick 1: %v", err)
	}
	// Second tick: only the "hot" entity is accessed → resets to 0.
	if err := mgr.TickDecayWithEntities(ctx, nil, []string{hot}); err != nil {
		t.Fatalf("tick 2: %v", err)
	}

	hotEnt := getEntity(t, store, hot)
	if hotEnt.TurnsSince != 0 {
		t.Errorf("hot.TurnsSince=%d, want 0 after reset", hotEnt.TurnsSince)
	}
	coldEnt := getEntity(t, store, cold)
	if coldEnt.TurnsSince != 2 {
		t.Errorf("cold.TurnsSince=%d, want 2 after two unaccessed ticks", coldEnt.TurnsSince)
	}
	if coldEnt.Retention >= 1.0 {
		t.Errorf("cold.Retention=%v, want < 1.0 (decayed)", coldEnt.Retention)
	}
}

// TestManagerTickDecay_LegacyWrapper confirms the old TickDecay signature
// still works AND bumps entities (via the wrapper into
// TickDecayWithEntities with a nil entity access set). The session-end
// path uses TickDecayWithEntities directly; every other caller still
// flows through TickDecay.
func TestManagerTickDecay_LegacyWrapper(t *testing.T) {
	ctx := context.Background()
	store := memory.NewMemStore()
	d := newTestDecayer()
	mgr := NewMemoryManager(store, d, 0.10, 0.05)

	insertFact(t, store, "fA", "clusterA", "fact for A")
	id := upsertEntity(t, store, "wrapper-entity", 0)

	if err := mgr.TickDecay(ctx, []string{"clusterA"}); err != nil {
		t.Fatalf("TickDecay: %v", err)
	}

	// Cluster reset.
	cA, _ := store.GetCluster(ctx, "clusterA")
	if cA.TurnsSince != 0 {
		t.Errorf("clusterA.TurnsSince=%d, want 0 (in access set)", cA.TurnsSince)
	}
	// Entity bumped (entity access set was nil).
	ent := getEntity(t, store, id)
	if ent.TurnsSince != 1 {
		t.Errorf("entity.TurnsSince=%d, want 1 (legacy wrapper still ticks entities)", ent.TurnsSince)
	}
}
