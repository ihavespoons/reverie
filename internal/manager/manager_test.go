package manager

import (
	"context"
	"math"
	"testing"

	"personal/reverie/internal/memory"
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
