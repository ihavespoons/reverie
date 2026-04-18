// Package manager implements reinforcement and decay-clock management for
// memory clusters. It sits between the Store and the MCP tool handlers.
package manager

import (
	"context"
	"fmt"

	"personal/reverie/internal/memory"
)

// MemoryManager reinforces utility/frequency on accessed clusters and
// advances the decay clock. Reinforcement is strictly tied to usage
// (per the Oblivion paper).
type MemoryManager interface {
	// Reinforce updates utility and frequency for the clusters owning the
	// memories named in credits. `credits[memoryID] = score` in [0,1].
	// For each cluster containing any credited memory, applies:
	//   U_new = U_old + alpha * (maxCreditForCluster - U_old)
	//   F_new = min(1, F_old + beta * (1 - F_old))
	Reinforce(ctx context.Context, credits map[string]float64) error

	// TickDecay increments turns_since on every cluster, then resets
	// turns_since to 0 for the clusters named in accessedClusterIDs.
	// Should be called once per write-path turn.
	TickDecay(ctx context.Context, accessedClusterIDs []string) error
}

// decayer is the local interface shape MemoryManager needs from a Decayer.
// Declared here (not imported from internal/decay) so the packages can be
// developed in parallel. Structural typing wires them at the call site.
type decayer interface {
	Retention(c memory.ClusterNode) float64
	GateC(c memory.ClusterNode) bool
	Temperature() float64
	Threshold() float64
}

// memoryManager is the concrete, unexported implementation of MemoryManager.
type memoryManager struct {
	store         memory.Store
	d             decayer
	utilityAlpha  float64 // EMA learning rate for utility
	frequencyBeta float64 // frequency saturation rate
}

// NewMemoryManager constructs a MemoryManager. utilityAlpha (alpha) is the EMA
// learning rate for utility (default 0.10); frequencyBeta (beta) is the
// frequency saturation rate (default 0.05). Both must be in (0,1].
func NewMemoryManager(store memory.Store, d decayer, utilityAlpha, frequencyBeta float64) MemoryManager {
	if utilityAlpha <= 0 || utilityAlpha > 1 {
		utilityAlpha = 0.10
	}
	if frequencyBeta <= 0 || frequencyBeta > 1 {
		frequencyBeta = 0.05
	}
	return &memoryManager{
		store:         store,
		d:             d,
		utilityAlpha:  utilityAlpha,
		frequencyBeta: frequencyBeta,
	}
}

// Reinforce updates utility and frequency for each cluster that owns at least
// one credited memory. Credits map memory IDs to a score in [0,1]. For each
// cluster, the maximum credit across its constituent memories is taken and
// applied via an exponential moving average (EMA):
//
//	U_new = U_old + alpha * (maxCredit - U_old)
//	F_new = min(1, F_old + beta * (1 - F_old))
//
// Memories that do not exist in the store are silently skipped.
func (mm *memoryManager) Reinforce(ctx context.Context, credits map[string]float64) error {
	if len(credits) == 0 {
		return nil
	}

	// Phase 1: resolve each memory ID to its cluster and compute the max
	// credit per cluster.
	clusterCredits := make(map[string]float64) // clusterID -> maxCredit
	for memID, score := range credits {
		fact, err := mm.store.GetFact(ctx, memID)
		if err != nil {
			return fmt.Errorf("manager: reinforce: get fact %s: %w", memID, err)
		}
		if fact == nil {
			// Unknown memory — skip silently.
			continue
		}
		clusterID := fact.ClusterID
		if cur, ok := clusterCredits[clusterID]; !ok || score > cur {
			clusterCredits[clusterID] = score
		}
	}

	// Phase 2: apply EMA update to each affected cluster.
	for clusterID, maxCredit := range clusterCredits {
		cluster, err := mm.store.GetCluster(ctx, clusterID)
		if err != nil {
			return fmt.Errorf("manager: reinforce: get cluster %s: %w", clusterID, err)
		}
		if cluster == nil {
			// Cluster does not exist — skip. Should not happen in practice
			// since the fact references it, but be defensive.
			continue
		}

		// EMA utility update: U_new = U_old + alpha * (maxCredit - U_old)
		newUtility := cluster.Utility + mm.utilityAlpha*(maxCredit-cluster.Utility)

		// Frequency saturation: F_new = min(1, F_old + beta * (1 - F_old))
		newFrequency := cluster.Frequency + mm.frequencyBeta*(1-cluster.Frequency)
		if newFrequency > 1.0 {
			newFrequency = 1.0
		}

		if err := mm.store.UpdateClusterState(ctx, clusterID, newUtility, newFrequency, cluster.TurnsSince); err != nil {
			return fmt.Errorf("manager: reinforce: update cluster %s: %w", clusterID, err)
		}
	}

	return nil
}

// TickDecay delegates to Store.TickAllClusters: increments turns_since on every
// cluster by 1, then resets to 0 for the clusters in accessedClusterIDs.
func (mm *memoryManager) TickDecay(ctx context.Context, accessedClusterIDs []string) error {
	if err := mm.store.TickAllClusters(ctx, accessedClusterIDs); err != nil {
		return fmt.Errorf("manager: tick decay: %w", err)
	}
	return nil
}
