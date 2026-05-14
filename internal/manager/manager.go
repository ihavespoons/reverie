// Package manager implements reinforcement and decay-clock management for
// memory clusters. It sits between the Store and the MCP tool handlers.
package manager

import (
	"context"
	"fmt"
	"time"

	"github.com/diffsec/reverie/internal/memory"
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
	// Should be called once per write-path turn. Entities are bumped as
	// well, with an empty access set (the write path doesn't know which
	// entities were touched). Thin wrapper around TickDecayWithEntities.
	TickDecay(ctx context.Context, accessedClusterIDs []string) error

	// TickDecayWithEntities is the entity-aware variant of TickDecay used
	// at session-end. Order of operations: TickAllClusters →
	// TickAllEntities → SetLastTick. A failure on either tick is
	// surfaced immediately and SetLastTick is NOT recorded.
	TickDecayWithEntities(ctx context.Context, accessedClusterIDs, accessedEntityIDs []string) error
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

// TickDecay is a thin wrapper around TickDecayWithEntities that bumps
// clusters (resetting accessedClusterIDs) and ALL entities (with an
// empty access set). The write path uses this — it doesn't know which
// entities the turn touched. The session-end path calls
// TickDecayWithEntities directly so it can attribute entity hits to the
// memories in the session buffer.
func (mm *memoryManager) TickDecay(ctx context.Context, accessedClusterIDs []string) error {
	return mm.TickDecayWithEntities(ctx, accessedClusterIDs, nil)
}

// TickDecayWithEntities runs both decay tick paths in order: clusters
// first, then entities, then records the last-tick timestamp. Either
// tick failure short-circuits and skips SetLastTick — the next attempt
// will retry from a known-bad state rather than papering over the
// partial commit. A SetLastTick failure after both ticks succeed is
// surfaced as a wrapped error; the ticks themselves already committed.
func (mm *memoryManager) TickDecayWithEntities(ctx context.Context, accessedClusterIDs, accessedEntityIDs []string) error {
	if err := mm.store.TickAllClusters(ctx, accessedClusterIDs); err != nil {
		return fmt.Errorf("manager: tick decay: clusters: %w", err)
	}
	if err := mm.store.TickAllEntities(ctx, accessedEntityIDs); err != nil {
		return fmt.Errorf("manager: tick decay: entities: %w", err)
	}
	if err := mm.store.SetLastTick(ctx, time.Now().UTC()); err != nil {
		return fmt.Errorf("manager: tick decay: record last tick: %w", err)
	}
	return nil
}
