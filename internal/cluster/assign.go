// Package cluster implements online nearest-centroid cluster assignment and
// centroid running-mean updates for L1 Procedural Memory. It's a thin layer
// over the Store -- no state of its own.
package cluster

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"

	"personal/reverie/internal/memory"
	"personal/reverie/pkg/vecmath"
)

// Assigner decides which cluster a new memory belongs to, creating a fresh
// cluster if no existing one is similar enough.
type Assigner interface {
	// Assign returns the cluster id for the given vector. If no existing
	// cluster's centroid is above minSimilarity, a new cluster is created
	// with the vector as its centroid and isNew=true. If isNew=true, do NOT
	// call AfterInsert -- the new cluster is already initialized with
	// itemCount=1 and centroid=vec.
	Assign(ctx context.Context, vec []float32) (clusterID string, isNew bool, err error)

	// AfterInsert updates the cluster's centroid via running mean after a
	// successful InsertFact/InsertEpisode for a vector that was assigned
	// to an EXISTING cluster (isNew=false from Assign). The running mean:
	//   new_centroid = (old_centroid * n + vec) / (n + 1)
	//   itemCount    = n + 1
	AfterInsert(ctx context.Context, clusterID string, vec []float32) error
}

// clusterStore is the subset of Store the Assigner depends on. Declared
// locally so the memory package can evolve in parallel. The concrete
// Store implementation satisfies this by structural typing.
type clusterStore interface {
	ListClusters(ctx context.Context) ([]memory.ClusterNode, error)
	GetCluster(ctx context.Context, id string) (*memory.ClusterNode, error)
	CreateCluster(ctx context.Context, c memory.ClusterNode) error
	UpdateClusterCentroid(ctx context.Context, id string, centroid []float32, itemCount int) error
}

// assigner is the unexported concrete implementation of Assigner.
type assigner struct {
	store              clusterStore
	minSimilarity      float64
	coldStartUtility   float64
	coldStartFrequency float64
}

// NewAssigner constructs an Assigner with the given store, minimum assignment
// similarity (typically 0.60), and cold-start utility/frequency (typically 0.5/0.5).
func NewAssigner(store clusterStore, minSimilarity, coldStartUtility, coldStartFrequency float64) Assigner {
	return &assigner{
		store:              store,
		minSimilarity:      minSimilarity,
		coldStartUtility:   coldStartUtility,
		coldStartFrequency: coldStartFrequency,
	}
}

// Assign implements Assigner.Assign.
func (a *assigner) Assign(ctx context.Context, vec []float32) (string, bool, error) {
	clusters, err := a.store.ListClusters(ctx)
	if err != nil {
		return "", false, fmt.Errorf("cluster: assign: list clusters: %w", err)
	}

	// No clusters exist -- create the first one.
	if len(clusters) == 0 {
		id, err := a.createCluster(ctx, vec)
		if err != nil {
			return "", false, err
		}
		return id, true, nil
	}

	// Find the nearest cluster by cosine similarity.
	bestID := ""
	bestSim := float32(-1)

	for i := range clusters {
		c := &clusters[i]
		// Skip clusters with nil or zero-length centroids.
		if len(c.Centroid) == 0 {
			continue
		}
		sim := vecmath.Cosine(vec, c.Centroid)
		if sim > bestSim {
			bestSim = sim
			bestID = c.ID
		}
	}

	// If no valid cluster was found (all had nil centroids), create a new one.
	if bestID == "" {
		id, err := a.createCluster(ctx, vec)
		if err != nil {
			return "", false, err
		}
		return id, true, nil
	}

	// If the best similarity is above threshold, return the existing cluster.
	if float64(bestSim) >= a.minSimilarity {
		return bestID, false, nil
	}

	// Below threshold -- create a new cluster.
	id, err := a.createCluster(ctx, vec)
	if err != nil {
		return "", false, err
	}
	return id, true, nil
}

// AfterInsert implements Assigner.AfterInsert.
func (a *assigner) AfterInsert(ctx context.Context, clusterID string, vec []float32) error {
	c, err := a.store.GetCluster(ctx, clusterID)
	if err != nil {
		return fmt.Errorf("cluster: after insert: get cluster %q: %w", clusterID, err)
	}
	if c == nil {
		return fmt.Errorf("cluster: after insert: cluster %q not found", clusterID)
	}

	newCentroid := UpdateCentroid(c.Centroid, c.ItemCount, vec)
	newCount := c.ItemCount + 1

	if err := a.store.UpdateClusterCentroid(ctx, clusterID, newCentroid, newCount); err != nil {
		return fmt.Errorf("cluster: after insert: update centroid %q: %w", clusterID, err)
	}
	return nil
}

// createCluster builds a new ClusterNode with the given vector as its centroid,
// persists it, and returns its ID.
func (a *assigner) createCluster(ctx context.Context, vec []float32) (string, error) {
	id := uuid.New().String()
	now := time.Now().UTC()

	node := memory.ClusterNode{
		ID:         id,
		Summary:    "cluster-" + id[:8],
		Domain:     "",
		MetaInstr:  "",
		ItemCount:  1,
		Centroid:   append([]float32(nil), vec...), // defensive copy
		Utility:    a.coldStartUtility,
		Frequency:  a.coldStartFrequency,
		TurnsSince: 0,
		LastAccess: now,
		CreatedAt:  now,
	}

	if err := a.store.CreateCluster(ctx, node); err != nil {
		return "", fmt.Errorf("cluster: assign: create cluster: %w", err)
	}
	return id, nil
}
