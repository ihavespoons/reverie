package memory

import (
	"context"
	"errors"
	"fmt"
)

// ErrEmptyCluster is returned by RecomputeCentroid when the cluster has zero
// members (or zero members with embeddings). The caller decides whether to
// delete the empty cluster row; RecomputeCentroid leaves it in place.
var ErrEmptyCluster = errors.New("memory: cluster has no members with embeddings")

// RecomputeCentroid loads all non-superseded facts and all episodes in the
// given cluster, averages their embeddings element-wise, and writes the
// result via Store.UpdateClusterCentroid.
//
// Members with nil or empty embeddings are skipped. If there are no members
// at all, or every member lacks an embedding, ErrEmptyCluster is returned and
// the cluster row is left untouched (caller decides whether to delete it).
//
// Paging: for large clusters we page through members in chunks to keep memory
// bounded and avoid loading every embedding twice (we accumulate into a running
// sum). Facts are iterated first, then episodes.
func RecomputeCentroid(ctx context.Context, store Store, clusterID string) error {
	// Use a large-but-bounded page size. clusterDetailMaxLimit is defined in
	// the mcpserver package; we pick an independent value here.
	const pageSize = 200

	var (
		sum       []float32
		dim       int
		withEmbed int
	)

	// Fold a single embedding into the running sum. Initializes sum/dim on
	// first non-empty embedding; later embeddings of differing length are
	// skipped (shouldn't happen in practice — embed model is fixed per store).
	fold := func(vec []float32) {
		if len(vec) == 0 {
			return
		}
		if sum == nil {
			sum = make([]float32, len(vec))
			dim = len(vec)
		}
		if len(vec) != dim {
			return
		}
		for i, v := range vec {
			sum[i] += v
		}
		withEmbed++
	}

	// Iterate facts.
	offset := 0
	for {
		facts, err := store.ListFactsByCluster(ctx, clusterID, pageSize, offset)
		if err != nil {
			return fmt.Errorf("memory: recompute centroid: list facts: %w", err)
		}
		if len(facts) == 0 {
			break
		}
		for i := range facts {
			fold(facts[i].Embedding)
		}
		if len(facts) < pageSize {
			break
		}
		offset += len(facts)
	}

	// Iterate episodes.
	offset = 0
	for {
		episodes, err := store.ListEpisodesByCluster(ctx, clusterID, pageSize, offset)
		if err != nil {
			return fmt.Errorf("memory: recompute centroid: list episodes: %w", err)
		}
		if len(episodes) == 0 {
			break
		}
		for i := range episodes {
			fold(episodes[i].Embedding)
		}
		if len(episodes) < pageSize {
			break
		}
		offset += len(episodes)
	}

	// Total member count includes members without embeddings (per spec:
	// centroid math uses only embedded members, but item_count is the full
	// non-superseded membership count). Fetch counts so the item_count stays
	// consistent with ListFactsByCluster/ListEpisodesByCluster semantics.
	factCount, err := store.CountFactsByCluster(ctx, clusterID)
	if err != nil {
		return fmt.Errorf("memory: recompute centroid: count facts: %w", err)
	}
	episodeCount, err := store.CountEpisodesByCluster(ctx, clusterID)
	if err != nil {
		return fmt.Errorf("memory: recompute centroid: count episodes: %w", err)
	}
	itemCount := factCount + episodeCount

	if itemCount == 0 || withEmbed == 0 {
		return ErrEmptyCluster
	}

	// Finalize mean.
	avg := make([]float32, dim)
	n := float32(withEmbed)
	for i, s := range sum {
		avg[i] = s / n
	}

	if err := store.UpdateClusterCentroid(ctx, clusterID, avg, itemCount); err != nil {
		return fmt.Errorf("memory: recompute centroid: update: %w", err)
	}
	return nil
}
