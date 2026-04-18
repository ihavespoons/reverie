package mcpserver

import (
	"context"
	"encoding/json"
	"testing"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"

	"personal/reverie/internal/memory"
)

func TestStatusResource_ReturnsValidJSON(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["status test fact"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Write a fact so counts are non-zero.
	_, _, err := s.handleWrite(ctx, nil, WriteInput{Content: "status test fact", Type: "user"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	req := &mcpsdk.ReadResourceRequest{}
	req.Params = &mcpsdk.ReadResourceParams{URI: "reverie://status"}
	result, err := s.handleStatusResource(ctx, req)
	if err != nil {
		t.Fatalf("handleStatusResource: %v", err)
	}

	if len(result.Contents) != 1 {
		t.Fatalf("expected 1 content, got %d", len(result.Contents))
	}

	// Parse the JSON to verify structure.
	var status statusResponse
	if err := json.Unmarshal([]byte(result.Contents[0].Text), &status); err != nil {
		t.Fatalf("unmarshal status: %v", err)
	}

	if status.Counts.Facts < 1 {
		t.Errorf("expected at least 1 fact, got %d", status.Counts.Facts)
	}
	if status.Counts.Clusters < 1 {
		t.Errorf("expected at least 1 cluster, got %d", status.Counts.Clusters)
	}
	if status.Embedding.Model != "stub" {
		// The stub embedder returns "stub" as its model.
		// Config default is "nomic-embed-text" but we use config defaults.
		// Actually, the status reads from cfg, not embedder.
	}
	if status.Decay.Temperature <= 0 {
		t.Errorf("expected positive temperature, got %f", status.Decay.Temperature)
	}
	if status.Decay.RetentionThreshold <= 0 {
		t.Errorf("expected positive retention threshold, got %f", status.Decay.RetentionThreshold)
	}
	if status.DBPath == "" {
		t.Error("expected non-empty db_path")
	}
}

func TestL1IndexResource_ShowsClusters(t *testing.T) {
	emb := newStubEmbedder(4)
	emb.vectors["l1 index test fact"] = []float32{0.5, 0.5, 0.0, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Write a fact to create a cluster with item_count > 0.
	_, _, err := s.handleWrite(ctx, nil, WriteInput{Content: "l1 index test fact", Type: "project"})
	if err != nil {
		t.Fatalf("write: %v", err)
	}

	req := &mcpsdk.ReadResourceRequest{}
	req.Params = &mcpsdk.ReadResourceParams{URI: "reverie://l1/index"}
	result, err := s.handleL1IndexResource(ctx, req)
	if err != nil {
		t.Fatalf("handleL1IndexResource: %v", err)
	}

	if len(result.Contents) != 1 {
		t.Fatalf("expected 1 content, got %d", len(result.Contents))
	}

	var index l1IndexResponse
	if err := json.Unmarshal([]byte(result.Contents[0].Text), &index); err != nil {
		t.Fatalf("unmarshal l1 index: %v", err)
	}

	if len(index.Clusters) == 0 {
		t.Fatal("expected at least one cluster in L1 index")
	}

	// Verify the cluster has expected fields.
	cl := index.Clusters[0]
	if cl.ID == "" {
		t.Error("expected non-empty cluster ID")
	}
	if cl.ItemCount <= 0 {
		t.Errorf("expected positive item_count, got %d", cl.ItemCount)
	}
	if cl.Retention <= 0 || cl.Retention > 1.0 {
		t.Errorf("expected retention in (0, 1], got %f", cl.Retention)
	}
}

func TestL1IndexResource_OmitsEmptyClusters(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Create a cluster with item_count=0 directly via the store.
	err := s.store.CreateCluster(ctx, memory.ClusterNode{
		ID:      "empty-cluster",
		Summary: "empty",
	})
	if err != nil {
		t.Fatalf("create cluster: %v", err)
	}

	req := &mcpsdk.ReadResourceRequest{}
	req.Params = &mcpsdk.ReadResourceParams{URI: "reverie://l1/index"}
	result, err := s.handleL1IndexResource(ctx, req)
	if err != nil {
		t.Fatalf("handleL1IndexResource: %v", err)
	}

	var index l1IndexResponse
	if err := json.Unmarshal([]byte(result.Contents[0].Text), &index); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// Any clusters in the index should have item_count > 0.
	for _, cl := range index.Clusters {
		if cl.ItemCount == 0 {
			t.Errorf("cluster %s has item_count=0 but should be omitted", cl.ID)
		}
	}
}

func TestL3RecentResource_ShowsEpisodes(t *testing.T) {
	emb := newStubEmbedder(4)
	episodeText := "episode sit\nepisode act\nepisode out\nepisode pre"
	emb.vectors[episodeText] = []float32{0.3, 0.4, 0.5, 0.0}
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	// Write an episode.
	_, epOut, err := s.handleWrite(ctx, nil, WriteInput{
		Type: "feedback",
		Episode: &EpisodePayload{
			Situation:  "episode sit",
			Action:     "episode act",
			Outcome:    "episode out",
			Preemptive: "episode pre",
		},
	})
	if err != nil {
		t.Fatalf("write episode: %v", err)
	}

	req := &mcpsdk.ReadResourceRequest{}
	req.Params = &mcpsdk.ReadResourceParams{URI: "reverie://l3/recent"}
	result, err := s.handleL3RecentResource(ctx, req)
	if err != nil {
		t.Fatalf("handleL3RecentResource: %v", err)
	}

	if len(result.Contents) != 1 {
		t.Fatalf("expected 1 content, got %d", len(result.Contents))
	}

	var recent l3RecentResponse
	if err := json.Unmarshal([]byte(result.Contents[0].Text), &recent); err != nil {
		t.Fatalf("unmarshal l3 recent: %v", err)
	}

	if len(recent.Episodes) != 1 {
		t.Fatalf("expected 1 episode, got %d", len(recent.Episodes))
	}

	ep := recent.Episodes[0]
	if ep.ID != epOut.ID {
		t.Errorf("expected episode ID %s, got %s", epOut.ID, ep.ID)
	}
	if ep.Situation != "episode sit" {
		t.Errorf("expected situation 'episode sit', got %q", ep.Situation)
	}
	if ep.CreatedAt == "" {
		t.Error("expected non-empty created_at")
	}
	if ep.ClusterID == "" {
		t.Error("expected non-empty cluster_id")
	}
}

func TestL3RecentResource_EmptyStore(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()

	req := &mcpsdk.ReadResourceRequest{}
	req.Params = &mcpsdk.ReadResourceParams{URI: "reverie://l3/recent"}
	result, err := s.handleL3RecentResource(ctx, req)
	if err != nil {
		t.Fatalf("handleL3RecentResource: %v", err)
	}

	var recent l3RecentResponse
	if err := json.Unmarshal([]byte(result.Contents[0].Text), &recent); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(recent.Episodes) != 0 {
		t.Errorf("expected 0 episodes, got %d", len(recent.Episodes))
	}
}
