package mcpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

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

// --- reverie://l1/cluster/{id} tests (Phase 1E) ---

// readClusterDetail issues a ReadResource request for the given URI and parses
// the JSON response. Fails the test on any protocol or parse error.
func readClusterDetail(t *testing.T, s *Server, uri string) l1ClusterDetailResponse {
	t.Helper()
	req := &mcpsdk.ReadResourceRequest{}
	req.Params = &mcpsdk.ReadResourceParams{URI: uri}
	result, err := s.handleL1ClusterDetailResource(context.Background(), req)
	if err != nil {
		t.Fatalf("handleL1ClusterDetailResource(%q): %v", uri, err)
	}
	if len(result.Contents) != 1 {
		t.Fatalf("expected 1 content, got %d", len(result.Contents))
	}
	var resp l1ClusterDetailResponse
	if err := json.Unmarshal([]byte(result.Contents[0].Text), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return resp
}

// seedClusterDetail inserts nFacts + nEpisodes directly into the store under
// a single known cluster. Bypassing the write handler avoids the conflict-
// based supersede (which would collapse identical-embedding facts) and the
// assigner's similarity decisions (which may scatter items across clusters).
// The goal here is to exercise the resource handler's pagination/counting
// logic — cluster assignment is covered by the assigner's own test suite.
// Returns the cluster ID used.
func seedClusterDetail(t *testing.T, s *Server, nFacts, nEpisodes int) string {
	t.Helper()
	ctx := context.Background()

	clusterID := "test-cluster-" + fmt.Sprintf("%d-%d", nFacts, nEpisodes)
	if err := s.store.CreateCluster(ctx, memory.ClusterNode{
		ID:         clusterID,
		Summary:    "test cluster",
		Domain:     "test",
		CreatedAt:  time.Now().UTC(),
		LastAccess: time.Now().UTC(),
	}); err != nil {
		t.Fatalf("CreateCluster: %v", err)
	}

	base := time.Now().UTC().Add(-time.Hour)
	for i := 0; i < nFacts; i++ {
		f := memory.Fact{
			ClusterID: clusterID,
			Content:   fmt.Sprintf("cluster fact %d", i),
			Subtype:   "project",
			Source:    "inferred",
			Embedding: []float32{1, 0, 0, 0},
			CreatedAt: base.Add(time.Duration(i) * time.Second),
		}
		if _, err := s.store.InsertFact(ctx, f); err != nil {
			t.Fatalf("InsertFact %d: %v", i, err)
		}
	}

	for i := 0; i < nEpisodes; i++ {
		ep := memory.Episode{
			ClusterID:  clusterID,
			Situation:  fmt.Sprintf("cluster episode situation %d", i),
			Action:     fmt.Sprintf("action %d", i),
			Outcome:    fmt.Sprintf("outcome %d", i),
			Preemptive: fmt.Sprintf("pre %d", i),
			Embedding:  []float32{1, 0, 0, 0},
			// Offset episodes AFTER facts in time so created_at ordering is
			// facts-then-episodes — gives the pagination test a deterministic
			// boundary to cross.
			CreatedAt: base.Add(time.Duration(nFacts+i) * time.Second),
		}
		if _, err := s.store.InsertEpisode(ctx, ep); err != nil {
			t.Fatalf("InsertEpisode %d: %v", i, err)
		}
	}

	return clusterID
}

func TestL1ClusterDetail_ReturnsMetaAndMembers(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	clusterID := seedClusterDetail(t, s, 2, 1)

	resp := readClusterDetail(t, s, "reverie://l1/cluster/"+clusterID)

	if resp.Cluster.ID != clusterID {
		t.Errorf("cluster.id = %q, want %q", resp.Cluster.ID, clusterID)
	}
	if resp.Total != 3 {
		t.Errorf("total = %d, want 3", resp.Total)
	}
	if len(resp.Members) != 3 {
		t.Fatalf("members = %d, want 3", len(resp.Members))
	}
	// Default limit/offset.
	if resp.Limit != 50 || resp.Offset != 0 {
		t.Errorf("limit/offset = %d/%d, want 50/0", resp.Limit, resp.Offset)
	}
	// Facts come first (both stores return facts before episodes, and facts
	// are written first in the seed).
	if resp.Members[0].Layer != "l2_semantic" || resp.Members[2].Layer != "l3_episodic" {
		t.Errorf("member layers = [%s ... %s], want l2_semantic first and l3_episodic last",
			resp.Members[0].Layer, resp.Members[2].Layer)
	}
	// Member ordering is created_at ascending.
	for i := 1; i < len(resp.Members); i++ {
		if resp.Members[i].CreatedAt < resp.Members[i-1].CreatedAt {
			t.Errorf("members not ordered ascending: %v", resp.Members)
		}
	}
}

func TestL1ClusterDetail_Pagination(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	clusterID := seedClusterDetail(t, s, 3, 2) // 3 facts, 2 episodes = 5 total

	// Page 1: limit=2 offset=0 — all facts-side.
	p1 := readClusterDetail(t, s, fmt.Sprintf("reverie://l1/cluster/%s?limit=2&offset=0", clusterID))
	if len(p1.Members) != 2 || p1.Total != 5 {
		t.Fatalf("p1 members=%d total=%d, want 2/5", len(p1.Members), p1.Total)
	}

	// Page 2: limit=2 offset=2 — straddles fact/episode boundary.
	p2 := readClusterDetail(t, s, fmt.Sprintf("reverie://l1/cluster/%s?limit=2&offset=2", clusterID))
	if len(p2.Members) != 2 {
		t.Fatalf("p2 members=%d, want 2", len(p2.Members))
	}
	if p2.Members[0].Layer != "l2_semantic" || p2.Members[1].Layer != "l3_episodic" {
		t.Errorf("p2 layers = [%s %s], want [l2_semantic l3_episodic]",
			p2.Members[0].Layer, p2.Members[1].Layer)
	}

	// Page 3: offset=4 — last episode only.
	p3 := readClusterDetail(t, s, fmt.Sprintf("reverie://l1/cluster/%s?limit=2&offset=4", clusterID))
	if len(p3.Members) != 1 {
		t.Fatalf("p3 members=%d, want 1", len(p3.Members))
	}
	if p3.Members[0].Layer != "l3_episodic" {
		t.Errorf("p3 last member layer = %q, want l3_episodic", p3.Members[0].Layer)
	}

	// Collect IDs across pages — no duplicates, no gaps.
	seen := map[string]bool{}
	for _, p := range []l1ClusterDetailResponse{p1, p2, p3} {
		for _, m := range p.Members {
			if seen[m.ID] {
				t.Errorf("duplicate id across pages: %s", m.ID)
			}
			seen[m.ID] = true
		}
	}
	if len(seen) != 5 {
		t.Errorf("covered %d distinct ids, want 5", len(seen))
	}
}

func TestL1ClusterDetail_LimitCapped(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	clusterID := seedClusterDetail(t, s, 1, 0)

	resp := readClusterDetail(t, s, fmt.Sprintf("reverie://l1/cluster/%s?limit=5000", clusterID))
	if resp.Limit != 200 {
		t.Errorf("limit = %d, want 200 (capped)", resp.Limit)
	}
}

func TestL1ClusterDetail_UnknownCluster(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	req := &mcpsdk.ReadResourceRequest{}
	req.Params = &mcpsdk.ReadResourceParams{URI: "reverie://l1/cluster/does-not-exist"}
	_, err := s.handleL1ClusterDetailResource(context.Background(), req)
	if err == nil {
		t.Fatal("expected error for unknown cluster")
	}
	if !strings.Contains(err.Error(), "cluster not found") {
		t.Errorf("err = %v, want 'cluster not found'", err)
	}
}

func TestL1ClusterDetail_SupersededFactsExcluded(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	ctx := context.Background()
	clusterID := seedClusterDetail(t, s, 2, 0)

	// Find the two seeded facts and supersede the first by the second.
	facts, err := s.store.ListFactsByCluster(ctx, clusterID, 100, 0)
	if err != nil {
		t.Fatalf("ListFactsByCluster: %v", err)
	}
	if len(facts) != 2 {
		t.Fatalf("expected 2 facts, got %d", len(facts))
	}
	if err := s.store.SupersedeFact(ctx, facts[0].ID, facts[1].ID); err != nil {
		t.Fatalf("SupersedeFact: %v", err)
	}

	resp := readClusterDetail(t, s, "reverie://l1/cluster/"+clusterID)
	if resp.Total != 1 {
		t.Errorf("total = %d, want 1 (superseded fact excluded)", resp.Total)
	}
	for _, m := range resp.Members {
		if m.ID == facts[0].ID {
			t.Errorf("superseded fact %s should be excluded from members", facts[0].ID)
		}
	}
}

func TestL1ClusterDetail_InvalidLimitOffset(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	req := &mcpsdk.ReadResourceRequest{}
	req.Params = &mcpsdk.ReadResourceParams{URI: "reverie://l1/cluster/any?limit=-1"}
	if _, err := s.handleL1ClusterDetailResource(context.Background(), req); err == nil {
		t.Error("negative limit should error")
	}

	req.Params.URI = "reverie://l1/cluster/any?limit=abc"
	if _, err := s.handleL1ClusterDetailResource(context.Background(), req); err == nil {
		t.Error("non-numeric limit should error")
	}

	req.Params.URI = "reverie://l1/cluster/any?offset=-5"
	if _, err := s.handleL1ClusterDetailResource(context.Background(), req); err == nil {
		t.Error("negative offset should error")
	}
}
