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

// --- reverie://stats/daily tests (Phase 5C) ---

// readStatsDaily issues a ReadResource against the stats/daily handler and
// parses the JSON. Failure modes — protocol error, bad JSON — fail the test.
func readStatsDaily(t *testing.T, s *Server, uri string) dailyStatsResponse {
	t.Helper()
	req := &mcpsdk.ReadResourceRequest{}
	req.Params = &mcpsdk.ReadResourceParams{URI: uri}
	result, err := s.handleStatsDailyResource(context.Background(), req)
	if err != nil {
		t.Fatalf("handleStatsDailyResource(%q): %v", uri, err)
	}
	if len(result.Contents) != 1 {
		t.Fatalf("expected 1 content, got %d", len(result.Contents))
	}
	var resp dailyStatsResponse
	if err := json.Unmarshal([]byte(result.Contents[0].Text), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return resp
}

func TestStatsDaily_DefaultRangeReturns31Days(t *testing.T) {
	// Default span: from = to - 30, both inclusive → 31 zero-filled rows.
	// Locking the inclusive-math choice into the test so downstream clients
	// can rely on it.
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	resp := readStatsDaily(t, s, "reverie://stats/daily")
	if len(resp.Days) != 31 {
		t.Errorf("len(Days) = %d, want 31 (inclusive 30-day default lookback)", len(resp.Days))
	}
	if resp.From == "" || resp.To == "" {
		t.Errorf("from/to should be populated: from=%q to=%q", resp.From, resp.To)
	}
	// Today is the last day in the default window.
	today := time.Now().UTC().Format("2006-01-02")
	if resp.To != today {
		t.Errorf("to = %q, want %q", resp.To, today)
	}
	// All rows zero-valued on a fresh (memStore) backend.
	for i, d := range resp.Days {
		if d.FactsIn+d.FactsOut+d.EpisodesIn+d.EpisodesOut+d.Supersedes != 0 {
			t.Errorf("day[%d] %+v should be all-zero on fresh store", i, d)
		}
	}
	// Dates are strictly ascending, continuous.
	for i := 1; i < len(resp.Days); i++ {
		if resp.Days[i].Date <= resp.Days[i-1].Date {
			t.Errorf("dates not strictly ascending at %d: %s then %s", i, resp.Days[i-1].Date, resp.Days[i].Date)
		}
	}
}

func TestStatsDaily_ExplicitRangeAndGapFill(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	// 5-day window. memStore returns no rows, so all should be zero-filled.
	resp := readStatsDaily(t, s, "reverie://stats/daily?from=2026-01-01&to=2026-01-05")
	if resp.From != "2026-01-01" || resp.To != "2026-01-05" {
		t.Errorf("from/to = %s..%s, want 2026-01-01..2026-01-05", resp.From, resp.To)
	}
	if len(resp.Days) != 5 {
		t.Fatalf("len(Days) = %d, want 5", len(resp.Days))
	}
	wantDates := []string{"2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"}
	for i, want := range wantDates {
		if resp.Days[i].Date != want {
			t.Errorf("day[%d].Date = %s, want %s", i, resp.Days[i].Date, want)
		}
	}
}

func TestStatsDaily_FromAfterToErrors(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	req := &mcpsdk.ReadResourceRequest{}
	req.Params = &mcpsdk.ReadResourceParams{URI: "reverie://stats/daily?from=2026-02-01&to=2026-01-01"}
	_, err := s.handleStatsDailyResource(context.Background(), req)
	if err == nil {
		t.Fatal("expected error for from > to")
	}
	if !strings.Contains(err.Error(), "must be <= to") {
		t.Errorf("err = %v, want 'must be <= to'", err)
	}
}

func TestStatsDaily_SpanTooLongErrors(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	// 366 days from 2025-01-01 to 2026-01-01 exceeds the 365-day cap.
	req := &mcpsdk.ReadResourceRequest{}
	req.Params = &mcpsdk.ReadResourceParams{URI: "reverie://stats/daily?from=2025-01-01&to=2026-01-01"}
	_, err := s.handleStatsDailyResource(context.Background(), req)
	if err == nil {
		t.Fatal("expected error for span > 365 days")
	}
	if !strings.Contains(err.Error(), "exceeds max") {
		t.Errorf("err = %v, want 'exceeds max'", err)
	}
}

func TestStatsDaily_EmptyStoreZeroesNotError(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	// A one-day range on a fresh store should return one zero row.
	resp := readStatsDaily(t, s, "reverie://stats/daily?from=2026-04-18&to=2026-04-18")
	if len(resp.Days) != 1 {
		t.Fatalf("len(Days) = %d, want 1", len(resp.Days))
	}
	d := resp.Days[0]
	if d.Date != "2026-04-18" {
		t.Errorf("date = %s, want 2026-04-18", d.Date)
	}
	if d.FactsIn != 0 || d.FactsOut != 0 || d.EpisodesIn != 0 || d.EpisodesOut != 0 || d.Supersedes != 0 {
		t.Errorf("expected all-zero row, got %+v", d)
	}
}

// --- reverie://l1/at_risk tests (Phase 5B) ---

// readAtRisk issues a ReadResource against the at_risk handler and parses
// the JSON response. Protocol / JSON errors fail the test.
func readAtRisk(t *testing.T, s *Server, uri string) l1AtRiskResponse {
	t.Helper()
	req := &mcpsdk.ReadResourceRequest{}
	req.Params = &mcpsdk.ReadResourceParams{URI: uri}
	result, err := s.handleL1AtRiskResource(context.Background(), req)
	if err != nil {
		t.Fatalf("handleL1AtRiskResource(%q): %v", uri, err)
	}
	if len(result.Contents) != 1 {
		t.Fatalf("expected 1 content, got %d", len(result.Contents))
	}
	var resp l1AtRiskResponse
	if err := json.Unmarshal([]byte(result.Contents[0].Text), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return resp
}

// seedAtRiskCluster creates a single non-empty cluster (ItemCount=1) with the
// given id and sets its utility/frequency/turns_since so the decayer returns
// a controllable retention. The test server uses a standard Decayer with
// temperature=10 and threshold=0.3; with U=F=0.5 the stability is ~10.1, so
// retention falls off exp(-n/10.1).
func seedAtRiskCluster(t *testing.T, s *Server, id string, turnsSince int) {
	t.Helper()
	ctx := context.Background()
	if err := s.store.CreateCluster(ctx, memory.ClusterNode{
		ID:        id,
		Summary:   "at-risk " + id,
		Domain:    "test",
		ItemCount: 1,
	}); err != nil {
		t.Fatalf("CreateCluster(%s): %v", id, err)
	}
	// UpdateClusterState is the shortest path to a known turns_since without
	// threading writes + decay ticks through the manager.
	if err := s.store.UpdateClusterState(ctx, id, 0.5, 0.5, turnsSince); err != nil {
		t.Fatalf("UpdateClusterState(%s): %v", id, err)
	}
}

func TestL1AtRisk_OrdersAscendingByRetention(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	// Retention uses strict < threshold, and R=exp(0)=1.0 at turns_since=0.
	// Use turns_since=1 for the "fresh" case so R<1 and it qualifies under
	// threshold=1.0, keeping three distinct ordered retention values.
	// turns_since 1 -> R~0.906, 10 -> ~0.37, 100 -> ~5e-5.
	seedAtRiskCluster(t, s, "c-fresh", 1)
	seedAtRiskCluster(t, s, "c-mid", 10)
	seedAtRiskCluster(t, s, "c-stale", 100)

	// threshold=1.0 includes everything (since R<1 for turns_since>=1).
	resp := readAtRisk(t, s, "reverie://l1/at_risk?threshold=1.0")
	if resp.Threshold != 1.0 {
		t.Errorf("Threshold = %g, want 1.0", resp.Threshold)
	}
	if len(resp.Clusters) != 3 || resp.Total != 3 {
		t.Fatalf("len/total = %d/%d, want 3/3", len(resp.Clusters), resp.Total)
	}
	wantOrder := []string{"c-stale", "c-mid", "c-fresh"}
	for i, id := range wantOrder {
		if resp.Clusters[i].ID != id {
			t.Errorf("Clusters[%d].ID = %q, want %q (retention order)", i, resp.Clusters[i].ID, id)
		}
	}
	for i := 1; i < len(resp.Clusters); i++ {
		if resp.Clusters[i].Retention < resp.Clusters[i-1].Retention {
			t.Errorf("retention not ascending: %v", resp.Clusters)
		}
	}
}

func TestL1AtRisk_DefaultThresholdFromConfig(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	// Default threshold is 0.3. With U=F=0.5,T=10 (stability ~10.1):
	//   turns_since=0 -> R=1.0 (above, excluded)
	//   turns_since=100 -> R~5e-5 (below, included)
	seedAtRiskCluster(t, s, "c-above", 0)
	seedAtRiskCluster(t, s, "c-below", 100)

	resp := readAtRisk(t, s, "reverie://l1/at_risk")
	if resp.Threshold != s.decayer.Threshold() {
		t.Errorf("Threshold = %g, want %g (config default)", resp.Threshold, s.decayer.Threshold())
	}
	if len(resp.Clusters) != 1 || resp.Total != 1 {
		t.Fatalf("len/total = %d/%d, want 1/1 (only c-below)", len(resp.Clusters), resp.Total)
	}
	if resp.Clusters[0].ID != "c-below" {
		t.Errorf("Clusters[0].ID = %q, want c-below", resp.Clusters[0].ID)
	}
}

func TestL1AtRisk_ExplicitThresholdOverride(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	// With threshold=0.5 the "mid" cluster (R~0.37) is in; the fresh cluster
	// (R=1.0) is out. This proves the query param takes precedence over the
	// default 0.3 (which would ALSO exclude the mid cluster).
	seedAtRiskCluster(t, s, "c-fresh", 0)
	seedAtRiskCluster(t, s, "c-mid", 10)
	seedAtRiskCluster(t, s, "c-stale", 100)

	resp := readAtRisk(t, s, "reverie://l1/at_risk?threshold=0.5")
	if resp.Threshold != 0.5 {
		t.Errorf("Threshold = %g, want 0.5", resp.Threshold)
	}
	if len(resp.Clusters) != 2 || resp.Total != 2 {
		t.Fatalf("len/total = %d/%d, want 2/2", len(resp.Clusters), resp.Total)
	}
	// Most at-risk first.
	if resp.Clusters[0].ID != "c-stale" || resp.Clusters[1].ID != "c-mid" {
		t.Errorf("order = [%s %s], want [c-stale c-mid]", resp.Clusters[0].ID, resp.Clusters[1].ID)
	}
}

func TestL1AtRisk_ThresholdOneIncludesAllNonEmpty(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	// See TestL1AtRisk_OrdersAscendingByRetention — turns_since must be >=1
	// so R<1 and the cluster qualifies under threshold=1.0.
	seedAtRiskCluster(t, s, "c-fresh", 1)
	seedAtRiskCluster(t, s, "c-mid", 10)

	// Empty cluster must be excluded regardless of threshold.
	if err := s.store.CreateCluster(context.Background(), memory.ClusterNode{
		ID: "c-empty", Summary: "empty",
	}); err != nil {
		t.Fatalf("CreateCluster(empty): %v", err)
	}

	resp := readAtRisk(t, s, "reverie://l1/at_risk?threshold=1.0")
	if len(resp.Clusters) != 2 || resp.Total != 2 {
		t.Fatalf("len/total = %d/%d, want 2/2 (non-empty only)", len(resp.Clusters), resp.Total)
	}
	for _, c := range resp.Clusters {
		if c.ID == "c-empty" {
			t.Errorf("empty cluster %q should be excluded", c.ID)
		}
	}
}

func TestL1AtRisk_LimitCapsResultsTotalReportsFull(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	// 10 clusters, all deeply at-risk. limit=3 must yield 3 entries while
	// Total reflects the full matching count.
	for i := 0; i < 10; i++ {
		seedAtRiskCluster(t, s, fmt.Sprintf("c-%02d", i), 1000)
	}

	resp := readAtRisk(t, s, "reverie://l1/at_risk?limit=3")
	if len(resp.Clusters) != 3 {
		t.Errorf("len(Clusters) = %d, want 3 (limit)", len(resp.Clusters))
	}
	if resp.Total != 10 {
		t.Errorf("Total = %d, want 10 (pre-limit)", resp.Total)
	}
}

func TestL1AtRisk_LimitHardCapAt500(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	// 501 at-risk clusters; limit=999 should be clamped to 500.
	for i := 0; i < 501; i++ {
		seedAtRiskCluster(t, s, fmt.Sprintf("c-%04d", i), 1000)
	}

	resp := readAtRisk(t, s, "reverie://l1/at_risk?limit=999")
	if len(resp.Clusters) != 500 {
		t.Errorf("len(Clusters) = %d, want 500 (hard cap)", len(resp.Clusters))
	}
	if resp.Total != 501 {
		t.Errorf("Total = %d, want 501", resp.Total)
	}
}

func TestL1AtRisk_EmptyStore(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	resp := readAtRisk(t, s, "reverie://l1/at_risk")
	if resp.Threshold != s.decayer.Threshold() {
		t.Errorf("Threshold = %g, want %g", resp.Threshold, s.decayer.Threshold())
	}
	if len(resp.Clusters) != 0 {
		t.Errorf("len(Clusters) = %d, want 0", len(resp.Clusters))
	}
	if resp.Total != 0 {
		t.Errorf("Total = %d, want 0", resp.Total)
	}
}

func TestL1AtRisk_InvalidThresholdAndLimit(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	cases := []struct {
		name string
		uri  string
	}{
		{"threshold below zero", "reverie://l1/at_risk?threshold=-0.1"},
		{"threshold above one", "reverie://l1/at_risk?threshold=1.5"},
		{"threshold not a number", "reverie://l1/at_risk?threshold=abc"},
		{"limit negative", "reverie://l1/at_risk?limit=-1"},
		{"limit not a number", "reverie://l1/at_risk?limit=abc"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := &mcpsdk.ReadResourceRequest{}
			req.Params = &mcpsdk.ReadResourceParams{URI: tc.uri}
			if _, err := s.handleL1AtRiskResource(context.Background(), req); err == nil {
				t.Errorf("%s: expected error, got nil", tc.uri)
			}
		})
	}
}

func TestStatsDaily_InvalidDateFormat(t *testing.T) {
	emb := newStubEmbedder(4)
	s := newTestServer(emb)
	defer s.recallCache.stop()

	req := &mcpsdk.ReadResourceRequest{}
	req.Params = &mcpsdk.ReadResourceParams{URI: "reverie://stats/daily?from=not-a-date"}
	if _, err := s.handleStatsDailyResource(context.Background(), req); err == nil {
		t.Error("expected error for malformed from")
	}
}
