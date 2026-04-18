package mcpserver

import (
	"context"
	"crypto/sha256"
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/google/uuid"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"

	"personal/reverie/internal/memory"
)

// validSubtypes is the set of allowed fact subtypes.
var validSubtypes = map[string]bool{
	"user":      true,
	"feedback":  true,
	"project":   true,
	"reference": true,
}

// --- memory_recall ---

// RecallInput is the input schema for the memory_recall tool.
type RecallInput struct {
	Query string   `json:"query" jsonschema:"Natural-language query to search memories"`
	Limit int      `json:"limit,omitempty" jsonschema:"Maximum number of results (default 10)"`
	Hints []string `json:"hints,omitempty" jsonschema:"Optional hint strings to augment the query"`
	Round int      `json:"round,omitempty" jsonschema:"Recall round: 0 for OR-logic (default); 1+ for AND-logic refinement"`
}

// RecallCandidate is a single candidate in the recall result.
type RecallCandidate struct {
	ID         string   `json:"id"`
	Content    string   `json:"content"`
	Layer      string   `json:"layer"`
	Subtype    string   `json:"subtype,omitempty"`
	Similarity float32  `json:"similarity"`
	Retention  float64  `json:"retention"`
	GateBPass  bool     `json:"gate_b_pass"`
	GateCPass  bool     `json:"gate_c_pass"`
	LinkedIDs  []string `json:"linked_ids,omitempty"`
	ClusterID  string   `json:"cluster_id"`
	Tags       []string `json:"tags,omitempty"`
}

// RecallOutput is the output schema for the memory_recall tool.
type RecallOutput struct {
	RecallID   string            `json:"recall_id"`
	Round      int               `json:"round"`
	Candidates []RecallCandidate `json:"candidates"`
}

func (s *Server) handleRecall(ctx context.Context, _ *mcpsdk.CallToolRequest, in RecallInput) (*mcpsdk.CallToolResult, RecallOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, RecallOutput{}, nil
	}

	if in.Query == "" {
		return nil, RecallOutput{}, fmt.Errorf("query is required")
	}

	limit := in.Limit
	if limit <= 0 {
		limit = 10
	}

	// Compute query embedding.
	vecs, err := s.embedder.Embed(ctx, []string{in.Query})
	if err != nil {
		return nil, RecallOutput{}, fmt.Errorf("embed query: %w", err)
	}
	if len(vecs) == 0 || len(vecs[0]) == 0 {
		return nil, RecallOutput{}, fmt.Errorf("embedder returned empty vector")
	}
	queryVec := vecs[0]

	// Global cosine search.
	candidates, err := s.store.GlobalSearch(ctx, queryVec, limit)
	if err != nil {
		return nil, RecallOutput{}, fmt.Errorf("global search: %w", err)
	}

	threshold := s.cfg.Memory.SimilarityThreshold
	if threshold <= 0 {
		threshold = 0.70
	}

	out := RecallOutput{
		RecallID:   uuid.New().String(),
		Round:      in.Round,
		Candidates: make([]RecallCandidate, len(candidates)),
	}

	for i, c := range candidates {
		var retention float64
		var gateCPass bool

		clusterID := c.ClusterID()
		cl, err := s.store.GetCluster(ctx, clusterID)
		if err != nil {
			s.logger.Warn("memory_recall: get cluster failed", "cluster_id", clusterID, "err", err)
		}
		if cl != nil {
			retention = s.decayer.Retention(*cl)
			gateCPass = s.decayer.GateC(*cl)
		}
		// If cluster not found (shouldn't happen): retention=0, gate_c_pass=false.

		rc := RecallCandidate{
			ID:         c.ID(),
			Content:    c.Content(),
			Layer:      string(c.Layer()),
			Similarity: c.Similarity,
			Retention:  retention,
			GateBPass:  float64(c.Similarity) > threshold,
			GateCPass:  gateCPass,
			ClusterID:  clusterID,
		}

		// Set subtype and tags for facts; tags for episodes.
		if c.Fact != nil {
			rc.Subtype = c.Fact.Subtype
			rc.Tags = normalizeTagsSlice(c.Fact.Tags)
		} else if c.Episode != nil {
			rc.Tags = normalizeTagsSlice(c.Episode.Tags)
		} else {
			rc.Tags = normalizeTagsSlice(nil)
		}

		// Fetch cross-type linked IDs.
		if c.Fact != nil {
			links, linkErr := s.store.GetFactLinks(ctx, c.Fact.ID)
			if linkErr != nil {
				s.logger.Warn("memory_recall: get fact links failed", "fact_id", c.Fact.ID, "err", linkErr)
			}
			for _, l := range links {
				rc.LinkedIDs = append(rc.LinkedIDs, l.EpisodeID)
			}
		} else if c.Episode != nil {
			links, linkErr := s.store.GetEpisodeLinks(ctx, c.Episode.ID)
			if linkErr != nil {
				s.logger.Warn("memory_recall: get episode links failed", "episode_id", c.Episode.ID, "err", linkErr)
			}
			for _, l := range links {
				rc.LinkedIDs = append(rc.LinkedIDs, l.FactID)
			}
		}

		out.Candidates[i] = rc
	}

	// Stash in recall cache for memory_apply_judgment.
	s.recallCache.put(out.RecallID, &cachedRecall{
		queryVec:   queryVec,
		candidates: candidates,
		round:      in.Round,
		createdAt:  time.Now(),
	})

	s.logger.Info("memory_recall", "recall_id", out.RecallID, "candidates", len(out.Candidates), "round", in.Round)
	return nil, out, nil
}

// --- memory_write ---

// EpisodePayload is the structured input for writing an L3 episodic memory.
type EpisodePayload struct {
	Situation     string   `json:"situation" jsonschema:"what triggered this episode"`
	Action        string   `json:"action" jsonschema:"what was done"`
	Outcome       string   `json:"outcome" jsonschema:"what happened as a result"`
	Preemptive    string   `json:"preemptive" jsonschema:"actionable lesson: what to do next time"`
	LinkedFactIDs []string `json:"linked_fact_ids,omitempty" jsonschema:"fact IDs this episode cross-references"`
}

// WriteInput is the input schema for the memory_write tool.
type WriteInput struct {
	Content string          `json:"content,omitempty" jsonschema:"The content of the memory (required for L2 facts)"`
	Type    string          `json:"type" jsonschema:"Memory subtype: user, feedback, project, or reference"`
	Tags    []string        `json:"tags,omitempty" jsonschema:"Optional tags for categorization"`
	Source  string          `json:"source,omitempty" jsonschema:"Source attribution (default: inferred)"`
	Episode *EpisodePayload `json:"episode,omitempty" jsonschema:"if set, writes an L3 episode instead of an L2 fact"`
}

// WriteOutput is the output schema for the memory_write tool.
type WriteOutput struct {
	ID    string `json:"id"`
	Layer string `json:"layer"`
}

func (s *Server) handleWrite(ctx context.Context, _ *mcpsdk.CallToolRequest, in WriteInput) (*mcpsdk.CallToolResult, WriteOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, WriteOutput{}, nil
	}

	hasContent := in.Content != ""
	hasEpisode := in.Episode != nil

	// Validate: exactly one of content or episode must be provided.
	if hasContent == hasEpisode {
		if hasContent {
			return nil, WriteOutput{}, fmt.Errorf("provide either content (L2 fact) or episode (L3 episode), not both")
		}
		return nil, WriteOutput{}, fmt.Errorf("either content (L2 fact) or episode (L3 episode) is required")
	}

	if !validSubtypes[in.Type] {
		return nil, WriteOutput{}, fmt.Errorf("invalid type %q: must be one of user, feedback, project, reference", in.Type)
	}

	if hasEpisode {
		return s.writeEpisode(ctx, in)
	}
	return s.writeFact(ctx, in)
}

// writeEpisode handles the L3 episode write path.
func (s *Server) writeEpisode(ctx context.Context, in WriteInput) (*mcpsdk.CallToolResult, WriteOutput, error) {
	ep := in.Episode

	// Compute embedding over the concatenated episode fields.
	embedText := ep.Situation + "\n" + ep.Action + "\n" + ep.Outcome + "\n" + ep.Preemptive
	vecs, err := s.embedder.Embed(ctx, []string{embedText})
	if err != nil {
		return nil, WriteOutput{}, fmt.Errorf("embed episode: %w", err)
	}
	if len(vecs) == 0 || len(vecs[0]) == 0 {
		return nil, WriteOutput{}, fmt.Errorf("embedder returned empty vector")
	}
	vec := vecs[0]

	// Cluster assignment.
	clusterID, isNew, err := s.assigner.Assign(ctx, vec)
	if err != nil {
		return nil, WriteOutput{}, fmt.Errorf("cluster assign: %w", err)
	}

	h := sha256.Sum256([]byte(embedText))
	contentHash := fmt.Sprintf("%x", h)

	episode := memory.Episode{
		ClusterID:     clusterID,
		Situation:     ep.Situation,
		Action:        ep.Action,
		Outcome:       ep.Outcome,
		Preemptive:    ep.Preemptive,
		ContentHash:   contentHash,
		Embedding:     vec,
		LinkedFactIDs: ep.LinkedFactIDs,
		Tags:          in.Tags,
	}

	id, err := s.store.InsertEpisode(ctx, episode)
	if err != nil {
		return nil, WriteOutput{}, fmt.Errorf("insert episode: %w", err)
	}

	// Update centroid for existing clusters.
	if !isNew {
		if afterErr := s.assigner.AfterInsert(ctx, clusterID, vec); afterErr != nil {
			s.logger.Warn("memory_write: after insert (episode) failed", "cluster_id", clusterID, "err", afterErr)
		}
	}

	// Tick decay for the cluster.
	if tickErr := s.mgr.TickDecay(ctx, []string{clusterID}); tickErr != nil {
		s.logger.Warn("memory_write: tick decay (episode) failed", "err", tickErr)
	}

	s.logger.Info("memory_write", "id", id, "layer", "l3_episodic", "cluster_id", clusterID, "is_new_cluster", isNew)
	return nil, WriteOutput{ID: id, Layer: string(memory.TypeL3Episodic)}, nil
}

// writeFact handles the L2 fact write path with cluster assignment and conflict detection.
func (s *Server) writeFact(ctx context.Context, in WriteInput) (*mcpsdk.CallToolResult, WriteOutput, error) {
	// Compute embedding for the content.
	vecs, err := s.embedder.Embed(ctx, []string{in.Content})
	if err != nil {
		return nil, WriteOutput{}, fmt.Errorf("embed content: %w", err)
	}
	if len(vecs) == 0 || len(vecs[0]) == 0 {
		return nil, WriteOutput{}, fmt.Errorf("embedder returned empty vector")
	}
	vec := vecs[0]

	// Cluster assignment.
	clusterID, isNew, err := s.assigner.Assign(ctx, vec)
	if err != nil {
		return nil, WriteOutput{}, fmt.Errorf("cluster assign: %w", err)
	}

	source := in.Source
	if source == "" {
		source = "inferred"
	}

	h := sha256.Sum256([]byte(in.Content))
	contentHash := fmt.Sprintf("%x", h)

	fact := memory.Fact{
		Content:     in.Content,
		ContentHash: contentHash,
		ClusterID:   clusterID,
		Subtype:     in.Type,
		Source:      source,
		Embedding:   vec,
		Confidence:  1.0,
		Tags:        in.Tags,
	}

	// Conflict detection: find near-duplicate facts in the same subtype.
	conflictThreshold := s.cfg.Memory.ConflictThreshold
	if conflictThreshold <= 0 {
		conflictThreshold = 0.92
	}
	similar, err := s.store.FindSimilarFacts(ctx, in.Type, vec, float32(conflictThreshold), 1)
	if err != nil {
		s.logger.Warn("memory_write: find similar facts failed", "err", err)
	}

	// Insert the new fact.
	id, err := s.store.InsertFact(ctx, fact)
	if err != nil {
		return nil, WriteOutput{}, fmt.Errorf("insert fact: %w", err)
	}

	// Supersede the old fact if a conflict was found.
	if len(similar) > 0 {
		oldFact := similar[0].Fact
		if oldFact != nil && oldFact.ID != id {
			oldContent := oldFact.Content
			if len(oldContent) > 80 {
				oldContent = oldContent[:80] + "..."
			}
			if superErr := s.store.SupersedeFact(ctx, oldFact.ID, id); superErr != nil {
				s.logger.Warn("memory_write: supersede fact failed", "old_id", oldFact.ID, "new_id", id, "err", superErr)
			} else {
				s.logger.Info("memory_write: superseded fact", "old_id", oldFact.ID, "new_id", id, "old_content", oldContent)
			}
		}
	}

	// Update centroid for existing clusters.
	if !isNew {
		if afterErr := s.assigner.AfterInsert(ctx, clusterID, vec); afterErr != nil {
			s.logger.Warn("memory_write: after insert (fact) failed", "cluster_id", clusterID, "err", afterErr)
		}
	}

	// Tick decay for the cluster.
	if tickErr := s.mgr.TickDecay(ctx, []string{clusterID}); tickErr != nil {
		s.logger.Warn("memory_write: tick decay (fact) failed", "err", tickErr)
	}

	s.logger.Info("memory_write", "id", id, "subtype", in.Type, "content_len", len(in.Content), "cluster_id", clusterID, "is_new_cluster", isNew)
	return nil, WriteOutput{ID: id, Layer: string(memory.TypeL2Semantic)}, nil
}

// --- memory_reinforce ---

// ReinforceInput is the input schema for the memory_reinforce tool.
type ReinforceInput struct {
	MemoryIDs []string `json:"memory_ids" jsonschema:"IDs of memories to reinforce"`
}

// ReinforceOutput is the output schema for the memory_reinforce tool.
type ReinforceOutput struct {
	Reinforced int `json:"reinforced"`
}

func (s *Server) handleReinforce(ctx context.Context, _ *mcpsdk.CallToolRequest, in ReinforceInput) (*mcpsdk.CallToolResult, ReinforceOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, ReinforceOutput{}, nil
	}

	if len(in.MemoryIDs) == 0 {
		return nil, ReinforceOutput{}, fmt.Errorf("memory_ids is required and must not be empty")
	}

	// Build credits map — default each id to 1.0 (full credit).
	credits := make(map[string]float64, len(in.MemoryIDs))
	for _, id := range in.MemoryIDs {
		credits[id] = 1.0
	}

	// Update cluster utility/frequency via MemoryManager.
	if err := s.mgr.Reinforce(ctx, credits); err != nil {
		return nil, ReinforceOutput{}, fmt.Errorf("reinforce: %w", err)
	}

	// Also update fact-level accessed_at timestamps.
	if err := s.store.TouchAccessed(ctx, in.MemoryIDs); err != nil {
		return nil, ReinforceOutput{}, fmt.Errorf("touch accessed: %w", err)
	}

	s.logger.Info("memory_reinforce", "count", len(in.MemoryIDs))
	return nil, ReinforceOutput{Reinforced: len(in.MemoryIDs)}, nil
}

// --- memory_forget ---

// ForgetInput is the input schema for the memory_forget tool.
type ForgetInput struct {
	ID    string `json:"id,omitempty" jsonschema:"ID of a specific memory to delete"`
	Query string `json:"query,omitempty" jsonschema:"Query to find candidates for deletion (returns candidates without deleting)"`
}

// ForgetCandidate is a candidate returned for confirmation when query is used.
type ForgetCandidate struct {
	ID      string `json:"id"`
	Content string `json:"content"`
	Layer   string `json:"layer"`
}

// ForgetOutput is the output schema for the memory_forget tool.
type ForgetOutput struct {
	Deleted    int               `json:"deleted"`
	Candidates []ForgetCandidate `json:"candidates,omitempty"`
}

func (s *Server) handleForget(ctx context.Context, _ *mcpsdk.CallToolRequest, in ForgetInput) (*mcpsdk.CallToolResult, ForgetOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, ForgetOutput{}, nil
	}

	hasID := in.ID != ""
	hasQuery := in.Query != ""

	if hasID == hasQuery {
		return nil, ForgetOutput{}, fmt.Errorf("exactly one of id or query must be provided")
	}

	if hasID {
		// Try fact first; if not found, try episode; if neither, error.
		fact, err := s.store.GetFact(ctx, in.ID)
		if err != nil {
			return nil, ForgetOutput{}, fmt.Errorf("get fact: %w", err)
		}
		if fact != nil {
			if delErr := s.store.DeleteFact(ctx, in.ID); delErr != nil {
				return nil, ForgetOutput{}, fmt.Errorf("delete fact: %w", delErr)
			}
			s.logger.Info("memory_forget", "deleted_id", in.ID, "layer", "l2_semantic")
			return nil, ForgetOutput{Deleted: 1}, nil
		}

		ep, err := s.store.GetEpisode(ctx, in.ID)
		if err != nil {
			return nil, ForgetOutput{}, fmt.Errorf("get episode: %w", err)
		}
		if ep != nil {
			if delErr := s.store.DeleteEpisode(ctx, in.ID); delErr != nil {
				return nil, ForgetOutput{}, fmt.Errorf("delete episode: %w", delErr)
			}
			s.logger.Info("memory_forget", "deleted_id", in.ID, "layer", "l3_episodic")
			return nil, ForgetOutput{Deleted: 1}, nil
		}

		return nil, ForgetOutput{}, fmt.Errorf("memory not found: %s", in.ID)
	}

	// Query mode: search for candidates but do not delete.
	vecs, err := s.embedder.Embed(ctx, []string{in.Query})
	if err != nil {
		return nil, ForgetOutput{}, fmt.Errorf("embed query: %w", err)
	}
	if len(vecs) == 0 || len(vecs[0]) == 0 {
		return nil, ForgetOutput{}, fmt.Errorf("embedder returned empty vector")
	}

	candidates, err := s.store.GlobalSearch(ctx, vecs[0], 5)
	if err != nil {
		return nil, ForgetOutput{}, fmt.Errorf("global search: %w", err)
	}

	out := ForgetOutput{Deleted: 0}
	for _, c := range candidates {
		out.Candidates = append(out.Candidates, ForgetCandidate{
			ID:      c.ID(),
			Content: c.Content(),
			Layer:   string(c.Layer()),
		})
	}

	s.logger.Info("memory_forget", "query", in.Query, "candidates", len(out.Candidates))
	return nil, out, nil
}

// --- memory_list ---

// ListInput is the input schema for the memory_list tool.
type ListInput struct {
	Layer   string   `json:"layer,omitempty" jsonschema:"l2 (default, facts) or l3 (episodes)"`
	Subtype string   `json:"subtype,omitempty" jsonschema:"Filter by subtype: user, feedback, project, reference"`
	Limit   int      `json:"limit,omitempty" jsonschema:"Maximum number of results (default 25)"`
	Offset  int      `json:"offset,omitempty" jsonschema:"Number of results to skip (for pagination)"`
	Sort    string   `json:"sort,omitempty" jsonschema:"Sort order: created (default) or accessed"`
	TagsAny []string `json:"tags_any,omitempty" jsonschema:"Filter to memories with at least one of these tags"`
}

// ListMemory is a memory entry in the list result, without the embedding vector
// to avoid bloating the tool response.
type ListMemory struct {
	ID         string   `json:"id"`
	Content    string   `json:"content"`
	Layer      string   `json:"layer"`
	Subtype    string   `json:"subtype,omitempty"`
	Source     string   `json:"source,omitempty"`
	Confidence float64  `json:"confidence,omitempty"`
	CreatedAt  string   `json:"created_at"`
	AccessedAt string   `json:"accessed_at"`
	ClusterID  string   `json:"cluster_id"`
	Tags       []string `json:"tags,omitempty"`
}

// ListOutput is the output schema for the memory_list tool.
type ListOutput struct {
	Memories []ListMemory `json:"memories"`
}

func (s *Server) handleList(ctx context.Context, _ *mcpsdk.CallToolRequest, in ListInput) (*mcpsdk.CallToolResult, ListOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, ListOutput{}, nil
	}

	filter := memory.ListFilter{
		Limit:   in.Limit,
		Offset:  in.Offset,
		Sort:    in.Sort,
		TagsAny: in.TagsAny,
	}

	if in.Layer == "l3" {
		// L3 episodes path.
		episodes, err := s.store.ListEpisodes(ctx, filter)
		if err != nil {
			return nil, ListOutput{}, fmt.Errorf("list episodes: %w", err)
		}

		out := ListOutput{
			Memories: make([]ListMemory, len(episodes)),
		}
		for i, ep := range episodes {
			content := memory.Candidate{Episode: &ep}.Content()
			out.Memories[i] = ListMemory{
				ID:         ep.ID,
				Content:    content,
				Layer:      string(memory.TypeL3Episodic),
				CreatedAt:  ep.CreatedAt.Format("2006-01-02T15:04:05Z"),
				AccessedAt: ep.AccessedAt.Format("2006-01-02T15:04:05Z"),
				ClusterID:  ep.ClusterID,
				Tags:       normalizeTagsSlice(ep.Tags),
			}
		}

		s.logger.Info("memory_list", "count", len(out.Memories), "layer", "l3")
		return nil, out, nil
	}

	// Default: L2 facts path.
	if in.Subtype != "" {
		filter.Subtype = &in.Subtype
	}

	facts, err := s.store.ListFacts(ctx, filter)
	if err != nil {
		return nil, ListOutput{}, fmt.Errorf("list facts: %w", err)
	}

	out := ListOutput{
		Memories: make([]ListMemory, len(facts)),
	}
	for i, f := range facts {
		out.Memories[i] = ListMemory{
			ID:         f.ID,
			Content:    f.Content,
			Layer:      string(memory.TypeL2Semantic),
			Subtype:    f.Subtype,
			Source:     f.Source,
			Confidence: f.Confidence,
			CreatedAt:  f.CreatedAt.Format("2006-01-02T15:04:05Z"),
			AccessedAt: f.AccessedAt.Format("2006-01-02T15:04:05Z"),
			ClusterID:  f.ClusterID,
			Tags:       normalizeTagsSlice(f.Tags),
		}
	}

	s.logger.Info("memory_list", "count", len(out.Memories), "subtype", in.Subtype)
	return nil, out, nil
}

// normalizeTagsSlice returns a non-nil []string so that JSON encoding produces
// [] instead of null. Per the Phase 1 behavior spec, tool outputs should emit
// an explicit empty array for the tags field rather than a null.
func normalizeTagsSlice(tags []string) []string {
	if tags == nil {
		return []string{}
	}
	return tags
}

// --- memory_apply_judgment ---

// Verdict is a single keep/drop verdict from the memory-judge subagent.
type Verdict struct {
	MemoryID string `json:"memory_id"`
	Keep     bool   `json:"keep"`
	Reason   string `json:"reason,omitempty"`
}

// ApplyJudgmentInput is the input schema for the memory_apply_judgment tool.
type ApplyJudgmentInput struct {
	RecallID string    `json:"recall_id" jsonschema:"the recall_id returned by memory_recall"`
	Verdicts []Verdict `json:"verdicts" jsonschema:"per-candidate keep/drop verdicts from the memory-judge subagent"`
}

// MemoryRefResp is a memory reference in the apply judgment output, without embeddings.
type MemoryRefResp struct {
	ID         string  `json:"id"`
	Content    string  `json:"content"`
	Layer      string  `json:"layer"`
	Subtype    string  `json:"subtype"`
	Similarity float32 `json:"similarity"`
	Retention  float64 `json:"retention"`
}

// ApplyJudgmentOutput is the output schema for the memory_apply_judgment tool.
type ApplyJudgmentOutput struct {
	Memories     []MemoryRefResp `json:"memories"`
	AppliedLogic string          `json:"applied_logic"` // "OR" or "AND"
}

func (s *Server) handleApplyJudgment(ctx context.Context, _ *mcpsdk.CallToolRequest, in ApplyJudgmentInput) (*mcpsdk.CallToolResult, ApplyJudgmentOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, ApplyJudgmentOutput{}, nil
	}

	if in.RecallID == "" {
		return nil, ApplyJudgmentOutput{}, fmt.Errorf("recall_id is required")
	}

	cached, ok := s.recallCache.get(in.RecallID)
	if !ok {
		return nil, ApplyJudgmentOutput{}, fmt.Errorf("recall_id not found or expired")
	}

	// Build verdict lookup: memoryID -> keep.
	verdictMap := make(map[string]bool, len(in.Verdicts))
	for _, v := range in.Verdicts {
		verdictMap[v.MemoryID] = v.Keep
	}

	threshold := s.cfg.Memory.SimilarityThreshold
	if threshold <= 0 {
		threshold = 0.70
	}

	logic := "OR"
	if cached.round >= 1 {
		logic = "AND"
	}

	// scoredCandidate holds a candidate that passed the gate filter along with
	// its composite score for ranking.
	type scoredCandidate struct {
		candidate memory.Candidate
		retention float64
		composite float64
	}

	var passed []scoredCandidate

	for _, c := range cached.candidates {
		// Gate A: from verdicts. Missing verdict defaults to false.
		gateA := verdictMap[c.ID()]

		// Gate B: similarity threshold.
		gateB := float64(c.Similarity) > threshold

		// Gate C: retention threshold from decayer.
		var retention float64
		var gateC bool
		cl, err := s.store.GetCluster(ctx, c.ClusterID())
		if err != nil {
			s.logger.Warn("apply_judgment: get cluster failed", "cluster_id", c.ClusterID(), "err", err)
		}
		if cl != nil {
			retention = s.decayer.Retention(*cl)
			gateC = s.decayer.GateC(*cl)
		}

		var pass bool
		if cached.round == 0 {
			pass = gateA || gateB || gateC
		} else {
			pass = gateA && gateB && gateC
		}

		if pass {
			// Composite score: similarity * max(retention, 0.01).
			ret := math.Max(retention, 0.01)
			composite := float64(c.Similarity) * ret
			passed = append(passed, scoredCandidate{
				candidate: c,
				retention: retention,
				composite: composite,
			})
		}
	}

	// Sort by composite score descending.
	sort.Slice(passed, func(i, j int) bool {
		return passed[i].composite > passed[j].composite
	})

	// Budget cap.
	budgetMax := s.cfg.Memory.CacheBudgetMax
	if budgetMax <= 0 {
		budgetMax = 50
	}
	if len(passed) > budgetMax {
		passed = passed[:budgetMax]
	}

	// Touch accessed for passing candidates.
	if len(passed) > 0 {
		passedIDs := make([]string, len(passed))
		for i, sc := range passed {
			passedIDs[i] = sc.candidate.ID()
		}
		if err := s.store.TouchAccessed(ctx, passedIDs); err != nil {
			s.logger.Warn("apply_judgment: touch accessed failed", "err", err)
		}
	}

	// Build output (strip embeddings).
	memories := make([]MemoryRefResp, len(passed))
	for i, sc := range passed {
		ref := MemoryRefResp{
			ID:         sc.candidate.ID(),
			Content:    sc.candidate.Content(),
			Layer:      string(sc.candidate.Layer()),
			Similarity: sc.candidate.Similarity,
			Retention:  sc.retention,
		}
		if sc.candidate.Fact != nil {
			ref.Subtype = sc.candidate.Fact.Subtype
		}
		memories[i] = ref
	}

	s.logger.Info("memory_apply_judgment", "recall_id", in.RecallID, "round", cached.round, "logic", logic, "passed", len(memories))
	return nil, ApplyJudgmentOutput{Memories: memories, AppliedLogic: logic}, nil
}

// --- memory_decay_tick ---

// DecayTickInput is the input schema for the memory_decay_tick tool.
type DecayTickInput struct {
	TurnsElapsed int  `json:"turns_elapsed,omitempty" jsonschema:"Number of turns elapsed (ignored in Phase 2)"`
	SessionEnd   bool `json:"session_end,omitempty" jsonschema:"If true, treat as session-end tick (bumps all clusters)"`
}

// DecayTickOutput is the output schema for the memory_decay_tick tool.
type DecayTickOutput struct {
	Ticked bool `json:"ticked"`
}

// --- memory_update_cluster ---

// UpdateClusterInput is the input schema for the memory_update_cluster tool.
type UpdateClusterInput struct {
	ClusterID string `json:"cluster_id" jsonschema:"the cluster ID to update"`
	Summary   string `json:"summary,omitempty" jsonschema:"new summary for the cluster"`
	Domain    string `json:"domain,omitempty" jsonschema:"domain label (e.g. go-patterns, user-prefs)"`
	MetaInstr string `json:"meta_instr,omitempty" jsonschema:"meta-instruction (e.g. when dealing with X, always do Y)"`
}

// UpdateClusterOutput is the output schema for the memory_update_cluster tool.
type UpdateClusterOutput struct {
	ID        string  `json:"id"`
	Summary   string  `json:"summary"`
	Domain    string  `json:"domain"`
	MetaInstr string  `json:"meta_instr"`
	ItemCount int     `json:"item_count"`
	Utility   float64 `json:"utility"`
	Frequency float64 `json:"frequency"`
}

func (s *Server) handleUpdateCluster(ctx context.Context, _ *mcpsdk.CallToolRequest, in UpdateClusterInput) (*mcpsdk.CallToolResult, UpdateClusterOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, UpdateClusterOutput{}, nil
	}

	if in.ClusterID == "" {
		return nil, UpdateClusterOutput{}, fmt.Errorf("cluster_id is required")
	}

	// Fetch the existing cluster.
	cl, err := s.store.GetCluster(ctx, in.ClusterID)
	if err != nil {
		return nil, UpdateClusterOutput{}, fmt.Errorf("get cluster: %w", err)
	}
	if cl == nil {
		return nil, UpdateClusterOutput{}, fmt.Errorf("cluster %q not found", in.ClusterID)
	}

	// Build updated values: only overwrite non-empty fields from input.
	summary := cl.Summary
	if in.Summary != "" {
		summary = in.Summary
	}
	domain := cl.Domain
	if in.Domain != "" {
		domain = in.Domain
	}
	metaInstr := cl.MetaInstr
	if in.MetaInstr != "" {
		metaInstr = in.MetaInstr
	}

	if err := s.store.UpdateClusterMeta(ctx, in.ClusterID, summary, domain, metaInstr); err != nil {
		return nil, UpdateClusterOutput{}, fmt.Errorf("update cluster meta: %w", err)
	}

	s.logger.Info("memory_update_cluster", "cluster_id", in.ClusterID, "summary", summary, "domain", domain)

	return nil, UpdateClusterOutput{
		ID:        cl.ID,
		Summary:   summary,
		Domain:    domain,
		MetaInstr: metaInstr,
		ItemCount: cl.ItemCount,
		Utility:   cl.Utility,
		Frequency: cl.Frequency,
	}, nil
}

func (s *Server) handleDecayTick(ctx context.Context, _ *mcpsdk.CallToolRequest, in DecayTickInput) (*mcpsdk.CallToolResult, DecayTickOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, DecayTickOutput{}, nil
	}

	if in.SessionEnd {
		s.logger.Info("memory_decay_tick: session end, ticking all clusters")
	}

	// TickDecay with nil accessedIDs bumps all clusters with none reset.
	if err := s.mgr.TickDecay(ctx, nil); err != nil {
		return nil, DecayTickOutput{}, fmt.Errorf("tick decay: %w", err)
	}

	s.logger.Info("memory_decay_tick", "session_end", in.SessionEnd)
	return nil, DecayTickOutput{Ticked: true}, nil
}

// --- memory_reassign_cluster ---

// ReassignClusterInput is the input schema for the memory_reassign_cluster tool.
type ReassignClusterInput struct {
	MemoryID        string `json:"memory_id" jsonschema:"fact or episode ID"`
	TargetClusterID string `json:"target_cluster_id" jsonschema:"destination cluster ID (must exist)"`
}

// ReassignClusterOutput is the output schema for the memory_reassign_cluster tool.
type ReassignClusterOutput struct {
	MemoryID          string `json:"memory_id"`
	OldClusterID      string `json:"old_cluster_id"`
	NewClusterID      string `json:"new_cluster_id"`
	OldClusterDeleted bool   `json:"old_cluster_deleted"`
}

func (s *Server) handleReassignCluster(ctx context.Context, _ *mcpsdk.CallToolRequest, in ReassignClusterInput) (*mcpsdk.CallToolResult, ReassignClusterOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, ReassignClusterOutput{}, nil
	}

	if in.MemoryID == "" {
		return nil, ReassignClusterOutput{}, fmt.Errorf("memory_id is required")
	}
	if in.TargetClusterID == "" {
		return nil, ReassignClusterOutput{}, fmt.Errorf("target_cluster_id is required")
	}

	// Look up memory: fact first, then episode. Capture old cluster id.
	var oldClusterID string
	var layer string
	fact, err := s.store.GetFact(ctx, in.MemoryID)
	if err != nil {
		return nil, ReassignClusterOutput{}, fmt.Errorf("get fact: %w", err)
	}
	if fact != nil {
		oldClusterID = fact.ClusterID
		layer = string(memory.TypeL2Semantic)
	} else {
		ep, getErr := s.store.GetEpisode(ctx, in.MemoryID)
		if getErr != nil {
			return nil, ReassignClusterOutput{}, fmt.Errorf("get episode: %w", getErr)
		}
		if ep == nil {
			return nil, ReassignClusterOutput{}, fmt.Errorf("memory not found: %s", in.MemoryID)
		}
		oldClusterID = ep.ClusterID
		layer = string(memory.TypeL3Episodic)
	}

	// Same-cluster reassign is a no-op error.
	if oldClusterID == in.TargetClusterID {
		return nil, ReassignClusterOutput{}, fmt.Errorf("memory %s already in cluster %s", in.MemoryID, in.TargetClusterID)
	}

	// Validate target cluster exists.
	target, err := s.store.GetCluster(ctx, in.TargetClusterID)
	if err != nil {
		return nil, ReassignClusterOutput{}, fmt.Errorf("get target cluster: %w", err)
	}
	if target == nil {
		return nil, ReassignClusterOutput{}, fmt.Errorf("cluster not found: %s", in.TargetClusterID)
	}

	// Move the memory. SetMemoryCluster bumps accessed_at as part of the write.
	if err := s.store.SetMemoryCluster(ctx, in.MemoryID, in.TargetClusterID); err != nil {
		return nil, ReassignClusterOutput{}, fmt.Errorf("set memory cluster: %w", err)
	}

	// Recompute centroid for the new cluster. The memory is now a member, so
	// it will be folded into the average.
	if err := memory.RecomputeCentroid(ctx, s.store, in.TargetClusterID); err != nil && err != memory.ErrEmptyCluster {
		return nil, ReassignClusterOutput{}, fmt.Errorf("recompute target centroid: %w", err)
	}

	// Recompute centroid for the old cluster. If it's now empty, delete it.
	oldDeleted := false
	recErr := memory.RecomputeCentroid(ctx, s.store, oldClusterID)
	switch {
	case recErr == memory.ErrEmptyCluster:
		if err := s.store.DeleteCluster(ctx, oldClusterID); err != nil {
			return nil, ReassignClusterOutput{}, fmt.Errorf("delete empty old cluster: %w", err)
		}
		oldDeleted = true
	case recErr != nil:
		return nil, ReassignClusterOutput{}, fmt.Errorf("recompute old centroid: %w", recErr)
	}

	// Treat the target cluster as accessed — reset its turns_since. TickDecay
	// increments every cluster then resets the listed ones, matching the
	// pattern used by handleWrite after InsertFact/InsertEpisode.
	if err := s.mgr.TickDecay(ctx, []string{in.TargetClusterID}); err != nil {
		s.logger.Warn("memory_reassign_cluster: tick decay failed", "cluster_id", in.TargetClusterID, "err", err)
	}

	s.logger.Info("memory_reassign_cluster",
		"memory_id", in.MemoryID,
		"layer", layer,
		"old_cluster_id", oldClusterID,
		"new_cluster_id", in.TargetClusterID,
		"old_cluster_deleted", oldDeleted,
	)

	return nil, ReassignClusterOutput{
		MemoryID:          in.MemoryID,
		OldClusterID:      oldClusterID,
		NewClusterID:      in.TargetClusterID,
		OldClusterDeleted: oldDeleted,
	}, nil
}

// --- memory_split_cluster ---

// ClusterMeta carries optional L1 metadata to apply to each new cluster
// created by memory_split_cluster. All fields are optional.
type ClusterMeta struct {
	Summary   string `json:"summary,omitempty"`
	Domain    string `json:"domain,omitempty"`
	MetaInstr string `json:"meta_instr,omitempty"`
}

// SplitClusterInput is the input schema for the memory_split_cluster tool.
type SplitClusterInput struct {
	ClusterID string        `json:"cluster_id" jsonschema:"cluster to split (must exist)"`
	Groups    [][]string    `json:"groups" jsonschema:"non-overlapping partitions; each group becomes a new cluster"`
	Metas     []ClusterMeta `json:"metas,omitempty" jsonschema:"optional metadata for each new cluster (summary, domain, meta_instr). Must be same length as groups if provided."`
}

// SplitClusterOutput is the output schema for the memory_split_cluster tool.
type SplitClusterOutput struct {
	SourceClusterID   string   `json:"source_cluster_id"`
	NewClusterIDs     []string `json:"new_cluster_ids"`
	SourceDeleted     bool     `json:"source_deleted"`
	RemainingInSource int      `json:"remaining_in_source"`
}

// handleSplitCluster partitions a source cluster's members into one or more
// new clusters based on explicit ID groups. Any members not appearing in a
// group remain in the source. If all members are partitioned, the source is
// deleted; otherwise its centroid is recomputed.
//
// The operation is not currently wrapped in a transaction — steps are executed
// sequentially. A future pass (once Phase 2B's withTx helper is available) can
// rewrap for full all-or-nothing atomicity.
func (s *Server) handleSplitCluster(ctx context.Context, _ *mcpsdk.CallToolRequest, in SplitClusterInput) (*mcpsdk.CallToolResult, SplitClusterOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, SplitClusterOutput{}, nil
	}

	if in.ClusterID == "" {
		return nil, SplitClusterOutput{}, fmt.Errorf("cluster_id is required")
	}
	if len(in.Groups) == 0 {
		return nil, SplitClusterOutput{}, fmt.Errorf("groups is required and must not be empty")
	}
	if len(in.Metas) > 0 && len(in.Metas) != len(in.Groups) {
		return nil, SplitClusterOutput{}, fmt.Errorf("metas length (%d) must match groups length (%d)", len(in.Metas), len(in.Groups))
	}

	// Validate source cluster exists.
	source, err := s.store.GetCluster(ctx, in.ClusterID)
	if err != nil {
		return nil, SplitClusterOutput{}, fmt.Errorf("get source cluster: %w", err)
	}
	if source == nil {
		return nil, SplitClusterOutput{}, fmt.Errorf("cluster not found: %s", in.ClusterID)
	}

	// Build the source membership set once: all non-superseded fact IDs plus
	// all episode IDs currently in the cluster. Paged to keep memory bounded
	// on large clusters.
	membership, err := s.collectClusterMembership(ctx, in.ClusterID)
	if err != nil {
		return nil, SplitClusterOutput{}, err
	}

	// Validate groups: non-empty, no overlap across groups, every member is
	// in the source cluster. Build the union set so we can compute "remaining".
	union := make(map[string]struct{})
	for i, g := range in.Groups {
		if len(g) == 0 {
			return nil, SplitClusterOutput{}, fmt.Errorf("group %d is empty", i)
		}
		for _, id := range g {
			if _, ok := membership[id]; !ok {
				return nil, SplitClusterOutput{}, fmt.Errorf("memory %s not a member of cluster %s", id, in.ClusterID)
			}
			if _, dup := union[id]; dup {
				return nil, SplitClusterOutput{}, fmt.Errorf("memory %s appears in more than one group", id)
			}
			union[id] = struct{}{}
		}
	}

	// For each group, create a new cluster and reparent its members.
	newClusterIDs := make([]string, len(in.Groups))
	for i, g := range in.Groups {
		newID := uuid.New().String()

		node := memory.ClusterNode{ID: newID}
		if i < len(in.Metas) {
			node.Summary = in.Metas[i].Summary
			node.Domain = in.Metas[i].Domain
			node.MetaInstr = in.Metas[i].MetaInstr
		}
		if err := s.store.CreateCluster(ctx, node); err != nil {
			return nil, SplitClusterOutput{}, fmt.Errorf("create new cluster for group %d: %w", i, err)
		}

		// Reparent each member. SetMemoryCluster also bumps accessed_at.
		for _, memID := range g {
			if err := s.store.SetMemoryCluster(ctx, memID, newID); err != nil {
				return nil, SplitClusterOutput{}, fmt.Errorf("reassign %s to new cluster: %w", memID, err)
			}
		}

		// Recompute the new cluster's centroid from its fresh membership.
		if err := memory.RecomputeCentroid(ctx, s.store, newID); err != nil && err != memory.ErrEmptyCluster {
			return nil, SplitClusterOutput{}, fmt.Errorf("recompute centroid for new cluster %d: %w", i, err)
		}

		newClusterIDs[i] = newID
	}

	// Compute remaining members of the source (membership minus union of groups).
	remaining := len(membership) - len(union)

	// Handle the source cluster: delete if fully partitioned, otherwise
	// recompute its centroid with the survivors.
	sourceDeleted := false
	if remaining == 0 {
		if err := s.store.DeleteCluster(ctx, in.ClusterID); err != nil {
			return nil, SplitClusterOutput{}, fmt.Errorf("delete source cluster: %w", err)
		}
		sourceDeleted = true
	} else {
		if err := memory.RecomputeCentroid(ctx, s.store, in.ClusterID); err != nil && err != memory.ErrEmptyCluster {
			return nil, SplitClusterOutput{}, fmt.Errorf("recompute source centroid: %w", err)
		}
	}

	// Treat each new cluster as accessed — reset turns_since=0. Matches the
	// pattern used by handleReassignCluster and handleWrite.
	if err := s.mgr.TickDecay(ctx, newClusterIDs); err != nil {
		s.logger.Warn("memory_split_cluster: tick decay failed", "err", err)
	}

	s.logger.Info("memory_split_cluster",
		"source_cluster_id", in.ClusterID,
		"groups", len(in.Groups),
		"new_cluster_ids", newClusterIDs,
		"source_deleted", sourceDeleted,
		"remaining_in_source", remaining,
	)

	return nil, SplitClusterOutput{
		SourceClusterID:   in.ClusterID,
		NewClusterIDs:     newClusterIDs,
		SourceDeleted:     sourceDeleted,
		RemainingInSource: remaining,
	}, nil
}

// collectClusterMembership returns the set of all member IDs (facts and
// episodes) of a cluster. Pages through ListFactsByCluster /
// ListEpisodesByCluster to stay bounded in memory on large clusters.
func (s *Server) collectClusterMembership(ctx context.Context, clusterID string) (map[string]struct{}, error) {
	const pageSize = 200
	members := make(map[string]struct{})

	offset := 0
	for {
		facts, err := s.store.ListFactsByCluster(ctx, clusterID, pageSize, offset)
		if err != nil {
			return nil, fmt.Errorf("list facts by cluster: %w", err)
		}
		if len(facts) == 0 {
			break
		}
		for i := range facts {
			members[facts[i].ID] = struct{}{}
		}
		if len(facts) < pageSize {
			break
		}
		offset += len(facts)
	}

	offset = 0
	for {
		episodes, err := s.store.ListEpisodesByCluster(ctx, clusterID, pageSize, offset)
		if err != nil {
			return nil, fmt.Errorf("list episodes by cluster: %w", err)
		}
		if len(episodes) == 0 {
			break
		}
		for i := range episodes {
			members[episodes[i].ID] = struct{}{}
		}
		if len(episodes) < pageSize {
			break
		}
		offset += len(episodes)
	}

	return members, nil
}

// --- memory_get ---

// GetInput is the input schema for the memory_get tool.
type GetInput struct {
	ID string `json:"id" jsonschema:"memory ID (fact or episode)"`
}

// LinkRef is a cross-type link reference returned by memory_get.
type LinkRef struct {
	ID       string `json:"id"`
	Layer    string `json:"layer"`
	LinkType string `json:"link_type"`
}

// GetOutput is the output schema for the memory_get tool. It is the full
// single-record view: basic fields, fact-only supersede chain, episode-only
// structured fields, and cross-type links.
type GetOutput struct {
	ID             string   `json:"id"`
	Layer          string   `json:"layer"` // "l2_semantic" or "l3_episodic"
	Content        string   `json:"content"`
	Subtype        string   `json:"subtype,omitempty"`
	Source         string   `json:"source,omitempty"`
	Confidence     float64  `json:"confidence,omitempty"`
	Tags           []string `json:"tags"`
	ClusterID      string   `json:"cluster_id"`
	ClusterSummary string   `json:"cluster_summary,omitempty"`
	CreatedAt      string   `json:"created_at"`
	AccessedAt     string   `json:"accessed_at"`
	// Fact-only fields.
	ValidFrom    string   `json:"valid_from,omitempty"`
	SupersededBy *string  `json:"superseded_by,omitempty"`
	Supersedes   []string `json:"supersedes,omitempty"`
	// Cross-type links.
	Links []LinkRef `json:"links,omitempty"`
	// Episode-only fields.
	Situation  string `json:"situation,omitempty"`
	Action     string `json:"action,omitempty"`
	Outcome    string `json:"outcome,omitempty"`
	Preemptive string `json:"preemptive,omitempty"`
}

func (s *Server) handleGet(ctx context.Context, _ *mcpsdk.CallToolRequest, in GetInput) (*mcpsdk.CallToolResult, GetOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, GetOutput{}, nil
	}

	if in.ID == "" {
		return nil, GetOutput{}, fmt.Errorf("id is required")
	}

	// Try fact first. Note: GetFact returns the row regardless of superseded
	// state, so this is the history-view that the spec requires.
	fact, err := s.store.GetFact(ctx, in.ID)
	if err != nil {
		return nil, GetOutput{}, fmt.Errorf("get fact: %w", err)
	}
	if fact != nil {
		return s.buildGetOutputFact(ctx, fact)
	}

	ep, err := s.store.GetEpisode(ctx, in.ID)
	if err != nil {
		return nil, GetOutput{}, fmt.Errorf("get episode: %w", err)
	}
	if ep != nil {
		return s.buildGetOutputEpisode(ctx, ep)
	}

	return nil, GetOutput{}, fmt.Errorf("memory not found: %s", in.ID)
}

// buildGetOutputFact assembles the full GetOutput for an L2 fact, including
// supersede chain and linked episodes.
func (s *Server) buildGetOutputFact(ctx context.Context, f *memory.Fact) (*mcpsdk.CallToolResult, GetOutput, error) {
	out := GetOutput{
		ID:           f.ID,
		Layer:        string(memory.TypeL2Semantic),
		Content:      f.Content,
		Subtype:      f.Subtype,
		Source:       f.Source,
		Confidence:   f.Confidence,
		Tags:         normalizeTagsSlice(f.Tags),
		ClusterID:    f.ClusterID,
		CreatedAt:    f.CreatedAt.Format("2006-01-02T15:04:05Z"),
		AccessedAt:   f.AccessedAt.Format("2006-01-02T15:04:05Z"),
		ValidFrom:    f.ValidFrom.Format("2006-01-02T15:04:05Z"),
		SupersededBy: f.SupersededBy,
	}

	s.populateClusterSummary(ctx, f.ClusterID, &out)

	supersedes, err := s.store.GetFactSupersedes(ctx, f.ID)
	if err != nil {
		return nil, GetOutput{}, fmt.Errorf("get fact supersedes: %w", err)
	}
	if len(supersedes) > 0 {
		out.Supersedes = supersedes
	}

	// Cross-type links: episodes linked to this fact.
	links, err := s.store.GetFactLinks(ctx, f.ID)
	if err != nil {
		return nil, GetOutput{}, fmt.Errorf("get fact links: %w", err)
	}
	for _, l := range links {
		out.Links = append(out.Links, LinkRef{
			ID:       l.EpisodeID,
			Layer:    string(memory.TypeL3Episodic),
			LinkType: l.LinkType,
		})
	}

	s.logger.Info("memory_get", "id", f.ID, "layer", "l2_semantic")
	return nil, out, nil
}

// buildGetOutputEpisode assembles the full GetOutput for an L3 episode,
// including linked facts.
func (s *Server) buildGetOutputEpisode(ctx context.Context, ep *memory.Episode) (*mcpsdk.CallToolResult, GetOutput, error) {
	// Render Content the same way memory_list / memory_recall does for
	// episodes: reuse Candidate.Content() which joins situation/action/
	// outcome/preemptive with newlines and "Label: " prefixes.
	content := memory.Candidate{Episode: ep}.Content()

	out := GetOutput{
		ID:         ep.ID,
		Layer:      string(memory.TypeL3Episodic),
		Content:    content,
		Tags:       normalizeTagsSlice(ep.Tags),
		ClusterID:  ep.ClusterID,
		CreatedAt:  ep.CreatedAt.Format("2006-01-02T15:04:05Z"),
		AccessedAt: ep.AccessedAt.Format("2006-01-02T15:04:05Z"),
		Situation:  ep.Situation,
		Action:     ep.Action,
		Outcome:    ep.Outcome,
		Preemptive: ep.Preemptive,
	}

	s.populateClusterSummary(ctx, ep.ClusterID, &out)

	// Cross-type links: facts linked to this episode.
	links, err := s.store.GetEpisodeLinks(ctx, ep.ID)
	if err != nil {
		return nil, GetOutput{}, fmt.Errorf("get episode links: %w", err)
	}
	for _, l := range links {
		out.Links = append(out.Links, LinkRef{
			ID:       l.FactID,
			Layer:    string(memory.TypeL2Semantic),
			LinkType: l.LinkType,
		})
	}

	s.logger.Info("memory_get", "id", ep.ID, "layer", "l3_episodic")
	return nil, out, nil
}

// populateClusterSummary looks up the cluster and fills ClusterSummary if the
// cluster exists and has a summary. A missing cluster is logged but not fatal
// because memory_get is a read-only inspection tool.
func (s *Server) populateClusterSummary(ctx context.Context, clusterID string, out *GetOutput) {
	if clusterID == "" {
		return
	}
	cl, err := s.store.GetCluster(ctx, clusterID)
	if err != nil {
		s.logger.Warn("memory_get: get cluster failed", "cluster_id", clusterID, "err", err)
		return
	}
	if cl != nil {
		out.ClusterSummary = cl.Summary
	}
}
