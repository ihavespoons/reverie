package mcpserver

import (
	"context"
	"crypto/sha256"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/diffsec/reverie/internal/memory"
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
	Query     string   `json:"query" jsonschema:"Natural-language query to search memories"`
	Limit     int      `json:"limit,omitempty" jsonschema:"Maximum number of results (default 10)"`
	Hints     []string `json:"hints,omitempty" jsonschema:"Optional hint strings to augment the query"`
	Round     int      `json:"round,omitempty" jsonschema:"Recall round: 0 for OR-logic (default); 1+ for AND-logic refinement"`
	ClusterID string   `json:"cluster_id,omitempty" jsonschema:"restrict to members of this cluster"`
	Subtype   string   `json:"subtype,omitempty" jsonschema:"restrict to this fact subtype"`
	Layer     string   `json:"layer,omitempty" jsonschema:"l2, l3, or both (default)"`
	TagsAny   []string `json:"tags_any,omitempty" jsonschema:"restrict to memories with any of these tags"`
	SessionID string   `json:"session_id,omitempty" jsonschema:"attach this call to a session; buffer updates on recall/write/reinforce"`

	// ExpandViaGraph: when true and round==0, recall additionally walks
	// memory_edges and entity_mentions from the vector seeds to surface
	// graph neighbors. Default false. Ignored on round>=1.
	ExpandViaGraph bool `json:"expand_via_graph,omitempty" jsonschema:"opt-in graph expansion on top of vector recall"`

	// GraphHops: BFS depth budget for graph expansion. Clamped to [1,3].
	// Defaults to 2 when ExpandViaGraph is true. Memory->entity->memory
	// counts as 2 hops.
	GraphHops int `json:"graph_hops,omitempty" jsonschema:"BFS depth for graph expansion (1..3, default 2)"`
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

	// Distance: 0 for vector hits, >=1 for graph-expanded neighbors.
	Distance int `json:"distance"`

	// CompositeScore: the final ranking score. For vector hits this equals
	// Similarity. For graph hits it is seed_similarity * neighbor_retention
	// * (decay_per_hop ^ Distance), taking the MAX across seeds when the
	// neighbor is reachable from multiple seeds.
	CompositeScore float64 `json:"composite_score"`
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

	// Normalize layer filter up front so invalid inputs fail fast before search.
	layerFilter, err := normalizeLayerFilter(in.Layer)
	if err != nil {
		return nil, RecallOutput{}, err
	}

	// Global cosine search.
	vectorCandidates, err := s.store.GlobalSearch(ctx, queryVec, limit)
	if err != nil {
		return nil, RecallOutput{}, fmt.Errorf("global search: %w", err)
	}

	// Capture seed similarities BEFORE filtering so a filtered-out vector hit
	// can still seed graph expansion (per the design doc).
	seedSimilarity := make(map[string]float64, len(vectorCandidates))
	for _, c := range vectorCandidates {
		seedSimilarity[c.ID()] = float64(c.Similarity)
	}

	// Apply optional filters post-cosine, pre-ranking. Empty filters pass through.
	filteredCandidates := vectorCandidates
	if in.ClusterID != "" || in.Subtype != "" || layerFilter != layerBoth || len(in.TagsAny) > 0 {
		filtered := vectorCandidates[:0]
		for _, c := range vectorCandidates {
			if passesRecallFilters(c, in.ClusterID, in.Subtype, layerFilter, in.TagsAny) {
				filtered = append(filtered, c)
			}
		}
		filteredCandidates = filtered
	}

	threshold := s.cfg.Memory.SimilarityThreshold
	if threshold <= 0 {
		threshold = 0.70
	}

	// Decide whether graph expansion is honored: only on round 0.
	doGraphExpansion := in.ExpandViaGraph && in.Round == 0
	if in.ExpandViaGraph && in.Round >= 1 {
		s.logger.Info("memory_recall: expand_via_graph ignored on round>=1", "round", in.Round)
	}

	// retentionCache and gateCCache eliminate redundant cluster lookups
	// across the vector and graph populate paths.
	retentionCache := make(map[string]float64)
	gateCCache := make(map[string]bool)
	clusterRetention := func(clusterID string) (float64, bool) {
		if r, ok := retentionCache[clusterID]; ok {
			return r, gateCCache[clusterID]
		}
		cl, gerr := s.store.GetCluster(ctx, clusterID)
		if gerr != nil {
			s.logger.Warn("memory_recall: get cluster failed", "cluster_id", clusterID, "err", gerr)
			retentionCache[clusterID] = 0
			gateCCache[clusterID] = false
			return 0, false
		}
		if cl == nil {
			retentionCache[clusterID] = 0
			gateCCache[clusterID] = false
			return 0, false
		}
		r := s.decayer.Retention(*cl)
		g := s.decayer.GateC(*cl)
		retentionCache[clusterID] = r
		gateCCache[clusterID] = g
		return r, g
	}

	// Build vector candidate map first.
	byID := make(map[string]RecallCandidate)
	// retentionByID tracks the per-candidate retention used downstream
	// (session-buffer scoring needs this aligned with the final output).
	retentionByID := make(map[string]float64)
	// origCandidateByID maps id -> original Candidate (used by the
	// session-buffer update so we can call recallBufferEntry).
	origCandidateByID := make(map[string]memory.Candidate)

	buildCandidateFromMemory := func(c memory.Candidate, similarity float32, distance int, composite float64, gateBOverride *bool) RecallCandidate {
		clusterID := c.ClusterID()
		retention, gateCPass := clusterRetention(clusterID)

		gateBPass := float64(similarity) > threshold
		if gateBOverride != nil {
			gateBPass = *gateBOverride
		}

		rc := RecallCandidate{
			ID:             c.ID(),
			Content:        c.Content(),
			Layer:          string(c.Layer()),
			Similarity:     similarity,
			Retention:      retention,
			GateBPass:      gateBPass,
			GateCPass:      gateCPass,
			ClusterID:      clusterID,
			Distance:       distance,
			CompositeScore: composite,
		}

		if c.Fact != nil {
			rc.Subtype = c.Fact.Subtype
			rc.Tags = normalizeTagsSlice(c.Fact.Tags)
		} else if c.Episode != nil {
			rc.Tags = normalizeTagsSlice(c.Episode.Tags)
		} else {
			rc.Tags = normalizeTagsSlice(nil)
		}

		// Cross-type linked IDs (edge_type='evidence').
		var ownerID string
		if c.Fact != nil {
			ownerID = c.Fact.ID
		} else if c.Episode != nil {
			ownerID = c.Episode.ID
		}
		if ownerID != "" {
			edges, linkErr := s.store.ListEdges(ctx, ownerID, 1)
			if linkErr != nil {
				s.logger.Warn("memory_recall: list edges failed", "memory_id", ownerID, "err", linkErr)
			}
			for _, e := range edges {
				if e.Edge.EdgeType != "evidence" {
					continue
				}
				other := e.Edge.DstID
				if e.Edge.SrcID != ownerID {
					other = e.Edge.SrcID
				}
				rc.LinkedIDs = append(rc.LinkedIDs, other)
			}
		}

		retentionByID[rc.ID] = retention
		origCandidateByID[rc.ID] = c
		return rc
	}

	for _, c := range filteredCandidates {
		rc := buildCandidateFromMemory(c, c.Similarity, 0, float64(c.Similarity), nil)
		byID[rc.ID] = rc
	}

	// Graph expansion (round 0 only).
	if doGraphExpansion {
		hops := in.GraphHops
		if hops <= 0 {
			hops = 2
		}
		if hops < 1 {
			hops = 1
		}
		if hops > 3 {
			hops = 3
		}

		decayPerHop := s.cfg.Memory.GraphDecayPerHop
		if decayPerHop <= 0 {
			decayPerHop = 0.8
		}
		maxVisited := s.cfg.Memory.GraphMaxVisited
		if maxVisited == 0 {
			maxVisited = 2000
		}
		minRet := s.cfg.Memory.GraphMinRetentionForExpansion
		if minRet < 0 {
			minRet = 0.05
		}
		// Note: minRet==0 is treated as "no filter" by the store.

		seedIDs := make([]string, 0, len(seedSimilarity))
		for id := range seedSimilarity {
			seedIDs = append(seedIDs, id)
		}

		if len(seedIDs) > 0 {
			hits, gerr := s.store.ExpandViaGraph(ctx, seedIDs, hops, minRet, maxVisited)
			if gerr != nil {
				return nil, RecallOutput{}, fmt.Errorf("graph expansion: %w", gerr)
			}

			falseVal := false
			// Aggregate hits by neighbor id, keeping the MAX composite
			// across seeds.
			type bestHit struct {
				distance  int
				composite float64
				layer     string
			}
			bestByNeighbor := make(map[string]bestHit)
			for _, h := range hits {
				seedSim := seedSimilarity[h.SeedID]
				// Fetch neighbor (fact or episode) for retention lookup.
				var neighborClusterID string
				switch h.NeighborLayer {
				case string(memory.TypeL2Semantic):
					f, fErr := s.store.GetFact(ctx, h.NeighborID)
					if fErr != nil || f == nil {
						continue
					}
					neighborClusterID = f.ClusterID
				case string(memory.TypeL3Episodic):
					ep, eErr := s.store.GetEpisode(ctx, h.NeighborID)
					if eErr != nil || ep == nil {
						continue
					}
					neighborClusterID = ep.ClusterID
				default:
					continue
				}
				neighborRetention, _ := clusterRetention(neighborClusterID)
				composite := seedSim * neighborRetention * math.Pow(decayPerHop, float64(h.Distance))

				bh, exists := bestByNeighbor[h.NeighborID]
				if !exists || composite > bh.composite {
					bestByNeighbor[h.NeighborID] = bestHit{
						distance:  h.Distance,
						composite: composite,
						layer:     h.NeighborLayer,
					}
				}
			}

			for nID, bh := range bestByNeighbor {
				if existing, ok := byID[nID]; ok {
					// Collision with vector hit: keep higher composite.
					if bh.composite > existing.CompositeScore {
						// Graph wins: rebuild with graph provenance.
						c, cerr := loadCandidateByID(ctx, s.store, nID, bh.layer)
						if cerr != nil || c == nil {
							s.logger.Warn("memory_recall: load graph candidate failed", "id", nID, "err", cerr)
							continue
						}
						// Apply optional filters uniformly.
						if !passesRecallFilters(*c, in.ClusterID, in.Subtype, layerFilter, in.TagsAny) {
							delete(byID, nID)
							continue
						}
						rc := buildCandidateFromMemory(*c, 0, bh.distance, bh.composite, &falseVal)
						byID[nID] = rc
					}
					continue
				}
				// New (graph-only) entry.
				c, cerr := loadCandidateByID(ctx, s.store, nID, bh.layer)
				if cerr != nil || c == nil {
					s.logger.Warn("memory_recall: load graph candidate failed", "id", nID, "err", cerr)
					continue
				}
				// Apply optional filters uniformly to graph-only candidates.
				if !passesRecallFilters(*c, in.ClusterID, in.Subtype, layerFilter, in.TagsAny) {
					continue
				}
				rc := buildCandidateFromMemory(*c, 0, bh.distance, bh.composite, &falseVal)
				byID[nID] = rc
			}
		}
	}

	// Flatten + sort by composite score descending.
	merged := make([]RecallCandidate, 0, len(byID))
	for _, rc := range byID {
		merged = append(merged, rc)
	}
	sort.Slice(merged, func(i, j int) bool {
		if merged[i].CompositeScore != merged[j].CompositeScore {
			return merged[i].CompositeScore > merged[j].CompositeScore
		}
		// Deterministic tiebreak by ID.
		return merged[i].ID < merged[j].ID
	})
	if len(merged) > limit {
		merged = merged[:limit]
	}

	out := RecallOutput{
		RecallID:   uuid.New().String(),
		Round:      in.Round,
		Candidates: merged,
	}

	// Stash in recall cache for memory_apply_judgment. Only vector
	// candidates (with similarity) are cached -- graph hits have no
	// query-cosine and apply_judgment expects similarity-bearing rows.
	s.recallCache.put(out.RecallID, &cachedRecall{
		queryVec:   queryVec,
		candidates: filteredCandidates,
		round:      in.Round,
		createdAt:  time.Now(),
	})

	// Session buffer update: append each returned candidate with the
	// composite score, mirroring the pre-7C behavior (vector path) and
	// extending it to graph hits where the underlying Candidate is known.
	budgetMax := s.bufferBudgetMax()
	if err := s.applySessionMutation(ctx, in.SessionID, func(wm *memory.WorkingMemory) {
		for _, rc := range merged {
			origC, ok := origCandidateByID[rc.ID]
			if !ok {
				continue
			}
			// For graph-only hits, similarity is 0; recallBufferEntry
			// uses Candidate.Similarity which equals rc.Similarity here
			// (set on the vector path). For graph hits Similarity=0
			// would zero the buffer score; use composite instead.
			c := origC
			if rc.Distance > 0 {
				// Substitute composite for buffer scoring purposes.
				// Build a synthetic Candidate by copying and overriding similarity.
				synth := origC
				synth.Similarity = float32(rc.CompositeScore)
				c = synth
			}
			memory.AppendToBuffer(wm, recallBufferEntry(c, retentionByID[rc.ID]), budgetMax)
		}
	}); err != nil {
		return nil, RecallOutput{}, err
	}

	s.logger.Info("memory_recall", "recall_id", out.RecallID, "candidates", len(out.Candidates), "round", in.Round, "session_id", in.SessionID)
	return nil, out, nil
}

// loadCandidateByID looks up a memory ID in the layer indicated and
// returns a Candidate wrapping it (Similarity=0 since this is a
// non-vector path). Returns (nil, nil) if not found.
func loadCandidateByID(ctx context.Context, store memory.Store, id, layer string) (*memory.Candidate, error) {
	switch layer {
	case string(memory.TypeL2Semantic):
		f, err := store.GetFact(ctx, id)
		if err != nil {
			return nil, err
		}
		if f == nil {
			return nil, nil
		}
		return &memory.Candidate{Fact: f}, nil
	case string(memory.TypeL3Episodic):
		ep, err := store.GetEpisode(ctx, id)
		if err != nil {
			return nil, err
		}
		if ep == nil {
			return nil, nil
		}
		return &memory.Candidate{Episode: ep}, nil
	}
	return nil, nil
}

// layerFilter represents the normalized Recall layer filter.
type layerFilter int

const (
	layerBoth layerFilter = iota
	layerL2Only
	layerL3Only
)

// normalizeLayerFilter maps the raw input string to a layerFilter value.
// Accepts "", "both" (pass-through); "l2", "L2", "l2_semantic" (L2-only);
// "l3", "L3", "l3_episodic" (L3-only). Any other value is an error.
func normalizeLayerFilter(in string) (layerFilter, error) {
	switch strings.ToLower(in) {
	case "", "both":
		return layerBoth, nil
	case "l2", "l2_semantic":
		return layerL2Only, nil
	case "l3", "l3_episodic":
		return layerL3Only, nil
	default:
		return layerBoth, fmt.Errorf("invalid layer %q: must be one of l2, l3, or both", in)
	}
}

// passesRecallFilters returns true if the candidate satisfies every active
// filter. Empty filter fields are pass-through. Semantics per the 2E spec:
//   - cluster_id: candidate's cluster_id must match.
//   - subtype: fact subtype must match; episodes always fail this filter.
//   - layer: L2-only excludes episodes; L3-only excludes facts.
//   - tags_any: union — at least one of the candidate's tags must appear in
//     the filter set (empty filter set is pass-through).
func passesRecallFilters(c memory.Candidate, clusterID, subtype string, layer layerFilter, tagsAny []string) bool {
	if clusterID != "" && c.ClusterID() != clusterID {
		return false
	}
	if subtype != "" {
		if c.Fact == nil || c.Fact.Subtype != subtype {
			return false
		}
	}
	switch layer {
	case layerL2Only:
		if c.Fact == nil {
			return false
		}
	case layerL3Only:
		if c.Episode == nil {
			return false
		}
	}
	if len(tagsAny) > 0 {
		var candTags []string
		switch {
		case c.Fact != nil:
			candTags = c.Fact.Tags
		case c.Episode != nil:
			candTags = c.Episode.Tags
		}
		if !tagsOverlap(candTags, tagsAny) {
			return false
		}
	}
	return true
}

// tagsOverlap returns true if a and b share at least one tag.
func tagsOverlap(a, b []string) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	set := make(map[string]struct{}, len(a))
	for _, t := range a {
		set[t] = struct{}{}
	}
	for _, t := range b {
		if _, ok := set[t]; ok {
			return true
		}
	}
	return false
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
	Content    string          `json:"content,omitempty" jsonschema:"The content of the memory (required for L2 facts)"`
	Type       string          `json:"type" jsonschema:"Memory subtype: user, feedback, project, or reference"`
	Tags       []string        `json:"tags,omitempty" jsonschema:"Optional tags for categorization"`
	Source     string          `json:"source,omitempty" jsonschema:"Source attribution (default: inferred)"`
	Confidence *float64        `json:"confidence,omitempty" jsonschema:"Confidence in [0,1]; defaults to 1.0. Facts only — rejected for episodes."`
	Episode    *EpisodePayload `json:"episode,omitempty" jsonschema:"if set, writes an L3 episode instead of an L2 fact"`
	DryRun     bool            `json:"dry_run,omitempty" jsonschema:"if true, preview cluster assignment and supersede candidate without writing"`
	SessionID  string          `json:"session_id,omitempty" jsonschema:"attach this call to a session; buffer updates on recall/write/reinforce"`
}

// WriteOutput is the output schema for the memory_write tool.
type WriteOutput struct {
	ID      string        `json:"id,omitempty"`
	Layer   string        `json:"layer"`
	DryRun  bool          `json:"dry_run"`
	Preview *WritePreview `json:"preview,omitempty"`
}

// WritePreview describes what a memory_write would do when DryRun=true. It is
// populated only on dry-run responses; committed writes leave Preview nil.
type WritePreview struct {
	ProposedClusterID    string              `json:"proposed_cluster_id"`
	ProposedClusterIsNew bool                `json:"proposed_cluster_is_new"`
	ProposedSupersedes   *SupersedeCandidate `json:"proposed_supersedes,omitempty"`
	ContentHash          string              `json:"content_hash"`
}

// SupersedeCandidate describes the existing fact that a proposed write would
// supersede. Populated only for fact dry-runs when a near-duplicate exists
// above the configured conflict threshold.
type SupersedeCandidate struct {
	ID         string  `json:"id"`
	Content    string  `json:"content"`
	Similarity float32 `json:"similarity"`
	Subtype    string  `json:"subtype"`
	CreatedAt  string  `json:"created_at"`
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

	// Confidence is a fact-only field. Reject it on episode writes rather than
	// silently dropping (same bug-avoidance policy as 1D tags wiring).
	if in.Confidence != nil {
		if hasEpisode {
			return nil, WriteOutput{}, fmt.Errorf("confidence is not supported for episode writes")
		}
		if *in.Confidence < 0.0 || *in.Confidence > 1.0 {
			return nil, WriteOutput{}, fmt.Errorf("confidence %v out of range [0,1]", *in.Confidence)
		}
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

	// Dry-run short-circuit: episodes have no supersede detection. Populate
	// preview with cluster assignment and content hash; skip insert/tick/
	// centroid side effects.
	if in.DryRun {
		s.logger.Info("memory_write", "dry_run", true, "layer", "l3_episodic", "cluster_id", clusterID, "is_new_cluster", isNew)
		return nil, WriteOutput{
			Layer:  string(memory.TypeL3Episodic),
			DryRun: true,
			Preview: &WritePreview{
				ProposedClusterID:    clusterID,
				ProposedClusterIsNew: isNew,
				ContentHash:          contentHash,
			},
		}, nil
	}

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

	// Session buffer update: append the new memory ID with score 1.0.
	if err := s.appendWriteToSessionBuffer(ctx, in.SessionID, id, memory.TypeL3Episodic, embedText); err != nil {
		return nil, WriteOutput{}, err
	}

	s.logger.Info("memory_write", "id", id, "layer", "l3_episodic", "cluster_id", clusterID, "is_new_cluster", isNew, "session_id", in.SessionID)
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

	confidence := 1.0
	if in.Confidence != nil {
		confidence = *in.Confidence
	}

	fact := memory.Fact{
		Content:     in.Content,
		ContentHash: contentHash,
		ClusterID:   clusterID,
		Subtype:     in.Type,
		Source:      source,
		Embedding:   vec,
		Confidence:  confidence,
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

	// Dry-run short-circuit: return the cluster assignment and any conflict
	// candidate without inserting the fact, updating centroid, or ticking
	// decay. Embedding-cache side effects (via CachedProvider) are accepted
	// per spec — they mirror recall behavior and don't touch the store.
	if in.DryRun {
		preview := &WritePreview{
			ProposedClusterID:    clusterID,
			ProposedClusterIsNew: isNew,
			ContentHash:          contentHash,
		}
		if len(similar) > 0 && similar[0].Fact != nil {
			oldFact := similar[0].Fact
			preview.ProposedSupersedes = &SupersedeCandidate{
				ID:         oldFact.ID,
				Content:    oldFact.Content,
				Similarity: similar[0].Similarity,
				Subtype:    oldFact.Subtype,
				CreatedAt:  oldFact.CreatedAt.UTC().Format(time.RFC3339),
			}
		}
		s.logger.Info("memory_write", "dry_run", true, "subtype", in.Type, "content_len", len(in.Content), "cluster_id", clusterID, "is_new_cluster", isNew, "has_supersede_candidate", preview.ProposedSupersedes != nil)
		return nil, WriteOutput{
			Layer:   string(memory.TypeL2Semantic),
			DryRun:  true,
			Preview: preview,
		}, nil
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

	// Session buffer update: append the new memory ID with score 1.0.
	if err := s.appendWriteToSessionBuffer(ctx, in.SessionID, id, memory.TypeL2Semantic, in.Content); err != nil {
		return nil, WriteOutput{}, err
	}

	s.logger.Info("memory_write", "id", id, "subtype", in.Type, "content_len", len(in.Content), "cluster_id", clusterID, "is_new_cluster", isNew, "session_id", in.SessionID)
	return nil, WriteOutput{ID: id, Layer: string(memory.TypeL2Semantic)}, nil
}

// --- memory_reinforce ---

// ReinforceInput is the input schema for the memory_reinforce tool.
type ReinforceInput struct {
	MemoryIDs []string `json:"memory_ids" jsonschema:"IDs of memories to reinforce"`
	SessionID string   `json:"session_id,omitempty" jsonschema:"attach this call to a session; buffer updates on recall/write/reinforce"`
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

	// Session buffer update: bump the score of any buffer entries whose ID
	// was reinforced. We use a simple capped +0.1 bump (see
	// reinforceNewScore); re-recalling the underlying retention here would
	// require re-fetching each memory + its cluster, which is out of scope
	// for a reinforce call.
	if err := s.applySessionMutation(ctx, in.SessionID, func(wm *memory.WorkingMemory) {
		if len(wm.Buffer) == 0 {
			return
		}
		updates := make(map[string]float64, len(in.MemoryIDs))
		reinforceSet := make(map[string]struct{}, len(in.MemoryIDs))
		for _, id := range in.MemoryIDs {
			reinforceSet[id] = struct{}{}
		}
		for _, ref := range wm.Buffer {
			if _, ok := reinforceSet[ref.ID]; ok {
				updates[ref.ID] = reinforceNewScore(ref.Score)
			}
		}
		memory.RescoreBuffer(wm, updates)
	}); err != nil {
		return nil, ReinforceOutput{}, err
	}

	s.logger.Info("memory_reinforce", "count", len(in.MemoryIDs), "session_id", in.SessionID)
	return nil, ReinforceOutput{Reinforced: len(in.MemoryIDs)}, nil
}

// --- memory_forget ---

// ForgetInput is the input schema for the memory_forget tool.
type ForgetInput struct {
	ID    string   `json:"id,omitempty" jsonschema:"ID of a specific memory to delete"`
	IDs   []string `json:"ids,omitempty" jsonschema:"Batch delete multiple memories by ID"`
	Query string   `json:"query,omitempty" jsonschema:"Query to find candidates for deletion (returns candidates without deleting)"`
}

// ForgetCandidate is a candidate returned for confirmation when query is used.
type ForgetCandidate struct {
	ID      string `json:"id"`
	Content string `json:"content"`
	Layer   string `json:"layer"`
}

// ForgetFailure describes a per-ID failure during a batch forget.
type ForgetFailure struct {
	ID     string `json:"id"`
	Reason string `json:"reason"`
}

// ForgetOutput is the output schema for the memory_forget tool.
type ForgetOutput struct {
	Deleted    int               `json:"deleted"`
	Failed     []ForgetFailure   `json:"failed,omitempty"`
	Candidates []ForgetCandidate `json:"candidates,omitempty"`
}

// forgetOne attempts to delete a single memory by ID, fact-then-episode.
// Returns (true, "") on success; (false, reason) on not-found or store error.
func (s *Server) forgetOne(ctx context.Context, id string) (bool, string) {
	if id == "" {
		return false, "empty id"
	}

	fact, err := s.store.GetFact(ctx, id)
	if err != nil {
		return false, fmt.Sprintf("get fact: %v", err)
	}
	if fact != nil {
		if delErr := s.store.DeleteFact(ctx, id); delErr != nil {
			return false, fmt.Sprintf("delete fact: %v", delErr)
		}
		s.logger.Info("memory_forget", "deleted_id", id, "layer", "l2_semantic")
		return true, ""
	}

	ep, err := s.store.GetEpisode(ctx, id)
	if err != nil {
		return false, fmt.Sprintf("get episode: %v", err)
	}
	if ep != nil {
		if delErr := s.store.DeleteEpisode(ctx, id); delErr != nil {
			return false, fmt.Sprintf("delete episode: %v", delErr)
		}
		s.logger.Info("memory_forget", "deleted_id", id, "layer", "l3_episodic")
		return true, ""
	}

	return false, fmt.Sprintf("memory not found: %s", id)
}

func (s *Server) handleForget(ctx context.Context, _ *mcpsdk.CallToolRequest, in ForgetInput) (*mcpsdk.CallToolResult, ForgetOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, ForgetOutput{}, nil
	}

	hasID := in.ID != ""
	hasIDs := in.IDs != nil
	hasQuery := in.Query != ""

	// Exactly one of (ID, IDs, Query) must be set.
	modes := 0
	if hasID {
		modes++
	}
	if hasIDs {
		modes++
	}
	if hasQuery {
		modes++
	}
	if modes != 1 {
		return nil, ForgetOutput{}, fmt.Errorf("provide exactly one of id, ids, query")
	}

	// Non-nil but empty slice is a caller bug — reject.
	if hasIDs && len(in.IDs) == 0 {
		return nil, ForgetOutput{}, fmt.Errorf("ids must not be empty")
	}

	if hasID {
		ok, reason := s.forgetOne(ctx, in.ID)
		if ok {
			return nil, ForgetOutput{Deleted: 1}, nil
		}
		return nil, ForgetOutput{}, fmt.Errorf("%s", reason)
	}

	if hasIDs {
		out := ForgetOutput{}
		for _, id := range in.IDs {
			ok, reason := s.forgetOne(ctx, id)
			if ok {
				out.Deleted++
				continue
			}
			out.Failed = append(out.Failed, ForgetFailure{ID: id, Reason: reason})
		}
		s.logger.Info("memory_forget", "batch_size", len(in.IDs), "deleted", out.Deleted, "failed", len(out.Failed))
		return nil, out, nil
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
	RecallID  string    `json:"recall_id" jsonschema:"the recall_id returned by memory_recall"`
	Verdicts  []Verdict `json:"verdicts" jsonschema:"per-candidate keep/drop verdicts from the memory-judge subagent"`
	SessionID string    `json:"session_id,omitempty" jsonschema:"attach this call to a session; buffer updates on recall/write/reinforce"`
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

	// Session buffer update: replace the buffer entries for this recall
	// with the filtered (kept) set so rejected candidates no longer occupy
	// budget.
	if err := s.applySessionMutation(ctx, in.SessionID, func(wm *memory.WorkingMemory) {
		keepIDs := make([]string, len(memories))
		for i, m := range memories {
			keepIDs[i] = m.ID
		}
		memory.ReplaceBufferFiltered(wm, keepIDs)
	}); err != nil {
		return nil, ApplyJudgmentOutput{}, err
	}

	s.logger.Info("memory_apply_judgment", "recall_id", in.RecallID, "round", cached.round, "logic", logic, "passed", len(memories), "session_id", in.SessionID)
	return nil, ApplyJudgmentOutput{Memories: memories, AppliedLogic: logic}, nil
}

// --- memory_decay_tick ---

// DecayTickInput is the input schema for the memory_decay_tick tool.
//
// Phase 3C: turns_elapsed and session_end were removed because they had no
// effect on behavior. They are retained here solely to detect callers that
// still send the old shape so we can return an explicit error naming the
// removed flags — the MCP SDK's JSON decoder silently drops unknown fields,
// so pure deletion would leave callers with a confusing no-op. The Note
// field is purely an optional audit annotation; it does not influence the
// tick.
type DecayTickInput struct {
	Note string `json:"note,omitempty" jsonschema:"optional log annotation"`

	// Deprecated-rejected: these fields were removed in Phase 3C. Setting
	// either returns an error. Do not read these values elsewhere.
	TurnsElapsed int  `json:"turns_elapsed,omitempty" jsonschema:"removed in Phase 3C; setting returns an error"`
	SessionEnd   bool `json:"session_end,omitempty" jsonschema:"removed in Phase 3C; setting returns an error"`
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

	// Phase 3C: reject old-shape callers explicitly so they discover the
	// breaking change instead of experiencing a silent no-op.
	if in.SessionEnd || in.TurnsElapsed != 0 {
		return nil, DecayTickOutput{}, fmt.Errorf("memory_decay_tick: the session_end and turns_elapsed fields were removed in Phase 3C; call memory_decay_tick without arguments")
	}

	// TickDecay with nil accessedIDs bumps all clusters with none reset.
	if err := s.mgr.TickDecay(ctx, nil); err != nil {
		return nil, DecayTickOutput{}, fmt.Errorf("tick decay: %w", err)
	}

	if in.Note != "" {
		s.logger.Info("memory_decay_tick", "note", in.Note)
	} else {
		s.logger.Info("memory_decay_tick")
	}
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

// --- memory_merge_clusters ---

// MergeClustersInput is the input schema for the memory_merge_clusters tool.
type MergeClustersInput struct {
	SourceClusterIDs []string `json:"source_cluster_ids" jsonschema:"clusters to merge and delete (must be non-empty)"`
	TargetClusterID  string   `json:"target_cluster_id" jsonschema:"destination cluster (must exist, cannot be in source list)"`
}

// MergeClustersOutput is the output schema for the memory_merge_clusters tool.
type MergeClustersOutput struct {
	TargetClusterID string   `json:"target_cluster_id"`
	MergedCount     int      `json:"merged_count"`
	DeletedClusters []string `json:"deleted_clusters"`
	NewItemCount    int      `json:"new_item_count"`
}

func (s *Server) handleMergeClusters(ctx context.Context, _ *mcpsdk.CallToolRequest, in MergeClustersInput) (*mcpsdk.CallToolResult, MergeClustersOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, MergeClustersOutput{}, nil
	}

	if len(in.SourceClusterIDs) == 0 {
		return nil, MergeClustersOutput{}, fmt.Errorf("source_cluster_ids must be non-empty")
	}
	if in.TargetClusterID == "" {
		return nil, MergeClustersOutput{}, fmt.Errorf("target_cluster_id is required")
	}

	// Validate up-front (pre-mutation): target not in sources, and all IDs
	// exist. This is the atomicity guarantee for the real store: if validation
	// passes, every subsequent op is a bulk UPDATE/DELETE that either fully
	// succeeds or is rolled back inside the store's transaction.
	for _, src := range in.SourceClusterIDs {
		if src == in.TargetClusterID {
			return nil, MergeClustersOutput{}, fmt.Errorf("target cluster %s cannot be in source list", in.TargetClusterID)
		}
	}

	target, err := s.store.GetCluster(ctx, in.TargetClusterID)
	if err != nil {
		return nil, MergeClustersOutput{}, fmt.Errorf("get target cluster: %w", err)
	}
	if target == nil {
		return nil, MergeClustersOutput{}, fmt.Errorf("cluster not found: %s", in.TargetClusterID)
	}
	// Dedup sources while preserving order — repeated IDs would cause the
	// second delete to be a no-op and inflate nothing, but an explicit check
	// keeps the reported DeletedClusters list accurate.
	seen := make(map[string]struct{}, len(in.SourceClusterIDs))
	sources := make([]string, 0, len(in.SourceClusterIDs))
	for _, src := range in.SourceClusterIDs {
		if src == "" {
			return nil, MergeClustersOutput{}, fmt.Errorf("source cluster id cannot be empty")
		}
		if _, dup := seen[src]; dup {
			continue
		}
		seen[src] = struct{}{}

		cl, err := s.store.GetCluster(ctx, src)
		if err != nil {
			return nil, MergeClustersOutput{}, fmt.Errorf("get source cluster %s: %w", src, err)
		}
		if cl == nil {
			return nil, MergeClustersOutput{}, fmt.Errorf("cluster not found: %s", src)
		}
		sources = append(sources, src)
	}

	// Perform moves and deletes. Each MoveAllClusterMembers is atomic inside
	// the store (sqlite wraps both UPDATEs in a transaction). With the
	// pre-validation above, no source can disappear between validation and
	// move, so the loop cannot observe a partial-failure state.
	totalMoved := 0
	deleted := make([]string, 0, len(sources))
	for _, src := range sources {
		moved, err := s.store.MoveAllClusterMembers(ctx, src, in.TargetClusterID)
		if err != nil {
			return nil, MergeClustersOutput{}, fmt.Errorf("move members from %s: %w", src, err)
		}
		totalMoved += moved

		if err := s.store.DeleteCluster(ctx, src); err != nil {
			return nil, MergeClustersOutput{}, fmt.Errorf("delete source cluster %s: %w", src, err)
		}
		deleted = append(deleted, src)
	}

	// Recompute target centroid AFTER moves land. Centroid drift on an error
	// here is recoverable — membership is correct and a subsequent
	// recompute/tick will heal it.
	if err := memory.RecomputeCentroid(ctx, s.store, in.TargetClusterID); err != nil && err != memory.ErrEmptyCluster {
		return nil, MergeClustersOutput{}, fmt.Errorf("recompute target centroid: %w", err)
	}

	// Reset turns_since on the target — merge is an access, same as reassign.
	if err := s.mgr.TickDecay(ctx, []string{in.TargetClusterID}); err != nil {
		s.logger.Warn("memory_merge_clusters: tick decay failed", "cluster_id", in.TargetClusterID, "err", err)
	}

	// Report the post-merge item count.
	newItemCount := 0
	postTarget, err := s.store.GetCluster(ctx, in.TargetClusterID)
	if err != nil {
		return nil, MergeClustersOutput{}, fmt.Errorf("get target cluster (post): %w", err)
	}
	if postTarget != nil {
		newItemCount = postTarget.ItemCount
	}

	s.logger.Info("memory_merge_clusters",
		"target_cluster_id", in.TargetClusterID,
		"merged_count", totalMoved,
		"deleted_clusters", deleted,
		"new_item_count", newItemCount,
	)

	return nil, MergeClustersOutput{
		TargetClusterID: in.TargetClusterID,
		MergedCount:     totalMoved,
		DeletedClusters: deleted,
		NewItemCount:    newItemCount,
	}, nil
}

// --- memory_update_content ---

// UpdateContentInput is the input schema for the memory_update_content tool.
//
// Tags is a tri-state pointer: nil preserves the current tag set; a non-nil
// pointer replaces it (an explicit empty slice clears tags). This mirrors the
// Phase 2D spec.
type UpdateContentInput struct {
	ID      string          `json:"id" jsonschema:"fact or episode ID"`
	Content string          `json:"content,omitempty" jsonschema:"new content (for facts)"`
	Episode *EpisodePayload `json:"episode,omitempty" jsonschema:"new episode fields (for episodes); omit linked_fact_ids to preserve existing"`
	Tags    *[]string       `json:"tags,omitempty" jsonschema:"replace tags; omit to preserve"`
}

// UpdateContentOutput is the output schema for the memory_update_content tool.
type UpdateContentOutput struct {
	ID         string `json:"id"`
	Layer      string `json:"layer"`
	Reembedded bool   `json:"reembedded"`
}

func (s *Server) handleUpdateContent(ctx context.Context, _ *mcpsdk.CallToolRequest, in UpdateContentInput) (*mcpsdk.CallToolResult, UpdateContentOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, UpdateContentOutput{}, nil
	}

	if in.ID == "" {
		return nil, UpdateContentOutput{}, fmt.Errorf("id is required")
	}

	hasContent := in.Content != ""
	hasEpisode := in.Episode != nil
	hasTags := in.Tags != nil

	// Both content and episode payloads simultaneously is nonsense (layer
	// ambiguity).
	if hasContent && hasEpisode {
		return nil, UpdateContentOutput{}, fmt.Errorf("provide content OR episode, not both")
	}
	// No-op guard: caller supplied nothing to change.
	if !hasContent && !hasEpisode && !hasTags {
		return nil, UpdateContentOutput{}, fmt.Errorf("nothing to update: provide content, episode, or tags")
	}

	// Resolve ID: fact first, then episode. A superseded fact is allowed
	// because update preserves superseded_by (operator may want to fix a typo
	// in history).
	fact, err := s.store.GetFact(ctx, in.ID)
	if err != nil {
		return nil, UpdateContentOutput{}, fmt.Errorf("get fact: %w", err)
	}
	if fact != nil {
		if hasEpisode {
			return nil, UpdateContentOutput{}, fmt.Errorf("layer mismatch: %s is a fact but episode payload was provided", in.ID)
		}
		return s.updateFactContent(ctx, fact, in)
	}

	ep, err := s.store.GetEpisode(ctx, in.ID)
	if err != nil {
		return nil, UpdateContentOutput{}, fmt.Errorf("get episode: %w", err)
	}
	if ep != nil {
		if hasContent {
			return nil, UpdateContentOutput{}, fmt.Errorf("layer mismatch: %s is an episode but content was provided", in.ID)
		}
		return s.updateEpisodeContent(ctx, ep, in)
	}

	return nil, UpdateContentOutput{}, fmt.Errorf("memory not found: %s", in.ID)
}

// updateFactContent handles the L2 fact amendment path. It re-embeds the new
// content, hashes it, and delegates to Store.UpdateFactContent. It does NOT
// reassign cluster and does NOT trigger conflict detection / supersede.
func (s *Server) updateFactContent(ctx context.Context, existing *memory.Fact, in UpdateContentInput) (*mcpsdk.CallToolResult, UpdateContentOutput, error) {
	// When tags are the only change, content is empty — keep the existing content.
	newContent := existing.Content
	if in.Content != "" {
		newContent = in.Content
	}

	vecs, err := s.embedder.Embed(ctx, []string{newContent})
	if err != nil {
		return nil, UpdateContentOutput{}, fmt.Errorf("embed content: %w", err)
	}
	if len(vecs) == 0 || len(vecs[0]) == 0 {
		return nil, UpdateContentOutput{}, fmt.Errorf("embedder returned empty vector")
	}
	vec := vecs[0]

	h := sha256.Sum256([]byte(newContent))
	contentHash := fmt.Sprintf("%x", h)

	if err := s.store.UpdateFactContent(ctx, existing.ID, newContent, contentHash, vec, in.Tags); err != nil {
		return nil, UpdateContentOutput{}, fmt.Errorf("update fact content: %w", err)
	}

	s.logger.Info("memory_update_content",
		"id", existing.ID,
		"layer", "l2_semantic",
		"content_changed", in.Content != "",
		"tags_changed", in.Tags != nil,
	)
	return nil, UpdateContentOutput{
		ID:         existing.ID,
		Layer:      string(memory.TypeL2Semantic),
		Reembedded: true,
	}, nil
}

// updateEpisodeContent handles the L3 episode amendment path. It re-embeds
// the structured fields, hashes them, and delegates to Store.UpdateEpisodeContent.
// Tags preservation is threaded through by copying the existing tag set when
// in.Tags is nil. Linked fact IDs are replaced only when the caller set
// ep.LinkedFactIDs explicitly (nil preserves, non-nil replaces, empty clears).
func (s *Server) updateEpisodeContent(ctx context.Context, existing *memory.Episode, in UpdateContentInput) (*mcpsdk.CallToolResult, UpdateContentOutput, error) {
	epIn := in.Episode

	// Episode fields: caller supplies the full new tuple. We still accept the
	// empty-string convention as "the caller wants empty here"; spec does not
	// carve out partial updates for episodes.
	newSituation := epIn.Situation
	newAction := epIn.Action
	newOutcome := epIn.Outcome
	newPreemptive := epIn.Preemptive

	// Match writeEpisode's join convention so re-embeds are self-consistent.
	embedText := newSituation + "\n" + newAction + "\n" + newOutcome + "\n" + newPreemptive
	vecs, err := s.embedder.Embed(ctx, []string{embedText})
	if err != nil {
		return nil, UpdateContentOutput{}, fmt.Errorf("embed episode: %w", err)
	}
	if len(vecs) == 0 || len(vecs[0]) == 0 {
		return nil, UpdateContentOutput{}, fmt.Errorf("embedder returned empty vector")
	}
	vec := vecs[0]

	h := sha256.Sum256([]byte(embedText))
	contentHash := fmt.Sprintf("%x", h)

	// Tags tri-state: translate from the pointer form to the concrete slice
	// the store expects. When the pointer is nil, we pass the existing tag
	// set through so normalizeTags yields the same rows.
	var tagsForStore []string
	if in.Tags != nil {
		tagsForStore = *in.Tags
	} else {
		tagsForStore = existing.Tags
	}

	update := memory.Episode{
		Situation:   newSituation,
		Action:      newAction,
		Outcome:     newOutcome,
		Preemptive:  newPreemptive,
		Embedding:   vec,
		ContentHash: contentHash,
		Tags:        tagsForStore,
	}

	if err := s.store.UpdateEpisodeContent(ctx, existing.ID, update); err != nil {
		return nil, UpdateContentOutput{}, fmt.Errorf("update episode content: %w", err)
	}

	// Replace links only when the caller provided a non-nil slice. nil =
	// preserve (no mutation), empty slice = clear.
	linksChanged := false
	if epIn.LinkedFactIDs != nil {
		if err := s.store.ReplaceEpisodeLinks(ctx, existing.ID, epIn.LinkedFactIDs); err != nil {
			return nil, UpdateContentOutput{}, fmt.Errorf("replace episode links: %w", err)
		}
		linksChanged = true
	}

	s.logger.Info("memory_update_content",
		"id", existing.ID,
		"layer", "l3_episodic",
		"tags_changed", in.Tags != nil,
		"links_changed", linksChanged,
	)
	return nil, UpdateContentOutput{
		ID:         existing.ID,
		Layer:      string(memory.TypeL3Episodic),
		Reembedded: true,
	}, nil
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

	// Cross-type links: evidence edges from this fact's row in memory_edges.
	// Preserves the pre-Phase-7 user-visible shape of memory_get.Links by
	// filtering to edge_type='evidence' and resolving the "other" endpoint's
	// layer via fact/episode lookup.
	edges, err := s.store.ListEdges(ctx, f.ID, 1)
	if err != nil {
		return nil, GetOutput{}, fmt.Errorf("list edges: %w", err)
	}
	for _, e := range edges {
		if e.Edge.EdgeType != "evidence" {
			continue
		}
		other := e.Edge.DstID
		if e.Edge.SrcID != f.ID {
			other = e.Edge.SrcID
		}
		layer := resolveLayerString(ctx, s, other)
		out.Links = append(out.Links, LinkRef{
			ID:       other,
			Layer:    layer,
			LinkType: e.Edge.EdgeType,
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

	// Cross-type links: evidence edges from this episode's row in memory_edges.
	// Preserves the pre-Phase-7 user-visible shape of memory_get.Links by
	// filtering to edge_type='evidence' and resolving the "other" endpoint's
	// layer via fact/episode lookup.
	edges, err := s.store.ListEdges(ctx, ep.ID, 1)
	if err != nil {
		return nil, GetOutput{}, fmt.Errorf("list edges: %w", err)
	}
	for _, e := range edges {
		if e.Edge.EdgeType != "evidence" {
			continue
		}
		other := e.Edge.DstID
		if e.Edge.SrcID != ep.ID {
			other = e.Edge.SrcID
		}
		layer := resolveLayerString(ctx, s, other)
		out.Links = append(out.Links, LinkRef{
			ID:       other,
			Layer:    layer,
			LinkType: e.Edge.EdgeType,
		})
	}

	s.logger.Info("memory_get", "id", ep.ID, "layer", "l3_episodic")
	return nil, out, nil
}

// resolveLayerString classifies a memory ID as "l2_semantic", "l3_episodic",
// or "entity" by trying each table in turn. Returns "" if none match (e.g.,
// orphan reference). Used to populate LinkRef.Layer / EdgeDetail.OtherLayer.
func resolveLayerString(ctx context.Context, s *Server, id string) string {
	if f, err := s.store.GetFact(ctx, id); err == nil && f != nil {
		return string(memory.TypeL2Semantic)
	}
	if ep, err := s.store.GetEpisode(ctx, id); err == nil && ep != nil {
		return string(memory.TypeL3Episodic)
	}
	if ent, err := s.store.GetEntity(ctx, id); err == nil && ent.ID != "" {
		return "entity"
	}
	return ""
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

// --- memory_unsupersede ---

// UnsupersedeInput is the input schema for the memory_unsupersede tool.
type UnsupersedeInput struct {
	FactID string `json:"fact_id" jsonschema:"ID of the superseded fact to revive"`
}

// UnsupersedeOutput is the output schema for the memory_unsupersede tool.
type UnsupersedeOutput struct {
	FactID                 string `json:"fact_id"`
	PreviouslySupersededBy string `json:"previously_superseded_by"`
	// Warning is populated when the formerly-superseding fact is still
	// active (its superseded_by is nil); the operator has two coexisting
	// facts and should decide how to reconcile them.
	Warning string `json:"warning,omitempty"`
}

// handleUnsupersede clears the superseded_by pointer on a fact, reviving it so
// it participates in ListFacts / GlobalSearch again. It does NOT recompute
// cluster centroids because membership is unchanged — only a flag flips.
func (s *Server) handleUnsupersede(ctx context.Context, _ *mcpsdk.CallToolRequest, in UnsupersedeInput) (*mcpsdk.CallToolResult, UnsupersedeOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, UnsupersedeOutput{}, nil
	}

	if in.FactID == "" {
		return nil, UnsupersedeOutput{}, fmt.Errorf("fact_id is required")
	}

	// GetFact returns (nil, nil) for IDs that aren't facts (including
	// episode IDs — they're stored in a different table). That's enough of
	// a not-found signal for the spec: "ID does not name a fact" is
	// indistinguishable from "ID does not exist" at this layer.
	fact, err := s.store.GetFact(ctx, in.FactID)
	if err != nil {
		return nil, UnsupersedeOutput{}, fmt.Errorf("get fact: %w", err)
	}
	if fact == nil {
		return nil, UnsupersedeOutput{}, fmt.Errorf("fact not found: %s", in.FactID)
	}
	if fact.SupersededBy == nil {
		return nil, UnsupersedeOutput{}, fmt.Errorf("fact is not superseded")
	}

	prev, err := s.store.ClearFactSuperseded(ctx, in.FactID)
	if err != nil {
		return nil, UnsupersedeOutput{}, fmt.Errorf("clear fact superseded: %w", err)
	}

	out := UnsupersedeOutput{
		FactID:                 in.FactID,
		PreviouslySupersededBy: prev,
	}

	// If the superseder is itself still active (its superseded_by is nil),
	// we now have two coexisting facts in the same subtype that originally
	// conflicted. Warn the operator so they can reconcile.
	super, err := s.store.GetFact(ctx, prev)
	if err != nil {
		// Don't fail the overall op — the revival succeeded; log and
		// leave Warning empty.
		s.logger.Warn("memory_unsupersede: get superseder failed", "id", prev, "err", err)
	} else if super != nil && super.SupersededBy == nil {
		out.Warning = fmt.Sprintf(
			"both fact %s and fact %s are now active and may be treated as duplicates; consider memory_forget or memory_update_content on one of them",
			in.FactID, prev,
		)
	}

	s.logger.Info("memory_unsupersede", "fact_id", in.FactID, "previously_superseded_by", prev, "warning", out.Warning != "")
	return nil, out, nil
}

// --- Phase 7 knowledge graph tools ---
//
// Six MCP tools that replace the retired memory_link / memory_unlink /
// memory_list_links surface from Phase 4A. The new tools cover the full
// knowledge graph: typed/weighted edges between any two memory or entity
// IDs (memory_edge_add / memory_edge_remove / memory_edge_list), and a
// first-class entity layer (memory_entity_upsert / memory_entity_mention /
// memory_entity_neighbors). See docs/design/phase-7-knowledge-graph.md
// for the locked decisions and tool-shape source of truth.

// edgePreviewMax is the max length of content_preview fields returned by
// memory_edge_list and memory_entity_neighbors.
const edgePreviewMax = 120

// truncatePreview trims s to at most n bytes. A char here is a byte for
// simplicity (same convention used elsewhere in the codebase for length
// limits).
func truncatePreview(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

// --- memory_edge_add ---

// EdgeAddInput is the input schema for the memory_edge_add tool.
type EdgeAddInput struct {
	SrcID    string   `json:"src_id" jsonschema:"source memory ID (fact, episode, or entity)"`
	DstID    string   `json:"dst_id" jsonschema:"destination memory ID"`
	EdgeType string   `json:"edge_type" jsonschema:"free-form; canonical types in README"`
	Weight   *float64 `json:"weight,omitempty" jsonschema:"default 1.0"`
}

// EdgeAddOutput is the output schema for the memory_edge_add tool.
type EdgeAddOutput struct {
	SrcID    string  `json:"src_id"`
	DstID    string  `json:"dst_id"`
	EdgeType string  `json:"edge_type"`
	Weight   float64 `json:"weight"`
	Created  bool    `json:"created"`
}

func (s *Server) handleEdgeAdd(ctx context.Context, _ *mcpsdk.CallToolRequest, in EdgeAddInput) (*mcpsdk.CallToolResult, EdgeAddOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, EdgeAddOutput{}, nil
	}

	if in.SrcID == "" {
		return nil, EdgeAddOutput{}, fmt.Errorf("src_id is required")
	}
	if in.DstID == "" {
		return nil, EdgeAddOutput{}, fmt.Errorf("dst_id is required")
	}
	if in.EdgeType == "" {
		return nil, EdgeAddOutput{}, fmt.Errorf("edge_type is required")
	}

	// Validate both IDs resolve to a fact, episode, or entity. Conservative
	// "fail loud on unknown IDs" stance preserves the old handleLink behavior.
	if resolveLayerString(ctx, s, in.SrcID) == "" {
		return nil, EdgeAddOutput{}, fmt.Errorf("src_id not found: %s", in.SrcID)
	}
	if resolveLayerString(ctx, s, in.DstID) == "" {
		return nil, EdgeAddOutput{}, fmt.Errorf("dst_id not found: %s", in.DstID)
	}

	weight := 1.0
	if in.Weight != nil {
		weight = *in.Weight
	}

	edge := memory.Edge{
		SrcID:    in.SrcID,
		DstID:    in.DstID,
		EdgeType: in.EdgeType,
		Weight:   weight,
	}
	created, err := s.store.AddEdge(ctx, edge)
	if err != nil {
		return nil, EdgeAddOutput{}, fmt.Errorf("add edge: %w", err)
	}

	// Per the design doc's locked decision: idempotent repeat with a
	// different weight does NOT overwrite — return the stored row's
	// weight, not the caller's input weight.
	canonicalWeight := weight
	if !created {
		existingEdges, lookupErr := s.store.ListEdges(ctx, in.SrcID, 1)
		if lookupErr != nil {
			return nil, EdgeAddOutput{}, fmt.Errorf("resolve stored edge weight: %w", lookupErr)
		}
		for _, ewd := range existingEdges {
			if ewd.Edge.SrcID == in.SrcID &&
				ewd.Edge.DstID == in.DstID &&
				ewd.Edge.EdgeType == in.EdgeType {
				canonicalWeight = ewd.Edge.Weight
				break
			}
		}
	}

	s.logger.Info("memory_edge_add",
		"src_id", in.SrcID, "dst_id", in.DstID, "edge_type", in.EdgeType,
		"weight", canonicalWeight, "created", created)
	return nil, EdgeAddOutput{
		SrcID:    in.SrcID,
		DstID:    in.DstID,
		EdgeType: in.EdgeType,
		Weight:   canonicalWeight,
		Created:  created,
	}, nil
}

// --- memory_edge_remove ---

// EdgeRemoveInput is the input schema for the memory_edge_remove tool.
type EdgeRemoveInput struct {
	SrcID    string `json:"src_id" jsonschema:"source memory ID"`
	DstID    string `json:"dst_id" jsonschema:"destination memory ID"`
	EdgeType string `json:"edge_type" jsonschema:"edge type label"`
}

// EdgeRemoveOutput is the output schema for the memory_edge_remove tool.
type EdgeRemoveOutput struct {
	Deleted bool `json:"deleted"`
}

func (s *Server) handleEdgeRemove(ctx context.Context, _ *mcpsdk.CallToolRequest, in EdgeRemoveInput) (*mcpsdk.CallToolResult, EdgeRemoveOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, EdgeRemoveOutput{}, nil
	}

	if in.SrcID == "" {
		return nil, EdgeRemoveOutput{}, fmt.Errorf("src_id is required")
	}
	if in.DstID == "" {
		return nil, EdgeRemoveOutput{}, fmt.Errorf("dst_id is required")
	}
	if in.EdgeType == "" {
		return nil, EdgeRemoveOutput{}, fmt.Errorf("edge_type is required")
	}

	deleted, err := s.store.RemoveEdge(ctx, in.SrcID, in.DstID, in.EdgeType)
	if err != nil {
		return nil, EdgeRemoveOutput{}, fmt.Errorf("remove edge: %w", err)
	}

	s.logger.Info("memory_edge_remove",
		"src_id", in.SrcID, "dst_id", in.DstID, "edge_type", in.EdgeType, "deleted", deleted)
	return nil, EdgeRemoveOutput{Deleted: deleted}, nil
}

// --- memory_edge_list ---

// EdgeListInput is the input schema for the memory_edge_list tool.
type EdgeListInput struct {
	MemoryID string `json:"memory_id" jsonschema:"fact, episode, or entity ID"`
	Hops     int    `json:"hops,omitempty" jsonschema:"1..3, default 1"`
}

// EdgeDetail describes a single edge from the perspective of the seed
// memory. OtherID is the endpoint that BFS reached at this depth — at
// hops=1 it is the non-seed endpoint of the single edge row; at hops>=2
// the seed itself no longer participates directly, and OtherID is the
// newly-reached node (i.e., the endpoint that was not already in the
// visited set at the moment BFS emitted this row).
type EdgeDetail struct {
	OtherID        string  `json:"other_id"`
	OtherLayer     string  `json:"other_layer"`
	EdgeType       string  `json:"edge_type"`
	Weight         float64 `json:"weight"`
	Distance       int     `json:"distance"`
	ContentPreview string  `json:"content_preview"`
}

// EdgeListOutput is the output schema for the memory_edge_list tool.
type EdgeListOutput struct {
	MemoryID string       `json:"memory_id"`
	Layer    string       `json:"layer"`
	Edges    []EdgeDetail `json:"edges"`
}

func (s *Server) handleEdgeList(ctx context.Context, _ *mcpsdk.CallToolRequest, in EdgeListInput) (*mcpsdk.CallToolResult, EdgeListOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, EdgeListOutput{}, nil
	}

	if in.MemoryID == "" {
		return nil, EdgeListOutput{}, fmt.Errorf("memory_id is required")
	}

	hops := in.Hops
	if hops == 0 {
		hops = 1
	}
	if hops < 1 || hops > 3 {
		return nil, EdgeListOutput{}, fmt.Errorf("hops out of range: %d (must be 1..3)", hops)
	}

	seedLayer := resolveLayerString(ctx, s, in.MemoryID)
	if seedLayer == "" {
		return nil, EdgeListOutput{}, fmt.Errorf("memory not found: %s", in.MemoryID)
	}

	rawEdges, err := s.store.ListEdges(ctx, in.MemoryID, hops)
	if err != nil {
		return nil, EdgeListOutput{}, fmt.Errorf("list edges: %w", err)
	}

	// Mirror the store's BFS visited-set so we can determine OtherID for
	// hops >= 2: the endpoint that is NOT in visited at row-emission time
	// is the newly-reached "other". The store guarantees exactly one of
	// (src, dst) is in visited per emitted row, so this is unambiguous.
	visited := map[string]struct{}{in.MemoryID: {}}
	details := make([]EdgeDetail, 0, len(rawEdges))
	for _, ewd := range rawEdges {
		var other string
		if _, ok := visited[ewd.Edge.SrcID]; ok {
			other = ewd.Edge.DstID
		} else {
			other = ewd.Edge.SrcID
		}
		visited[other] = struct{}{}

		otherLayer := resolveLayerString(ctx, s, other)
		preview := buildContentPreview(ctx, s, other, otherLayer)

		details = append(details, EdgeDetail{
			OtherID:        other,
			OtherLayer:     otherLayer,
			EdgeType:       ewd.Edge.EdgeType,
			Weight:         ewd.Edge.Weight,
			Distance:       ewd.Distance,
			ContentPreview: preview,
		})
	}

	s.logger.Info("memory_edge_list",
		"memory_id", in.MemoryID, "layer", seedLayer, "hops", hops, "count", len(details))
	return nil, EdgeListOutput{
		MemoryID: in.MemoryID,
		Layer:    seedLayer,
		Edges:    details,
	}, nil
}

// buildContentPreview returns a 120-char preview for the memory at id based
// on its layer. Empty string when the layer is unknown or the row is missing.
func buildContentPreview(ctx context.Context, s *Server, id, layer string) string {
	switch layer {
	case string(memory.TypeL2Semantic):
		f, err := s.store.GetFact(ctx, id)
		if err != nil || f == nil {
			return ""
		}
		return truncatePreview(f.Content, edgePreviewMax)
	case string(memory.TypeL3Episodic):
		ep, err := s.store.GetEpisode(ctx, id)
		if err != nil || ep == nil {
			return ""
		}
		return truncatePreview(memory.Candidate{Episode: ep}.Content(), edgePreviewMax)
	case "entity":
		ent, err := s.store.GetEntity(ctx, id)
		if err != nil || ent.ID == "" {
			return ""
		}
		return truncatePreview(ent.Name+" ("+ent.EntityType+")", edgePreviewMax)
	}
	return ""
}

// --- memory_entity_upsert ---

// EntityUpsertInput is the input schema for the memory_entity_upsert tool.
type EntityUpsertInput struct {
	Name       string `json:"name" jsonschema:"entity name"`
	EntityType string `json:"entity_type" jsonschema:"file/repo/library/concept/person/command/..."`
}

// EntityUpsertOutput is the output schema for the memory_entity_upsert tool.
type EntityUpsertOutput struct {
	EntityID            string `json:"entity_id"`
	Created             bool   `json:"created"`
	MatchedBySimilarity bool   `json:"matched_by_similarity"`
}

func (s *Server) handleEntityUpsert(ctx context.Context, _ *mcpsdk.CallToolRequest, in EntityUpsertInput) (*mcpsdk.CallToolResult, EntityUpsertOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, EntityUpsertOutput{}, nil
	}

	if in.Name == "" {
		return nil, EntityUpsertOutput{}, fmt.Errorf("name is required")
	}
	if in.EntityType == "" {
		return nil, EntityUpsertOutput{}, fmt.Errorf("entity_type is required")
	}

	// Embed the deterministic text per the locked decision: name + " (" + entity_type + ")".
	// Best-effort: if the embedder is unavailable, fall through with a nil
	// embedding so the entity can still be stored and found by exact match.
	embedText := in.Name + " (" + in.EntityType + ")"
	var embedding []float32
	vecs, err := s.embedder.Embed(ctx, []string{embedText})
	if err != nil {
		s.logger.Warn("memory_entity_upsert: embed failed; inserting without embedding",
			"name", in.Name, "entity_type", in.EntityType, "err", err)
	} else if len(vecs) > 0 && len(vecs[0]) > 0 {
		embedding = vecs[0]
	}

	id, created, matched, err := s.store.UpsertEntity(ctx, in.Name, in.EntityType, embedding)
	if err != nil {
		return nil, EntityUpsertOutput{}, fmt.Errorf("upsert entity: %w", err)
	}

	s.logger.Info("memory_entity_upsert",
		"entity_id", id, "name", in.Name, "entity_type", in.EntityType,
		"created", created, "matched_by_similarity", matched)
	return nil, EntityUpsertOutput{
		EntityID:            id,
		Created:             created,
		MatchedBySimilarity: matched,
	}, nil
}

// --- memory_entity_mention ---

// EntityMentionInput is the input schema for the memory_entity_mention tool.
type EntityMentionInput struct {
	MemoryID  string   `json:"memory_id" jsonschema:"fact or episode ID"`
	EntityIDs []string `json:"entity_ids" jsonschema:"one or more entity IDs"`
	Role      string   `json:"role,omitempty" jsonschema:"advisory: subject/object/mention"`
}

// EntityMentionOutput is the output schema for the memory_entity_mention tool.
type EntityMentionOutput struct {
	Inserted int `json:"inserted"`
}

func (s *Server) handleEntityMention(ctx context.Context, _ *mcpsdk.CallToolRequest, in EntityMentionInput) (*mcpsdk.CallToolResult, EntityMentionOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, EntityMentionOutput{}, nil
	}

	if in.MemoryID == "" {
		return nil, EntityMentionOutput{}, fmt.Errorf("memory_id is required")
	}
	if len(in.EntityIDs) == 0 {
		return nil, EntityMentionOutput{}, fmt.Errorf("entity_ids must be non-empty")
	}

	// Validate memory_id resolves to a fact or episode (NOT entity).
	memLayer := resolveLayerString(ctx, s, in.MemoryID)
	switch memLayer {
	case string(memory.TypeL2Semantic), string(memory.TypeL3Episodic):
		// ok
	case "entity":
		return nil, EntityMentionOutput{}, fmt.Errorf("memory_id %s is an entity; mentions are memory->entity only", in.MemoryID)
	default:
		return nil, EntityMentionOutput{}, fmt.Errorf("memory not found: %s", in.MemoryID)
	}

	// Validate every entity_id resolves to an entity.
	for _, eid := range in.EntityIDs {
		if eid == "" {
			return nil, EntityMentionOutput{}, fmt.Errorf("entity_ids contains empty string")
		}
		ent, err := s.store.GetEntity(ctx, eid)
		if err != nil {
			return nil, EntityMentionOutput{}, fmt.Errorf("get entity %s: %w", eid, err)
		}
		if ent.ID == "" {
			return nil, EntityMentionOutput{}, fmt.Errorf("entity not found: %s", eid)
		}
	}

	inserted, err := s.store.AddEntityMentions(ctx, in.MemoryID, in.EntityIDs, in.Role)
	if err != nil {
		return nil, EntityMentionOutput{}, fmt.Errorf("add entity mentions: %w", err)
	}

	s.logger.Info("memory_entity_mention",
		"memory_id", in.MemoryID, "entity_count", len(in.EntityIDs),
		"role", in.Role, "inserted", inserted)
	return nil, EntityMentionOutput{Inserted: inserted}, nil
}

// --- memory_entity_neighbors ---

// EntityNeighborsInput is the input schema for the memory_entity_neighbors tool.
type EntityNeighborsInput struct {
	EntityID string `json:"entity_id" jsonschema:"seed entity ID"`
	Hops     int    `json:"hops,omitempty" jsonschema:"1..3, default 1"`
}

// NeighborMemoryOut is a memory neighbor returned by memory_entity_neighbors.
type NeighborMemoryOut struct {
	ID             string `json:"id"`
	Layer          string `json:"layer"`
	ContentPreview string `json:"content_preview"`
	Distance       int    `json:"distance"`
}

// NeighborEntityOut is an entity neighbor returned by memory_entity_neighbors.
type NeighborEntityOut struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	EntityType string `json:"entity_type"`
	Distance   int    `json:"distance"`
}

// EntityNeighborsOutput is the output schema for the memory_entity_neighbors tool.
type EntityNeighborsOutput struct {
	EntityID string              `json:"entity_id"`
	Memories []NeighborMemoryOut `json:"memories"`
	Entities []NeighborEntityOut `json:"entities"`
}

func (s *Server) handleEntityNeighbors(ctx context.Context, _ *mcpsdk.CallToolRequest, in EntityNeighborsInput) (*mcpsdk.CallToolResult, EntityNeighborsOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, EntityNeighborsOutput{}, nil
	}

	if in.EntityID == "" {
		return nil, EntityNeighborsOutput{}, fmt.Errorf("entity_id is required")
	}

	hops := in.Hops
	if hops == 0 {
		hops = 1
	}
	if hops < 1 || hops > 3 {
		return nil, EntityNeighborsOutput{}, fmt.Errorf("hops out of range: %d (must be 1..3)", hops)
	}

	ent, err := s.store.GetEntity(ctx, in.EntityID)
	if err != nil {
		return nil, EntityNeighborsOutput{}, fmt.Errorf("get entity: %w", err)
	}
	if ent.ID == "" {
		return nil, EntityNeighborsOutput{}, fmt.Errorf("entity not found: %s", in.EntityID)
	}

	memories, entities, err := s.store.ListEntityNeighbors(ctx, in.EntityID, hops)
	if err != nil {
		return nil, EntityNeighborsOutput{}, fmt.Errorf("list entity neighbors: %w", err)
	}

	memOut := make([]NeighborMemoryOut, 0, len(memories))
	for _, m := range memories {
		memOut = append(memOut, NeighborMemoryOut{
			ID:             m.ID,
			Layer:          string(m.Layer),
			ContentPreview: m.ContentPreview,
			Distance:       m.Distance,
		})
	}
	entOut := make([]NeighborEntityOut, 0, len(entities))
	for _, e := range entities {
		entOut = append(entOut, NeighborEntityOut{
			ID:         e.ID,
			Name:       e.Name,
			EntityType: e.EntityType,
			Distance:   e.Distance,
		})
	}

	s.logger.Info("memory_entity_neighbors",
		"entity_id", in.EntityID, "hops", hops,
		"memories", len(memOut), "entities", len(entOut))
	return nil, EntityNeighborsOutput{
		EntityID: in.EntityID,
		Memories: memOut,
		Entities: entOut,
	}, nil
}

// --- Session tools (Phase 6c) ---
//
// The four handlers below expose the session lifecycle: init (create or
// resume), snapshot (explicit checkpoint), restore (pure read), and end
// (scoped decay tick + optional episode + close). They reuse applySession-
// Mutation / AppendToBuffer / etc. for the buffer plumbing — see
// session_helpers.go and internal/memory/session_buffer.go.

// sessionTimeFormat is the ISO8601 UTC format emitted in session tool outputs.
// Matches the design doc's fixed-width "2006-01-02T15:04:05Z" (no sub-second,
// always Z) so harness-side parsing stays simple.
const sessionTimeFormat = "2006-01-02T15:04:05Z"

// formatSessionTime renders t as ISO8601 UTC. Zero times are rendered as an
// empty string so callers can detect "not set" without importing time.
func formatSessionTime(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return t.UTC().Format(sessionTimeFormat)
}

// formatSessionTimePtr returns a *string formatted per formatSessionTime, or
// nil when the input pointer is nil. Used for ClosedAt in outputs.
func formatSessionTimePtr(t *time.Time) *string {
	if t == nil {
		return nil
	}
	out := formatSessionTime(*t)
	return &out
}

// mergeTagSlices concatenates two tag slices for handoff to the store's
// UpdateSessionMeta, which runs normalizeTags internally (lowercase, trim,
// dedup, sort, length-check). Either or both inputs may be nil. Used by
// session_init when resuming with new tags — the spec calls for MERGE, not
// REPLACE. Dedup/sort happen inside the store; we just append.
func mergeTagSlices(existing, incoming []string) []string {
	combined := make([]string, 0, len(existing)+len(incoming))
	combined = append(combined, existing...)
	combined = append(combined, incoming...)
	return combined
}

// --- memory_session_init ---

// SessionInitInput is the input schema for memory_session_init.
type SessionInitInput struct {
	SessionID   string   `json:"session_id" jsonschema:"stable session identifier (client-generated)"`
	ProjectHint string   `json:"project_hint,omitempty" jsonschema:"freeform hint describing the project scope (replaces on resume if non-empty)"`
	Tags        []string `json:"tags,omitempty" jsonschema:"tags to associate with this session (merged with existing tags on resume)"`
}

// SessionInitOutput is the output schema for memory_session_init.
type SessionInitOutput struct {
	SessionID   string             `json:"session_id"`
	Created     bool               `json:"created"`
	Buffer      []memory.MemoryRef `json:"buffer"`
	ProjectHint string             `json:"project_hint"`
	Tags        []string           `json:"tags"`
	CreatedAt   string             `json:"created_at"`
	ClosedAt    *string            `json:"closed_at,omitempty"`
}

func (s *Server) handleSessionInit(ctx context.Context, _ *mcpsdk.CallToolRequest, in SessionInitInput) (*mcpsdk.CallToolResult, SessionInitOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, SessionInitOutput{}, nil
	}

	if in.SessionID == "" {
		return nil, SessionInitOutput{}, fmt.Errorf("session_id is required")
	}

	existing, err := s.store.GetSession(ctx, in.SessionID)
	if err != nil {
		return nil, SessionInitOutput{}, fmt.Errorf("get session: %w", err)
	}

	if existing == nil {
		// New session: insert with an empty buffer sized to the current budget.
		budgetMax := s.bufferBudgetMax()
		sess := memory.Session{
			ID:          in.SessionID,
			ProjectHint: in.ProjectHint,
			Tags:        in.Tags,
			WorkingMem: memory.WorkingMemory{
				Buffer:    []memory.MemoryRef{},
				BudgetMax: budgetMax,
			},
		}
		if err := s.store.CreateSession(ctx, sess); err != nil {
			return nil, SessionInitOutput{}, fmt.Errorf("create session: %w", err)
		}
		// Re-fetch so the server-assigned timestamps are authoritative.
		created, err := s.store.GetSession(ctx, in.SessionID)
		if err != nil || created == nil {
			return nil, SessionInitOutput{}, fmt.Errorf("reload created session: %w", err)
		}
		s.logger.Info("memory_session_init", "session_id", in.SessionID, "created", true)
		return nil, SessionInitOutput{
			SessionID:   created.ID,
			Created:     true,
			Buffer:      []memory.MemoryRef{},
			ProjectHint: created.ProjectHint,
			Tags:        normalizeTagsSlice(created.Tags),
			CreatedAt:   formatSessionTime(created.CreatedAt),
			ClosedAt:    nil,
		}, nil
	}

	// Existing session.
	if existing.ClosedAt != nil {
		return nil, SessionInitOutput{}, fmt.Errorf("session closed, cannot resume: %s", in.SessionID)
	}

	// Resume: merge project_hint (replace if non-empty) and tags (union).
	newProjectHint := existing.ProjectHint
	if in.ProjectHint != "" {
		newProjectHint = in.ProjectHint
	}

	mergedTags := existing.Tags
	// Only touch tags when the caller actually provided them (non-nil).
	if in.Tags != nil {
		mergedTags = mergeTagSlices(existing.Tags, in.Tags)
	}

	// Write meta only if something changed. This keeps updated_at stable for
	// no-op resumes (which matters for downstream tooling that watches it).
	if newProjectHint != existing.ProjectHint || in.Tags != nil {
		if err := s.store.UpdateSessionMeta(ctx, in.SessionID, newProjectHint, mergedTags); err != nil {
			return nil, SessionInitOutput{}, fmt.Errorf("update session meta: %w", err)
		}
	}

	// Pull fresh state so the output reflects any meta update.
	resumed, err := s.store.GetSession(ctx, in.SessionID)
	if err != nil || resumed == nil {
		return nil, SessionInitOutput{}, fmt.Errorf("reload resumed session: %w", err)
	}

	buf := resumed.WorkingMem.Buffer
	if buf == nil {
		buf = []memory.MemoryRef{}
	}

	s.logger.Info("memory_session_init", "session_id", in.SessionID, "created", false, "buffer_len", len(buf))
	return nil, SessionInitOutput{
		SessionID:   resumed.ID,
		Created:     false,
		Buffer:      buf,
		ProjectHint: resumed.ProjectHint,
		Tags:        normalizeTagsSlice(resumed.Tags),
		CreatedAt:   formatSessionTime(resumed.CreatedAt),
		ClosedAt:    formatSessionTimePtr(resumed.ClosedAt),
	}, nil
}

// --- memory_session_snapshot ---

// SessionSnapshotInput is the input schema for memory_session_snapshot.
type SessionSnapshotInput struct {
	SessionID string `json:"session_id" jsonschema:"session to checkpoint"`
}

// SessionSnapshotOutput is the output schema for memory_session_snapshot.
type SessionSnapshotOutput struct {
	Persisted bool   `json:"persisted"`
	UpdatedAt string `json:"updated_at"`
}

func (s *Server) handleSessionSnapshot(ctx context.Context, _ *mcpsdk.CallToolRequest, in SessionSnapshotInput) (*mcpsdk.CallToolResult, SessionSnapshotOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, SessionSnapshotOutput{}, nil
	}

	if in.SessionID == "" {
		return nil, SessionSnapshotOutput{}, fmt.Errorf("session_id is required")
	}

	sess, err := s.store.GetSession(ctx, in.SessionID)
	if err != nil {
		return nil, SessionSnapshotOutput{}, fmt.Errorf("get session: %w", err)
	}
	if sess == nil {
		return nil, SessionSnapshotOutput{}, fmt.Errorf("session not found: %s", in.SessionID)
	}
	if sess.ClosedAt != nil {
		return nil, SessionSnapshotOutput{}, fmt.Errorf("session closed: %s", in.SessionID)
	}

	// Idempotent checkpoint: persist the current buffer even if it hasn't
	// changed. UpdateSessionBuffer bumps updated_at unconditionally so the
	// caller can observe that the checkpoint landed.
	if err := s.store.UpdateSessionBuffer(ctx, in.SessionID, sess.WorkingMem); err != nil {
		return nil, SessionSnapshotOutput{}, fmt.Errorf("update session buffer: %w", err)
	}

	// Re-read so the new updated_at is authoritative (the server assigns it).
	fresh, err := s.store.GetSession(ctx, in.SessionID)
	if err != nil || fresh == nil {
		return nil, SessionSnapshotOutput{}, fmt.Errorf("reload session after snapshot: %w", err)
	}

	s.logger.Info("memory_session_snapshot", "session_id", in.SessionID, "buffer_len", len(fresh.WorkingMem.Buffer))
	return nil, SessionSnapshotOutput{
		Persisted: true,
		UpdatedAt: formatSessionTime(fresh.UpdatedAt),
	}, nil
}

// --- memory_session_restore ---

// SessionRestoreInput is the input schema for memory_session_restore.
type SessionRestoreInput struct {
	SessionID string `json:"session_id" jsonschema:"session to load"`
}

// SessionRestoreOutput is the output schema for memory_session_restore.
type SessionRestoreOutput struct {
	Buffer      []memory.MemoryRef `json:"buffer"`
	ProjectHint string             `json:"project_hint"`
	Tags        []string           `json:"tags"`
	UpdatedAt   string             `json:"updated_at"`
	ClosedAt    *string            `json:"closed_at,omitempty"`
}

func (s *Server) handleSessionRestore(ctx context.Context, _ *mcpsdk.CallToolRequest, in SessionRestoreInput) (*mcpsdk.CallToolResult, SessionRestoreOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, SessionRestoreOutput{}, nil
	}

	if in.SessionID == "" {
		return nil, SessionRestoreOutput{}, fmt.Errorf("session_id is required")
	}

	sess, err := s.store.GetSession(ctx, in.SessionID)
	if err != nil {
		return nil, SessionRestoreOutput{}, fmt.Errorf("get session: %w", err)
	}
	if sess == nil {
		return nil, SessionRestoreOutput{}, fmt.Errorf("session not found: %s", in.SessionID)
	}

	buf := sess.WorkingMem.Buffer
	if buf == nil {
		buf = []memory.MemoryRef{}
	}

	s.logger.Info("memory_session_restore", "session_id", in.SessionID, "buffer_len", len(buf), "closed", sess.ClosedAt != nil)
	return nil, SessionRestoreOutput{
		Buffer:      buf,
		ProjectHint: sess.ProjectHint,
		Tags:        normalizeTagsSlice(sess.Tags),
		UpdatedAt:   formatSessionTime(sess.UpdatedAt),
		ClosedAt:    formatSessionTimePtr(sess.ClosedAt),
	}, nil
}

// --- memory_session_end ---

// SessionEndInput is the input schema for memory_session_end.
type SessionEndInput struct {
	SessionID string          `json:"session_id" jsonschema:"session to close"`
	Episode   *EpisodePayload `json:"episode,omitempty" jsonschema:"if set, write an L3 episode summarizing the session"`
	// EpisodeType lets the caller choose the subtype for the summary episode.
	// Defaults to "feedback" when unset — the best default for retrospective
	// "here's what we learned" tuples. Facts-only taxonomy values are rejected
	// downstream by writeEpisode's handleWrite wrapper.
	EpisodeType string   `json:"episode_type,omitempty" jsonschema:"subtype for the optional summary episode (default: feedback)"`
	EpisodeTags []string `json:"episode_tags,omitempty" jsonschema:"extra tags to apply to the summary episode; session:<id> is always added"`
}

// SessionEndOutput is the output schema for memory_session_end.
type SessionEndOutput struct {
	SessionID      string `json:"session_id"`
	EpisodeID      string `json:"episode_id,omitempty"`
	ClustersTicked int    `json:"clusters_ticked"`
}

func (s *Server) handleSessionEnd(ctx context.Context, _ *mcpsdk.CallToolRequest, in SessionEndInput) (*mcpsdk.CallToolResult, SessionEndOutput, error) {
	if s.cfg.Server.Disabled {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: `{"status":"disabled"}`}},
		}, SessionEndOutput{}, nil
	}

	if in.SessionID == "" {
		return nil, SessionEndOutput{}, fmt.Errorf("session_id is required")
	}

	sess, err := s.store.GetSession(ctx, in.SessionID)
	if err != nil {
		return nil, SessionEndOutput{}, fmt.Errorf("get session: %w", err)
	}
	if sess == nil {
		return nil, SessionEndOutput{}, fmt.Errorf("session not found: %s", in.SessionID)
	}
	if sess.ClosedAt != nil {
		return nil, SessionEndOutput{}, fmt.Errorf("session already closed: %s", in.SessionID)
	}

	// Step 1: resolve cluster IDs from the buffer's live fact/episode refs.
	// Missing memories (deleted since they landed in the buffer) are skipped
	// silently — they carry no live cluster membership to tick.
	clusterSet := make(map[string]struct{})
	bufferFactIDs := make([]string, 0, len(sess.WorkingMem.Buffer))
	for _, ref := range sess.WorkingMem.Buffer {
		switch ref.Layer {
		case memory.TypeL2Semantic:
			fact, err := s.store.GetFact(ctx, ref.ID)
			if err != nil {
				return nil, SessionEndOutput{}, fmt.Errorf("get fact %s: %w", ref.ID, err)
			}
			if fact == nil {
				continue // deleted; skip
			}
			if fact.ClusterID != "" {
				clusterSet[fact.ClusterID] = struct{}{}
			}
			bufferFactIDs = append(bufferFactIDs, fact.ID)
		case memory.TypeL3Episodic:
			ep, err := s.store.GetEpisode(ctx, ref.ID)
			if err != nil {
				return nil, SessionEndOutput{}, fmt.Errorf("get episode %s: %w", ref.ID, err)
			}
			if ep == nil {
				continue // deleted; skip
			}
			if ep.ClusterID != "" {
				clusterSet[ep.ClusterID] = struct{}{}
			}
		default:
			// Unknown layer — shouldn't happen for a buffer entry, skip.
			continue
		}
	}

	clusterIDs := make([]string, 0, len(clusterSet))
	for cid := range clusterSet {
		clusterIDs = append(clusterIDs, cid)
	}
	// Sort for determinism (useful for logs and tests).
	sort.Strings(clusterIDs)

	// Step 2a: resolve buffered-memory mentions → entity IDs so the
	// scoped tick can reset those entities' turns_since alongside the
	// clusters. Empty buffer → empty set → entities decay normally.
	bufferMemoryIDs := make([]string, 0, len(sess.WorkingMem.Buffer))
	for _, ref := range sess.WorkingMem.Buffer {
		bufferMemoryIDs = append(bufferMemoryIDs, ref.ID)
	}
	entityIDs, err := s.store.ListEntitiesByMemoryIDs(ctx, bufferMemoryIDs)
	if err != nil {
		return nil, SessionEndOutput{}, fmt.Errorf("list entities by memory ids: %w", err)
	}
	sort.Strings(entityIDs)

	// Step 2b: scoped decay tick — bump all clusters and entities,
	// reset the touched sets to 0.
	if err := s.mgr.TickDecayWithEntities(ctx, clusterIDs, entityIDs); err != nil {
		return nil, SessionEndOutput{}, fmt.Errorf("tick decay: %w", err)
	}

	out := SessionEndOutput{
		SessionID:      in.SessionID,
		ClustersTicked: len(clusterIDs),
	}

	// Step 3: optional summary episode. We delegate to handleWrite so
	// embedding, clustering, tick-decay, and validation are shared with the
	// normal write path. Passing SessionID="" is deliberate: we're closing
	// the session on the next step and don't want another buffer mutation
	// against it.
	if in.Episode != nil {
		ep := *in.Episode // shallow copy so we can safely mutate

		// Auto-link: ensure every L2 fact currently in the buffer is linked
		// from this summary episode. Dedup against any fact IDs the caller
		// explicitly supplied.
		existingLinks := make(map[string]struct{}, len(ep.LinkedFactIDs))
		for _, fid := range ep.LinkedFactIDs {
			existingLinks[fid] = struct{}{}
		}
		for _, fid := range bufferFactIDs {
			if _, ok := existingLinks[fid]; !ok {
				ep.LinkedFactIDs = append(ep.LinkedFactIDs, fid)
				existingLinks[fid] = struct{}{}
			}
		}

		// Auto-tag: always attach "session:<id>" so the episode is
		// cross-session discoverable. The rest of the tags come from
		// EpisodeTags verbatim (normalization happens downstream).
		sessionTag := "session:" + in.SessionID
		epTags := make([]string, 0, len(in.EpisodeTags)+1)
		epTags = append(epTags, in.EpisodeTags...)
		hasSessionTag := false
		for _, t := range in.EpisodeTags {
			if strings.EqualFold(strings.TrimSpace(t), sessionTag) {
				hasSessionTag = true
				break
			}
		}
		if !hasSessionTag {
			epTags = append(epTags, sessionTag)
		}

		subtype := in.EpisodeType
		if subtype == "" {
			subtype = "feedback"
		}

		_, writeOut, err := s.handleWrite(ctx, nil, WriteInput{
			Type:    subtype,
			Episode: &ep,
			Tags:    epTags,
			// SessionID intentionally left empty: we are closing this session
			// after the write and must not re-enter applySessionMutation
			// against it.
		})
		if err != nil {
			return nil, SessionEndOutput{}, fmt.Errorf("write session episode: %w", err)
		}
		out.EpisodeID = writeOut.ID
	}

	// Step 4: close the session.
	if err := s.store.CloseSession(ctx, in.SessionID); err != nil {
		return nil, SessionEndOutput{}, fmt.Errorf("close session: %w", err)
	}

	s.logger.Info("memory_session_end",
		"session_id", in.SessionID,
		"clusters_ticked", out.ClustersTicked,
		"entities_ticked", len(entityIDs),
		"episode_written", out.EpisodeID != "",
	)
	return nil, out, nil
}
