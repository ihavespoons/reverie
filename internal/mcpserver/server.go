package mcpserver

import (
	"context"
	"log/slog"
	"os"
	"time"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/diffsec/reverie/internal/cluster"
	"github.com/diffsec/reverie/internal/config"
	"github.com/diffsec/reverie/internal/decay"
	"github.com/diffsec/reverie/internal/embed"
	"github.com/diffsec/reverie/internal/manager"
	"github.com/diffsec/reverie/internal/memory"
)

// Server wraps the MCP SDK server and wires tool handlers to the memory store,
// embedding provider, decayer, and memory manager. It is the main entry point
// for the reverie MCP surface.
type Server struct {
	store       memory.Store
	embedder    embed.EmbeddingProvider
	decayer     decay.Decayer
	mgr         manager.MemoryManager
	assigner    cluster.Assigner
	cfg         *config.Config
	recallCache *recallCache
	logger      *slog.Logger
}

// NewServer creates a configured MCP server with all tools registered.
// If logger is nil, a discard logger is used.
func NewServer(store memory.Store, embedder embed.EmbeddingProvider, decayer decay.Decayer, mgr manager.MemoryManager, assigner cluster.Assigner, cfg *config.Config, logger *slog.Logger) *Server {
	if logger == nil {
		logger = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError + 1}))
	}

	ttl := time.Duration(cfg.Server.RecallCacheTTLSeconds) * time.Second
	if ttl <= 0 {
		ttl = 300 * time.Second
	}

	return &Server{
		store:       store,
		embedder:    embedder,
		decayer:     decayer,
		mgr:         mgr,
		assigner:    assigner,
		cfg:         cfg,
		recallCache: newRecallCache(ttl),
		logger:      logger,
	}
}

// Run constructs the MCP SDK server, registers all Phase 1 tools, and runs
// over stdio. It blocks until the context is cancelled or the client disconnects.
func (s *Server) Run(ctx context.Context) error {
	defer s.recallCache.stop()

	srv := mcpsdk.NewServer(
		&mcpsdk.Implementation{
			Name:    "reverie",
			Version: "0.1.0",
		},
		&mcpsdk.ServerOptions{
			Instructions: "Reverie is a persistent memory system for coding agents. " +
				"Use memory_recall to search, memory_apply_judgment to filter with Gate A verdicts, " +
				"memory_write to store, memory_reinforce to boost, memory_update_cluster to curate L1 metadata, " +
				"memory_forget to delete, and memory_list to browse. " +
				"Resources are available: reverie://status (system health), reverie://l1/index (cluster meta-index), " +
				"reverie://l3/recent (recent episodes). Use the session_start prompt at the beginning of a session.",
			Logger: s.logger,
		},
	)

	s.registerTools(srv)
	s.registerResources(srv)
	s.registerPrompts(srv)

	s.logger.Info("starting MCP stdio server")
	return srv.Run(ctx, &mcpsdk.StdioTransport{})
}

// registerTools adds all tool definitions to the SDK server.
func (s *Server) registerTools(srv *mcpsdk.Server) {
	openWorld := false

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name:        "memory_recall",
		Description: "Search persistent memory by natural-language query. Returns ranked candidates with similarity scores and gate pass flags. Use round=0 (default) for permissive OR-logic recall; round=1+ for refinement AND-logic.",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:  true,
			OpenWorldHint: &openWorld,
			Title:         "Recall memories",
		},
	}, s.handleRecall)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name:        "memory_write",
		Description: "Write a new memory. For L2 facts, provide content (text). For L3 episodes, provide an episode payload (situation/action/outcome/preemptive). Type must be one of: user, feedback, project, reference. Returns the new memory ID and layer.",
		Annotations: &mcpsdk.ToolAnnotations{
			IdempotentHint: true,
			OpenWorldHint:  &openWorld,
			Title:          "Write memory",
		},
	}, s.handleWrite)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name:        "memory_reinforce",
		Description: "Boost the utility of memories that were actually used. Pass the IDs of memories you referenced in your response. This updates their access timestamps and (Phase 2) utility scores.",
		Annotations: &mcpsdk.ToolAnnotations{
			IdempotentHint: true,
			OpenWorldHint:  &openWorld,
			Title:          "Reinforce memories",
		},
	}, s.handleReinforce)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name:        "memory_forget",
		Description: "Delete a memory by ID, or search for candidates to delete by query. If id is given, deletes immediately. If query is given (without id), returns candidates for confirmation (no deletion). Exactly one of id or query must be provided.",
		Annotations: &mcpsdk.ToolAnnotations{
			OpenWorldHint: &openWorld,
			Title:         "Forget memory",
		},
	}, s.handleForget)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name:        "memory_list",
		Description: "List memories with optional filtering by layer (l2 facts or l3 episodes) and subtype (user, feedback, project, reference). Supports pagination via limit/offset and sorting by created or accessed timestamp.",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:  true,
			OpenWorldHint: &openWorld,
			Title:         "List memories",
		},
	}, s.handleList)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name:        "memory_get",
		Description: "Fetch a single memory (fact or episode) by ID. Returns the full record including cluster metadata, supersede chain (facts: both predecessors via 'supersedes' and successor via 'superseded_by'), cross-type links, and episode-specific fields. Superseded facts are returned when requested by ID — this is the history view.",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:  true,
			OpenWorldHint: &openWorld,
			Title:         "Get memory by ID",
		},
	}, s.handleGet)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name: "memory_apply_judgment",
		Description: "Apply Gate A verdicts from a memory-judge subagent to a previous memory_recall result. " +
			"Pass the recall_id from memory_recall and per-candidate keep/drop verdicts. " +
			"Round 0 uses OR logic (permissive: gate_a OR gate_b OR gate_c). " +
			"Round 1+ uses AND logic (strict: gate_a AND gate_b AND gate_c). " +
			"Returns the final filtered memory set ranked by composite score.",
		Annotations: &mcpsdk.ToolAnnotations{
			OpenWorldHint: &openWorld,
			Title:         "Apply judgment to recall",
		},
	}, s.handleApplyJudgment)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name:        "memory_update_cluster",
		Description: "Update a cluster's L1 metadata (summary, domain label, meta-instruction). Use this to curate the L1 meta-index. The cluster's utility and frequency are managed automatically by memory_reinforce.",
		Annotations: &mcpsdk.ToolAnnotations{
			IdempotentHint: true,
			OpenWorldHint:  &openWorld,
			Title:          "Update cluster metadata",
		},
	}, s.handleUpdateCluster)

	readOnlyFalse := false
	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name: "memory_reassign_cluster",
		Description: "Move a single fact or episode into a different cluster. Use this to correct auto-clustering mistakes. " +
			"Recomputes centroids for both the old and new clusters; if the old cluster is emptied by the move, it is deleted. " +
			"The memory's ID is preserved so supersede chains and episode links stay intact.",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:  readOnlyFalse,
			OpenWorldHint: &openWorld,
			Title:         "Reassign memory to a cluster",
		},
	}, s.handleReassignCluster)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name: "memory_split_cluster",
		Description: "Partition a cluster's members into new clusters by explicit ID groups. " +
			"Each group becomes a new cluster; optional Metas[] provides summary/domain/meta_instr per group. " +
			"Members not listed in any group remain in the source cluster. " +
			"If every member is partitioned the source cluster is deleted; otherwise its centroid is recomputed. " +
			"Groups must be non-empty and non-overlapping, and every listed ID must currently be a member of the source cluster.",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:  readOnlyFalse,
			OpenWorldHint: &openWorld,
			Title:         "Split cluster into new clusters",
		},
	}, s.handleSplitCluster)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name: "memory_merge_clusters",
		Description: "Merge N source clusters into a single target cluster. Every non-superseded fact and every episode in each source is reparented to the target; the source clusters are then deleted. " +
			"Member IDs are preserved so supersede chains and episode links stay intact. The target's centroid is recomputed from its new membership. " +
			"Errors if the target is listed as a source, if the source list is empty, or if any source or the target does not exist.",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:  readOnlyFalse,
			OpenWorldHint: &openWorld,
			Title:         "Merge clusters",
		},
	}, s.handleMergeClusters)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name: "memory_update_content",
		Description: "Amend a fact's content or an episode's situation/action/outcome/preemptive fields in place. " +
			"Re-embeds and re-hashes the memory but preserves its ID, cluster_id, created_at, valid_from, " +
			"superseded_by, and (unless explicitly replaced) tags and episode links. " +
			"Does NOT trigger conflict detection/supersede and does NOT reassign cluster — those are separate tools. " +
			"Provide content (facts) OR episode (episodes), not both. tags is a tri-state: omit to preserve, empty array to clear. " +
			"For episodes, linked_fact_ids follows the same tri-state convention.",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:  readOnlyFalse,
			OpenWorldHint: &openWorld,
			Title:         "Update memory content",
		},
	}, s.handleUpdateContent)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name: "memory_decay_tick",
		Description: "Advance the decay clock. Internal tool for session-end hooks and admin use. " +
			"Not typically called by agents. Increments turns_since on all clusters.",
		Annotations: &mcpsdk.ToolAnnotations{
			OpenWorldHint: &openWorld,
			Title:         "Decay tick",
		},
	}, s.handleDecayTick)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name: "memory_unsupersede",
		Description: "Reverse an auto-supersede by clearing the superseded_by pointer on a fact so it is active again. " +
			"Use this when a near-duplicate write was actually a distinct fact and should not have been hidden. " +
			"Errors if the fact does not exist or is not currently superseded. " +
			"Returns a warning when the superseder is still active — the operator then has two coexisting facts and may want to memory_forget or memory_update_content one of them.",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:  readOnlyFalse,
			OpenWorldHint: &openWorld,
			Title:         "Unsupersede fact",
		},
	}, s.handleUnsupersede)

	// --- Phase 7 knowledge graph tools ---

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name:        "memory_edge_add",
		Description: "Add a typed directed edge between two memories or entities.",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:   readOnlyFalse,
			IdempotentHint: true,
			OpenWorldHint:  &openWorld,
			Title:          "Add knowledge-graph edge",
		},
	}, s.handleEdgeAdd)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name:        "memory_edge_remove",
		Description: "Remove a specific edge (idempotent on missing).",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:   readOnlyFalse,
			IdempotentHint: true,
			OpenWorldHint:  &openWorld,
			Title:          "Remove knowledge-graph edge",
		},
	}, s.handleEdgeRemove)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name:        "memory_edge_list",
		Description: "List edges incident to a memory or entity, up to N hops (1-3).",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:  true,
			OpenWorldHint: &openWorld,
			Title:         "List edges",
		},
	}, s.handleEdgeList)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name:        "memory_entity_upsert",
		Description: "Create or dedupe an entity by (name, entity_type), with similarity fallback.",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:   readOnlyFalse,
			IdempotentHint: true,
			OpenWorldHint:  &openWorld,
			Title:          "Upsert entity",
		},
	}, s.handleEntityUpsert)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name:        "memory_entity_mention",
		Description: "Attach a memory to one or more entities (idempotent).",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:   readOnlyFalse,
			IdempotentHint: true,
			OpenWorldHint:  &openWorld,
			Title:          "Add entity mentions",
		},
	}, s.handleEntityMention)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name:        "memory_entity_neighbors",
		Description: "Walk the graph from an entity to nearby memories and entities.",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:  true,
			OpenWorldHint: &openWorld,
			Title:         "Entity neighbors",
		},
	}, s.handleEntityNeighbors)

	// --- Phase 6c session tools ---

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name: "memory_session_init",
		Description: "Open or resume a session. Creates a new session row if session_id is unknown; otherwise returns the persisted buffer for resume. " +
			"On resume, project_hint replaces (if non-empty) and tags merge with the existing set. " +
			"Closed sessions cannot be resumed — a fresh session_id is required.",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:   readOnlyFalse,
			IdempotentHint: true,
			OpenWorldHint:  &openWorld,
			Title:          "Initialize or resume a session",
		},
	}, s.handleSessionInit)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name: "memory_session_snapshot",
		Description: "Force a checkpoint write of the current session buffer. Normally implicit after each mutation; " +
			"call this tool to request an explicit persist, e.g. before the harness checkpoints itself.",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:   readOnlyFalse,
			IdempotentHint: true,
			OpenWorldHint:  &openWorld,
			Title:          "Checkpoint session buffer",
		},
	}, s.handleSessionSnapshot)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name: "memory_session_restore",
		Description: "Read the persisted state for a session: buffer, tags, project_hint, updated_at, and closed_at. " +
			"Pure read — does not reopen a closed session. Use session_init for the create/resume path.",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:  true,
			OpenWorldHint: &openWorld,
			Title:         "Restore session state",
		},
	}, s.handleSessionRestore)

	mcpsdk.AddTool(srv, &mcpsdk.Tool{
		Name:        "memory_session_end",
		Description: "Close a session. Runs a scoped decay tick (bumps all clusters, resets turns_since to 0 only for clusters referenced by the session buffer), optionally writes an L3 episode summary (auto-tagged 'session:<id>', auto-linked to every fact in the buffer), and marks the session closed. Closed sessions are read-only.",
		Annotations: &mcpsdk.ToolAnnotations{
			ReadOnlyHint:  readOnlyFalse,
			OpenWorldHint: &openWorld,
			Title:         "End a session",
		},
	}, s.handleSessionEnd)
}
