package mcpserver

import (
	"context"
	"log/slog"
	"os"
	"time"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"

	"personal/reverie/internal/cluster"
	"personal/reverie/internal/config"
	"personal/reverie/internal/decay"
	"personal/reverie/internal/embed"
	"personal/reverie/internal/manager"
	"personal/reverie/internal/memory"
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
		Name: "memory_decay_tick",
		Description: "Advance the decay clock. Internal tool for session-end hooks and admin use. " +
			"Not typically called by agents. Increments turns_since on all clusters.",
		Annotations: &mcpsdk.ToolAnnotations{
			OpenWorldHint: &openWorld,
			Title:         "Decay tick",
		},
	}, s.handleDecayTick)
}
