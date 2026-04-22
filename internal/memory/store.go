package memory

import (
	"context"
	"time"
)

// Store defines the persistence interface for the reverie memory system.
// It covers L2 fact operations, L3 episode operations, cross-type links,
// cluster management, temporal conflict resolution, and brute-force cosine search.
type Store interface {
	// --- L2 Fact operations ---

	// InsertFact persists a new L2 semantic fact and returns its assigned ID.
	// The fact's Embedding field should already be populated before insertion.
	// If a fact with the same ContentHash already exists and is not superseded,
	// implementations should return the existing ID (idempotency).
	// If fact.ClusterID is empty, the default cluster is used.
	InsertFact(ctx context.Context, f Fact) (id string, err error)

	// GetFact retrieves a single fact by its ID. Returns nil and no error if
	// the fact does not exist.
	GetFact(ctx context.Context, id string) (*Fact, error)

	// DeleteFact removes a fact by ID. Returns no error if the fact does not
	// exist (idempotent delete).
	DeleteFact(ctx context.Context, id string) error

	// ListFacts returns facts matching the given filter criteria, ordered by
	// the filter's Sort field. Use this for browsing/audit via memory_list.
	// Superseded facts are excluded.
	ListFacts(ctx context.Context, filter ListFilter) ([]Fact, error)

	// --- L3 Episode operations ---

	// InsertEpisode persists a new L3 episodic memory and returns its assigned ID.
	// If episode.ClusterID is empty, the default cluster is used.
	InsertEpisode(ctx context.Context, e Episode) (id string, err error)

	// GetEpisode retrieves a single episode by its ID. Returns nil and no error
	// if the episode does not exist.
	GetEpisode(ctx context.Context, id string) (*Episode, error)

	// DeleteEpisode removes an episode by ID. Returns no error if the episode
	// does not exist (idempotent delete). Associated fact_episode_links are
	// cascade-deleted.
	DeleteEpisode(ctx context.Context, id string) error

	// ListEpisodes returns episodes matching the given filter criteria, ordered
	// by the filter's Sort field. The Subtype field of ListFilter is ignored
	// for episodes (episodes do not have subtypes); all other fields apply.
	ListEpisodes(ctx context.Context, filter ListFilter) ([]Episode, error)

	// ListFactsByCluster returns non-superseded facts in the given cluster,
	// ordered by created_at ascending (oldest first — stable for pagination).
	// limit/offset control the page window. A negative limit is clamped to 0.
	ListFactsByCluster(ctx context.Context, clusterID string, limit, offset int) ([]Fact, error)

	// ListEpisodesByCluster returns episodes in the given cluster, ordered by
	// created_at ascending (oldest first — stable for pagination). limit/offset
	// control the page window. A negative limit is clamped to 0.
	ListEpisodesByCluster(ctx context.Context, clusterID string, limit, offset int) ([]Episode, error)

	// CountFactsByCluster returns the total number of non-superseded facts in
	// the given cluster (used for paginated membership responses).
	CountFactsByCluster(ctx context.Context, clusterID string) (int, error)

	// CountEpisodesByCluster returns the total number of episodes in the given
	// cluster (used for paginated membership responses).
	CountEpisodesByCluster(ctx context.Context, clusterID string) (int, error)

	// --- Fact <-> Episode cross-type links ---

	// LinkFactEpisode creates a link between a fact and an episode with the given
	// link type (e.g. "evidence"). Duplicate links are silently ignored.
	// Returns created=true when a new row was inserted, created=false when the
	// link already existed (idempotent no-op).
	LinkFactEpisode(ctx context.Context, factID, episodeID, linkType string) (created bool, err error)

	// UnlinkFactEpisode removes the link between a fact and an episode.
	// Returns deleted=true when a row was removed, deleted=false when no such
	// link existed (idempotent).
	UnlinkFactEpisode(ctx context.Context, factID, episodeID string) (deleted bool, err error)

	// GetFactLinks returns the episodes linked to the given fact, eager-loaded
	// via a single JOIN query.
	GetFactLinks(ctx context.Context, factID string) ([]EpisodeLink, error)

	// GetEpisodeLinks returns the facts linked to the given episode, eager-loaded
	// via a single JOIN query.
	GetEpisodeLinks(ctx context.Context, episodeID string) ([]FactLink, error)

	// --- Search ---

	// GlobalSearch performs a brute-force cosine similarity search across all
	// L2 facts and L3 episodes. It returns the top `limit` candidates
	// ranked by descending similarity to queryVec. Superseded facts are excluded.
	GlobalSearch(ctx context.Context, queryVec []float32, limit int) ([]Candidate, error)

	// TouchAccessed updates the accessed_at timestamp for the given memory IDs
	// to the current time. Works for both facts and episodes.
	TouchAccessed(ctx context.Context, ids []string) error

	// --- Cluster operations ---

	// GetCluster returns the cluster with the given id, or (nil, nil) if not found.
	GetCluster(ctx context.Context, id string) (*ClusterNode, error)

	// ListClusters returns all clusters ordered by id. On a fresh store with no
	// facts inserted, this returns an empty slice.
	ListClusters(ctx context.Context) ([]ClusterNode, error)

	// CreateCluster persists a new cluster node. Returns an error if a cluster
	// with the same ID already exists.
	CreateCluster(ctx context.Context, c ClusterNode) error

	// UpdateClusterCentroid updates the centroid vector and item count for the
	// cluster with the given ID. Returns an error if the cluster does not exist.
	UpdateClusterCentroid(ctx context.Context, id string, centroid []float32, itemCount int) error

	// UpdateClusterMeta sets the summary, domain, and meta_instr fields for a cluster.
	// Returns an error if the cluster does not exist.
	UpdateClusterMeta(ctx context.Context, id string, summary, domain, metaInstr string) error

	// UpdateClusterState writes the utility, frequency, and turnsSince fields for
	// the cluster atomically. LastAccess is updated to time.Now().UTC().
	// Returns an error if the cluster does not exist.
	UpdateClusterState(ctx context.Context, id string, utility, frequency float64, turnsSince int) error

	// SetMemoryCluster updates the cluster_id of the fact or episode identified
	// by memoryID. Implementations try the fact table first, then the episode
	// table. Returns an error ("memory not found: <id>") if neither row exists.
	// accessed_at is bumped to time.Now().UTC() as part of the same write —
	// reassignment is a touch.
	SetMemoryCluster(ctx context.Context, memoryID, clusterID string) error

	// DeleteCluster removes the cluster row with the given id. Returns nil
	// (idempotent) if the cluster does not exist. Returns an error
	// ("cluster not empty") if any non-superseded fact or any episode still
	// references the cluster — the caller must move members first.
	DeleteCluster(ctx context.Context, id string) error

	// MoveAllClusterMembers reparents every non-superseded fact and every
	// episode currently assigned to sourceClusterID into targetClusterID in a
	// single logical operation and returns the total number of rows moved
	// (facts + episodes). Rows' accessed_at is bumped to now() as part of the
	// write. Returns 0 with no error if the source cluster has no members.
	// Implementations should perform this atomically (single transaction for
	// durable stores) so partial moves cannot be observed.
	MoveAllClusterMembers(ctx context.Context, sourceClusterID, targetClusterID string) (moved int, err error)

	// TickAllClusters increments turns_since by 1 for all clusters, then sets
	// turns_since=0 for the clusters named in accessedIDs. last_access is
	// updated to time.Now().UTC() for the accessed ones. Single transaction.
	TickAllClusters(ctx context.Context, accessedIDs []string) error

	// --- Embedding update (for reindex) ---

	// UpdateFactEmbedding replaces the embedding vector for a fact.
	// Used by reindex after switching embedding models.
	UpdateFactEmbedding(ctx context.Context, id string, embedding []float32) error

	// UpdateEpisodeEmbedding replaces the embedding vector for an episode.
	// Used by reindex after switching embedding models.
	UpdateEpisodeEmbedding(ctx context.Context, id string, embedding []float32) error

	// UpdateFactContent amends an L2 fact's content, content_hash, embedding,
	// and (optionally) tags in place. accessed_at is bumped to
	// time.Now().UTC(). ID, cluster_id, created_at, valid_from, source,
	// confidence, subtype, and superseded_by are preserved.
	//
	// Tags semantics: a nil `tags` pointer preserves the existing tag set; a
	// non-nil pointer replaces it — an empty (but non-nil) slice clears tags.
	// Implementations normalize the replacement slice via normalizeTags.
	//
	// Returns an error if the fact does not exist.
	UpdateFactContent(ctx context.Context, id, content, contentHash string, embedding []float32, tags *[]string) error

	// UpdateEpisodeContent amends an L3 episode's situation/action/outcome/
	// preemptive fields, embedding, content_hash, and (optionally) tags in
	// place. accessed_at is bumped to time.Now().UTC(). ID, cluster_id, and
	// created_at are preserved. Cross-type links are NOT touched here — the
	// caller uses ReplaceEpisodeLinks for that.
	//
	// Tags semantics: the caller signals the tri-state via e.Tags. Because Go
	// passes slices as (ptr, len, cap) the store cannot distinguish nil from
	// empty on its own; the handler is responsible for leaving e.Tags as the
	// existing tag set when preservation is desired, and for supplying a
	// (possibly empty) slice when replacement is desired. Implementations
	// normalize e.Tags via normalizeTags before writing.
	UpdateEpisodeContent(ctx context.Context, id string, e Episode) error

	// ReplaceEpisodeLinks deletes all fact_episode_links for the given
	// episode and then inserts one row per factID. Callers pass an empty
	// slice to clear links; nil is treated the same as empty (no rows).
	// The handler is responsible for the "nil means preserve" convention —
	// it should not call this method at all when no link change is wanted.
	ReplaceEpisodeLinks(ctx context.Context, episodeID string, factIDs []string) error

	// --- Temporal conflict resolution ---

	// SupersedeFact sets the superseded_by field of the old fact to point to the
	// new fact. Returns an error if the old fact does not exist.
	SupersedeFact(ctx context.Context, oldID, newID string) error

	// ClearFactSuperseded reverses a supersede by clearing the superseded_by
	// column on the given fact. Returns the previous superseded_by value on
	// success. accessed_at is bumped to time.Now().UTC() as part of the write.
	//
	// Error cases, distinguished by error message so handlers can react:
	//   - "fact not found: <id>" when no row exists for the given id.
	//   - "fact is not superseded" when the row exists but superseded_by is
	//     already NULL. The caller should treat this as operator confusion
	//     rather than a generic not-found.
	ClearFactSuperseded(ctx context.Context, id string) (previouslySupersededBy string, err error)

	// GetFactSupersedes returns the IDs of facts whose superseded_by equals the
	// given id. In other words: the history of facts that this fact replaced.
	// Returns an empty slice (not nil) when the fact supersedes nothing.
	GetFactSupersedes(ctx context.Context, id string) ([]string, error)

	// FindSimilarFacts returns non-superseded facts of the given subtype whose
	// embedding has cosine similarity >= threshold to queryVec. Results are
	// ordered by descending similarity and capped at limit.
	FindSimilarFacts(ctx context.Context, subtype string, queryVec []float32, threshold float32, limit int) ([]Candidate, error)

	// --- Observability (Phase 5C) ---

	// ListDailyStats returns the per-day activity rows inside the inclusive
	// [from, to] range (YYYY-MM-DD, UTC). Rows are sorted ascending by date.
	// The returned slice contains only dates for which a daily_stats row
	// exists — callers that need zero-filled gaps must expand the range
	// themselves. Implementations without trigger-driven stats (e.g. the
	// in-memory test store) return an empty slice with no error.
	ListDailyStats(ctx context.Context, from, to string) ([]DailyStats, error)

	// GetLastTick returns the timestamp of the last successful decay tick.
	// Returns a zero-value time.Time (check via .IsZero()) when no tick has
	// ever run — the decay_state row is seeded with NULL by migration 3.
	GetLastTick(ctx context.Context) (time.Time, error)

	// SetLastTick persists the timestamp of the most recent decay tick. The
	// store writes it as ISO8601 UTC (RFC3339) into decay_state.last_tick
	// WHERE id=1. Callers pass time.Now().UTC() after a successful tick.
	SetLastTick(ctx context.Context, t time.Time) error

	// SupersedeLongestChain returns the length of the longest supersede
	// chain in the facts table. A solitary fact (no supersede relationship)
	// contributes a chain of length 0 — only chains with at least one
	// superseded_by edge are counted. Implementations should apply a short
	// internal timeout (per Phase 5A spec: 100ms) and return
	// context.DeadlineExceeded if the computation would exceed it, so the
	// status handler can degrade gracefully.
	SupersedeLongestChain(ctx context.Context) (int, error)

	// CountSupersededFacts returns the number of facts whose superseded_by
	// column is non-NULL. Used by reverie://status to report the total
	// superseded population.
	CountSupersededFacts(ctx context.Context) (int, error)

	// --- Session CRUD (Phase 6b) ---

	// GetSession returns the session by id, or (nil, nil) if not found.
	GetSession(ctx context.Context, id string) (*Session, error)

	// CreateSession inserts a new session row. Returns an error if a session
	// with the same id already exists. Implementations normalize tags with
	// the same normalizeTags helper used by facts/episodes.
	CreateSession(ctx context.Context, s Session) error

	// UpdateSessionBuffer serializes the session's WorkingMemory (buffer +
	// budget only — clusters/interaction/taskmeta are ignored per the
	// Phase 6a ownership split) to the working_memory column and bumps
	// updated_at. Returns an error if the session does not exist.
	UpdateSessionBuffer(ctx context.Context, id string, wm WorkingMemory) error

	// UpdateSessionMeta replaces the project_hint and tags fields for the
	// session. Implementations normalize tags via normalizeTags. Returns an
	// error if the session does not exist.
	UpdateSessionMeta(ctx context.Context, id string, projectHint string, tags []string) error

	// CloseSession sets closed_at to the current time on the session.
	// Calling CloseSession on an already-closed session is a no-op (returns
	// nil) — idempotency matches the rest of the store's delete/close
	// conventions. Returns an error if the session does not exist.
	CloseSession(ctx context.Context, id string) error

	// Close releases any resources held by the store (e.g., database connections).
	Close() error
}

// RetentionBucket is a single bar in a retention histogram: the half-open
// range [Min, Max) along the retention axis (0..1) and the count of clusters
// whose retention falls inside it. The last bucket in a histogram is
// inclusive of its Max so retention=1.0 lands somewhere.
type RetentionBucket struct {
	Min   float64 `json:"min"`
	Max   float64 `json:"max"`
	Count int     `json:"count"`
}

// DailyStats is a single row of the daily_stats table — per-day counters for
// memory activity. Fields mirror the table columns. The SQLite store
// maintains these via triggers on facts/episodes; see migration 3.
type DailyStats struct {
	Date        string `json:"date"` // "YYYY-MM-DD" in UTC
	FactsIn     int    `json:"facts_in"`
	FactsOut    int    `json:"facts_out"`
	EpisodesIn  int    `json:"episodes_in"`
	EpisodesOut int    `json:"episodes_out"`
	Supersedes  int    `json:"supersedes"`
}

// ListFilter specifies criteria for listing facts and episodes.
type ListFilter struct {
	// Subtype filters by the auto-memory taxonomy classification.
	// If nil, all subtypes are returned. Ignored for episode listings
	// (episodes do not have subtypes).
	Subtype *string `json:"subtype"`

	// Limit caps the number of results. Zero means implementation default.
	Limit int `json:"limit"`

	// Offset skips the first N results (for pagination).
	Offset int `json:"offset"`

	// Sort determines ordering: "created" (by created_at) or "accessed" (by accessed_at).
	// Empty string defaults to "created".
	Sort string `json:"sort"`

	// TagsAny filters to memories containing at least one of these tags.
	// Nil or empty means no tag filter. Values are normalized (lowercased and
	// trimmed) before matching; an all-empty slice therefore matches nothing
	// and is treated as "no filter".
	TagsAny []string `json:"tags_any"`
}
