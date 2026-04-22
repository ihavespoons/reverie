// Package memory defines the core data types and interfaces for the reverie
// memory system. It implements a three-layer memory model based on the Oblivion
// paper: L1 clusters (procedural), L2 semantic facts, and L3 episodic memories.
package memory

import (
	"strings"
	"time"
)

// MemoryType identifies which layer a memory belongs to.
type MemoryType string

const (
	TypeL1Cluster  MemoryType = "l1_cluster"
	TypeL2Semantic MemoryType = "l2_semantic"
	TypeL3Episodic MemoryType = "l3_episodic"
)

// ClusterNode represents an L1 procedural memory cluster. Clusters group
// related facts and episodes under a shared centroid, summary, and
// meta-instruction. They are always-resident in working memory.
type ClusterNode struct {
	ID         string    `json:"id"`
	Summary    string    `json:"summary"`
	Domain     string    `json:"domain"`
	MetaInstr  string    `json:"meta_instr"`
	ItemCount  int       `json:"item_count"`
	Centroid   []float32 `json:"centroid"`
	Utility    float64   `json:"utility"`     // U_t(c) in [0,1]
	Frequency  float64   `json:"frequency"`   // F_t(c) in [0,1]
	TurnsSince int       `json:"turns_since"` // n_t(c)
	LastAccess time.Time `json:"last_access"`
	CreatedAt  time.Time `json:"created_at"`
}

// Fact represents an L2 semantic memory — a timestamped piece of knowledge
// with optional temporal drift tracking via SupersededBy.
type Fact struct {
	ID           string    `json:"id"`
	ClusterID    string    `json:"cluster_id"`
	Content      string    `json:"content"`
	ContentHash  string    `json:"content_hash"`
	Source       string    `json:"source"`
	Embedding    []float32 `json:"embedding"`
	Confidence   float64   `json:"confidence"`
	ValidFrom    time.Time `json:"valid_from"`
	SupersededBy *string   `json:"superseded_by"` // points to newer fact if drifted
	CreatedAt    time.Time `json:"created_at"`
	AccessedAt   time.Time `json:"accessed_at"`
	// Subtype carries the auto-memory taxonomy classification.
	// Valid values: "user", "feedback", "project", "reference".
	Subtype string `json:"subtype"`
	// Tags is an optional set of categorization labels. The store normalizes
	// (lowercases/trims/dedupes/sorts) and validates (<=16 tags, <=32 chars
	// each) on insert. Reads always return a non-nil slice, empty when no
	// tags are stored.
	Tags []string `json:"tags"`
}

// Episode represents an L3 episodic memory — a structured tuple of
// situation/action/outcome/preemptive, optionally linked to L2 facts.
type Episode struct {
	ID            string    `json:"id"`
	ClusterID     string    `json:"cluster_id"`
	Situation     string    `json:"situation"`
	Action        string    `json:"action"`
	Outcome       string    `json:"outcome"`
	Preemptive    string    `json:"preemptive"`
	ContentHash   string    `json:"content_hash"`
	Embedding     []float32 `json:"embedding"`
	LinkedFactIDs []string  `json:"linked_fact_ids"`
	CreatedAt     time.Time `json:"created_at"`
	AccessedAt    time.Time `json:"accessed_at"`
	// Tags is an optional set of categorization labels, normalized and
	// validated at write time. Same rules as Fact.Tags.
	Tags []string `json:"tags"`
}

// WorkingMemory is the bounded in-session cache. It holds a sliding window
// of recent turns, all L1 clusters (always-resident), and a dynamic buffer
// of L2/L3 references under a configurable budget.
type WorkingMemory struct {
	InteractionCtx []Turn        `json:"interaction_ctx"` // K recent raw turns (ring-buffered)
	Clusters       []ClusterNode `json:"clusters"`        // ALL L1 — permanent residents
	Buffer         []MemoryRef   `json:"buffer"`          // dynamic L2/L3 refs under budget
	TaskMeta       *TaskMeta     `json:"task_meta"`
	BudgetUsed     int           `json:"budget_used"` // item-count budget
	BudgetMax      int           `json:"budget_max"`  // item-count budget ceiling
}

// MemoryRef is a lightweight reference to a memory in the working memory buffer.
// It points to either an L2 fact or L3 episode by ID and layer.
type MemoryRef struct {
	ID      string     `json:"id"`
	Layer   MemoryType `json:"layer"`
	Score   float64    `json:"score"`   // composite relevance score
	Content string     `json:"content"` // denormalized for display without re-fetch
}

// Turn represents a single interaction turn in the sliding window.
type Turn struct {
	Role      string    `json:"role"`    // "user" or "assistant"
	Content   string    `json:"content"` // raw text
	Timestamp time.Time `json:"timestamp"`
}

// TaskMeta carries metadata about the current task context, used by the
// executor to scope recalls and inform write-path classification.
type TaskMeta struct {
	ProjectHint string   `json:"project_hint"`
	SessionID   string   `json:"session_id"`
	Tags        []string `json:"tags"`
}

// SubQuery represents a decomposed recall query. The caller is responsible
// for query decomposition — this type is used to pass structured sub-queries
// to the recall pipeline.
type SubQuery struct {
	Text   string  `json:"text"`
	Weight float64 `json:"weight"` // relative importance [0,1]
}

// RetentionScore holds the computed retention value for a memory along with
// the inputs used to derive it (for transparency in recall results).
type RetentionScore struct {
	Value      float64 `json:"value"`       // R_t(c) = exp(-n/S)
	TurnsSince int     `json:"turns_since"` // n
	Utility    float64 `json:"utility"`     // U_t(c)
	Frequency  float64 `json:"frequency"`   // F_t(c)
}

// Candidate is a search result that may be a fact (L2) or an episode (L3).
// Exactly one of Fact or Episode is non-nil.
type Candidate struct {
	Fact       *Fact    `json:"fact,omitempty"`
	Episode    *Episode `json:"episode,omitempty"`
	Similarity float32  `json:"similarity"`
}

// ID returns the underlying memory's id.
func (c Candidate) ID() string {
	if c.Fact != nil {
		return c.Fact.ID
	}
	if c.Episode != nil {
		return c.Episode.ID
	}
	return ""
}

// Content returns the displayable content. For episodes, joins
// situation/action/outcome/preemptive with newlines.
func (c Candidate) Content() string {
	if c.Fact != nil {
		return c.Fact.Content
	}
	if c.Episode != nil {
		parts := make([]string, 0, 4)
		if c.Episode.Situation != "" {
			parts = append(parts, "Situation: "+c.Episode.Situation)
		}
		if c.Episode.Action != "" {
			parts = append(parts, "Action: "+c.Episode.Action)
		}
		if c.Episode.Outcome != "" {
			parts = append(parts, "Outcome: "+c.Episode.Outcome)
		}
		if c.Episode.Preemptive != "" {
			parts = append(parts, "Preemptive: "+c.Episode.Preemptive)
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

// Layer returns TypeL2Semantic for facts, TypeL3Episodic for episodes.
func (c Candidate) Layer() MemoryType {
	if c.Fact != nil {
		return TypeL2Semantic
	}
	if c.Episode != nil {
		return TypeL3Episodic
	}
	return ""
}

// ClusterID returns the cluster id (works for both fact and episode).
func (c Candidate) ClusterID() string {
	if c.Fact != nil {
		return c.Fact.ClusterID
	}
	if c.Episode != nil {
		return c.Episode.ClusterID
	}
	return ""
}

// Embedding returns the vector (nil if unset).
func (c Candidate) Embedding() []float32 {
	if c.Fact != nil {
		return c.Fact.Embedding
	}
	if c.Episode != nil {
		return c.Episode.Embedding
	}
	return nil
}

// AccessedAt returns the last-access time.
func (c Candidate) AccessedAt() time.Time {
	if c.Fact != nil {
		return c.Fact.AccessedAt
	}
	if c.Episode != nil {
		return c.Episode.AccessedAt
	}
	return time.Time{}
}

// Session is the persisted working-memory checkpoint for a named session.
// ID is the client-generated identifier; ProjectHint and Tags mirror the
// TaskMeta fields; WorkingMem carries the buffer + budget metadata (the
// JSON blob persisted to sessions.working_memory only includes buffer +
// budget — cluster/interaction/task metadata is owned elsewhere per the
// Phase 6a ownership split). ClosedAt is nil for active sessions; once set,
// the session is read-only and subsequent tool calls using this SessionID
// must error.
type Session struct {
	ID          string        `json:"id"`
	ProjectHint string        `json:"project_hint"`
	Tags        []string      `json:"tags"`
	WorkingMem  WorkingMemory `json:"working_mem"`
	CreatedAt   time.Time     `json:"created_at"`
	UpdatedAt   time.Time     `json:"updated_at"`
	ClosedAt    *time.Time    `json:"closed_at,omitempty"`
}

// EpisodeLink describes a cross-type link from a fact to an episode.
type EpisodeLink struct {
	EpisodeID string   `json:"episode_id"`
	LinkType  string   `json:"link_type"`
	Episode   *Episode `json:"episode,omitempty"` // eager-loaded for convenience
}

// FactLink describes a cross-type link from an episode to a fact.
type FactLink struct {
	FactID   string `json:"fact_id"`
	LinkType string `json:"link_type"`
	Fact     *Fact  `json:"fact,omitempty"` // eager-loaded for convenience
}
