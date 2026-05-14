package memory

import (
	"context"
	"crypto/sha256"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/diffsec/reverie/pkg/ebbinghaus"
	"github.com/diffsec/reverie/pkg/vecmath"
	"github.com/google/uuid"
)

// edgeRow is the in-memory mirror of a memory_edges row.
type edgeRow struct {
	SrcID     string
	DstID     string
	EdgeType  string
	Weight    float64
	CreatedAt time.Time
}

// entityRow is the in-memory mirror of an entities row.
type entityRow struct {
	ID         string
	Name       string
	EntityType string
	Embedding  []float32
	Utility    float64
	Frequency  float64
	TurnsSince int
	Retention  float64
	LastAccess time.Time
	CreatedAt  time.Time
}

// mentionRow is the in-memory mirror of an entity_mentions row. Role is
// stored verbatim; an empty string is the in-memory equivalent of a SQL
// NULL role.
type mentionRow struct {
	MemoryID string
	EntityID string
	Role     string
}

// memStore is an in-memory Store for use in unit tests of packages that depend
// on memory.Store. It mirrors the semantics of sqliteStore: idempotency on
// content_hash, superseded filtering, sort/limit/offset in ListFacts, and
// brute-force cosine GlobalSearch.
type memStore struct {
	mu    sync.RWMutex
	facts map[string]Fact
	// order preserves insertion order for deterministic iteration.
	order        []string
	episodes     map[string]Episode
	episodeOrder []string
	// Phase 7 knowledge-graph slices replace the old fact_episode_links
	// store. edges holds memory_edges rows; entities holds first-class
	// entity nodes; mentions holds memory->entity references.
	edges    []edgeRow
	entities []entityRow
	mentions []mentionRow
	clusters map[string]ClusterNode
	// lastTick mirrors the sqlite decay_state.last_tick singleton. Zero value
	// = never ticked.
	lastTick time.Time
	// sessions holds persisted session checkpoints (Phase 6b). Keyed by
	// Session.ID; the stored value is the authoritative snapshot.
	sessions map[string]Session
}

// NewMemStore returns a thread-safe in-memory Store.
func NewMemStore() Store {
	return &memStore{
		facts:    make(map[string]Fact),
		episodes: make(map[string]Episode),
		clusters: make(map[string]ClusterNode),
		sessions: make(map[string]Session),
	}
}

func (m *memStore) InsertFact(_ context.Context, f Fact) (string, error) {
	// Normalize + validate tags. Done outside the lock (no shared state) so a
	// rejection doesn't hold the write mutex.
	normTags, err := normalizeTags(f.Tags)
	if err != nil {
		return "", fmt.Errorf("mem store: insert fact: %w", err)
	}
	f.Tags = normTags

	m.mu.Lock()
	defer m.mu.Unlock()

	// Defaults.
	if f.ID == "" {
		f.ID = uuid.New().String()
	}
	if f.ContentHash == "" {
		h := sha256.Sum256([]byte(f.Content))
		f.ContentHash = fmt.Sprintf("%x", h)
	}
	now := time.Now().UTC()
	if f.CreatedAt.IsZero() {
		f.CreatedAt = now
	}
	if f.AccessedAt.IsZero() {
		f.AccessedAt = now
	}
	if f.ValidFrom.IsZero() {
		f.ValidFrom = now
	}
	if f.Confidence == 0 {
		f.Confidence = 1.0
	}
	if f.ClusterID == "" {
		f.ClusterID = "default"
	}

	// Idempotency: check for existing non-superseded fact with same hash.
	for _, existing := range m.facts {
		if existing.ContentHash == f.ContentHash && existing.SupersededBy == nil {
			return existing.ID, nil
		}
	}

	// Ensure the cluster exists (mirrors sqliteStore FK behavior).
	if _, ok := m.clusters[f.ClusterID]; !ok {
		now := time.Now().UTC()
		m.clusters[f.ClusterID] = ClusterNode{
			ID:         f.ClusterID,
			Summary:    f.ClusterID,
			LastAccess: now,
			CreatedAt:  now,
		}
	}

	m.facts[f.ID] = f
	m.order = append(m.order, f.ID)
	return f.ID, nil
}

func (m *memStore) GetFact(_ context.Context, id string) (*Fact, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	f, ok := m.facts[id]
	if !ok {
		return nil, nil
	}
	if f.Tags == nil {
		f.Tags = []string{}
	}
	return &f, nil
}

func (m *memStore) DeleteFact(_ context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.facts, id)
	// Remove from order slice.
	for i, oid := range m.order {
		if oid == id {
			m.order = append(m.order[:i], m.order[i+1:]...)
			break
		}
	}
	// Cascade memory_edges (either direction) and entity_mentions for this
	// memory id. Entities themselves are not cascade-deleted.
	m.cascadeDeleteMemory(id)
	return nil
}

// cascadeDeleteMemory removes any memory_edges row touching id (src or
// dst) and any entity_mentions row whose memory_id matches id. Must be
// called with m.mu held. Entities themselves are intentionally untouched
// (multi-memory references are common; orphaned entities fade through
// decay).
func (m *memStore) cascadeDeleteMemory(id string) {
	keptEdges := m.edges[:0]
	for _, e := range m.edges {
		if e.SrcID == id || e.DstID == id {
			continue
		}
		keptEdges = append(keptEdges, e)
	}
	m.edges = keptEdges

	keptMentions := m.mentions[:0]
	for _, mn := range m.mentions {
		if mn.MemoryID == id {
			continue
		}
		keptMentions = append(keptMentions, mn)
	}
	m.mentions = keptMentions
}

func (m *memStore) ListFacts(_ context.Context, filter ListFilter) ([]Fact, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var results []Fact
	for _, id := range m.order {
		f := m.facts[id]
		// Exclude superseded.
		if f.SupersededBy != nil {
			continue
		}
		// Subtype filter.
		if filter.Subtype != nil && f.Subtype != *filter.Subtype {
			continue
		}
		// TagsAny filter.
		if !tagMatchesAny(f.Tags, filter.TagsAny) {
			continue
		}
		if f.Tags == nil {
			f.Tags = []string{}
		}
		results = append(results, f)
	}

	// Sort.
	switch filter.Sort {
	case "accessed":
		sort.Slice(results, func(i, j int) bool {
			return results[i].AccessedAt.After(results[j].AccessedAt)
		})
	default: // "created" or empty
		sort.Slice(results, func(i, j int) bool {
			return results[i].CreatedAt.After(results[j].CreatedAt)
		})
	}

	// Limit and offset.
	limit := filter.Limit
	if limit <= 0 {
		limit = 25
	}
	if limit > 1000 {
		limit = 1000
	}

	offset := min(filter.Offset, len(results))
	results = results[offset:]
	if len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

func (m *memStore) GlobalSearch(_ context.Context, queryVec []float32, limit int) ([]Candidate, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var candidates []Candidate

	// Scan facts.
	for _, f := range m.facts {
		if f.SupersededBy != nil {
			continue
		}
		if len(f.Embedding) == 0 {
			continue
		}
		sim := vecmath.Cosine(queryVec, f.Embedding)
		fc := f // copy for pointer stability
		candidates = append(candidates, Candidate{Fact: &fc, Similarity: sim})
	}

	// Scan episodes.
	for _, ep := range m.episodes {
		if len(ep.Embedding) == 0 {
			continue
		}
		sim := vecmath.Cosine(queryVec, ep.Embedding)
		epc := ep // copy for pointer stability
		candidates = append(candidates, Candidate{Episode: &epc, Similarity: sim})
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Similarity > candidates[j].Similarity
	})

	if limit > 0 && len(candidates) > limit {
		candidates = candidates[:limit]
	}
	return candidates, nil
}

func (m *memStore) TouchAccessed(_ context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now().UTC()
	idSet := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		idSet[id] = struct{}{}
	}

	for id, f := range m.facts {
		if _, ok := idSet[id]; ok {
			f.AccessedAt = now
			m.facts[id] = f
		}
	}
	for id, ep := range m.episodes {
		if _, ok := idSet[id]; ok {
			ep.AccessedAt = now
			m.episodes[id] = ep
		}
	}
	return nil
}

func (m *memStore) GetCluster(_ context.Context, id string) (*ClusterNode, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	c, ok := m.clusters[id]
	if !ok {
		return nil, nil
	}
	return &c, nil
}

func (m *memStore) ListClusters(_ context.Context) ([]ClusterNode, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	clusters := make([]ClusterNode, 0, len(m.clusters))
	for _, c := range m.clusters {
		clusters = append(clusters, c)
	}
	sort.Slice(clusters, func(i, j int) bool {
		return clusters[i].ID < clusters[j].ID
	})
	return clusters, nil
}

func (m *memStore) UpdateClusterMeta(_ context.Context, id string, summary, domain, metaInstr string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	c, ok := m.clusters[id]
	if !ok {
		return fmt.Errorf("mem store: update cluster meta: cluster %q not found", id)
	}
	c.Summary = summary
	c.Domain = domain
	c.MetaInstr = metaInstr
	m.clusters[id] = c
	return nil
}

func (m *memStore) UpdateClusterState(_ context.Context, id string, utility, frequency float64, turnsSince int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	c, ok := m.clusters[id]
	if !ok {
		return fmt.Errorf("mem store: update cluster state: cluster %q not found", id)
	}
	c.Utility = utility
	c.Frequency = frequency
	c.TurnsSince = turnsSince
	c.LastAccess = time.Now().UTC()
	m.clusters[id] = c
	return nil
}

func (m *memStore) TickAllClusters(_ context.Context, accessedIDs []string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Step 1: increment turns_since for all clusters.
	for id, c := range m.clusters {
		c.TurnsSince++
		m.clusters[id] = c
	}

	// Step 2: reset turns_since=0 and update last_access for accessed clusters.
	if len(accessedIDs) > 0 {
		now := time.Now().UTC()
		for _, id := range accessedIDs {
			if c, ok := m.clusters[id]; ok {
				c.TurnsSince = 0
				c.LastAccess = now
				m.clusters[id] = c
			}
		}
	}

	return nil
}

// --- Episode operations ---

func (m *memStore) InsertEpisode(_ context.Context, e Episode) (string, error) {
	// Normalize + validate tags outside the lock.
	normTags, err := normalizeTags(e.Tags)
	if err != nil {
		return "", fmt.Errorf("mem store: insert episode: %w", err)
	}
	e.Tags = normTags

	m.mu.Lock()
	defer m.mu.Unlock()

	if e.ID == "" {
		e.ID = uuid.New().String()
	}
	if e.ContentHash == "" {
		h := sha256.Sum256([]byte(e.Situation + e.Action + e.Outcome + e.Preemptive))
		e.ContentHash = fmt.Sprintf("%x", h)
	}
	now := time.Now().UTC()
	if e.CreatedAt.IsZero() {
		e.CreatedAt = now
	}
	if e.AccessedAt.IsZero() {
		e.AccessedAt = now
	}
	if e.ClusterID == "" {
		e.ClusterID = "default"
	}

	// Ensure the cluster exists.
	if _, ok := m.clusters[e.ClusterID]; !ok {
		now := time.Now().UTC()
		m.clusters[e.ClusterID] = ClusterNode{
			ID:         e.ClusterID,
			Summary:    e.ClusterID,
			LastAccess: now,
			CreatedAt:  now,
		}
	}

	m.episodes[e.ID] = e
	m.episodeOrder = append(m.episodeOrder, e.ID)

	// Insert evidence edges for any linked fact IDs. Direction is pinned:
	// src=fact, dst=episode, edge_type='evidence'. Mirrors the sqlite
	// store's writeEpisode rewire.
	now2 := time.Now().UTC()
	for _, factID := range e.LinkedFactIDs {
		m.addEdgeIfAbsent(edgeRow{
			SrcID:     factID,
			DstID:     e.ID,
			EdgeType:  "evidence",
			Weight:    1.0,
			CreatedAt: now2,
		})
	}

	return e.ID, nil
}

// addEdgeIfAbsent inserts an edge unless one with the same
// (src,dst,type) tuple already exists. Must be called with m.mu held.
func (m *memStore) addEdgeIfAbsent(e edgeRow) bool {
	for _, existing := range m.edges {
		if existing.SrcID == e.SrcID && existing.DstID == e.DstID && existing.EdgeType == e.EdgeType {
			return false
		}
	}
	m.edges = append(m.edges, e)
	return true
}

func (m *memStore) GetEpisode(_ context.Context, id string) (*Episode, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	ep, ok := m.episodes[id]
	if !ok {
		return nil, nil
	}

	// Load linked fact IDs from the edges slice. Direction matches the
	// InsertEpisode write path: src=factID, dst=episodeID,
	// edge_type='evidence'.
	ep.LinkedFactIDs = nil
	for _, edge := range m.edges {
		if edge.DstID == id && edge.EdgeType == "evidence" {
			ep.LinkedFactIDs = append(ep.LinkedFactIDs, edge.SrcID)
		}
	}

	if ep.Tags == nil {
		ep.Tags = []string{}
	}

	return &ep, nil
}

func (m *memStore) DeleteEpisode(_ context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.episodes, id)
	for i, oid := range m.episodeOrder {
		if oid == id {
			m.episodeOrder = append(m.episodeOrder[:i], m.episodeOrder[i+1:]...)
			break
		}
	}
	// Cascade memory_edges + entity_mentions, same rules as DeleteFact.
	m.cascadeDeleteMemory(id)
	return nil
}

func (m *memStore) ListEpisodes(_ context.Context, filter ListFilter) ([]Episode, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var results []Episode
	for _, id := range m.episodeOrder {
		ep := m.episodes[id]
		if !tagMatchesAny(ep.Tags, filter.TagsAny) {
			continue
		}
		if ep.Tags == nil {
			ep.Tags = []string{}
		}
		results = append(results, ep)
	}

	// Sort.
	switch filter.Sort {
	case "accessed":
		sort.Slice(results, func(i, j int) bool {
			return results[i].AccessedAt.After(results[j].AccessedAt)
		})
	default: // "created" or empty
		sort.Slice(results, func(i, j int) bool {
			return results[i].CreatedAt.After(results[j].CreatedAt)
		})
	}

	// Limit and offset.
	limit := filter.Limit
	if limit <= 0 {
		limit = 25
	}
	if limit > 1000 {
		limit = 1000
	}

	offset := min(filter.Offset, len(results))
	results = results[offset:]
	if len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

// --- Cluster membership (paginated) ---

func (m *memStore) ListFactsByCluster(_ context.Context, clusterID string, limit, offset int) ([]Fact, error) {
	if limit < 0 {
		limit = 0
	}
	if offset < 0 {
		offset = 0
	}
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Collect non-superseded facts in the cluster, preserving insertion order
	// (which equals creation order).
	members := []Fact{}
	for _, id := range m.order {
		f := m.facts[id]
		if f.ClusterID != clusterID {
			continue
		}
		if f.SupersededBy != nil {
			continue
		}
		if f.Tags == nil {
			f.Tags = []string{}
		}
		members = append(members, f)
	}

	// created_at ascending. m.order is insertion order; do a stable sort on
	// CreatedAt to be safe against any non-monotonic inserts.
	sort.SliceStable(members, func(i, j int) bool {
		return members[i].CreatedAt.Before(members[j].CreatedAt)
	})

	off := min(offset, len(members))
	members = members[off:]
	if len(members) > limit {
		members = members[:limit]
	}
	return members, nil
}

func (m *memStore) ListEpisodesByCluster(_ context.Context, clusterID string, limit, offset int) ([]Episode, error) {
	if limit < 0 {
		limit = 0
	}
	if offset < 0 {
		offset = 0
	}
	m.mu.RLock()
	defer m.mu.RUnlock()

	members := []Episode{}
	for _, id := range m.episodeOrder {
		ep := m.episodes[id]
		if ep.ClusterID != clusterID {
			continue
		}
		if ep.Tags == nil {
			ep.Tags = []string{}
		}
		members = append(members, ep)
	}

	sort.SliceStable(members, func(i, j int) bool {
		return members[i].CreatedAt.Before(members[j].CreatedAt)
	})

	off := min(offset, len(members))
	members = members[off:]
	if len(members) > limit {
		members = members[:limit]
	}
	return members, nil
}

func (m *memStore) CountFactsByCluster(_ context.Context, clusterID string) (int, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	n := 0
	for _, id := range m.order {
		f := m.facts[id]
		if f.ClusterID == clusterID && f.SupersededBy == nil {
			n++
		}
	}
	return n, nil
}

func (m *memStore) CountEpisodesByCluster(_ context.Context, clusterID string) (int, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	n := 0
	for _, id := range m.episodeOrder {
		ep := m.episodes[id]
		if ep.ClusterID == clusterID {
			n++
		}
	}
	return n, nil
}

// --- Knowledge graph (Phase 7) ---
//
// In-memory mirror of the sqliteStore KG surface. The slices (edges,
// entities, mentions) live on memStore; helpers operate under m.mu.

// AddEdge inserts a typed directed edge. Idempotent by
// (src,dst,edge_type) — duplicate returns created=false with the
// existing weight untouched.
func (m *memStore) AddEdge(_ context.Context, e Edge) (bool, error) {
	if e.Weight == 0 {
		e.Weight = 1.0
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if e.CreatedAt.IsZero() {
		e.CreatedAt = time.Now().UTC()
	}
	return m.addEdgeIfAbsent(edgeRow{
		SrcID:     e.SrcID,
		DstID:     e.DstID,
		EdgeType:  e.EdgeType,
		Weight:    e.Weight,
		CreatedAt: e.CreatedAt,
	}), nil
}

// RemoveEdge deletes the edge matching (src,dst,edge_type). Missing
// edges return deleted=false (idempotent).
func (m *memStore) RemoveEdge(_ context.Context, srcID, dstID, edgeType string) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for i, e := range m.edges {
		if e.SrcID == srcID && e.DstID == dstID && e.EdgeType == edgeType {
			m.edges = append(m.edges[:i], m.edges[i+1:]...)
			return true, nil
		}
	}
	return false, nil
}

// ListEdges performs an in-memory BFS up to hops levels deep starting
// from memoryID. hops is clamped to [1,3] silently (the tool surface
// validates earlier). Each edge contributes one EdgeWithDistance at the
// depth its non-seed endpoint was first reached.
func (m *memStore) ListEdges(_ context.Context, memoryID string, hops int) ([]EdgeWithDistance, error) {
	if hops < 1 {
		hops = 1
	}
	if hops > 3 {
		hops = 3
	}
	m.mu.RLock()
	defer m.mu.RUnlock()

	visited := map[string]struct{}{memoryID: {}}
	frontier := map[string]struct{}{memoryID: {}}
	var results []EdgeWithDistance

	for depth := 1; depth <= hops; depth++ {
		if len(frontier) == 0 {
			break
		}
		nextFrontier := map[string]struct{}{}
		for _, e := range m.edges {
			_, srcIn := frontier[e.SrcID]
			_, dstIn := frontier[e.DstID]
			if !srcIn && !dstIn {
				continue
			}
			var other string
			if srcIn {
				other = e.DstID
			} else {
				other = e.SrcID
			}
			if _, seen := visited[other]; seen {
				continue
			}
			visited[other] = struct{}{}
			nextFrontier[other] = struct{}{}
			results = append(results, EdgeWithDistance{
				Edge: Edge{
					SrcID:     e.SrcID,
					DstID:     e.DstID,
					EdgeType:  e.EdgeType,
					Weight:    e.Weight,
					CreatedAt: e.CreatedAt,
				},
				Distance: depth,
			})
		}
		frontier = nextFrontier
	}
	return results, nil
}

// UpsertEntity dedups exact then by cosine similarity within the same
// entity_type, mirroring the sqliteStore semantics. The caller-supplied
// embedding is stored verbatim.
func (m *memStore) UpsertEntity(_ context.Context, name, entityType string, embedding []float32) (string, bool, bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Exact dedup.
	for _, ent := range m.entities {
		if ent.Name == name && ent.EntityType == entityType {
			return ent.ID, false, false, nil
		}
	}

	// Similarity dedup within the same entity_type.
	if len(embedding) > 0 {
		var bestID string
		var bestSim float32
		for _, ent := range m.entities {
			if ent.EntityType != entityType {
				continue
			}
			if len(ent.Embedding) == 0 {
				continue
			}
			sim := vecmath.Cosine(embedding, ent.Embedding)
			if sim > bestSim {
				bestSim = sim
				bestID = ent.ID
			}
		}
		if bestID != "" && float64(bestSim) >= defaultEntitySimilarityThreshold {
			return bestID, false, true, nil
		}
	}

	id := uuid.New().String()
	now := time.Now().UTC()
	m.entities = append(m.entities, entityRow{
		ID:         id,
		Name:       name,
		EntityType: entityType,
		Embedding:  embedding,
		Utility:    0.5,
		Frequency:  0.5,
		TurnsSince: 0,
		Retention:  1.0,
		CreatedAt:  now,
		// LastAccess intentionally zero — entity has never been
		// accessed yet; sqlite stores NULL.
	})
	return id, true, false, nil
}

// GetEntity returns the entity with the given id, or zero-value Entity
// if not found (matches GetFact/GetEpisode's not-found convention).
func (m *memStore) GetEntity(_ context.Context, id string) (Entity, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, ent := range m.entities {
		if ent.ID == id {
			return entityRowToEntity(ent), nil
		}
	}
	return Entity{}, nil
}

// entityRowToEntity copies the in-memory row into the exported Entity
// struct.
func entityRowToEntity(r entityRow) Entity {
	return Entity{
		ID:         r.ID,
		Name:       r.Name,
		EntityType: r.EntityType,
		Embedding:  r.Embedding,
		Utility:    r.Utility,
		Frequency:  r.Frequency,
		TurnsSince: r.TurnsSince,
		Retention:  r.Retention,
		LastAccess: r.LastAccess,
		CreatedAt:  r.CreatedAt,
	}
}

// FindSimilarEntities returns entities (optionally filtered by type) whose
// embedding has cosine similarity >= threshold to the query vector. Sorted
// descending by similarity, capped at limit.
func (m *memStore) FindSimilarEntities(_ context.Context, embedding []float32, entityType string, threshold float64, limit int) ([]EntityWithScore, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var scored []EntityWithScore
	for _, ent := range m.entities {
		if entityType != "" && ent.EntityType != entityType {
			continue
		}
		if len(ent.Embedding) == 0 {
			continue
		}
		sim := vecmath.Cosine(embedding, ent.Embedding)
		if float64(sim) >= threshold {
			scored = append(scored, EntityWithScore{
				Entity:     entityRowToEntity(ent),
				Similarity: sim,
			})
		}
	}
	sort.Slice(scored, func(i, j int) bool { return scored[i].Similarity > scored[j].Similarity })
	if limit > 0 && len(scored) > limit {
		scored = scored[:limit]
	}
	return scored, nil
}

// TickAllEntities increments turns_since for every entity, resets
// accessed entities to (turns_since=0, retention=1.0, last_access=now),
// then recomputes retention for every row via the pure formula in
// pkg/ebbinghaus. Same shape as TickAllClusters.
func (m *memStore) TickAllEntities(_ context.Context, accessedIDs []string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	accessedSet := make(map[string]struct{}, len(accessedIDs))
	for _, id := range accessedIDs {
		accessedSet[id] = struct{}{}
	}
	now := time.Now().UTC()
	for i := range m.entities {
		m.entities[i].TurnsSince++
		if _, ok := accessedSet[m.entities[i].ID]; ok {
			m.entities[i].TurnsSince = 0
			m.entities[i].LastAccess = now
		}
		m.entities[i].Retention = ebbinghaus.Retention(
			m.entities[i].TurnsSince,
			m.entities[i].Utility,
			m.entities[i].Frequency,
			ebbinghaus.DefaultTemperature,
		)
	}
	return nil
}

// AddEntityMentions inserts (memory_id, entity_id) rows for every entity
// in entityIDs. Idempotent by (memory_id, entity_id); duplicates return
// inserted=0 for that entity.
func (m *memStore) AddEntityMentions(_ context.Context, memoryID string, entityIDs []string, role string) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	inserted := 0
outer:
	for _, eid := range entityIDs {
		for _, existing := range m.mentions {
			if existing.MemoryID == memoryID && existing.EntityID == eid {
				continue outer
			}
		}
		m.mentions = append(m.mentions, mentionRow{
			MemoryID: memoryID,
			EntityID: eid,
			Role:     role,
		})
		inserted++
	}
	return inserted, nil
}

// ListMemoriesByEntity returns memories mentioning the given entity. Each
// MemoryRef's Layer/Content is resolved against the facts and episodes
// maps; orphaned mentions whose memory id no longer exists are dropped.
func (m *memStore) ListMemoriesByEntity(_ context.Context, entityID string, limit int) ([]MemoryRef, error) {
	if limit <= 0 {
		limit = 25
	}
	m.mu.RLock()
	defer m.mu.RUnlock()

	refs := []MemoryRef{}
	for _, mn := range m.mentions {
		if mn.EntityID != entityID {
			continue
		}
		ref, ok := m.resolveMemoryRef(mn.MemoryID)
		if !ok {
			continue
		}
		refs = append(refs, ref)
		if len(refs) >= limit {
			break
		}
	}
	return refs, nil
}

// resolveMemoryRef returns a populated MemoryRef on hit. Must be called
// with m.mu held (read or write).
func (m *memStore) resolveMemoryRef(id string) (MemoryRef, bool) {
	if f, ok := m.facts[id]; ok {
		return MemoryRef{ID: id, Layer: TypeL2Semantic, Content: truncatePreview(f.Content)}, true
	}
	if ep, ok := m.episodes[id]; ok {
		preview := strings.TrimSpace(ep.Situation + " " + ep.Action + " " + ep.Outcome + " " + ep.Preemptive)
		return MemoryRef{ID: id, Layer: TypeL3Episodic, Content: truncatePreview(preview)}, true
	}
	return MemoryRef{}, false
}

// ListEntityNeighbors walks out from the seed entity. Memories with
// direct mentions of the seed are Distance=1; memory_edges contribute
// further reach up to hops levels deep.
func (m *memStore) ListEntityNeighbors(ctx context.Context, entityID string, hops int) ([]NeighborMemory, []NeighborEntity, error) {
	if hops < 1 {
		hops = 1
	}
	if hops > 3 {
		hops = 3
	}

	var memories []NeighborMemory
	var entities []NeighborEntity
	seenMem := map[string]struct{}{}

	m.mu.RLock()
	// Distance-1 memories from mentions on the seed.
	for _, mn := range m.mentions {
		if mn.EntityID != entityID {
			continue
		}
		ref, ok := m.resolveMemoryRef(mn.MemoryID)
		if !ok {
			continue
		}
		if _, dup := seenMem[mn.MemoryID]; dup {
			continue
		}
		seenMem[mn.MemoryID] = struct{}{}
		memories = append(memories, NeighborMemory{
			ID:             mn.MemoryID,
			Layer:          ref.Layer,
			ContentPreview: ref.Content,
			Distance:       1,
		})
	}
	// Snapshot the data needed for the edge walk while the read lock is
	// held; ListEdges takes the same lock so we release it first.
	entityIndex := make(map[string]entityRow, len(m.entities))
	for _, ent := range m.entities {
		entityIndex[ent.ID] = ent
	}
	m.mu.RUnlock()

	edges, err := m.ListEdges(ctx, entityID, hops)
	if err != nil {
		return nil, nil, err
	}
	seenEnt := map[string]struct{}{entityID: {}}
	for _, ewd := range edges {
		// Determine the "other" endpoint.
		var other string
		if _, ok := seenEnt[ewd.Edge.SrcID]; ok {
			other = ewd.Edge.DstID
		} else if _, ok := seenEnt[ewd.Edge.DstID]; ok {
			other = ewd.Edge.SrcID
		} else if _, isEnt := entityIndex[ewd.Edge.SrcID]; isEnt {
			other = ewd.Edge.DstID
		} else {
			other = ewd.Edge.SrcID
		}
		seenEnt[other] = struct{}{}

		if ent, ok := entityIndex[other]; ok {
			entities = append(entities, NeighborEntity{
				ID:         ent.ID,
				Name:       ent.Name,
				EntityType: ent.EntityType,
				Distance:   ewd.Distance,
			})
			continue
		}
		m.mu.RLock()
		ref, ok := m.resolveMemoryRef(other)
		m.mu.RUnlock()
		if !ok {
			continue
		}
		if _, dup := seenMem[other]; dup {
			continue
		}
		seenMem[other] = struct{}{}
		memories = append(memories, NeighborMemory{
			ID:             other,
			Layer:          ref.Layer,
			ContentPreview: ref.Content,
			Distance:       ewd.Distance,
		})
	}
	return memories, entities, nil
}

// ListEntitiesByMemoryIDs returns the deduped set of entity IDs mentioned
// by any of the supplied memory IDs. Empty/nil input yields an empty
// (non-nil) slice with no error. Implementation is a single linear pass
// over m.mentions plus a set lookup — fine for test-scale stores.
func (m *memStore) ListEntitiesByMemoryIDs(_ context.Context, memoryIDs []string) ([]string, error) {
	if len(memoryIDs) == 0 {
		return []string{}, nil
	}
	wanted := make(map[string]struct{}, len(memoryIDs))
	for _, id := range memoryIDs {
		wanted[id] = struct{}{}
	}
	m.mu.RLock()
	defer m.mu.RUnlock()

	seen := make(map[string]struct{})
	out := make([]string, 0)
	for _, mn := range m.mentions {
		if _, ok := wanted[mn.MemoryID]; !ok {
			continue
		}
		if _, dup := seen[mn.EntityID]; dup {
			continue
		}
		seen[mn.EntityID] = struct{}{}
		out = append(out, mn.EntityID)
	}
	return out, nil
}

// ExpandViaGraph implementation -- see docs/design/phase-7c-graph-aware-recall.md.
//
// Phase 7C: BFS walks memory_edges and entity_mentions outward from
// seedIDs to at most hops levels deep, returning one GraphHit per
// (neighbor, seed) pair at that pair's shortest distance. Hops clamped
// to [1,3] defensively. minRetention<=0 disables the retention
// pre-filter; maxVisited<=0 disables the global visited cap.
func (m *memStore) ExpandViaGraph(_ context.Context, seedIDs []string, hops int, minRetention float64, maxVisited int) ([]GraphHit, error) {
	if hops < 1 {
		hops = 1
	}
	if hops > 3 {
		hops = 3
	}
	if len(seedIDs) == 0 {
		return nil, nil
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	// pair key for (memID, seedID)
	type pairKey struct {
		mem  string
		seed string
	}

	bestPair := make(map[pairKey]int)
	globalVisited := make(map[string]struct{})
	type frontierEntry struct {
		mem  string
		seed string
	}
	var frontier []frontierEntry

	seedSet := make(map[string]struct{}, len(seedIDs))
	for _, s := range seedIDs {
		if _, dup := seedSet[s]; dup {
			continue
		}
		seedSet[s] = struct{}{}
		bestPair[pairKey{s, s}] = 0
		globalVisited[s] = struct{}{}
		frontier = append(frontier, frontierEntry{mem: s, seed: s})
	}

	results := make([]GraphHit, 0)

	// Cluster-retention cache for the duration of this BFS.
	clusterRetention := make(map[string]float64)
	getClusterRetention := func(clusterID string) float64 {
		if r, ok := clusterRetention[clusterID]; ok {
			return r
		}
		c, ok := m.clusters[clusterID]
		if !ok {
			clusterRetention[clusterID] = 0
			return 0
		}
		r := ebbinghaus.Retention(c.TurnsSince, c.Utility, c.Frequency, ebbinghaus.DefaultTemperature)
		clusterRetention[clusterID] = r
		return r
	}

	// resolveMemoryLayer returns ("l2_semantic", clusterID, true) for facts,
	// ("l3_episodic", clusterID, true) for episodes, ("", "", false)
	// otherwise (e.g., entity id or unknown).
	resolveMemoryLayer := func(id string) (string, string, bool) {
		if f, ok := m.facts[id]; ok {
			return string(TypeL2Semantic), f.ClusterID, true
		}
		if ep, ok := m.episodes[id]; ok {
			return string(TypeL3Episodic), ep.ClusterID, true
		}
		return "", "", false
	}

	entityRetentionByID := make(map[string]float64, len(m.entities))
	for _, ent := range m.entities {
		entityRetentionByID[ent.ID] = ent.Retention
	}

	capReached := func() bool {
		return maxVisited > 0 && len(globalVisited) >= maxVisited
	}

	for depth := 1; depth <= hops; depth++ {
		if len(frontier) == 0 {
			break
		}
		if capReached() {
			break
		}
		var nextFrontier []frontierEntry

		// Snapshot per-iteration frontier IDs for fast lookup.
		frontierMems := make(map[string][]string) // memID -> seedIDs reaching this mem at current frontier
		for _, fe := range frontier {
			frontierMems[fe.mem] = append(frontierMems[fe.mem], fe.seed)
		}

		// (a) memory_edges -- one-hop edge neighbors.
		stopOuter := false
		for _, e := range m.edges {
			// Determine which side(s) are in the frontier.
			seedsFromSrc, srcIn := frontierMems[e.SrcID]
			seedsFromDst, dstIn := frontierMems[e.DstID]
			if !srcIn && !dstIn {
				continue
			}
			// Walk pairs: (parent, other)
			tryReach := func(parent, other string, seeds []string) bool {
				// Skip entity destinations (entities are mediators only).
				layer, clusterID, isMem := resolveMemoryLayer(other)
				if !isMem {
					return false
				}
				if other == parent {
					return false
				}
				if minRetention > 0 {
					if r := getClusterRetention(clusterID); r < minRetention {
						return false
					}
				}
				for _, seedID := range seeds {
					if other == seedID {
						continue
					}
					key := pairKey{mem: other, seed: seedID}
					if _, exists := bestPair[key]; exists {
						continue
					}
					bestPair[key] = depth
					results = append(results, GraphHit{
						NeighborID:    other,
						NeighborLayer: layer,
						SeedID:        seedID,
						Distance:      depth,
					})
					nextFrontier = append(nextFrontier, frontierEntry{mem: other, seed: seedID})
					if _, gv := globalVisited[other]; !gv {
						globalVisited[other] = struct{}{}
						if capReached() {
							return true
						}
					}
				}
				return false
			}

			if srcIn {
				if tryReach(e.SrcID, e.DstID, seedsFromSrc) {
					stopOuter = true
					break
				}
			}
			if dstIn {
				if tryReach(e.DstID, e.SrcID, seedsFromDst) {
					stopOuter = true
					break
				}
			}
		}

		if stopOuter {
			break
		}

		// (b) memory->entity->memory -- 2-hop entity mediator. Only
		// produces results if depth+1 <= hops.
		if depth+1 <= hops && !capReached() {
			// Build memory->entityIDs map and entity->memoryIDs map
			// scoped to the current frontier.
			frontierMemSet := frontierMems
			// For each frontier memory, find its entities, then for each
			// entity find its memories.
			memToEntities := make(map[string][]string)
			for _, mn := range m.mentions {
				if _, ok := frontierMemSet[mn.MemoryID]; !ok {
					continue
				}
				if minRetention > 0 {
					if r, ok := entityRetentionByID[mn.EntityID]; ok && r < minRetention {
						continue
					}
				}
				memToEntities[mn.MemoryID] = append(memToEntities[mn.MemoryID], mn.EntityID)
			}
			// For each entity, gather memory_ids that mention it.
			entityToMems := make(map[string][]string)
			neededEntities := make(map[string]struct{})
			for _, eids := range memToEntities {
				for _, eid := range eids {
					neededEntities[eid] = struct{}{}
				}
			}
			for _, mn := range m.mentions {
				if _, want := neededEntities[mn.EntityID]; !want {
					continue
				}
				entityToMems[mn.EntityID] = append(entityToMems[mn.EntityID], mn.MemoryID)
			}

			stop := false
		outerEnt:
			for parentMem, seeds := range frontierMemSet {
				eids := memToEntities[parentMem]
				for _, eid := range eids {
					mids := entityToMems[eid]
					for _, n := range mids {
						if n == parentMem {
							continue
						}
						layer, clusterID, isMem := resolveMemoryLayer(n)
						if !isMem {
							continue
						}
						if minRetention > 0 {
							if r := getClusterRetention(clusterID); r < minRetention {
								continue
							}
						}
						for _, seedID := range seeds {
							if n == seedID {
								continue
							}
							key := pairKey{mem: n, seed: seedID}
							if _, exists := bestPair[key]; exists {
								continue
							}
							bestPair[key] = depth + 1
							results = append(results, GraphHit{
								NeighborID:    n,
								NeighborLayer: layer,
								SeedID:        seedID,
								Distance:      depth + 1,
							})
							if _, gv := globalVisited[n]; !gv {
								globalVisited[n] = struct{}{}
								if capReached() {
									stop = true
									break outerEnt
								}
							}
						}
					}
				}
			}
			if stop {
				break
			}
		}

		frontier = nextFrontier
	}

	return results, nil
}

// CountEntities returns the total number of entities. Mirrors the
// sqliteStore counter; used by reverie://status.
func (m *memStore) CountEntities(_ context.Context) (int, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.entities), nil
}

// CountEdges returns the total number of memory_edges rows.
func (m *memStore) CountEdges(_ context.Context) (int, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.edges), nil
}

// --- Cluster operations (Phase 3 additions) ---

func (m *memStore) CreateCluster(_ context.Context, c ClusterNode) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.clusters[c.ID]; ok {
		return fmt.Errorf("mem store: create cluster: cluster %q already exists", c.ID)
	}

	now := time.Now().UTC()
	if c.CreatedAt.IsZero() {
		c.CreatedAt = now
	}
	if c.LastAccess.IsZero() {
		c.LastAccess = now
	}

	m.clusters[c.ID] = c
	return nil
}

func (m *memStore) UpdateClusterCentroid(_ context.Context, id string, centroid []float32, itemCount int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	c, ok := m.clusters[id]
	if !ok {
		return fmt.Errorf("mem store: update cluster centroid: cluster %q not found", id)
	}
	c.Centroid = centroid
	c.ItemCount = itemCount
	m.clusters[id] = c
	return nil
}

// SetMemoryCluster updates the cluster_id of a fact (first) or an episode
// (fallback) and bumps accessed_at. Errors "memory not found: %s" if neither
// exists.
func (m *memStore) SetMemoryCluster(_ context.Context, memoryID, clusterID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now().UTC()

	if f, ok := m.facts[memoryID]; ok {
		f.ClusterID = clusterID
		f.AccessedAt = now
		m.facts[memoryID] = f
		return nil
	}
	if ep, ok := m.episodes[memoryID]; ok {
		ep.ClusterID = clusterID
		ep.AccessedAt = now
		m.episodes[memoryID] = ep
		return nil
	}
	return fmt.Errorf("memory not found: %s", memoryID)
}

// MoveAllClusterMembers reparents every non-superseded fact and every episode
// whose cluster_id equals sourceClusterID onto targetClusterID. The move is
// guarded by the store's write mutex, so observers outside the lock see
// either the pre-state or the post-state — never a half-move.
func (m *memStore) MoveAllClusterMembers(_ context.Context, sourceClusterID, targetClusterID string) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now().UTC()
	moved := 0

	for id, f := range m.facts {
		if f.ClusterID != sourceClusterID {
			continue
		}
		if f.SupersededBy != nil {
			continue
		}
		f.ClusterID = targetClusterID
		f.AccessedAt = now
		m.facts[id] = f
		moved++
	}

	for id, ep := range m.episodes {
		if ep.ClusterID != sourceClusterID {
			continue
		}
		ep.ClusterID = targetClusterID
		ep.AccessedAt = now
		m.episodes[id] = ep
		moved++
	}

	return moved, nil
}

// DeleteCluster is idempotent — a missing cluster returns nil. Refuses if any
// non-superseded fact or any episode still references the cluster.
func (m *memStore) DeleteCluster(_ context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.clusters[id]; !ok {
		return nil
	}

	factCount := 0
	for _, f := range m.facts {
		if f.ClusterID == id && f.SupersededBy == nil {
			factCount++
		}
	}
	episodeCount := 0
	for _, ep := range m.episodes {
		if ep.ClusterID == id {
			episodeCount++
		}
	}
	if factCount+episodeCount > 0 {
		return fmt.Errorf("cluster not empty: %s (has %d facts, %d episodes)", id, factCount, episodeCount)
	}

	delete(m.clusters, id)
	return nil
}

// --- Embedding update (for reindex) ---

func (m *memStore) UpdateFactEmbedding(_ context.Context, id string, embedding []float32) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	f, ok := m.facts[id]
	if !ok {
		return fmt.Errorf("mem store: update fact embedding: fact %q not found", id)
	}
	f.Embedding = embedding
	m.facts[id] = f
	return nil
}

func (m *memStore) UpdateEpisodeEmbedding(_ context.Context, id string, embedding []float32) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	ep, ok := m.episodes[id]
	if !ok {
		return fmt.Errorf("mem store: update episode embedding: episode %q not found", id)
	}
	ep.Embedding = embedding
	m.episodes[id] = ep
	return nil
}

// --- Content amendments (Phase 2D) ---

// UpdateFactContent updates content, content_hash, embedding, optionally tags,
// and accessed_at. A nil `tags` pointer preserves the existing tag set; a
// non-nil pointer (even to an empty slice) replaces it.
func (m *memStore) UpdateFactContent(_ context.Context, id, content, contentHash string, embedding []float32, tags *[]string) error {
	var normTags []string
	if tags != nil {
		nt, err := normalizeTags(*tags)
		if err != nil {
			return fmt.Errorf("mem store: update fact content: %w", err)
		}
		normTags = nt
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	f, ok := m.facts[id]
	if !ok {
		return fmt.Errorf("mem store: update fact content: fact %q not found", id)
	}
	f.Content = content
	f.ContentHash = contentHash
	f.Embedding = embedding
	f.AccessedAt = time.Now().UTC()
	if tags != nil {
		f.Tags = normTags
	}
	m.facts[id] = f
	return nil
}

// UpdateEpisodeContent amends situation/action/outcome/preemptive, embedding,
// content_hash, and tags on an episode. accessed_at is bumped. ID, cluster_id,
// and created_at are preserved. The handler is responsible for populating
// e.Tags with the existing set when preservation is desired (value semantics
// preclude nil-vs-empty discrimination here).
func (m *memStore) UpdateEpisodeContent(_ context.Context, id string, e Episode) error {
	normTags, err := normalizeTags(e.Tags)
	if err != nil {
		return fmt.Errorf("mem store: update episode content: %w", err)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	existing, ok := m.episodes[id]
	if !ok {
		return fmt.Errorf("mem store: update episode content: episode %q not found", id)
	}
	existing.Situation = e.Situation
	existing.Action = e.Action
	existing.Outcome = e.Outcome
	existing.Preemptive = e.Preemptive
	existing.Embedding = e.Embedding
	existing.ContentHash = e.ContentHash
	existing.Tags = normTags
	existing.AccessedAt = time.Now().UTC()
	m.episodes[id] = existing
	return nil
}

// ReplaceEpisodeLinks clears the evidence edges (src=*, dst=episodeID,
// type='evidence') and reinstalls one row per supplied factID. nil and
// empty slices both clear all evidence edges; callers that want to
// preserve existing links must not call this method.
func (m *memStore) ReplaceEpisodeLinks(_ context.Context, episodeID string, factIDs []string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	keep := m.edges[:0]
	for _, e := range m.edges {
		if e.DstID == episodeID && e.EdgeType == "evidence" {
			continue
		}
		keep = append(keep, e)
	}
	m.edges = keep
	now := time.Now().UTC()
	for _, factID := range factIDs {
		m.addEdgeIfAbsent(edgeRow{
			SrcID:     factID,
			DstID:     episodeID,
			EdgeType:  "evidence",
			Weight:    1.0,
			CreatedAt: now,
		})
	}
	return nil
}

// --- Temporal conflict resolution ---

func (m *memStore) SupersedeFact(_ context.Context, oldID, newID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	f, ok := m.facts[oldID]
	if !ok {
		return fmt.Errorf("mem store: supersede fact: fact %q not found", oldID)
	}
	f.SupersededBy = &newID
	m.facts[oldID] = f
	return nil
}

// ClearFactSuperseded reverses a supersede by clearing the SupersededBy
// pointer. Returns the previous value, or an error describing whether the
// fact was missing or simply not superseded.
func (m *memStore) ClearFactSuperseded(_ context.Context, id string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	f, ok := m.facts[id]
	if !ok {
		return "", fmt.Errorf("fact not found: %s", id)
	}
	if f.SupersededBy == nil || *f.SupersededBy == "" {
		return "", fmt.Errorf("fact is not superseded")
	}
	prev := *f.SupersededBy
	f.SupersededBy = nil
	f.AccessedAt = time.Now().UTC()
	m.facts[id] = f
	return prev, nil
}

func (m *memStore) GetFactSupersedes(_ context.Context, id string) ([]string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	ids := []string{}
	// Iterate order for deterministic results.
	for _, fid := range m.order {
		f := m.facts[fid]
		if f.SupersededBy != nil && *f.SupersededBy == id {
			ids = append(ids, f.ID)
		}
	}
	return ids, nil
}

func (m *memStore) FindSimilarFacts(_ context.Context, subtype string, queryVec []float32, threshold float32, limit int) ([]Candidate, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var candidates []Candidate
	for _, f := range m.facts {
		if f.SupersededBy != nil {
			continue
		}
		if len(f.Embedding) == 0 {
			continue
		}
		if f.Subtype != subtype {
			continue
		}
		sim := vecmath.Cosine(queryVec, f.Embedding)
		if sim >= threshold {
			fc := f // copy for pointer stability
			candidates = append(candidates, Candidate{Fact: &fc, Similarity: sim})
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Similarity > candidates[j].Similarity
	})

	if limit > 0 && len(candidates) > limit {
		candidates = candidates[:limit]
	}
	return candidates, nil
}

// ListDailyStats on the in-memory store always returns an empty slice.
// The daily_stats table in the SQLite store is populated by SQL triggers on
// facts/episodes; memStore has no equivalent trigger machinery, and no test
// that uses memStore needs historical activity telemetry. This method exists
// solely to satisfy the Store interface.
func (m *memStore) ListDailyStats(_ context.Context, _, _ string) ([]DailyStats, error) {
	return []DailyStats{}, nil
}

// GetLastTick returns the last-recorded decay tick time, or zero-value if
// SetLastTick has never been called on this store.
func (m *memStore) GetLastTick(_ context.Context) (time.Time, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.lastTick, nil
}

// SetLastTick records the given tick time. Stored in UTC to match sqliteStore.
func (m *memStore) SetLastTick(_ context.Context, t time.Time) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.lastTick = t.UTC()
	return nil
}

// SupersedeLongestChain walks the in-memory supersede edges to find the
// longest chain. Semantics match sqliteStore: for A->B->C (A superseded by B,
// B superseded by C, C terminal) the returned depth is 3 — every fact in the
// chain is counted, including the current head. A fact with no supersede
// relationships contributes 0. This is O(N*d) where d is the chain depth —
// fine for the in-memory test harness.
func (m *memStore) SupersedeLongestChain(_ context.Context) (int, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	max := 0
	// For each fact that IS superseded, walk forward through superseded_by
	// pointers until we hit the terminal. Count every node including the
	// terminal — that matches the CTE semantics in sqliteStore.
	for _, f := range m.facts {
		if f.SupersededBy == nil {
			continue
		}
		depth := 1 // count the starting (superseded) fact
		visited := map[string]bool{f.ID: true}
		cur := f
		for cur.SupersededBy != nil {
			nextID := *cur.SupersededBy
			if visited[nextID] {
				break // cycle guard
			}
			visited[nextID] = true
			next, ok := m.facts[nextID]
			if !ok {
				break // dangling pointer
			}
			depth++
			cur = next
		}
		if depth > max {
			max = depth
		}
	}
	return max, nil
}

// CountSupersededFacts returns the number of facts with a non-nil
// SupersededBy pointer.
func (m *memStore) CountSupersededFacts(_ context.Context) (int, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	n := 0
	for _, f := range m.facts {
		if f.SupersededBy != nil {
			n++
		}
	}
	return n, nil
}

// --- Session CRUD (Phase 6b) ---

func (m *memStore) GetSession(_ context.Context, id string) (*Session, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	s, ok := m.sessions[id]
	if !ok {
		return nil, nil
	}
	// Return a deep-ish copy so callers mutating the returned Session can't
	// corrupt the store. Buffer + tags slices are the observable mutability
	// risk; clone them.
	out := s
	if s.Tags != nil {
		out.Tags = append([]string{}, s.Tags...)
	}
	if s.WorkingMem.Buffer != nil {
		out.WorkingMem.Buffer = append([]MemoryRef{}, s.WorkingMem.Buffer...)
	}
	if s.ClosedAt != nil {
		closed := *s.ClosedAt
		out.ClosedAt = &closed
	}
	return &out, nil
}

func (m *memStore) CreateSession(_ context.Context, s Session) error {
	if s.ID == "" {
		return fmt.Errorf("mem store: create session: id is required")
	}
	normTags, err := normalizeTags(s.Tags)
	if err != nil {
		return fmt.Errorf("mem store: create session: %w", err)
	}
	s.Tags = normTags

	now := time.Now().UTC()
	if s.CreatedAt.IsZero() {
		s.CreatedAt = now
	}
	if s.UpdatedAt.IsZero() {
		s.UpdatedAt = now
	}
	if s.WorkingMem.Buffer == nil {
		s.WorkingMem.Buffer = []MemoryRef{}
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.sessions[s.ID]; exists {
		return fmt.Errorf("mem store: create session: session already exists: %s", s.ID)
	}
	m.sessions[s.ID] = s
	return nil
}

func (m *memStore) UpdateSessionBuffer(_ context.Context, id string, wm WorkingMemory) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	s, ok := m.sessions[id]
	if !ok {
		return fmt.Errorf("mem store: update session buffer: session %q not found", id)
	}
	// Persist only buffer + budget per the Phase 6a ownership split; other
	// fields (Clusters/InteractionCtx/TaskMeta) are ignored even if the
	// caller supplied them.
	s.WorkingMem = WorkingMemory{
		Buffer:     append([]MemoryRef{}, wm.Buffer...),
		BudgetUsed: wm.BudgetUsed,
		BudgetMax:  wm.BudgetMax,
	}
	if s.WorkingMem.Buffer == nil {
		s.WorkingMem.Buffer = []MemoryRef{}
	}
	s.UpdatedAt = time.Now().UTC()
	m.sessions[id] = s
	return nil
}

func (m *memStore) UpdateSessionMeta(_ context.Context, id string, projectHint string, tags []string) error {
	normTags, err := normalizeTags(tags)
	if err != nil {
		return fmt.Errorf("mem store: update session meta: %w", err)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	s, ok := m.sessions[id]
	if !ok {
		return fmt.Errorf("mem store: update session meta: session %q not found", id)
	}
	s.ProjectHint = projectHint
	s.Tags = normTags
	s.UpdatedAt = time.Now().UTC()
	m.sessions[id] = s
	return nil
}

func (m *memStore) CloseSession(_ context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	s, ok := m.sessions[id]
	if !ok {
		return fmt.Errorf("mem store: close session: session %q not found", id)
	}
	if s.ClosedAt != nil {
		return nil // idempotent
	}
	now := time.Now().UTC()
	s.ClosedAt = &now
	s.UpdatedAt = now
	m.sessions[id] = s
	return nil
}

func (m *memStore) Close() error {
	return nil
}
