package memory

import (
	"context"
	"crypto/sha256"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
	"personal/reverie/pkg/vecmath"
)

// linkRow represents a row in the fact_episode_links table for the in-memory store.
type linkRow struct {
	FactID    string
	EpisodeID string
	LinkType  string
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
	links        []linkRow
	clusters     map[string]ClusterNode
}

// NewMemStore returns a thread-safe in-memory Store.
func NewMemStore() Store {
	return &memStore{
		facts:    make(map[string]Fact),
		episodes: make(map[string]Episode),
		clusters: make(map[string]ClusterNode),
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
	// Cascade-delete links.
	m.deleteLinksByFact(id)
	return nil
}

// deleteLinksByFact removes all link rows referencing the given fact. Must be called with mu held.
func (m *memStore) deleteLinksByFact(factID string) {
	filtered := m.links[:0]
	for _, l := range m.links {
		if l.FactID != factID {
			filtered = append(filtered, l)
		}
	}
	m.links = filtered
}

// deleteLinksByEpisode removes all link rows referencing the given episode. Must be called with mu held.
func (m *memStore) deleteLinksByEpisode(episodeID string) {
	filtered := m.links[:0]
	for _, l := range m.links {
		if l.EpisodeID != episodeID {
			filtered = append(filtered, l)
		}
	}
	m.links = filtered
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

	// Insert cross-links for any linked fact IDs.
	for _, factID := range e.LinkedFactIDs {
		m.addLink(factID, e.ID, "evidence")
	}

	return e.ID, nil
}

func (m *memStore) GetEpisode(_ context.Context, id string) (*Episode, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	ep, ok := m.episodes[id]
	if !ok {
		return nil, nil
	}

	// Load linked fact IDs.
	ep.LinkedFactIDs = nil
	for _, l := range m.links {
		if l.EpisodeID == id {
			ep.LinkedFactIDs = append(ep.LinkedFactIDs, l.FactID)
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
	// Cascade-delete links.
	m.deleteLinksByEpisode(id)
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

// --- Fact <-> Episode cross-type links ---

// addLink adds a link row. Must be called with mu held.
func (m *memStore) addLink(factID, episodeID, linkType string) {
	// Dedup.
	for _, l := range m.links {
		if l.FactID == factID && l.EpisodeID == episodeID {
			return
		}
	}
	m.links = append(m.links, linkRow{
		FactID:    factID,
		EpisodeID: episodeID,
		LinkType:  linkType,
	})
}

func (m *memStore) LinkFactEpisode(_ context.Context, factID, episodeID, linkType string) error {
	if linkType == "" {
		linkType = "evidence"
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.addLink(factID, episodeID, linkType)
	return nil
}

func (m *memStore) GetFactLinks(_ context.Context, factID string) ([]EpisodeLink, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var links []EpisodeLink
	for _, l := range m.links {
		if l.FactID == factID {
			link := EpisodeLink{
				EpisodeID: l.EpisodeID,
				LinkType:  l.LinkType,
			}
			if ep, ok := m.episodes[l.EpisodeID]; ok {
				epc := ep
				link.Episode = &epc
			}
			links = append(links, link)
		}
	}
	return links, nil
}

func (m *memStore) GetEpisodeLinks(_ context.Context, episodeID string) ([]FactLink, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var links []FactLink
	for _, l := range m.links {
		if l.EpisodeID == episodeID {
			link := FactLink{
				FactID:   l.FactID,
				LinkType: l.LinkType,
			}
			if f, ok := m.facts[l.FactID]; ok {
				fc := f
				link.Fact = &fc
			}
			links = append(links, link)
		}
	}
	return links, nil
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

func (m *memStore) Close() error {
	return nil
}
