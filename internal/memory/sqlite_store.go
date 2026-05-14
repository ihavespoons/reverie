package memory

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/diffsec/reverie/pkg/ebbinghaus"
	"github.com/diffsec/reverie/pkg/vecmath"
	"github.com/google/uuid"
)

const (
	defaultClusterID      = "default"
	defaultClusterSummary = "default"
	defaultLimit          = 25
	maxLimit              = 1000
	timeFormat            = time.RFC3339
)

// sqliteStore implements the Store interface backed by a SQLite database.
type sqliteStore struct {
	db *sql.DB
}

// NewSQLiteStore returns a Store backed by the given SQLite database.
// The database should already have the schema applied (via db.Open).
func NewSQLiteStore(db *sql.DB) Store {
	return &sqliteStore{db: db}
}

// ensureDefaultCluster inserts the default cluster if it does not exist.
func (s *sqliteStore) ensureDefaultCluster(ctx context.Context) error {
	_, err := s.db.ExecContext(ctx,
		`INSERT OR IGNORE INTO clusters (id, summary) VALUES (?, ?)`,
		defaultClusterID, defaultClusterSummary,
	)
	if err != nil {
		return fmt.Errorf("sqlite store: ensure default cluster: %w", err)
	}
	return nil
}

func (s *sqliteStore) InsertFact(ctx context.Context, f Fact) (string, error) {
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
		f.ClusterID = defaultClusterID
	}

	// Normalize + validate tags up front so we reject bad input before any
	// side effects (including the idempotency lookup, though that's read-only).
	normTags, err := normalizeTags(f.Tags)
	if err != nil {
		return "", fmt.Errorf("sqlite store: insert fact: %w", err)
	}
	f.Tags = normTags
	tagsJSON, err := encodeTags(normTags)
	if err != nil {
		return "", fmt.Errorf("sqlite store: insert fact: %w", err)
	}

	// Idempotency: check for existing non-superseded fact with same hash.
	var existingID string
	err = s.db.QueryRowContext(ctx,
		`SELECT id FROM facts WHERE content_hash = ? AND superseded_by IS NULL LIMIT 1`,
		f.ContentHash,
	).Scan(&existingID)
	if err == nil {
		return existingID, nil
	}
	if err != sql.ErrNoRows {
		return "", fmt.Errorf("sqlite store: insert fact: check hash: %w", err)
	}

	// Ensure the cluster exists (FK constraint).
	if f.ClusterID == defaultClusterID {
		if err := s.ensureDefaultCluster(ctx); err != nil {
			return "", err
		}
	}

	embBlob := EncodeVector(f.Embedding)

	_, err = s.db.ExecContext(ctx,
		`INSERT INTO facts (id, cluster_id, content, embedding, content_hash, subtype, source, confidence, valid_from, superseded_by, created_at, accessed_at, tags)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		f.ID,
		f.ClusterID,
		f.Content,
		embBlob,
		f.ContentHash,
		nullableString(f.Subtype),
		f.Source,
		f.Confidence,
		f.ValidFrom.Format(timeFormat),
		f.SupersededBy,
		f.CreatedAt.Format(timeFormat),
		f.AccessedAt.Format(timeFormat),
		tagsJSON,
	)
	if err != nil {
		return "", fmt.Errorf("sqlite store: insert fact: %w", err)
	}
	return f.ID, nil
}

func (s *sqliteStore) GetFact(ctx context.Context, id string) (*Fact, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, cluster_id, content, embedding, content_hash, subtype, source, confidence, valid_from, superseded_by, created_at, accessed_at, tags
		 FROM facts WHERE id = ?`, id,
	)
	f, err := scanFact(row)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("sqlite store: get fact: %w", err)
	}
	return f, nil
}

func (s *sqliteStore) DeleteFact(ctx context.Context, id string) error {
	// Cascade: memory_edges and entity_mentions reference fact IDs by
	// polymorphic ID (no FK in the schema), so the application is on the
	// hook for clearing them. Wrap the three deletes in a single tx so a
	// failure mid-cascade can't leave dangling rows.
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite store: delete fact: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck
	if _, err := tx.ExecContext(ctx,
		`DELETE FROM memory_edges WHERE src_id = ? OR dst_id = ?`, id, id,
	); err != nil {
		return fmt.Errorf("sqlite store: delete fact: cascade edges: %w", err)
	}
	if _, err := tx.ExecContext(ctx,
		`DELETE FROM entity_mentions WHERE memory_id = ?`, id,
	); err != nil {
		return fmt.Errorf("sqlite store: delete fact: cascade mentions: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM facts WHERE id = ?`, id); err != nil {
		return fmt.Errorf("sqlite store: delete fact: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("sqlite store: delete fact: commit: %w", err)
	}
	return nil
}

func (s *sqliteStore) ListFacts(ctx context.Context, filter ListFilter) ([]Fact, error) {
	limit := filter.Limit
	if limit <= 0 {
		limit = defaultLimit
	}
	if limit > maxLimit {
		limit = maxLimit
	}

	orderCol := "created_at"
	if filter.Sort == "accessed" {
		orderCol = "accessed_at"
	}

	query := `SELECT id, cluster_id, content, embedding, content_hash, subtype, source, confidence, valid_from, superseded_by, created_at, accessed_at, tags
	          FROM facts WHERE superseded_by IS NULL`
	args := []any{}

	if filter.Subtype != nil {
		query += ` AND subtype = ?`
		args = append(args, *filter.Subtype)
	}

	// TagsAny is applied Go-side: the tags column is a JSON TEXT blob, which
	// keeps the schema free of JSON1-extension dependencies. The filter is
	// cheap relative to a fact's embedding cost.
	query += fmt.Sprintf(` ORDER BY %s DESC LIMIT ? OFFSET ?`, orderCol)
	args = append(args, limit, filter.Offset)

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: list facts: %w", err)
	}
	defer rows.Close()

	var facts []Fact
	for rows.Next() {
		f, err := scanFactRows(rows)
		if err != nil {
			return nil, fmt.Errorf("sqlite store: list facts: scan: %w", err)
		}
		if !tagMatchesAny(f.Tags, filter.TagsAny) {
			continue
		}
		facts = append(facts, *f)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite store: list facts: rows: %w", err)
	}
	return facts, nil
}

func (s *sqliteStore) GlobalSearch(ctx context.Context, queryVec []float32, limit int) ([]Candidate, error) {
	// Scan facts.
	factRows, err := s.db.QueryContext(ctx,
		`SELECT id, cluster_id, content, embedding, content_hash, subtype, source, confidence, valid_from, superseded_by, created_at, accessed_at, tags
		 FROM facts WHERE superseded_by IS NULL AND embedding IS NOT NULL`,
	)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: global search: facts: %w", err)
	}
	defer factRows.Close()

	var candidates []Candidate
	for factRows.Next() {
		f, err := scanFactRows(factRows)
		if err != nil {
			return nil, fmt.Errorf("sqlite store: global search: scan fact: %w", err)
		}
		sim := vecmath.Cosine(queryVec, f.Embedding)
		candidates = append(candidates, Candidate{Fact: f, Similarity: sim})
	}
	if err := factRows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite store: global search: fact rows: %w", err)
	}

	// Scan episodes.
	epRows, err := s.db.QueryContext(ctx,
		`SELECT id, cluster_id, situation, action, outcome, preemptive, embedding, content_hash, created_at, accessed_at, tags
		 FROM episodes WHERE embedding IS NOT NULL`,
	)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: global search: episodes: %w", err)
	}
	defer epRows.Close()

	for epRows.Next() {
		ep, err := scanEpisodeRows(epRows)
		if err != nil {
			return nil, fmt.Errorf("sqlite store: global search: scan episode: %w", err)
		}
		sim := vecmath.Cosine(queryVec, ep.Embedding)
		candidates = append(candidates, Candidate{Episode: ep, Similarity: sim})
	}
	if err := epRows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite store: global search: episode rows: %w", err)
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Similarity > candidates[j].Similarity
	})

	if limit > 0 && len(candidates) > limit {
		candidates = candidates[:limit]
	}
	return candidates, nil
}

func (s *sqliteStore) TouchAccessed(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	now := time.Now().UTC().Format(timeFormat)
	placeholders := make([]string, len(ids))
	args := []any{now}
	for i, id := range ids {
		placeholders[i] = "?"
		args = append(args, id)
	}
	ph := strings.Join(placeholders, ",")

	// Update facts.
	factQuery := fmt.Sprintf(
		`UPDATE facts SET accessed_at = ? WHERE id IN (%s)`, ph,
	)
	_, err := s.db.ExecContext(ctx, factQuery, args...)
	if err != nil {
		return fmt.Errorf("sqlite store: touch accessed: facts: %w", err)
	}

	// Update episodes.
	epQuery := fmt.Sprintf(
		`UPDATE episodes SET accessed_at = ? WHERE id IN (%s)`, ph,
	)
	_, err = s.db.ExecContext(ctx, epQuery, args...)
	if err != nil {
		return fmt.Errorf("sqlite store: touch accessed: episodes: %w", err)
	}
	return nil
}

func (s *sqliteStore) GetCluster(ctx context.Context, id string) (*ClusterNode, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, summary, domain, meta_instr, item_count, centroid,
		        utility, frequency, turns_since, last_access, created_at
		 FROM clusters WHERE id = ?`, id,
	)

	var c ClusterNode
	var summary, domain, metaInstr sql.NullString
	var centroidBlob []byte
	var lastAccessStr, createdStr string

	err := row.Scan(
		&c.ID, &summary, &domain, &metaInstr, &c.ItemCount, &centroidBlob,
		&c.Utility, &c.Frequency, &c.TurnsSince, &lastAccessStr, &createdStr,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("sqlite store: get cluster: %w", err)
	}

	if summary.Valid {
		c.Summary = summary.String
	}
	if domain.Valid {
		c.Domain = domain.String
	}
	if metaInstr.Valid {
		c.MetaInstr = metaInstr.String
	}
	c.Centroid = DecodeVector(centroidBlob)
	c.LastAccess, _ = time.Parse(timeFormat, lastAccessStr)
	c.CreatedAt, _ = time.Parse(timeFormat, createdStr)

	return &c, nil
}

func (s *sqliteStore) ListClusters(ctx context.Context) ([]ClusterNode, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, summary, domain, meta_instr, item_count, centroid,
		        utility, frequency, turns_since, last_access, created_at
		 FROM clusters ORDER BY id`,
	)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: list clusters: %w", err)
	}
	defer rows.Close()

	var clusters []ClusterNode
	for rows.Next() {
		var c ClusterNode
		var summary, domain, metaInstr sql.NullString
		var centroidBlob []byte
		var lastAccessStr, createdStr string

		err := rows.Scan(
			&c.ID, &summary, &domain, &metaInstr, &c.ItemCount, &centroidBlob,
			&c.Utility, &c.Frequency, &c.TurnsSince, &lastAccessStr, &createdStr,
		)
		if err != nil {
			return nil, fmt.Errorf("sqlite store: list clusters: scan: %w", err)
		}

		if summary.Valid {
			c.Summary = summary.String
		}
		if domain.Valid {
			c.Domain = domain.String
		}
		if metaInstr.Valid {
			c.MetaInstr = metaInstr.String
		}
		c.Centroid = DecodeVector(centroidBlob)
		c.LastAccess, _ = time.Parse(timeFormat, lastAccessStr)
		c.CreatedAt, _ = time.Parse(timeFormat, createdStr)

		clusters = append(clusters, c)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite store: list clusters: rows: %w", err)
	}
	return clusters, nil
}

func (s *sqliteStore) UpdateClusterMeta(ctx context.Context, id string, summary, domain, metaInstr string) error {
	res, err := s.db.ExecContext(ctx,
		`UPDATE clusters SET summary = ?, domain = ?, meta_instr = ? WHERE id = ?`,
		nullableString(summary), nullableString(domain), nullableString(metaInstr), id,
	)
	if err != nil {
		return fmt.Errorf("sqlite store: update cluster meta: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("sqlite store: update cluster meta: rows affected: %w", err)
	}
	if n == 0 {
		return fmt.Errorf("sqlite store: update cluster meta: cluster %q not found", id)
	}
	return nil
}

func (s *sqliteStore) UpdateClusterState(ctx context.Context, id string, utility, frequency float64, turnsSince int) error {
	now := time.Now().UTC().Format(timeFormat)
	res, err := s.db.ExecContext(ctx,
		`UPDATE clusters SET utility = ?, frequency = ?, turns_since = ?, last_access = ?
		 WHERE id = ?`,
		utility, frequency, turnsSince, now, id,
	)
	if err != nil {
		return fmt.Errorf("sqlite store: update cluster state: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("sqlite store: update cluster state: rows affected: %w", err)
	}
	if n == 0 {
		return fmt.Errorf("sqlite store: update cluster state: cluster %q not found", id)
	}
	return nil
}

func (s *sqliteStore) TickAllClusters(ctx context.Context, accessedIDs []string) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite store: tick all clusters: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	// Step 1: increment turns_since for all clusters.
	_, err = tx.ExecContext(ctx, `UPDATE clusters SET turns_since = turns_since + 1`)
	if err != nil {
		return fmt.Errorf("sqlite store: tick all clusters: increment: %w", err)
	}

	// Step 2: reset turns_since=0 and update last_access for accessed clusters.
	if len(accessedIDs) > 0 {
		now := time.Now().UTC().Format(timeFormat)
		placeholders := make([]string, len(accessedIDs))
		args := []any{now}
		for i, id := range accessedIDs {
			placeholders[i] = "?"
			args = append(args, id)
		}
		query := fmt.Sprintf(
			`UPDATE clusters SET turns_since = 0, last_access = ? WHERE id IN (%s)`,
			strings.Join(placeholders, ","),
		)
		_, err = tx.ExecContext(ctx, query, args...)
		if err != nil {
			return fmt.Errorf("sqlite store: tick all clusters: reset accessed: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("sqlite store: tick all clusters: commit: %w", err)
	}
	return nil
}

// --- Episode operations ---

func (s *sqliteStore) InsertEpisode(ctx context.Context, e Episode) (string, error) {
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
		e.ClusterID = defaultClusterID
	}

	// Normalize + validate tags before touching the DB.
	normTags, err := normalizeTags(e.Tags)
	if err != nil {
		return "", fmt.Errorf("sqlite store: insert episode: %w", err)
	}
	e.Tags = normTags
	tagsJSON, err := encodeTags(normTags)
	if err != nil {
		return "", fmt.Errorf("sqlite store: insert episode: %w", err)
	}

	// Ensure the cluster exists (FK constraint).
	if e.ClusterID == defaultClusterID {
		if err := s.ensureDefaultCluster(ctx); err != nil {
			return "", err
		}
	}

	embBlob := EncodeVector(e.Embedding)

	_, err = s.db.ExecContext(ctx,
		`INSERT INTO episodes (id, cluster_id, situation, action, outcome, preemptive, embedding, content_hash, created_at, accessed_at, tags)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		e.ID,
		e.ClusterID,
		e.Situation,
		e.Action,
		e.Outcome,
		e.Preemptive,
		embBlob,
		e.ContentHash,
		e.CreatedAt.Format(timeFormat),
		e.AccessedAt.Format(timeFormat),
		tagsJSON,
	)
	if err != nil {
		return "", fmt.Errorf("sqlite store: insert episode: %w", err)
	}

	// Insert evidence edges for any linked fact IDs. Direction is pinned:
	// fact -> episode, edge_type='evidence'. This mirrors the legacy
	// fact_episode_links semantics (fact_id, episode_id) and is the
	// invariant getEpisode/ReplaceEpisodeLinks expect.
	edgeCreated := time.Now().UTC().Format(timeFormat)
	for _, factID := range e.LinkedFactIDs {
		_, err := s.db.ExecContext(ctx,
			`INSERT OR IGNORE INTO memory_edges (src_id, dst_id, edge_type, created_at) VALUES (?, ?, ?, ?)`,
			factID, e.ID, "evidence", edgeCreated,
		)
		if err != nil {
			return "", fmt.Errorf("sqlite store: insert episode: link fact %s: %w", factID, err)
		}
	}

	return e.ID, nil
}

func (s *sqliteStore) GetEpisode(ctx context.Context, id string) (*Episode, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, cluster_id, situation, action, outcome, preemptive, embedding, content_hash, created_at, accessed_at, tags
		 FROM episodes WHERE id = ?`, id,
	)
	ep, err := scanEpisodeFrom(row)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("sqlite store: get episode: %w", err)
	}

	// Load linked fact IDs from memory_edges. Direction matches the
	// insert path in InsertEpisode: src_id=factID, dst_id=episodeID,
	// edge_type='evidence'.
	linkRows, err := s.db.QueryContext(ctx,
		`SELECT src_id FROM memory_edges WHERE dst_id = ? AND edge_type = 'evidence'`, id,
	)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: get episode: load links: %w", err)
	}
	defer linkRows.Close()
	for linkRows.Next() {
		var factID string
		if err := linkRows.Scan(&factID); err != nil {
			return nil, fmt.Errorf("sqlite store: get episode: scan link: %w", err)
		}
		ep.LinkedFactIDs = append(ep.LinkedFactIDs, factID)
	}
	if err := linkRows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite store: get episode: link rows: %w", err)
	}

	return ep, nil
}

func (s *sqliteStore) DeleteEpisode(ctx context.Context, id string) error {
	// Cascade memory_edges and entity_mentions in the same tx as the
	// episode row delete — same rationale as DeleteFact.
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite store: delete episode: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck
	if _, err := tx.ExecContext(ctx,
		`DELETE FROM memory_edges WHERE src_id = ? OR dst_id = ?`, id, id,
	); err != nil {
		return fmt.Errorf("sqlite store: delete episode: cascade edges: %w", err)
	}
	if _, err := tx.ExecContext(ctx,
		`DELETE FROM entity_mentions WHERE memory_id = ?`, id,
	); err != nil {
		return fmt.Errorf("sqlite store: delete episode: cascade mentions: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM episodes WHERE id = ?`, id); err != nil {
		return fmt.Errorf("sqlite store: delete episode: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("sqlite store: delete episode: commit: %w", err)
	}
	return nil
}

func (s *sqliteStore) ListEpisodes(ctx context.Context, filter ListFilter) ([]Episode, error) {
	limit := filter.Limit
	if limit <= 0 {
		limit = defaultLimit
	}
	if limit > maxLimit {
		limit = maxLimit
	}

	orderCol := "created_at"
	if filter.Sort == "accessed" {
		orderCol = "accessed_at"
	}

	query := fmt.Sprintf(
		`SELECT id, cluster_id, situation, action, outcome, preemptive, embedding, content_hash, created_at, accessed_at, tags
		 FROM episodes ORDER BY %s DESC LIMIT ? OFFSET ?`, orderCol,
	)

	rows, err := s.db.QueryContext(ctx, query, limit, filter.Offset)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: list episodes: %w", err)
	}
	defer rows.Close()

	var episodes []Episode
	for rows.Next() {
		ep, err := scanEpisodeRows(rows)
		if err != nil {
			return nil, fmt.Errorf("sqlite store: list episodes: scan: %w", err)
		}
		if !tagMatchesAny(ep.Tags, filter.TagsAny) {
			continue
		}
		episodes = append(episodes, *ep)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite store: list episodes: rows: %w", err)
	}
	return episodes, nil
}

// --- Cluster membership (paginated) ---

func (s *sqliteStore) ListFactsByCluster(ctx context.Context, clusterID string, limit, offset int) ([]Fact, error) {
	if limit < 0 {
		limit = 0
	}
	if offset < 0 {
		offset = 0
	}
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, cluster_id, content, embedding, content_hash, subtype, source, confidence, valid_from, superseded_by, created_at, accessed_at, tags
		 FROM facts WHERE cluster_id = ? AND superseded_by IS NULL
		 ORDER BY created_at ASC LIMIT ? OFFSET ?`,
		clusterID, limit, offset,
	)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: list facts by cluster: %w", err)
	}
	defer rows.Close()

	facts := []Fact{}
	for rows.Next() {
		f, err := scanFactRows(rows)
		if err != nil {
			return nil, fmt.Errorf("sqlite store: list facts by cluster: scan: %w", err)
		}
		facts = append(facts, *f)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite store: list facts by cluster: rows: %w", err)
	}
	return facts, nil
}

func (s *sqliteStore) ListEpisodesByCluster(ctx context.Context, clusterID string, limit, offset int) ([]Episode, error) {
	if limit < 0 {
		limit = 0
	}
	if offset < 0 {
		offset = 0
	}
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, cluster_id, situation, action, outcome, preemptive, embedding, content_hash, created_at, accessed_at, tags
		 FROM episodes WHERE cluster_id = ?
		 ORDER BY created_at ASC LIMIT ? OFFSET ?`,
		clusterID, limit, offset,
	)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: list episodes by cluster: %w", err)
	}
	defer rows.Close()

	episodes := []Episode{}
	for rows.Next() {
		ep, err := scanEpisodeRows(rows)
		if err != nil {
			return nil, fmt.Errorf("sqlite store: list episodes by cluster: scan: %w", err)
		}
		episodes = append(episodes, *ep)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite store: list episodes by cluster: rows: %w", err)
	}
	return episodes, nil
}

func (s *sqliteStore) CountFactsByCluster(ctx context.Context, clusterID string) (int, error) {
	var n int
	err := s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM facts WHERE cluster_id = ? AND superseded_by IS NULL`,
		clusterID,
	).Scan(&n)
	if err != nil {
		return 0, fmt.Errorf("sqlite store: count facts by cluster: %w", err)
	}
	return n, nil
}

func (s *sqliteStore) CountEpisodesByCluster(ctx context.Context, clusterID string) (int, error) {
	var n int
	err := s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM episodes WHERE cluster_id = ?`,
		clusterID,
	).Scan(&n)
	if err != nil {
		return 0, fmt.Errorf("sqlite store: count episodes by cluster: %w", err)
	}
	return n, nil
}

// --- Knowledge graph (Phase 7) ---
//
// All methods below operate on memory_edges, entities, and entity_mentions.
// These tables are polymorphic on ID (no FKs) — application code decides
// which row a given ID belongs to. Cascade-on-delete for facts/episodes
// lives in DeleteFact/DeleteEpisode above.

// defaultEntitySimilarityThreshold is the cosine-similarity cutoff used by
// UpsertEntity when no explicit threshold is plumbed through. The store
// interface deliberately omits a threshold parameter (the design doc locks
// reuse of cfg.Memory.SimilarityThreshold, which lives in the MCP layer);
// 0.55 is a conservative value that catches obvious typos under nomic-style
// embeddings without collapsing unrelated entities.
//
// TODO(phase-2): plumb cfg.Memory.SimilarityThreshold through the handler so
// this constant can be removed.
const defaultEntitySimilarityThreshold = 0.55

// AddEdge inserts a typed directed edge keyed by (src_id, dst_id,
// edge_type). Idempotent: a repeat call returns created=false without
// touching the existing row's weight.
func (s *sqliteStore) AddEdge(ctx context.Context, e Edge) (bool, error) {
	if e.Weight == 0 {
		e.Weight = 1.0
	}
	res, err := s.db.ExecContext(ctx,
		`INSERT OR IGNORE INTO memory_edges (src_id, dst_id, edge_type, weight, created_at)
		 VALUES (?, ?, ?, ?, datetime('now'))`,
		e.SrcID, e.DstID, e.EdgeType, e.Weight,
	)
	if err != nil {
		return false, fmt.Errorf("sqlite store: add edge: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("sqlite store: add edge: rows affected: %w", err)
	}
	return n > 0, nil
}

// RemoveEdge deletes the (src_id, dst_id, edge_type) row. Missing edges
// are not an error — deleted=false is returned instead.
func (s *sqliteStore) RemoveEdge(ctx context.Context, srcID, dstID, edgeType string) (bool, error) {
	res, err := s.db.ExecContext(ctx,
		`DELETE FROM memory_edges WHERE src_id = ? AND dst_id = ? AND edge_type = ?`,
		srcID, dstID, edgeType,
	)
	if err != nil {
		return false, fmt.Errorf("sqlite store: remove edge: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("sqlite store: remove edge: rows affected: %w", err)
	}
	return n > 0, nil
}

// ListEdges performs a Go-side BFS from memoryID over memory_edges up to
// `hops` levels deep. hops is silently clamped to [1,3] (the public tool
// surface validates earlier and rejects out-of-range; this clamp is
// defense-in-depth). Each edge contributes exactly one EdgeWithDistance at
// the depth its non-seed endpoint was first reached.
func (s *sqliteStore) ListEdges(ctx context.Context, memoryID string, hops int) ([]EdgeWithDistance, error) {
	if hops < 1 {
		hops = 1
	}
	if hops > 3 {
		hops = 3
	}

	visited := map[string]struct{}{memoryID: {}}
	frontier := []string{memoryID}
	var results []EdgeWithDistance

	for depth := 1; depth <= hops; depth++ {
		if len(frontier) == 0 {
			break
		}
		placeholders := make([]string, len(frontier))
		args := make([]any, 0, len(frontier)*2)
		for i, id := range frontier {
			placeholders[i] = "?"
			args = append(args, id)
		}
		// Add frontier a second time for the OR-arm.
		for _, id := range frontier {
			args = append(args, id)
		}
		query := fmt.Sprintf(
			`SELECT src_id, dst_id, edge_type, weight, created_at FROM memory_edges
			 WHERE src_id IN (%s) OR dst_id IN (%s)`,
			strings.Join(placeholders, ","),
			strings.Join(placeholders, ","),
		)
		rows, err := s.db.QueryContext(ctx, query, args...)
		if err != nil {
			return nil, fmt.Errorf("sqlite store: list edges: query: %w", err)
		}

		nextFrontierSet := map[string]struct{}{}
		var nextFrontier []string
		for rows.Next() {
			var e Edge
			var createdStr sql.NullString
			if err := rows.Scan(&e.SrcID, &e.DstID, &e.EdgeType, &e.Weight, &createdStr); err != nil {
				rows.Close()
				return nil, fmt.Errorf("sqlite store: list edges: scan: %w", err)
			}
			if createdStr.Valid {
				e.CreatedAt, _ = time.Parse(timeFormat, createdStr.String)
			}
			// Determine which side is the "other" endpoint.
			var other string
			if _, srcInFrontier := indexOfFrontier(frontier, e.SrcID); srcInFrontier {
				other = e.DstID
			} else {
				other = e.SrcID
			}
			if _, seen := visited[other]; seen {
				continue
			}
			visited[other] = struct{}{}
			if _, alreadyQueued := nextFrontierSet[other]; !alreadyQueued {
				nextFrontierSet[other] = struct{}{}
				nextFrontier = append(nextFrontier, other)
			}
			results = append(results, EdgeWithDistance{Edge: e, Distance: depth})
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return nil, fmt.Errorf("sqlite store: list edges: rows: %w", err)
		}
		rows.Close()
		frontier = nextFrontier
	}
	return results, nil
}

// indexOfFrontier returns (i, true) if id is in frontier; (0, false)
// otherwise. Tiny helper kept local; frontier is small (BFS-level wide).
func indexOfFrontier(frontier []string, id string) (int, bool) {
	for i, f := range frontier {
		if f == id {
			return i, true
		}
	}
	return 0, false
}

// UpsertEntity dedups by (name, entity_type) exact match, then by cosine
// similarity within the same entity_type (using
// defaultEntitySimilarityThreshold). Only inserts on miss; the caller-
// supplied embedding is stored verbatim. Layer text format
// "name + (entity_type)" is the caller's responsibility per the design
// doc's locked decision.
func (s *sqliteStore) UpsertEntity(ctx context.Context, name, entityType string, embedding []float32) (string, bool, bool, error) {
	// Step 1: exact dedup.
	var existingID string
	err := s.db.QueryRowContext(ctx,
		`SELECT id FROM entities WHERE name = ? AND entity_type = ?`,
		name, entityType,
	).Scan(&existingID)
	if err == nil {
		return existingID, false, false, nil
	}
	if err != sql.ErrNoRows {
		return "", false, false, fmt.Errorf("sqlite store: upsert entity: exact lookup: %w", err)
	}

	// Step 2: similarity dedup (skipped when caller did not supply an
	// embedding — same-type cosine is meaningless without one).
	if len(embedding) > 0 {
		matches, simErr := s.FindSimilarEntities(ctx, embedding, entityType, defaultEntitySimilarityThreshold, 1)
		if simErr != nil {
			return "", false, false, fmt.Errorf("sqlite store: upsert entity: similarity lookup: %w", simErr)
		}
		if len(matches) > 0 {
			return matches[0].Entity.ID, false, true, nil
		}
	}

	// Step 3: insert.
	id := uuid.New().String()
	now := time.Now().UTC().Format(timeFormat)
	embBlob := EncodeVector(embedding)
	_, err = s.db.ExecContext(ctx,
		`INSERT INTO entities (id, name, entity_type, embedding, utility, frequency, turns_since, retention, last_access, created_at)
		 VALUES (?, ?, ?, ?, 0.5, 0.5, 0, 1.0, NULL, ?)`,
		id, name, entityType, embBlob, now,
	)
	if err != nil {
		return "", false, false, fmt.Errorf("sqlite store: upsert entity: insert: %w", err)
	}
	return id, true, false, nil
}

// GetEntity returns the entity with the given ID or (zero Entity, nil)
// when no row matches — matches GetFact/GetEpisode's not-found convention.
func (s *sqliteStore) GetEntity(ctx context.Context, id string) (Entity, error) {
	var e Entity
	var embBlob []byte
	var lastAccess, createdAt sql.NullString
	err := s.db.QueryRowContext(ctx,
		`SELECT id, name, entity_type, embedding, utility, frequency, turns_since, retention, last_access, created_at
		 FROM entities WHERE id = ?`, id,
	).Scan(&e.ID, &e.Name, &e.EntityType, &embBlob, &e.Utility, &e.Frequency, &e.TurnsSince, &e.Retention, &lastAccess, &createdAt)
	if err == sql.ErrNoRows {
		return Entity{}, nil
	}
	if err != nil {
		return Entity{}, fmt.Errorf("sqlite store: get entity: %w", err)
	}
	e.Embedding = DecodeVector(embBlob)
	if lastAccess.Valid {
		e.LastAccess, _ = time.Parse(timeFormat, lastAccess.String)
	}
	if createdAt.Valid {
		e.CreatedAt, _ = time.Parse(timeFormat, createdAt.String)
	}
	return e, nil
}

// FindSimilarEntities returns entities (optionally filtered by entityType
// — empty string scans all types) whose embedding has cosine similarity
// >= threshold to the query vector, ordered by descending similarity and
// capped at limit. Entities with NULL embeddings are skipped.
func (s *sqliteStore) FindSimilarEntities(ctx context.Context, embedding []float32, entityType string, threshold float64, limit int) ([]EntityWithScore, error) {
	var rows *sql.Rows
	var err error
	if entityType != "" {
		rows, err = s.db.QueryContext(ctx,
			`SELECT id, name, entity_type, embedding, utility, frequency, turns_since, retention, last_access, created_at
			 FROM entities WHERE entity_type = ? AND embedding IS NOT NULL`,
			entityType,
		)
	} else {
		rows, err = s.db.QueryContext(ctx,
			`SELECT id, name, entity_type, embedding, utility, frequency, turns_since, retention, last_access, created_at
			 FROM entities WHERE embedding IS NOT NULL`,
		)
	}
	if err != nil {
		return nil, fmt.Errorf("sqlite store: find similar entities: %w", err)
	}
	defer rows.Close()

	var scored []EntityWithScore
	for rows.Next() {
		var e Entity
		var embBlob []byte
		var lastAccess, createdAt sql.NullString
		if err := rows.Scan(&e.ID, &e.Name, &e.EntityType, &embBlob, &e.Utility, &e.Frequency, &e.TurnsSince, &e.Retention, &lastAccess, &createdAt); err != nil {
			return nil, fmt.Errorf("sqlite store: find similar entities: scan: %w", err)
		}
		e.Embedding = DecodeVector(embBlob)
		if len(e.Embedding) == 0 {
			continue
		}
		if lastAccess.Valid {
			e.LastAccess, _ = time.Parse(timeFormat, lastAccess.String)
		}
		if createdAt.Valid {
			e.CreatedAt, _ = time.Parse(timeFormat, createdAt.String)
		}
		sim := vecmath.Cosine(embedding, e.Embedding)
		if float64(sim) >= threshold {
			scored = append(scored, EntityWithScore{Entity: e, Similarity: sim})
		}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite store: find similar entities: rows: %w", err)
	}

	sort.Slice(scored, func(i, j int) bool { return scored[i].Similarity > scored[j].Similarity })
	if limit > 0 && len(scored) > limit {
		scored = scored[:limit]
	}
	return scored, nil
}

// TickAllEntities increments turns_since for every entity, then resets
// (turns_since=0, retention=1.0, last_access=now) for entities in
// accessedIDs, then rewrites every entity's retention column to match its
// post-update state via retentionFromState. Single transaction.
func (s *sqliteStore) TickAllEntities(ctx context.Context, accessedIDs []string) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite store: tick all entities: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	if _, err := tx.ExecContext(ctx, `UPDATE entities SET turns_since = turns_since + 1`); err != nil {
		return fmt.Errorf("sqlite store: tick all entities: increment: %w", err)
	}

	if len(accessedIDs) > 0 {
		now := time.Now().UTC().Format(timeFormat)
		placeholders := make([]string, len(accessedIDs))
		args := []any{now}
		for i, id := range accessedIDs {
			placeholders[i] = "?"
			args = append(args, id)
		}
		query := fmt.Sprintf(
			`UPDATE entities SET turns_since = 0, retention = 1.0, last_access = ? WHERE id IN (%s)`,
			strings.Join(placeholders, ","),
		)
		if _, err := tx.ExecContext(ctx, query, args...); err != nil {
			return fmt.Errorf("sqlite store: tick all entities: reset accessed: %w", err)
		}
	}

	// Recompute retention for every row via the pure formula in
	// pkg/ebbinghaus (same source of truth the decay engine wraps).
	rows, err := tx.QueryContext(ctx, `SELECT id, utility, frequency, turns_since FROM entities`)
	if err != nil {
		return fmt.Errorf("sqlite store: tick all entities: load: %w", err)
	}
	type retUpdate struct {
		id  string
		ret float64
	}
	var updates []retUpdate
	for rows.Next() {
		var id string
		var utility, frequency float64
		var turnsSince int
		if err := rows.Scan(&id, &utility, &frequency, &turnsSince); err != nil {
			rows.Close()
			return fmt.Errorf("sqlite store: tick all entities: scan: %w", err)
		}
		updates = append(updates, retUpdate{
			id:  id,
			ret: ebbinghaus.Retention(turnsSince, utility, frequency, ebbinghaus.DefaultTemperature),
		})
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return fmt.Errorf("sqlite store: tick all entities: load rows: %w", err)
	}
	rows.Close()

	for _, u := range updates {
		if _, err := tx.ExecContext(ctx,
			`UPDATE entities SET retention = ? WHERE id = ?`, u.ret, u.id,
		); err != nil {
			return fmt.Errorf("sqlite store: tick all entities: update retention: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("sqlite store: tick all entities: commit: %w", err)
	}
	return nil
}

// AddEntityMentions inserts (memory_id, entity_id, role) rows for every
// entity in entityIDs. Idempotency is keyed by (memory_id, entity_id);
// duplicate rows return inserted=0 for that entity. Empty role is stored
// as SQL NULL so callers can distinguish "not specified" from a specific
// role string.
func (s *sqliteStore) AddEntityMentions(ctx context.Context, memoryID string, entityIDs []string, role string) (int, error) {
	if len(entityIDs) == 0 {
		return 0, nil
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("sqlite store: add entity mentions: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	inserted := 0
	for _, eid := range entityIDs {
		res, err := tx.ExecContext(ctx,
			`INSERT OR IGNORE INTO entity_mentions (memory_id, entity_id, role) VALUES (?, ?, ?)`,
			memoryID, eid, nullableString(role),
		)
		if err != nil {
			return 0, fmt.Errorf("sqlite store: add entity mentions: insert: %w", err)
		}
		n, err := res.RowsAffected()
		if err != nil {
			return 0, fmt.Errorf("sqlite store: add entity mentions: rows affected: %w", err)
		}
		inserted += int(n)
	}

	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("sqlite store: add entity mentions: commit: %w", err)
	}
	return inserted, nil
}

// ListMemoriesByEntity returns memories mentioning the given entity,
// capped at limit. Each MemoryRef's Layer is resolved by trying the facts
// table first and then the episodes table; Content is the layer's
// canonical content text truncated to 120 chars.
func (s *sqliteStore) ListMemoriesByEntity(ctx context.Context, entityID string, limit int) ([]MemoryRef, error) {
	if limit <= 0 {
		limit = defaultLimit
	}
	rows, err := s.db.QueryContext(ctx,
		`SELECT memory_id FROM entity_mentions WHERE entity_id = ? LIMIT ?`,
		entityID, limit,
	)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: list memories by entity: %w", err)
	}
	defer rows.Close()

	var ids []string
	for rows.Next() {
		var mid string
		if err := rows.Scan(&mid); err != nil {
			return nil, fmt.Errorf("sqlite store: list memories by entity: scan: %w", err)
		}
		ids = append(ids, mid)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite store: list memories by entity: rows: %w", err)
	}

	refs := make([]MemoryRef, 0, len(ids))
	for _, mid := range ids {
		ref, ok, err := s.resolveMemoryRef(ctx, mid)
		if err != nil {
			return nil, err
		}
		if !ok {
			continue
		}
		refs = append(refs, ref)
	}
	return refs, nil
}

// resolveMemoryRef looks up a memory id in the facts and episodes tables
// and returns a populated MemoryRef on hit. Returns ok=false when the id
// does not belong to either layer (e.g., it's an entity id, or an
// orphaned mention left after a delete somehow). Content is truncated to
// 120 chars to match the design doc's preview contract.
func (s *sqliteStore) resolveMemoryRef(ctx context.Context, id string) (MemoryRef, bool, error) {
	var content string
	err := s.db.QueryRowContext(ctx, `SELECT content FROM facts WHERE id = ?`, id).Scan(&content)
	if err == nil {
		return MemoryRef{ID: id, Layer: TypeL2Semantic, Content: truncatePreview(content)}, true, nil
	}
	if err != sql.ErrNoRows {
		return MemoryRef{}, false, fmt.Errorf("sqlite store: resolve memory ref: facts: %w", err)
	}

	var situation, action, outcome, preemptive string
	err = s.db.QueryRowContext(ctx,
		`SELECT situation, action, outcome, preemptive FROM episodes WHERE id = ?`, id,
	).Scan(&situation, &action, &outcome, &preemptive)
	if err == nil {
		preview := strings.TrimSpace(situation + " " + action + " " + outcome + " " + preemptive)
		return MemoryRef{ID: id, Layer: TypeL3Episodic, Content: truncatePreview(preview)}, true, nil
	}
	if err != sql.ErrNoRows {
		return MemoryRef{}, false, fmt.Errorf("sqlite store: resolve memory ref: episodes: %w", err)
	}
	return MemoryRef{}, false, nil
}

// truncatePreview clips s to 120 chars. Falls within the design doc's
// preview contract for entity-mention and edge-neighbor surfaces.
func truncatePreview(s string) string {
	const maxPreview = 120
	if len(s) <= maxPreview {
		return s
	}
	return s[:maxPreview]
}

// ListEntityNeighbors walks the graph out from an entity. Memories
// reached via entity_mentions on the seed are Distance=1 NeighborMemory
// rows. Edges between entities (memory_edges) are walked BFS-style up to
// hops levels; any id that resolves to facts/episodes becomes a
// NeighborMemory, any id that resolves to entities becomes a
// NeighborEntity.
func (s *sqliteStore) ListEntityNeighbors(ctx context.Context, entityID string, hops int) ([]NeighborMemory, []NeighborEntity, error) {
	if hops < 1 {
		hops = 1
	}
	if hops > 3 {
		hops = 3
	}

	var memories []NeighborMemory
	var entities []NeighborEntity
	seenMem := map[string]struct{}{}
	seenEnt := map[string]struct{}{entityID: {}}

	// Distance-1 memories from entity_mentions on the seed.
	mentionRows, err := s.db.QueryContext(ctx,
		`SELECT memory_id FROM entity_mentions WHERE entity_id = ?`, entityID,
	)
	if err != nil {
		return nil, nil, fmt.Errorf("sqlite store: list entity neighbors: mentions: %w", err)
	}
	var directMems []string
	for mentionRows.Next() {
		var mid string
		if err := mentionRows.Scan(&mid); err != nil {
			mentionRows.Close()
			return nil, nil, fmt.Errorf("sqlite store: list entity neighbors: scan mention: %w", err)
		}
		directMems = append(directMems, mid)
	}
	mentionRows.Close()
	for _, mid := range directMems {
		ref, ok, err := s.resolveMemoryRef(ctx, mid)
		if err != nil {
			return nil, nil, err
		}
		if !ok {
			continue
		}
		if _, dup := seenMem[mid]; dup {
			continue
		}
		seenMem[mid] = struct{}{}
		memories = append(memories, NeighborMemory{
			ID:             mid,
			Layer:          ref.Layer,
			ContentPreview: ref.Content,
			Distance:       1,
		})
	}

	// BFS over memory_edges starting at the seed entity.
	edges, err := s.ListEdges(ctx, entityID, hops)
	if err != nil {
		return nil, nil, err
	}
	for _, ewd := range edges {
		// The "other" endpoint is whichever side is not the seed AND
		// whichever side has not been visited (the BFS already
		// guarantees one of them is the newly-reached node, but
		// determining which requires another lookup against seenEnt).
		var other string
		if _, ok := seenEnt[ewd.Edge.SrcID]; ok {
			other = ewd.Edge.DstID
		} else if _, ok := seenEnt[ewd.Edge.DstID]; ok {
			other = ewd.Edge.SrcID
		} else {
			// Neither endpoint is a known entity from this walk yet
			// (multi-hop case where both endpoints are newly seen via
			// a memory hop). Pick the one that resolves to an entity.
			ent, eErr := s.GetEntity(ctx, ewd.Edge.SrcID)
			if eErr != nil {
				return nil, nil, eErr
			}
			if ent.ID != "" && ent.ID == ewd.Edge.SrcID {
				other = ewd.Edge.DstID
			} else {
				other = ewd.Edge.SrcID
			}
		}
		seenEnt[other] = struct{}{}

		// Resolve which layer the other endpoint belongs to.
		ent, eErr := s.GetEntity(ctx, other)
		if eErr != nil {
			return nil, nil, eErr
		}
		if ent.ID == other {
			entities = append(entities, NeighborEntity{
				ID:         ent.ID,
				Name:       ent.Name,
				EntityType: ent.EntityType,
				Distance:   ewd.Distance,
			})
			continue
		}
		ref, ok, err := s.resolveMemoryRef(ctx, other)
		if err != nil {
			return nil, nil, err
		}
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
// (non-nil) result with no error. One batched SELECT keeps this cheap
// for the session-end caller which can pass dozens of buffered memories.
func (s *sqliteStore) ListEntitiesByMemoryIDs(ctx context.Context, memoryIDs []string) ([]string, error) {
	if len(memoryIDs) == 0 {
		return []string{}, nil
	}
	placeholders := make([]string, len(memoryIDs))
	args := make([]any, len(memoryIDs))
	for i, id := range memoryIDs {
		placeholders[i] = "?"
		args[i] = id
	}
	query := fmt.Sprintf(
		`SELECT DISTINCT entity_id FROM entity_mentions WHERE memory_id IN (%s)`,
		strings.Join(placeholders, ","),
	)
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: list entities by memory ids: %w", err)
	}
	defer rows.Close()

	out := make([]string, 0)
	for rows.Next() {
		var eid string
		if err := rows.Scan(&eid); err != nil {
			return nil, fmt.Errorf("sqlite store: list entities by memory ids: scan: %w", err)
		}
		out = append(out, eid)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite store: list entities by memory ids: rows: %w", err)
	}
	return out, nil
}

// Phase 7C: ExpandViaGraph implementation -- see docs/design/phase-7c-graph-aware-recall.md.
//
// BFS walks memory_edges and entity_mentions outward from seedIDs to at
// most hops levels deep, returning one GraphHit per (neighbor, seed)
// pair at that pair's shortest distance. Hops clamped to [1,3]
// defensively. minRetention<=0 disables the retention pre-filter;
// maxVisited<=0 disables the global visited cap. Cluster retention is
// computed via the pure Ebbinghaus formula (the store stays decoupled
// from the decayer).
func (s *sqliteStore) ExpandViaGraph(ctx context.Context, seedIDs []string, hops int, minRetention float64, maxVisited int) ([]GraphHit, error) {
	if hops < 1 {
		hops = 1
	}
	if hops > 3 {
		hops = 3
	}
	if len(seedIDs) == 0 {
		return nil, nil
	}

	type pairKey struct {
		mem  string
		seed string
	}
	type frontierEntry struct {
		mem  string
		seed string
	}

	bestPair := make(map[pairKey]int)
	globalVisited := make(map[string]struct{})
	var frontier []frontierEntry

	for _, sid := range seedIDs {
		if _, dup := globalVisited[sid]; dup {
			continue
		}
		globalVisited[sid] = struct{}{}
		bestPair[pairKey{sid, sid}] = 0
		frontier = append(frontier, frontierEntry{mem: sid, seed: sid})
	}

	results := make([]GraphHit, 0)

	clusterRetention := make(map[string]float64)
	getClusterRetention := func(clusterID string) (float64, error) {
		if r, ok := clusterRetention[clusterID]; ok {
			return r, nil
		}
		cl, err := s.GetCluster(ctx, clusterID)
		if err != nil {
			return 0, err
		}
		if cl == nil {
			clusterRetention[clusterID] = 0
			return 0, nil
		}
		r := ebbinghaus.Retention(cl.TurnsSince, cl.Utility, cl.Frequency, ebbinghaus.DefaultTemperature)
		clusterRetention[clusterID] = r
		return r, nil
	}

	// resolveMemoryLayer returns (layer, clusterID, true) for facts/episodes,
	// or ("", "", false) when the id is an entity / unknown.
	memoryLayerCache := make(map[string]struct {
		layer     string
		clusterID string
		ok        bool
	})
	resolveMemoryLayer := func(id string) (string, string, bool, error) {
		if cached, ok := memoryLayerCache[id]; ok {
			return cached.layer, cached.clusterID, cached.ok, nil
		}
		var clusterID string
		err := s.db.QueryRowContext(ctx, `SELECT cluster_id FROM facts WHERE id = ?`, id).Scan(&clusterID)
		if err == nil {
			memoryLayerCache[id] = struct {
				layer     string
				clusterID string
				ok        bool
			}{string(TypeL2Semantic), clusterID, true}
			return string(TypeL2Semantic), clusterID, true, nil
		}
		if err != sql.ErrNoRows {
			return "", "", false, fmt.Errorf("sqlite store: expand via graph: facts lookup: %w", err)
		}
		err = s.db.QueryRowContext(ctx, `SELECT cluster_id FROM episodes WHERE id = ?`, id).Scan(&clusterID)
		if err == nil {
			memoryLayerCache[id] = struct {
				layer     string
				clusterID string
				ok        bool
			}{string(TypeL3Episodic), clusterID, true}
			return string(TypeL3Episodic), clusterID, true, nil
		}
		if err != sql.ErrNoRows {
			return "", "", false, fmt.Errorf("sqlite store: expand via graph: episodes lookup: %w", err)
		}
		memoryLayerCache[id] = struct {
			layer     string
			clusterID string
			ok        bool
		}{"", "", false}
		return "", "", false, nil
	}

	// Entity retention cache.
	entityRetention := make(map[string]float64)
	getEntityRetention := func(id string) (float64, bool, error) {
		if r, ok := entityRetention[id]; ok {
			return r, true, nil
		}
		var r float64
		err := s.db.QueryRowContext(ctx, `SELECT retention FROM entities WHERE id = ?`, id).Scan(&r)
		if err == nil {
			entityRetention[id] = r
			return r, true, nil
		}
		if err == sql.ErrNoRows {
			return 0, false, nil
		}
		return 0, false, fmt.Errorf("sqlite store: expand via graph: entity retention: %w", err)
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

		// Group current frontier by memory ID (a single memory may appear
		// with multiple seed parents in the frontier).
		frontierMems := make(map[string][]string)
		for _, fe := range frontier {
			frontierMems[fe.mem] = append(frontierMems[fe.mem], fe.seed)
		}
		// Build placeholder list once for batched IN-clauses.
		frontierIDs := make([]string, 0, len(frontierMems))
		for mid := range frontierMems {
			frontierIDs = append(frontierIDs, mid)
		}

		var nextFrontier []frontierEntry
		stopAll := false

		// (a) memory_edges -- one batched query.
		if len(frontierIDs) > 0 {
			placeholders := make([]string, len(frontierIDs))
			args := make([]any, 0, len(frontierIDs)*2)
			for i, id := range frontierIDs {
				placeholders[i] = "?"
				args = append(args, id)
			}
			for _, id := range frontierIDs {
				args = append(args, id)
			}
			query := fmt.Sprintf(
				`SELECT src_id, dst_id FROM memory_edges WHERE src_id IN (%s) OR dst_id IN (%s)`,
				strings.Join(placeholders, ","),
				strings.Join(placeholders, ","),
			)
			rows, err := s.db.QueryContext(ctx, query, args...)
			if err != nil {
				return nil, fmt.Errorf("sqlite store: expand via graph: edges query: %w", err)
			}
			type edgePair struct {
				src string
				dst string
			}
			var edgePairs []edgePair
			for rows.Next() {
				var ep edgePair
				if err := rows.Scan(&ep.src, &ep.dst); err != nil {
					rows.Close()
					return nil, fmt.Errorf("sqlite store: expand via graph: edges scan: %w", err)
				}
				edgePairs = append(edgePairs, ep)
			}
			if err := rows.Err(); err != nil {
				rows.Close()
				return nil, fmt.Errorf("sqlite store: expand via graph: edges rows: %w", err)
			}
			rows.Close()

			tryReach := func(parent, other string, seeds []string) (bool, error) {
				layer, clusterID, isMem, err := resolveMemoryLayer(other)
				if err != nil {
					return false, err
				}
				if !isMem {
					return false, nil
				}
				if other == parent {
					return false, nil
				}
				if minRetention > 0 {
					r, err := getClusterRetention(clusterID)
					if err != nil {
						return false, err
					}
					if r < minRetention {
						return false, nil
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
							return true, nil
						}
					}
				}
				return false, nil
			}

		edgeLoop:
			for _, ep := range edgePairs {
				seedsFromSrc, srcIn := frontierMems[ep.src]
				seedsFromDst, dstIn := frontierMems[ep.dst]
				if srcIn {
					stop, err := tryReach(ep.src, ep.dst, seedsFromSrc)
					if err != nil {
						return nil, err
					}
					if stop {
						stopAll = true
						break edgeLoop
					}
				}
				if dstIn {
					stop, err := tryReach(ep.dst, ep.src, seedsFromDst)
					if err != nil {
						return nil, err
					}
					if stop {
						stopAll = true
						break edgeLoop
					}
				}
			}
		}

		if stopAll {
			break
		}

		// (b) memory->entity->memory -- 2-hop entity mediator. Only
		// records hits if depth+1 <= hops.
		if depth+1 <= hops && !capReached() && len(frontierIDs) > 0 {
			// Batched: memory_id -> entity_id for all frontier mems.
			placeholders := make([]string, len(frontierIDs))
			args := make([]any, len(frontierIDs))
			for i, id := range frontierIDs {
				placeholders[i] = "?"
				args[i] = id
			}
			q1 := fmt.Sprintf(
				`SELECT memory_id, entity_id FROM entity_mentions WHERE memory_id IN (%s)`,
				strings.Join(placeholders, ","),
			)
			rows, err := s.db.QueryContext(ctx, q1, args...)
			if err != nil {
				return nil, fmt.Errorf("sqlite store: expand via graph: mentions out: %w", err)
			}
			memToEntities := make(map[string][]string)
			entitySet := make(map[string]struct{})
			for rows.Next() {
				var mid, eid string
				if err := rows.Scan(&mid, &eid); err != nil {
					rows.Close()
					return nil, fmt.Errorf("sqlite store: expand via graph: mentions out scan: %w", err)
				}
				if minRetention > 0 {
					r, ok, eerr := getEntityRetention(eid)
					if eerr != nil {
						rows.Close()
						return nil, eerr
					}
					if ok && r < minRetention {
						continue
					}
				}
				memToEntities[mid] = append(memToEntities[mid], eid)
				entitySet[eid] = struct{}{}
			}
			if err := rows.Err(); err != nil {
				rows.Close()
				return nil, fmt.Errorf("sqlite store: expand via graph: mentions out rows: %w", err)
			}
			rows.Close()

			if len(entitySet) > 0 {
				// Batched: entity_id -> memory_id for all gathered entities.
				ents := make([]string, 0, len(entitySet))
				for eid := range entitySet {
					ents = append(ents, eid)
				}
				ph2 := make([]string, len(ents))
				args2 := make([]any, len(ents))
				for i, e := range ents {
					ph2[i] = "?"
					args2[i] = e
				}
				q2 := fmt.Sprintf(
					`SELECT entity_id, memory_id FROM entity_mentions WHERE entity_id IN (%s)`,
					strings.Join(ph2, ","),
				)
				rows2, err := s.db.QueryContext(ctx, q2, args2...)
				if err != nil {
					return nil, fmt.Errorf("sqlite store: expand via graph: mentions in: %w", err)
				}
				entityToMems := make(map[string][]string)
				for rows2.Next() {
					var eid, mid string
					if err := rows2.Scan(&eid, &mid); err != nil {
						rows2.Close()
						return nil, fmt.Errorf("sqlite store: expand via graph: mentions in scan: %w", err)
					}
					entityToMems[eid] = append(entityToMems[eid], mid)
				}
				if err := rows2.Err(); err != nil {
					rows2.Close()
					return nil, fmt.Errorf("sqlite store: expand via graph: mentions in rows: %w", err)
				}
				rows2.Close()

				stopEnt := false
			outerEnt:
				for parentMem, seeds := range frontierMems {
					eids := memToEntities[parentMem]
					for _, eid := range eids {
						mids := entityToMems[eid]
						for _, n := range mids {
							if n == parentMem {
								continue
							}
							layer, clusterID, isMem, lerr := resolveMemoryLayer(n)
							if lerr != nil {
								return nil, lerr
							}
							if !isMem {
								continue
							}
							if minRetention > 0 {
								r, rerr := getClusterRetention(clusterID)
								if rerr != nil {
									return nil, rerr
								}
								if r < minRetention {
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
										stopEnt = true
										break outerEnt
									}
								}
							}
						}
					}
				}
				if stopEnt {
					break
				}
			}
		}

		frontier = nextFrontier
	}

	return results, nil
}

// CountEntities returns the total number of entity rows. Cheap COUNT(*)
// behind the SQLite covering index — fine for reverie://status.
func (s *sqliteStore) CountEntities(ctx context.Context) (int, error) {
	var n int
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM entities`).Scan(&n); err != nil {
		return 0, fmt.Errorf("sqlite store: count entities: %w", err)
	}
	return n, nil
}

// CountEdges returns the total number of memory_edges rows. Cheap
// COUNT(*) — used alongside CountEntities on reverie://status.
func (s *sqliteStore) CountEdges(ctx context.Context) (int, error) {
	var n int
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM memory_edges`).Scan(&n); err != nil {
		return 0, fmt.Errorf("sqlite store: count edges: %w", err)
	}
	return n, nil
}

// --- Cluster operations (Phase 3 additions) ---

func (s *sqliteStore) CreateCluster(ctx context.Context, c ClusterNode) error {
	now := time.Now().UTC()
	if c.CreatedAt.IsZero() {
		c.CreatedAt = now
	}
	if c.LastAccess.IsZero() {
		c.LastAccess = now
	}

	centroidBlob := EncodeVector(c.Centroid)

	_, err := s.db.ExecContext(ctx,
		`INSERT INTO clusters (id, summary, domain, meta_instr, item_count, centroid,
		                       utility, frequency, turns_since, last_access, created_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		c.ID,
		nullableString(c.Summary),
		nullableString(c.Domain),
		nullableString(c.MetaInstr),
		c.ItemCount,
		centroidBlob,
		c.Utility,
		c.Frequency,
		c.TurnsSince,
		c.LastAccess.Format(timeFormat),
		c.CreatedAt.Format(timeFormat),
	)
	if err != nil {
		return fmt.Errorf("sqlite store: create cluster: %w", err)
	}
	return nil
}

func (s *sqliteStore) UpdateClusterCentroid(ctx context.Context, id string, centroid []float32, itemCount int) error {
	centroidBlob := EncodeVector(centroid)
	res, err := s.db.ExecContext(ctx,
		`UPDATE clusters SET centroid = ?, item_count = ? WHERE id = ?`,
		centroidBlob, itemCount, id,
	)
	if err != nil {
		return fmt.Errorf("sqlite store: update cluster centroid: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("sqlite store: update cluster centroid: rows affected: %w", err)
	}
	if n == 0 {
		return fmt.Errorf("sqlite store: update cluster centroid: cluster %q not found", id)
	}
	return nil
}

// SetMemoryCluster updates the cluster_id of a fact or episode by id. The
// fact table is tried first; if no row matches, the episode table is tried.
// Both updates also bump accessed_at. Errors "memory not found: %s" if
// neither table has the id.
func (s *sqliteStore) SetMemoryCluster(ctx context.Context, memoryID, clusterID string) error {
	now := time.Now().UTC().Format(timeFormat)

	res, err := s.db.ExecContext(ctx,
		`UPDATE facts SET cluster_id = ?, accessed_at = ? WHERE id = ?`,
		clusterID, now, memoryID,
	)
	if err != nil {
		return fmt.Errorf("sqlite store: set memory cluster: facts: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("sqlite store: set memory cluster: facts rows affected: %w", err)
	}
	if n > 0 {
		return nil
	}

	res, err = s.db.ExecContext(ctx,
		`UPDATE episodes SET cluster_id = ?, accessed_at = ? WHERE id = ?`,
		clusterID, now, memoryID,
	)
	if err != nil {
		return fmt.Errorf("sqlite store: set memory cluster: episodes: %w", err)
	}
	n, err = res.RowsAffected()
	if err != nil {
		return fmt.Errorf("sqlite store: set memory cluster: episodes rows affected: %w", err)
	}
	if n > 0 {
		return nil
	}

	return fmt.Errorf("memory not found: %s", memoryID)
}

// withTx begins a transaction, invokes fn, and commits on success or rolls
// back on error. If fn panics the transaction is rolled back and the panic
// re-raised. Use this for multi-statement operations that must be atomic.
func (s *sqliteStore) withTx(ctx context.Context, fn func(*sql.Tx) error) (err error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite store: begin tx: %w", err)
	}
	defer func() {
		if p := recover(); p != nil {
			_ = tx.Rollback()
			panic(p)
		}
		if err != nil {
			_ = tx.Rollback()
			return
		}
		err = tx.Commit()
		if err != nil {
			err = fmt.Errorf("sqlite store: commit tx: %w", err)
		}
	}()
	return fn(tx)
}

// MoveAllClusterMembers reparents all non-superseded facts and all episodes
// from sourceClusterID into targetClusterID atomically via a single
// transaction. Returns the total number of rows moved. accessed_at is bumped
// on every moved row.
func (s *sqliteStore) MoveAllClusterMembers(ctx context.Context, sourceClusterID, targetClusterID string) (int, error) {
	var moved int
	err := s.withTx(ctx, func(tx *sql.Tx) error {
		now := time.Now().UTC().Format(timeFormat)

		factsRes, err := tx.ExecContext(ctx,
			`UPDATE facts SET cluster_id = ?, accessed_at = ?
			 WHERE cluster_id = ? AND superseded_by IS NULL`,
			targetClusterID, now, sourceClusterID,
		)
		if err != nil {
			return fmt.Errorf("sqlite store: move cluster members: facts: %w", err)
		}
		factsN, err := factsRes.RowsAffected()
		if err != nil {
			return fmt.Errorf("sqlite store: move cluster members: facts rows affected: %w", err)
		}

		epRes, err := tx.ExecContext(ctx,
			`UPDATE episodes SET cluster_id = ?, accessed_at = ?
			 WHERE cluster_id = ?`,
			targetClusterID, now, sourceClusterID,
		)
		if err != nil {
			return fmt.Errorf("sqlite store: move cluster members: episodes: %w", err)
		}
		epN, err := epRes.RowsAffected()
		if err != nil {
			return fmt.Errorf("sqlite store: move cluster members: episodes rows affected: %w", err)
		}

		moved = int(factsN + epN)
		return nil
	})
	if err != nil {
		return 0, err
	}
	return moved, nil
}

// DeleteCluster removes the cluster row with the given id. Idempotent: a
// missing cluster returns nil. Refuses to delete if any non-superseded fact
// or any episode still references the cluster (safety guard against
// orphaning members via FK violation).
func (s *sqliteStore) DeleteCluster(ctx context.Context, id string) error {
	var factCount, episodeCount int
	if err := s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM facts WHERE cluster_id = ? AND superseded_by IS NULL`,
		id,
	).Scan(&factCount); err != nil {
		return fmt.Errorf("sqlite store: delete cluster: count facts: %w", err)
	}
	if err := s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM episodes WHERE cluster_id = ?`, id,
	).Scan(&episodeCount); err != nil {
		return fmt.Errorf("sqlite store: delete cluster: count episodes: %w", err)
	}
	if factCount+episodeCount > 0 {
		return fmt.Errorf("cluster not empty: %s (has %d facts, %d episodes)", id, factCount, episodeCount)
	}

	if _, err := s.db.ExecContext(ctx, `DELETE FROM clusters WHERE id = ?`, id); err != nil {
		return fmt.Errorf("sqlite store: delete cluster: %w", err)
	}
	return nil
}

// --- Embedding update (for reindex) ---

func (s *sqliteStore) UpdateFactEmbedding(ctx context.Context, id string, embedding []float32) error {
	embBlob := EncodeVector(embedding)
	res, err := s.db.ExecContext(ctx,
		`UPDATE facts SET embedding = ? WHERE id = ?`,
		embBlob, id,
	)
	if err != nil {
		return fmt.Errorf("sqlite store: update fact embedding: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("sqlite store: update fact embedding: rows affected: %w", err)
	}
	if n == 0 {
		return fmt.Errorf("sqlite store: update fact embedding: fact %q not found", id)
	}
	return nil
}

func (s *sqliteStore) UpdateEpisodeEmbedding(ctx context.Context, id string, embedding []float32) error {
	embBlob := EncodeVector(embedding)
	res, err := s.db.ExecContext(ctx,
		`UPDATE episodes SET embedding = ? WHERE id = ?`,
		embBlob, id,
	)
	if err != nil {
		return fmt.Errorf("sqlite store: update episode embedding: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("sqlite store: update episode embedding: rows affected: %w", err)
	}
	if n == 0 {
		return fmt.Errorf("sqlite store: update episode embedding: episode %q not found", id)
	}
	return nil
}

// --- Content amendments (Phase 2D) ---

// UpdateFactContent updates content, content_hash, embedding, optionally tags,
// and accessed_at for the given fact. Other columns (cluster_id, created_at,
// valid_from, superseded_by, subtype, source, confidence) are preserved.
//
// Tags tri-state: a nil `tags` pointer preserves the existing row; a non-nil
// pointer replaces it — even when the pointed-to slice is empty (which clears
// the tag set).
func (s *sqliteStore) UpdateFactContent(ctx context.Context, id, content, contentHash string, embedding []float32, tags *[]string) error {
	now := time.Now().UTC().Format(timeFormat)
	embBlob := EncodeVector(embedding)

	if tags != nil {
		normTags, err := normalizeTags(*tags)
		if err != nil {
			return fmt.Errorf("sqlite store: update fact content: %w", err)
		}
		tagsJSON, err := encodeTags(normTags)
		if err != nil {
			return fmt.Errorf("sqlite store: update fact content: %w", err)
		}
		res, err := s.db.ExecContext(ctx,
			`UPDATE facts SET content = ?, content_hash = ?, embedding = ?, tags = ?, accessed_at = ?
			 WHERE id = ?`,
			content, contentHash, embBlob, tagsJSON, now, id,
		)
		if err != nil {
			return fmt.Errorf("sqlite store: update fact content: %w", err)
		}
		n, err := res.RowsAffected()
		if err != nil {
			return fmt.Errorf("sqlite store: update fact content: rows affected: %w", err)
		}
		if n == 0 {
			return fmt.Errorf("sqlite store: update fact content: fact %q not found", id)
		}
		return nil
	}

	res, err := s.db.ExecContext(ctx,
		`UPDATE facts SET content = ?, content_hash = ?, embedding = ?, accessed_at = ?
		 WHERE id = ?`,
		content, contentHash, embBlob, now, id,
	)
	if err != nil {
		return fmt.Errorf("sqlite store: update fact content: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("sqlite store: update fact content: rows affected: %w", err)
	}
	if n == 0 {
		return fmt.Errorf("sqlite store: update fact content: fact %q not found", id)
	}
	return nil
}

// UpdateEpisodeContent amends the situation/action/outcome/preemptive fields,
// the embedding, the content_hash, and the tags of an episode. accessed_at is
// bumped to now. ID, cluster_id, and created_at are preserved.
//
// The caller must pre-populate e.Tags with either the existing tag set (to
// preserve) or the new tag set (to replace). Implementations cannot
// distinguish a nil vs zero-length slice through the value form, so the
// handler is the source of truth on preservation.
func (s *sqliteStore) UpdateEpisodeContent(ctx context.Context, id string, e Episode) error {
	now := time.Now().UTC().Format(timeFormat)
	embBlob := EncodeVector(e.Embedding)

	normTags, err := normalizeTags(e.Tags)
	if err != nil {
		return fmt.Errorf("sqlite store: update episode content: %w", err)
	}
	tagsJSON, err := encodeTags(normTags)
	if err != nil {
		return fmt.Errorf("sqlite store: update episode content: %w", err)
	}

	res, err := s.db.ExecContext(ctx,
		`UPDATE episodes
		 SET situation = ?, action = ?, outcome = ?, preemptive = ?,
		     embedding = ?, content_hash = ?, tags = ?, accessed_at = ?
		 WHERE id = ?`,
		e.Situation, e.Action, e.Outcome, e.Preemptive,
		embBlob, e.ContentHash, tagsJSON, now, id,
	)
	if err != nil {
		return fmt.Errorf("sqlite store: update episode content: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("sqlite store: update episode content: rows affected: %w", err)
	}
	if n == 0 {
		return fmt.Errorf("sqlite store: update episode content: episode %q not found", id)
	}
	return nil
}

// ReplaceEpisodeLinks atomically replaces the set of evidence edges
// (memory_edges rows where dst_id=episodeID and edge_type='evidence') with
// one row per factID. nil is equivalent to an empty slice (clears links) —
// callers that want to preserve existing links must skip calling this
// method.
func (s *sqliteStore) ReplaceEpisodeLinks(ctx context.Context, episodeID string, factIDs []string) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite store: replace episode links: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	if _, err := tx.ExecContext(ctx,
		`DELETE FROM memory_edges WHERE dst_id = ? AND edge_type = 'evidence'`, episodeID,
	); err != nil {
		return fmt.Errorf("sqlite store: replace episode links: delete: %w", err)
	}

	now := time.Now().UTC().Format(timeFormat)
	for _, factID := range factIDs {
		if _, err := tx.ExecContext(ctx,
			`INSERT OR IGNORE INTO memory_edges (src_id, dst_id, edge_type, weight, created_at) VALUES (?, ?, ?, ?, ?)`,
			factID, episodeID, "evidence", 1.0, now,
		); err != nil {
			return fmt.Errorf("sqlite store: replace episode links: insert fact %s: %w", factID, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("sqlite store: replace episode links: commit: %w", err)
	}
	return nil
}

// --- Temporal conflict resolution ---

func (s *sqliteStore) SupersedeFact(ctx context.Context, oldID, newID string) error {
	res, err := s.db.ExecContext(ctx,
		`UPDATE facts SET superseded_by = ? WHERE id = ?`,
		newID, oldID,
	)
	if err != nil {
		return fmt.Errorf("sqlite store: supersede fact: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("sqlite store: supersede fact: rows affected: %w", err)
	}
	if n == 0 {
		return fmt.Errorf("sqlite store: supersede fact: fact %q not found", oldID)
	}
	return nil
}

// ClearFactSuperseded sets the superseded_by column back to NULL on the given
// fact, bumps accessed_at, and returns the previous superseded_by value.
//
// Errors:
//   - "fact not found: <id>" if no row exists for id.
//   - "fact is not superseded" if the row exists but superseded_by is NULL.
//
// Implementation note: this is a read-then-update without an explicit
// transaction. Contention on reversing a supersede is effectively zero
// (it's an operator-driven correction, not an automatic path), so the
// extra overhead of a tx is not warranted.
func (s *sqliteStore) ClearFactSuperseded(ctx context.Context, id string) (string, error) {
	var supersededBy sql.NullString
	err := s.db.QueryRowContext(ctx,
		`SELECT superseded_by FROM facts WHERE id = ?`, id,
	).Scan(&supersededBy)
	if err == sql.ErrNoRows {
		return "", fmt.Errorf("fact not found: %s", id)
	}
	if err != nil {
		return "", fmt.Errorf("sqlite store: clear fact superseded: %w", err)
	}
	if !supersededBy.Valid || supersededBy.String == "" {
		return "", fmt.Errorf("fact is not superseded")
	}

	now := time.Now().UTC().Format(timeFormat)
	if _, err := s.db.ExecContext(ctx,
		`UPDATE facts SET superseded_by = NULL, accessed_at = ? WHERE id = ?`,
		now, id,
	); err != nil {
		return "", fmt.Errorf("sqlite store: clear fact superseded: %w", err)
	}
	return supersededBy.String, nil
}

func (s *sqliteStore) GetFactSupersedes(ctx context.Context, id string) ([]string, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id FROM facts WHERE superseded_by = ?`, id,
	)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: get fact supersedes: %w", err)
	}
	defer rows.Close()

	ids := []string{}
	for rows.Next() {
		var fid string
		if err := rows.Scan(&fid); err != nil {
			return nil, fmt.Errorf("sqlite store: get fact supersedes: scan: %w", err)
		}
		ids = append(ids, fid)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite store: get fact supersedes: rows: %w", err)
	}
	return ids, nil
}

func (s *sqliteStore) FindSimilarFacts(ctx context.Context, subtype string, queryVec []float32, threshold float32, limit int) ([]Candidate, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, cluster_id, content, embedding, content_hash, subtype, source, confidence, valid_from, superseded_by, created_at, accessed_at, tags
		 FROM facts WHERE superseded_by IS NULL AND embedding IS NOT NULL AND subtype = ?`,
		subtype,
	)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: find similar facts: %w", err)
	}
	defer rows.Close()

	var candidates []Candidate
	for rows.Next() {
		f, err := scanFactRows(rows)
		if err != nil {
			return nil, fmt.Errorf("sqlite store: find similar facts: scan: %w", err)
		}
		sim := vecmath.Cosine(queryVec, f.Embedding)
		if sim >= threshold {
			candidates = append(candidates, Candidate{Fact: f, Similarity: sim})
		}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite store: find similar facts: rows: %w", err)
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Similarity > candidates[j].Similarity
	})

	if limit > 0 && len(candidates) > limit {
		candidates = candidates[:limit]
	}
	return candidates, nil
}

// ListDailyStats returns rows from daily_stats where date falls in [from, to]
// inclusive. Dates are "YYYY-MM-DD" UTC strings; the caller is responsible
// for validating the range and for zero-filling gaps. Rows are sorted
// ascending by date (oldest first). Returns an empty slice (not nil) when
// no rows match.
func (s *sqliteStore) ListDailyStats(ctx context.Context, from, to string) ([]DailyStats, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT date, facts_in, facts_out, episodes_in, episodes_out, supersedes
		 FROM daily_stats
		 WHERE date BETWEEN ? AND ?
		 ORDER BY date ASC`,
		from, to,
	)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: list daily stats: %w", err)
	}
	defer rows.Close()

	out := []DailyStats{}
	for rows.Next() {
		var d DailyStats
		if err := rows.Scan(&d.Date, &d.FactsIn, &d.FactsOut, &d.EpisodesIn, &d.EpisodesOut, &d.Supersedes); err != nil {
			return nil, fmt.Errorf("sqlite store: list daily stats: scan: %w", err)
		}
		out = append(out, d)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite store: list daily stats: rows: %w", err)
	}
	return out, nil
}

// GetLastTick returns the last decay tick timestamp from the decay_state
// singleton row. A NULL last_tick (fresh DB, no tick ever) is surfaced as a
// zero-value time.Time so the caller can check with .IsZero() without
// disambiguating sentinel errors. RFC3339 is the on-disk format (same as the
// rest of this package's timestamps).
func (s *sqliteStore) GetLastTick(ctx context.Context) (time.Time, error) {
	var raw sql.NullString
	err := s.db.QueryRowContext(ctx,
		`SELECT last_tick FROM decay_state WHERE id = 1`,
	).Scan(&raw)
	if err == sql.ErrNoRows {
		// Migration 3 seeds id=1, so this should not happen in production.
		// Treat it the same as a NULL last_tick rather than erroring — the
		// status handler just wants "zero or a timestamp".
		return time.Time{}, nil
	}
	if err != nil {
		return time.Time{}, fmt.Errorf("sqlite store: get last tick: %w", err)
	}
	if !raw.Valid || raw.String == "" {
		return time.Time{}, nil
	}
	t, perr := time.Parse(timeFormat, raw.String)
	if perr != nil {
		return time.Time{}, fmt.Errorf("sqlite store: get last tick: parse %q: %w", raw.String, perr)
	}
	return t, nil
}

// SetLastTick persists the tick timestamp in ISO8601 UTC (RFC3339) on the
// singleton decay_state row. The row is seeded by migration 3, so UPDATE
// without fallback INSERT is sufficient.
func (s *sqliteStore) SetLastTick(ctx context.Context, t time.Time) error {
	_, err := s.db.ExecContext(ctx,
		`UPDATE decay_state SET last_tick = ? WHERE id = 1`,
		t.UTC().Format(timeFormat),
	)
	if err != nil {
		return fmt.Errorf("sqlite store: set last tick: %w", err)
	}
	return nil
}

// SupersedeLongestChain computes the length of the longest chain of facts
// linked by superseded_by. The recursive CTE starts from terminals (facts
// that are themselves NOT superseded but that ARE the target of at least one
// supersede edge) at depth=1 and walks backward through superseded_by edges,
// incrementing depth per hop.
//
// Semantics: for A -> B -> C where A.superseded_by=B and B.superseded_by=C,
// C is the terminal head of the chain (depth=1), B is the intermediate
// (depth=2), and A is the original (depth=3). The function returns 3.
// Intuition: the number is the total count of facts participating in the
// longest chain, including the currently-active head.
//
// NOTE (5A timeout test skipped): the 100ms guardrail below is hard to
// exercise without injecting a fake slow query hook. Given the CTE is
// trivially fast at realistic scale and the handler already degrades
// gracefully on DeadlineExceeded, the integration test for that branch was
// intentionally skipped; see the 5A task brief in docs/design/.
func (s *sqliteStore) SupersedeLongestChain(ctx context.Context) (int, error) {
	childCtx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
	defer cancel()

	const query = `
WITH RECURSIVE chain(id, depth) AS (
  SELECT id, 1 FROM facts
    WHERE superseded_by IS NULL
      AND id IN (
        SELECT superseded_by FROM facts WHERE superseded_by IS NOT NULL
      )
  UNION ALL
  SELECT f.id, chain.depth + 1 FROM facts f
    JOIN chain ON f.superseded_by = chain.id
)
SELECT COALESCE(MAX(depth), 0) FROM chain`

	var n int
	err := s.db.QueryRowContext(childCtx, query).Scan(&n)
	if err != nil {
		// Propagate DeadlineExceeded untouched so the status handler can tag
		// the subsection degraded.
		if childCtx.Err() == context.DeadlineExceeded {
			return 0, context.DeadlineExceeded
		}
		return 0, fmt.Errorf("sqlite store: supersede longest chain: %w", err)
	}
	return n, nil
}

// CountSupersededFacts returns the number of facts with a non-NULL
// superseded_by. Used by the status resource's supersede subsection.
func (s *sqliteStore) CountSupersededFacts(ctx context.Context) (int, error) {
	var n int
	err := s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM facts WHERE superseded_by IS NOT NULL`,
	).Scan(&n)
	if err != nil {
		return 0, fmt.Errorf("sqlite store: count superseded facts: %w", err)
	}
	return n, nil
}

// --- Session CRUD (Phase 6b) ---

func (s *sqliteStore) GetSession(ctx context.Context, id string) (*Session, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, project_hint, tags, working_memory, created_at, updated_at, closed_at
		 FROM sessions WHERE id = ?`, id,
	)

	var sess Session
	var projectHint, wmRaw, tagsRaw, createdStr, updatedStr string
	var closedAt sql.NullString
	if err := row.Scan(&sess.ID, &projectHint, &tagsRaw, &wmRaw, &createdStr, &updatedStr, &closedAt); err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("sqlite store: get session: %w", err)
	}
	sess.ProjectHint = projectHint

	tags, err := decodeTags(tagsRaw)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: get session: %w", err)
	}
	sess.Tags = tags

	wm, err := decodeWorkingMemory(wmRaw)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: get session: %w", err)
	}
	sess.WorkingMem = wm

	sess.CreatedAt, _ = time.Parse(timeFormat, createdStr)
	// updated_at is stored as datetime('now') in SQLite's "YYYY-MM-DD HH:MM:SS"
	// format by the default; try that first, then RFC3339 for rows written
	// by the Go layer.
	if t, perr := time.Parse(timeFormat, updatedStr); perr == nil {
		sess.UpdatedAt = t
	} else if t2, perr2 := time.Parse("2006-01-02 15:04:05", updatedStr); perr2 == nil {
		sess.UpdatedAt = t2.UTC()
	}
	if closedAt.Valid && closedAt.String != "" {
		var ct time.Time
		if t, perr := time.Parse(timeFormat, closedAt.String); perr == nil {
			ct = t
		} else if t2, perr2 := time.Parse("2006-01-02 15:04:05", closedAt.String); perr2 == nil {
			ct = t2.UTC()
		}
		if !ct.IsZero() {
			sess.ClosedAt = &ct
		}
	}

	return &sess, nil
}

func (s *sqliteStore) CreateSession(ctx context.Context, sess Session) error {
	if sess.ID == "" {
		return fmt.Errorf("sqlite store: create session: id is required")
	}
	normTags, err := normalizeTags(sess.Tags)
	if err != nil {
		return fmt.Errorf("sqlite store: create session: %w", err)
	}
	tagsJSON, err := encodeTags(normTags)
	if err != nil {
		return fmt.Errorf("sqlite store: create session: %w", err)
	}
	wmJSON, err := encodeWorkingMemory(sess.WorkingMem)
	if err != nil {
		return fmt.Errorf("sqlite store: create session: %w", err)
	}

	now := time.Now().UTC()
	if sess.CreatedAt.IsZero() {
		sess.CreatedAt = now
	}
	if sess.UpdatedAt.IsZero() {
		sess.UpdatedAt = now
	}

	_, err = s.db.ExecContext(ctx,
		`INSERT INTO sessions (id, turn_counter, working_memory, project_hint, tags, created_at, updated_at, closed_at)
		 VALUES (?, 0, ?, ?, ?, ?, ?, NULL)`,
		sess.ID, wmJSON, sess.ProjectHint, tagsJSON,
		sess.CreatedAt.Format(timeFormat),
		sess.UpdatedAt.Format(timeFormat),
	)
	if err != nil {
		// SQLite raises a unique constraint error for duplicate primary
		// keys; surface that as a stable error message so callers can key
		// off it without importing the driver package.
		if strings.Contains(err.Error(), "UNIQUE") || strings.Contains(err.Error(), "constraint") {
			return fmt.Errorf("sqlite store: create session: session already exists: %s", sess.ID)
		}
		return fmt.Errorf("sqlite store: create session: %w", err)
	}
	return nil
}

func (s *sqliteStore) UpdateSessionBuffer(ctx context.Context, id string, wm WorkingMemory) error {
	wmJSON, err := encodeWorkingMemory(wm)
	if err != nil {
		return fmt.Errorf("sqlite store: update session buffer: %w", err)
	}
	res, err := s.db.ExecContext(ctx,
		`UPDATE sessions SET working_memory = ?, updated_at = ? WHERE id = ?`,
		wmJSON, time.Now().UTC().Format(timeFormat), id,
	)
	if err != nil {
		return fmt.Errorf("sqlite store: update session buffer: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("sqlite store: update session buffer: rows affected: %w", err)
	}
	if n == 0 {
		return fmt.Errorf("sqlite store: update session buffer: session %q not found", id)
	}
	return nil
}

func (s *sqliteStore) UpdateSessionMeta(ctx context.Context, id string, projectHint string, tags []string) error {
	normTags, err := normalizeTags(tags)
	if err != nil {
		return fmt.Errorf("sqlite store: update session meta: %w", err)
	}
	tagsJSON, err := encodeTags(normTags)
	if err != nil {
		return fmt.Errorf("sqlite store: update session meta: %w", err)
	}
	res, err := s.db.ExecContext(ctx,
		`UPDATE sessions SET project_hint = ?, tags = ?, updated_at = ? WHERE id = ?`,
		projectHint, tagsJSON, time.Now().UTC().Format(timeFormat), id,
	)
	if err != nil {
		return fmt.Errorf("sqlite store: update session meta: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("sqlite store: update session meta: rows affected: %w", err)
	}
	if n == 0 {
		return fmt.Errorf("sqlite store: update session meta: session %q not found", id)
	}
	return nil
}

func (s *sqliteStore) CloseSession(ctx context.Context, id string) error {
	// Look up to distinguish not-found from already-closed: the former is
	// an error (protects callers from typos), the latter is idempotent.
	var closedAt sql.NullString
	err := s.db.QueryRowContext(ctx,
		`SELECT closed_at FROM sessions WHERE id = ?`, id,
	).Scan(&closedAt)
	if err == sql.ErrNoRows {
		return fmt.Errorf("sqlite store: close session: session %q not found", id)
	}
	if err != nil {
		return fmt.Errorf("sqlite store: close session: %w", err)
	}
	if closedAt.Valid && closedAt.String != "" {
		return nil // already closed — idempotent
	}

	now := time.Now().UTC().Format(timeFormat)
	_, err = s.db.ExecContext(ctx,
		`UPDATE sessions SET closed_at = ?, updated_at = ? WHERE id = ?`,
		now, now, id,
	)
	if err != nil {
		return fmt.Errorf("sqlite store: close session: %w", err)
	}
	return nil
}

func (s *sqliteStore) Close() error {
	return s.db.Close()
}

// nullableString returns a *string if s is non-empty, else nil (for nullable TEXT cols).
func nullableString(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// scanner is an interface satisfied by both *sql.Row and *sql.Rows.
type scanner interface {
	Scan(dest ...any) error
}

// scanFactFrom scans a Fact from a scanner (Row or Rows).
func scanFactFrom(sc scanner) (*Fact, error) {
	var f Fact
	var embBlob []byte
	var subtype sql.NullString
	var supersededBy sql.NullString
	var validFromStr, createdStr, accessedStr string
	var tagsRaw sql.NullString

	err := sc.Scan(
		&f.ID, &f.ClusterID, &f.Content, &embBlob, &f.ContentHash,
		&subtype, &f.Source, &f.Confidence, &validFromStr,
		&supersededBy, &createdStr, &accessedStr, &tagsRaw,
	)
	if err != nil {
		return nil, err
	}

	f.Embedding = DecodeVector(embBlob)
	if subtype.Valid {
		f.Subtype = subtype.String
	}
	if supersededBy.Valid {
		f.SupersededBy = &supersededBy.String
	}

	f.ValidFrom, _ = time.Parse(timeFormat, validFromStr)
	f.CreatedAt, _ = time.Parse(timeFormat, createdStr)
	f.AccessedAt, _ = time.Parse(timeFormat, accessedStr)

	tags, decErr := decodeTags(tagsRaw.String)
	if decErr != nil {
		return nil, decErr
	}
	f.Tags = tags

	return &f, nil
}

// scanFact scans a single fact from a *sql.Row.
func scanFact(row *sql.Row) (*Fact, error) {
	return scanFactFrom(row)
}

// scanFactRows scans a single fact from *sql.Rows.
func scanFactRows(rows *sql.Rows) (*Fact, error) {
	return scanFactFrom(rows)
}

// scanEpisodeFrom scans an Episode from a scanner (Row or Rows).
func scanEpisodeFrom(sc scanner) (*Episode, error) {
	var ep Episode
	var embBlob []byte
	var createdStr, accessedStr string
	var tagsRaw sql.NullString

	err := sc.Scan(
		&ep.ID, &ep.ClusterID, &ep.Situation, &ep.Action, &ep.Outcome, &ep.Preemptive,
		&embBlob, &ep.ContentHash, &createdStr, &accessedStr, &tagsRaw,
	)
	if err != nil {
		return nil, err
	}

	ep.Embedding = DecodeVector(embBlob)
	ep.CreatedAt, _ = time.Parse(timeFormat, createdStr)
	ep.AccessedAt, _ = time.Parse(timeFormat, accessedStr)

	tags, decErr := decodeTags(tagsRaw.String)
	if decErr != nil {
		return nil, decErr
	}
	ep.Tags = tags

	return &ep, nil
}

// scanEpisodeRows scans a single episode from *sql.Rows.
func scanEpisodeRows(rows *sql.Rows) (*Episode, error) {
	return scanEpisodeFrom(rows)
}
