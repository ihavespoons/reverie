package memory

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
	"personal/reverie/pkg/vecmath"
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

	// Idempotency: check for existing non-superseded fact with same hash.
	var existingID string
	err := s.db.QueryRowContext(ctx,
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
		`INSERT INTO facts (id, cluster_id, content, embedding, content_hash, subtype, source, confidence, valid_from, superseded_by, created_at, accessed_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
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
	)
	if err != nil {
		return "", fmt.Errorf("sqlite store: insert fact: %w", err)
	}
	return f.ID, nil
}

func (s *sqliteStore) GetFact(ctx context.Context, id string) (*Fact, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, cluster_id, content, embedding, content_hash, subtype, source, confidence, valid_from, superseded_by, created_at, accessed_at
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
	_, err := s.db.ExecContext(ctx, `DELETE FROM facts WHERE id = ?`, id)
	if err != nil {
		return fmt.Errorf("sqlite store: delete fact: %w", err)
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

	query := `SELECT id, cluster_id, content, embedding, content_hash, subtype, source, confidence, valid_from, superseded_by, created_at, accessed_at
	          FROM facts WHERE superseded_by IS NULL`
	args := []any{}

	if filter.Subtype != nil {
		query += ` AND subtype = ?`
		args = append(args, *filter.Subtype)
	}

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
		`SELECT id, cluster_id, content, embedding, content_hash, subtype, source, confidence, valid_from, superseded_by, created_at, accessed_at
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
		`SELECT id, cluster_id, situation, action, outcome, preemptive, embedding, content_hash, created_at, accessed_at
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

	// Ensure the cluster exists (FK constraint).
	if e.ClusterID == defaultClusterID {
		if err := s.ensureDefaultCluster(ctx); err != nil {
			return "", err
		}
	}

	embBlob := EncodeVector(e.Embedding)

	_, err := s.db.ExecContext(ctx,
		`INSERT INTO episodes (id, cluster_id, situation, action, outcome, preemptive, embedding, content_hash, created_at, accessed_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
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
	)
	if err != nil {
		return "", fmt.Errorf("sqlite store: insert episode: %w", err)
	}

	// Insert cross-links for any linked fact IDs.
	for _, factID := range e.LinkedFactIDs {
		_, err := s.db.ExecContext(ctx,
			`INSERT OR IGNORE INTO fact_episode_links (fact_id, episode_id, link_type) VALUES (?, ?, ?)`,
			factID, e.ID, "evidence",
		)
		if err != nil {
			return "", fmt.Errorf("sqlite store: insert episode: link fact %s: %w", factID, err)
		}
	}

	return e.ID, nil
}

func (s *sqliteStore) GetEpisode(ctx context.Context, id string) (*Episode, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, cluster_id, situation, action, outcome, preemptive, embedding, content_hash, created_at, accessed_at
		 FROM episodes WHERE id = ?`, id,
	)
	ep, err := scanEpisodeFrom(row)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("sqlite store: get episode: %w", err)
	}

	// Load linked fact IDs.
	linkRows, err := s.db.QueryContext(ctx,
		`SELECT fact_id FROM fact_episode_links WHERE episode_id = ?`, id,
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
	_, err := s.db.ExecContext(ctx, `DELETE FROM episodes WHERE id = ?`, id)
	if err != nil {
		return fmt.Errorf("sqlite store: delete episode: %w", err)
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
		`SELECT id, cluster_id, situation, action, outcome, preemptive, embedding, content_hash, created_at, accessed_at
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
		episodes = append(episodes, *ep)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite store: list episodes: rows: %w", err)
	}
	return episodes, nil
}

// --- Fact <-> Episode cross-type links ---

func (s *sqliteStore) LinkFactEpisode(ctx context.Context, factID, episodeID, linkType string) error {
	if linkType == "" {
		linkType = "evidence"
	}
	_, err := s.db.ExecContext(ctx,
		`INSERT OR IGNORE INTO fact_episode_links (fact_id, episode_id, link_type) VALUES (?, ?, ?)`,
		factID, episodeID, linkType,
	)
	if err != nil {
		return fmt.Errorf("sqlite store: link fact episode: %w", err)
	}
	return nil
}

func (s *sqliteStore) GetFactLinks(ctx context.Context, factID string) ([]EpisodeLink, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT l.episode_id, l.link_type,
		        e.id, e.cluster_id, e.situation, e.action, e.outcome, e.preemptive,
		        e.embedding, e.content_hash, e.created_at, e.accessed_at
		 FROM fact_episode_links l
		 JOIN episodes e ON e.id = l.episode_id
		 WHERE l.fact_id = ?`, factID,
	)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: get fact links: %w", err)
	}
	defer rows.Close()

	var links []EpisodeLink
	for rows.Next() {
		var link EpisodeLink
		var ep Episode
		var embBlob []byte
		var createdStr, accessedStr string

		err := rows.Scan(
			&link.EpisodeID, &link.LinkType,
			&ep.ID, &ep.ClusterID, &ep.Situation, &ep.Action, &ep.Outcome, &ep.Preemptive,
			&embBlob, &ep.ContentHash, &createdStr, &accessedStr,
		)
		if err != nil {
			return nil, fmt.Errorf("sqlite store: get fact links: scan: %w", err)
		}
		ep.Embedding = DecodeVector(embBlob)
		ep.CreatedAt, _ = time.Parse(timeFormat, createdStr)
		ep.AccessedAt, _ = time.Parse(timeFormat, accessedStr)
		link.Episode = &ep
		links = append(links, link)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite store: get fact links: rows: %w", err)
	}
	return links, nil
}

func (s *sqliteStore) GetEpisodeLinks(ctx context.Context, episodeID string) ([]FactLink, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT l.fact_id, l.link_type,
		        f.id, f.cluster_id, f.content, f.embedding, f.content_hash,
		        f.subtype, f.source, f.confidence, f.valid_from,
		        f.superseded_by, f.created_at, f.accessed_at
		 FROM fact_episode_links l
		 JOIN facts f ON f.id = l.fact_id
		 WHERE l.episode_id = ?`, episodeID,
	)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: get episode links: %w", err)
	}
	defer rows.Close()

	var links []FactLink
	for rows.Next() {
		var link FactLink
		var f Fact
		var embBlob []byte
		var subtype, supersededBy sql.NullString
		var validFromStr, createdStr, accessedStr string

		err := rows.Scan(
			&link.FactID, &link.LinkType,
			&f.ID, &f.ClusterID, &f.Content, &embBlob, &f.ContentHash,
			&subtype, &f.Source, &f.Confidence, &validFromStr,
			&supersededBy, &createdStr, &accessedStr,
		)
		if err != nil {
			return nil, fmt.Errorf("sqlite store: get episode links: scan: %w", err)
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
		link.Fact = &f
		links = append(links, link)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite store: get episode links: rows: %w", err)
	}
	return links, nil
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

func (s *sqliteStore) FindSimilarFacts(ctx context.Context, subtype string, queryVec []float32, threshold float32, limit int) ([]Candidate, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, cluster_id, content, embedding, content_hash, subtype, source, confidence, valid_from, superseded_by, created_at, accessed_at
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

	err := sc.Scan(
		&f.ID, &f.ClusterID, &f.Content, &embBlob, &f.ContentHash,
		&subtype, &f.Source, &f.Confidence, &validFromStr,
		&supersededBy, &createdStr, &accessedStr,
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

	err := sc.Scan(
		&ep.ID, &ep.ClusterID, &ep.Situation, &ep.Action, &ep.Outcome, &ep.Preemptive,
		&embBlob, &ep.ContentHash, &createdStr, &accessedStr,
	)
	if err != nil {
		return nil, err
	}

	ep.Embedding = DecodeVector(embBlob)
	ep.CreatedAt, _ = time.Parse(timeFormat, createdStr)
	ep.AccessedAt, _ = time.Parse(timeFormat, accessedStr)

	return &ep, nil
}

// scanEpisode scans a single episode from a *sql.Row.
func scanEpisode(row *sql.Row) (*Episode, error) {
	return scanEpisodeFrom(row)
}

// scanEpisodeRows scans a single episode from *sql.Rows.
func scanEpisodeRows(rows *sql.Rows) (*Episode, error) {
	return scanEpisodeFrom(rows)
}
