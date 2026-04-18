package embed

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"fmt"
	"strings"

	"personal/reverie/internal/memory"
)

// cachedProvider wraps an EmbeddingProvider with a SHA-256-keyed SQLite cache
// backed by the embedding_cache table.
type cachedProvider struct {
	inner EmbeddingProvider
	db    *sql.DB
}

// NewCachedProvider wraps an EmbeddingProvider so that embeddings are cached in
// the embedding_cache table of the given database. Cache keys are
// sha256(model + "|" + text) in hex. Cached vectors are encoded/decoded using
// memory.EncodeVector / memory.DecodeVector.
func NewCachedProvider(inner EmbeddingProvider, db *sql.DB) EmbeddingProvider {
	return &cachedProvider{inner: inner, db: db}
}

// contentHash computes the cache key for a text under the given model.
func contentHash(model, text string) string {
	h := sha256.Sum256([]byte(model + "|" + text))
	return fmt.Sprintf("%x", h)
}

// Embed returns embeddings for the given texts, using the cache where possible.
// Only uncached texts are sent to the inner provider. Results are returned in
// the same order as the input texts.
func (c *cachedProvider) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	model := c.inner.Model()
	results := make([][]float32, len(texts))

	// Compute hashes for all texts.
	hashes := make([]string, len(texts))
	hashToIndices := make(map[string][]int, len(texts))
	for i, t := range texts {
		h := contentHash(model, t)
		hashes[i] = h
		hashToIndices[h] = append(hashToIndices[h], i)
	}

	// Look up cached embeddings in a single query.
	cached, err := c.lookupCached(model, hashes)
	if err != nil {
		return nil, fmt.Errorf("embed cache: lookup: %w", err)
	}

	// Fill in cached results and identify misses.
	var missIndices []int
	for i, h := range hashes {
		if vec, ok := cached[h]; ok {
			results[i] = vec
		} else {
			missIndices = append(missIndices, i)
		}
	}

	if len(missIndices) == 0 {
		return results, nil
	}

	// Collect the texts that need embedding.
	missTexts := make([]string, len(missIndices))
	for i, idx := range missIndices {
		missTexts[i] = texts[idx]
	}

	// Call the inner provider for misses.
	newVecs, err := c.inner.Embed(ctx, missTexts)
	if err != nil {
		return nil, fmt.Errorf("embed cache: inner embed: %w", err)
	}

	// Store new embeddings in the cache and fill results.
	for i, idx := range missIndices {
		results[idx] = newVecs[i]
		h := hashes[idx]
		if err := c.insertCached(model, h, newVecs[i]); err != nil {
			return nil, fmt.Errorf("embed cache: insert: %w", err)
		}
	}

	return results, nil
}

// lookupCached performs a single SELECT for a batch of content hashes.
func (c *cachedProvider) lookupCached(model string, hashes []string) (map[string][]float32, error) {
	if len(hashes) == 0 {
		return nil, nil
	}

	// De-duplicate hashes for the query.
	unique := make(map[string]struct{}, len(hashes))
	for _, h := range hashes {
		unique[h] = struct{}{}
	}

	placeholders := make([]string, 0, len(unique))
	args := make([]any, 0, len(unique)+1)
	args = append(args, model)
	for h := range unique {
		placeholders = append(placeholders, "?")
		args = append(args, h)
	}

	query := "SELECT content_hash, embedding FROM embedding_cache WHERE model = ? AND content_hash IN (" +
		strings.Join(placeholders, ",") + ")"

	rows, err := c.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	result := make(map[string][]float32)
	for rows.Next() {
		var hash string
		var blob []byte
		if err := rows.Scan(&hash, &blob); err != nil {
			return nil, err
		}
		result[hash] = memory.DecodeVector(blob)
	}
	return result, rows.Err()
}

// insertCached inserts a single embedding into the cache. Uses INSERT OR IGNORE
// to handle concurrent inserts of the same key gracefully.
func (c *cachedProvider) insertCached(model, hash string, vec []float32) error {
	_, err := c.db.Exec(
		"INSERT OR IGNORE INTO embedding_cache (content_hash, model, embedding) VALUES (?, ?, ?)",
		hash, model, memory.EncodeVector(vec),
	)
	return err
}

// Dimensions delegates to the inner provider.
func (c *cachedProvider) Dimensions() int {
	return c.inner.Dimensions()
}

// Model delegates to the inner provider.
func (c *cachedProvider) Model() string {
	return c.inner.Model()
}
