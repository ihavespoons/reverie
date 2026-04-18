package migrate

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/google/uuid"

	"personal/reverie/internal/cluster"
	"personal/reverie/internal/embed"
	"personal/reverie/internal/memory"
)

// ImportResult summarizes a migration run.
type ImportResult struct {
	FilesScanned int
	FactsCreated int
	Skipped      int      // duplicate content_hash
	Errors       []string // non-fatal per-file errors
}

// merge adds the counts from another result into this one.
func (r *ImportResult) merge(other *ImportResult) {
	r.FilesScanned += other.FilesScanned
	r.FactsCreated += other.FactsCreated
	r.Skipped += other.Skipped
	r.Errors = append(r.Errors, other.Errors...)
}

// Importer reads Claude Code auto-memory markdown files and writes them
// into reverie's store as L2 semantic facts.
type Importer struct {
	store    memory.Store
	embedder embed.EmbeddingProvider
	assigner cluster.Assigner
	logger   func(string, ...any) // printf-style logger
}

// NewImporter constructs an Importer.
func NewImporter(store memory.Store, embedder embed.EmbeddingProvider, assigner cluster.Assigner, logger func(string, ...any)) *Importer {
	return &Importer{
		store:    store,
		embedder: embedder,
		assigner: assigner,
		logger:   logger,
	}
}

// ImportDir processes all .md files in a single memory directory,
// skipping MEMORY.md index files.
func (imp *Importer) ImportDir(ctx context.Context, dir string) (*ImportResult, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read directory %q: %w", dir, err)
	}

	result := &ImportResult{}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()

		// Only process .md files.
		if !strings.HasSuffix(strings.ToLower(name), ".md") {
			continue
		}

		// Skip MEMORY.md index files (case-insensitive).
		if strings.EqualFold(name, "MEMORY.md") {
			continue
		}

		result.FilesScanned++
		path := filepath.Join(dir, name)

		if err := imp.importFile(ctx, path, dir, result); err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("%s: %v", path, err))
		}
	}

	return result, nil
}

// ImportAllProjects walks ~/.claude/projects/*/memory/ and imports
// all auto-memory files found across all projects.
func (imp *Importer) ImportAllProjects(ctx context.Context, claudeDir string) (*ImportResult, error) {
	result := &ImportResult{}

	// List project directories under claudeDir.
	entries, err := os.ReadDir(claudeDir)
	if err != nil {
		return nil, fmt.Errorf("read projects directory %q: %w", claudeDir, err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		memDir := filepath.Join(claudeDir, entry.Name(), "memory")
		info, err := os.Stat(memDir)
		if err != nil || !info.IsDir() {
			continue // no memory/ subdirectory; skip silently
		}

		imp.logger("scanning project %s\n", entry.Name())

		dirResult, err := imp.ImportDir(ctx, memDir)
		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("project %s: %v", entry.Name(), err))
			continue
		}
		result.merge(dirResult)
	}

	return result, nil
}

// importFile processes a single auto-memory .md file.
func (imp *Importer) importFile(ctx context.Context, path, dir string, result *ImportResult) error {
	parsed, err := ParseMemoryFile(path)
	if err != nil {
		return err
	}

	if parsed.Body == "" {
		imp.logger("  skipping %s (empty body)\n", filepath.Base(path))
		return nil
	}

	subtype := parsed.Type
	if subtype == "" {
		imp.logger("  warning: %s has no type field in frontmatter\n", filepath.Base(path))
	}

	imp.logger("  importing %s (%s)...\n", filepath.Base(path), subtype)

	// Compute embedding.
	vecs, err := imp.embedder.Embed(ctx, []string{parsed.Body})
	if err != nil {
		return fmt.Errorf("embed: %w", err)
	}
	vec := vecs[0]

	// Assign to a cluster.
	clusterID, isNew, err := imp.assigner.Assign(ctx, vec)
	if err != nil {
		return fmt.Errorf("cluster assign: %w", err)
	}

	// Derive a source tag from the directory name and filename.
	source := "migrated:" + filepath.Base(dir) + "/" + filepath.Base(path)

	// Construct the fact with a known ID so we can detect duplicates.
	factID := uuid.New().String()
	fact := memory.Fact{
		ID:         factID,
		Content:    parsed.Body,
		Subtype:    subtype,
		Source:     source,
		Embedding:  vec,
		ClusterID:  clusterID,
		Confidence: 1.0,
	}

	returnedID, err := imp.store.InsertFact(ctx, fact)
	if err != nil {
		return fmt.Errorf("insert fact: %w", err)
	}

	if returnedID != factID {
		// InsertFact returned an existing ID — content hash already exists.
		result.Skipped++
		imp.logger("  skipped %s (duplicate)\n", filepath.Base(path))
		return nil
	}

	result.FactsCreated++
	imp.logger("  created %s (id=%s, cluster=%s)\n", filepath.Base(path), returnedID, clusterID)

	// Update cluster centroid if assigned to an existing cluster.
	if !isNew {
		if err := imp.assigner.AfterInsert(ctx, clusterID, vec); err != nil {
			return fmt.Errorf("after insert: %w", err)
		}
	}

	return nil
}
