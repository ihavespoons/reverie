package migrate

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/diffsec/reverie/internal/cluster"
	"github.com/diffsec/reverie/internal/memory"
)

// stubEmbedder returns a fixed-length vector for any input text.
// Every text gets the same vector, which is sufficient for testing
// the import pipeline without a real embedding model.
type stubEmbedder struct {
	dims int
}

func (s *stubEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	vecs := make([][]float32, len(texts))
	for i := range texts {
		vec := make([]float32, s.dims)
		// Fill with a small non-zero value so cosine is computable.
		for j := range vec {
			vec[j] = 0.1
		}
		vecs[i] = vec
	}
	return vecs, nil
}

func (s *stubEmbedder) Dimensions() int { return s.dims }
func (s *stubEmbedder) Model() string   { return "stub" }

// newTestImporter creates an Importer backed by in-memory store and stub embedder.
func newTestImporter(t *testing.T) (*Importer, memory.Store) {
	t.Helper()
	store := memory.NewMemStore()
	embedder := &stubEmbedder{dims: 8}
	assigner := cluster.NewAssigner(store, 0.60, 0.5, 0.5)
	logger := func(format string, args ...any) {
		t.Logf(format, args...)
	}
	return NewImporter(store, embedder, assigner, logger), store
}

func TestImportDir_AllTypes(t *testing.T) {
	imp, store := newTestImporter(t)
	ctx := context.Background()

	result, err := imp.ImportDir(ctx, "testdata/memory")
	if err != nil {
		t.Fatalf("ImportDir: %v", err)
	}

	if result.FilesScanned != 4 {
		t.Errorf("FilesScanned = %d, want 4", result.FilesScanned)
	}
	if result.FactsCreated != 4 {
		t.Errorf("FactsCreated = %d, want 4", result.FactsCreated)
	}
	if result.Skipped != 0 {
		t.Errorf("Skipped = %d, want 0", result.Skipped)
	}
	if len(result.Errors) != 0 {
		t.Errorf("Errors = %v, want empty", result.Errors)
	}

	// Verify all subtypes were stored correctly.
	subtypes := map[string]bool{"user": false, "project": false, "reference": false, "feedback": false}
	facts, err := store.ListFacts(ctx, memory.ListFilter{Limit: 100})
	if err != nil {
		t.Fatalf("ListFacts: %v", err)
	}
	if len(facts) != 4 {
		t.Fatalf("stored fact count = %d, want 4", len(facts))
	}
	for _, f := range facts {
		subtypes[f.Subtype] = true
		if f.Confidence != 1.0 {
			t.Errorf("fact %q confidence = %f, want 1.0", f.ID, f.Confidence)
		}
		if f.Source == "" {
			t.Errorf("fact %q has empty source", f.ID)
		}
	}
	for sub, found := range subtypes {
		if !found {
			t.Errorf("subtype %q not found in stored facts", sub)
		}
	}
}

func TestImportDir_Idempotent(t *testing.T) {
	imp, _ := newTestImporter(t)
	ctx := context.Background()

	// First import.
	result1, err := imp.ImportDir(ctx, "testdata/memory")
	if err != nil {
		t.Fatalf("first ImportDir: %v", err)
	}
	if result1.FactsCreated != 4 {
		t.Errorf("first run: FactsCreated = %d, want 4", result1.FactsCreated)
	}

	// Second import — all should be skipped.
	result2, err := imp.ImportDir(ctx, "testdata/memory")
	if err != nil {
		t.Fatalf("second ImportDir: %v", err)
	}
	if result2.FactsCreated != 0 {
		t.Errorf("second run: FactsCreated = %d, want 0", result2.FactsCreated)
	}
	if result2.Skipped != 4 {
		t.Errorf("second run: Skipped = %d, want 4", result2.Skipped)
	}
}

func TestImportDir_MissingFrontmatter(t *testing.T) {
	tmpDir := t.TempDir()

	// Write a file with no frontmatter.
	noFrontmatter := filepath.Join(tmpDir, "broken.md")
	if err := os.WriteFile(noFrontmatter, []byte("Just plain text, no frontmatter.\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Write a valid file alongside it.
	valid := filepath.Join(tmpDir, "valid.md")
	validContent := `---
name: Valid
description: A valid file
type: user
---
Valid content here.
`
	if err := os.WriteFile(valid, []byte(validContent), 0o644); err != nil {
		t.Fatal(err)
	}

	imp, _ := newTestImporter(t)
	ctx := context.Background()

	result, err := imp.ImportDir(ctx, tmpDir)
	if err != nil {
		t.Fatalf("ImportDir: %v", err)
	}

	if result.FilesScanned != 2 {
		t.Errorf("FilesScanned = %d, want 2", result.FilesScanned)
	}
	if result.FactsCreated != 1 {
		t.Errorf("FactsCreated = %d, want 1", result.FactsCreated)
	}
	if len(result.Errors) != 1 {
		t.Errorf("Errors count = %d, want 1", len(result.Errors))
	}
}

func TestImportDir_EmptyDir(t *testing.T) {
	tmpDir := t.TempDir()

	imp, _ := newTestImporter(t)
	ctx := context.Background()

	result, err := imp.ImportDir(ctx, tmpDir)
	if err != nil {
		t.Fatalf("ImportDir: %v", err)
	}

	if result.FilesScanned != 0 {
		t.Errorf("FilesScanned = %d, want 0", result.FilesScanned)
	}
	if result.FactsCreated != 0 {
		t.Errorf("FactsCreated = %d, want 0", result.FactsCreated)
	}
	if len(result.Errors) != 0 {
		t.Errorf("Errors = %v, want empty", result.Errors)
	}
}

func TestImportDir_NonexistentDir(t *testing.T) {
	imp, _ := newTestImporter(t)
	ctx := context.Background()

	_, err := imp.ImportDir(ctx, "/nonexistent/directory/path")
	if err == nil {
		t.Fatal("expected error for nonexistent directory, got nil")
	}
}

func TestImportAllProjects(t *testing.T) {
	// Create a temp structure mimicking ~/.claude/projects/*/memory/.
	tmpDir := t.TempDir()

	// Project 1: two memory files.
	proj1Mem := filepath.Join(tmpDir, "-Users-test-Code-project-alpha", "memory")
	if err := os.MkdirAll(proj1Mem, 0o755); err != nil {
		t.Fatal(err)
	}
	writeTestMemoryFile(t, proj1Mem, "user_prefs.md", "user", "User prefers dark mode.")
	writeTestMemoryFile(t, proj1Mem, "project_arch.md", "project", "Uses microservices architecture.")

	// Project 2: one memory file + a MEMORY.md (should be skipped).
	proj2Mem := filepath.Join(tmpDir, "-Users-test-Code-project-beta", "memory")
	if err := os.MkdirAll(proj2Mem, 0o755); err != nil {
		t.Fatal(err)
	}
	writeTestMemoryFile(t, proj2Mem, "reference_docs.md", "reference", "Docs at docs.example.com.")
	if err := os.WriteFile(filepath.Join(proj2Mem, "MEMORY.md"), []byte("- [Ref](reference_docs.md)\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Project 3: no memory directory (should be silently skipped).
	if err := os.MkdirAll(filepath.Join(tmpDir, "-Users-test-Code-project-gamma"), 0o755); err != nil {
		t.Fatal(err)
	}

	imp, _ := newTestImporter(t)
	ctx := context.Background()

	result, err := imp.ImportAllProjects(ctx, tmpDir)
	if err != nil {
		t.Fatalf("ImportAllProjects: %v", err)
	}

	if result.FilesScanned != 3 {
		t.Errorf("FilesScanned = %d, want 3", result.FilesScanned)
	}
	if result.FactsCreated != 3 {
		t.Errorf("FactsCreated = %d, want 3", result.FactsCreated)
	}
	if result.Skipped != 0 {
		t.Errorf("Skipped = %d, want 0", result.Skipped)
	}
	if len(result.Errors) != 0 {
		t.Errorf("Errors = %v, want empty", result.Errors)
	}
}

func TestImportDir_SourceTag(t *testing.T) {
	imp, store := newTestImporter(t)
	ctx := context.Background()

	result, err := imp.ImportDir(ctx, "testdata/memory")
	if err != nil {
		t.Fatalf("ImportDir: %v", err)
	}
	if result.FactsCreated != 4 {
		t.Fatalf("FactsCreated = %d, want 4", result.FactsCreated)
	}

	facts, err := store.ListFacts(ctx, memory.ListFilter{Limit: 100})
	if err != nil {
		t.Fatalf("ListFacts: %v", err)
	}

	for _, f := range facts {
		if f.Source == "" {
			t.Errorf("fact %q has empty source", f.ID)
		}
		// Source should start with "migrated:memory/"
		if len(f.Source) < 15 {
			t.Errorf("fact source %q seems too short", f.Source)
		}
	}
}

func TestImportDir_EmptyBody(t *testing.T) {
	tmpDir := t.TempDir()

	// Write a file with valid frontmatter but empty body.
	emptyBody := filepath.Join(tmpDir, "empty.md")
	content := `---
name: Empty
description: No body
type: user
---
`
	if err := os.WriteFile(emptyBody, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	imp, _ := newTestImporter(t)
	ctx := context.Background()

	result, err := imp.ImportDir(ctx, tmpDir)
	if err != nil {
		t.Fatalf("ImportDir: %v", err)
	}

	// Empty body files are scanned but not imported (no fact, no error).
	if result.FilesScanned != 1 {
		t.Errorf("FilesScanned = %d, want 1", result.FilesScanned)
	}
	if result.FactsCreated != 0 {
		t.Errorf("FactsCreated = %d, want 0", result.FactsCreated)
	}
	if len(result.Errors) != 0 {
		t.Errorf("Errors = %v, want empty", result.Errors)
	}
}

func TestImportDir_SkipsNonMD(t *testing.T) {
	tmpDir := t.TempDir()

	// Write a .txt file — should not be scanned.
	if err := os.WriteFile(filepath.Join(tmpDir, "notes.txt"), []byte("not markdown"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Write a valid .md file.
	writeTestMemoryFile(t, tmpDir, "valid.md", "user", "Valid content.")

	imp, _ := newTestImporter(t)
	ctx := context.Background()

	result, err := imp.ImportDir(ctx, tmpDir)
	if err != nil {
		t.Fatalf("ImportDir: %v", err)
	}

	if result.FilesScanned != 1 {
		t.Errorf("FilesScanned = %d, want 1 (should skip .txt)", result.FilesScanned)
	}
}

// writeTestMemoryFile writes a minimal auto-memory file to the given directory.
func writeTestMemoryFile(t *testing.T, dir, name, typ, body string) {
	t.Helper()
	content := fmt.Sprintf("---\nname: Test %s\ndescription: Test description\ntype: %s\n---\n%s\n", name, typ, body)
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatalf("write test memory file %s: %v", name, err)
	}
}
