// Package mcpserver implements the MCP tool surface for reverie's memory system.
// It registers tools with the MCP SDK server and delegates to the memory store
// and embedding provider for recall, write, reinforce, forget, and list operations.
package mcpserver

import (
	"sync"
	"time"

	"github.com/diffsec/reverie/internal/memory"
)

// cachedRecall holds the result of a memory_recall invocation, keyed by recall_id.
// It is used by the future memory_apply_judgment tool (Phase 2) to look up the
// candidate set and combine Gate A verdicts with Gate B/C results.
type cachedRecall struct {
	queryVec   []float32
	candidates []memory.Candidate
	round      int
	createdAt  time.Time
}

// recallCache is a TTL-bounded in-memory cache for recall results. A background
// janitor goroutine periodically evicts expired entries.
type recallCache struct {
	mu      sync.Mutex
	entries map[string]*cachedRecall
	ttl     time.Duration
	done    chan struct{}
}

// newRecallCache creates a recall cache with the given TTL and starts a
// background janitor that runs every ttl/2 to evict expired entries.
func newRecallCache(ttl time.Duration) *recallCache {
	rc := &recallCache{
		entries: make(map[string]*cachedRecall),
		ttl:     ttl,
		done:    make(chan struct{}),
	}
	go rc.janitor()
	return rc
}

// put stores a recall result under the given id.
func (rc *recallCache) put(id string, entry *cachedRecall) {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	rc.entries[id] = entry
}

// get retrieves a recall result by id. Returns nil, false if not found or expired.
func (rc *recallCache) get(id string) (*cachedRecall, bool) {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	entry, ok := rc.entries[id]
	if !ok {
		return nil, false
	}
	if time.Since(entry.createdAt) > rc.ttl {
		delete(rc.entries, id)
		return nil, false
	}
	return entry, true
}

// stop terminates the background janitor goroutine.
func (rc *recallCache) stop() {
	close(rc.done)
}

// janitor runs in the background and evicts expired entries at half the TTL interval.
func (rc *recallCache) janitor() {
	interval := rc.ttl / 2
	if interval < time.Second {
		interval = time.Second
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-rc.done:
			return
		case now := <-ticker.C:
			rc.mu.Lock()
			for id, entry := range rc.entries {
				if now.Sub(entry.createdAt) > rc.ttl {
					delete(rc.entries, id)
				}
			}
			rc.mu.Unlock()
		}
	}
}
