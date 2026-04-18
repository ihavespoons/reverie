// Package embed defines the interface for text embedding providers.
// Reverie uses embeddings for vector similarity search across memories.
// The only external API call reverie makes is to an embedding provider
// (e.g., Voyage AI). Implementations live in separate files (e.g., voyage.go).
package embed

import "context"

// EmbeddingProvider generates vector embeddings from text. Implementations
// must be safe for concurrent use.
type EmbeddingProvider interface {
	// Embed returns embedding vectors for the given texts. The returned slice
	// has the same length as texts, with each inner slice having Dimensions()
	// elements. Implementations should batch requests where possible.
	Embed(ctx context.Context, texts []string) ([][]float32, error)

	// Dimensions returns the dimensionality of the embedding vectors produced
	// by this provider (e.g., 1024 for voyage-3).
	Dimensions() int

	// Model returns the model identifier used by this provider (e.g., "voyage-3").
	// This value is used as the model column in the embedding_cache table.
	Model() string
}
