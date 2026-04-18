package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

const (
	openAICompatDefaultBatch = 32
	openAICompatMaxBatch     = 256
)

// openAICompatProvider implements EmbeddingProvider against any OpenAI
// /v1/embeddings compatible endpoint. Works with:
//   - Ollama (http://localhost:11434/v1)
//   - LM Studio (http://localhost:1234/v1)
//   - OpenAI (https://api.openai.com/v1)
//   - Together, Fireworks, and anything else that speaks the same shape.
type openAICompatProvider struct {
	endpoint  string // full URL to /embeddings
	apiKey    string
	model     string
	batchSize int
	dims      int
	client    *http.Client
}

// NewOpenAICompatProvider creates an embedding provider speaking the OpenAI
// /v1/embeddings protocol. baseURL should include the /v1 suffix (e.g.,
// "http://localhost:11434/v1"). apiKey may be empty for local servers like
// Ollama and LM Studio. dims is the embedding dimensionality produced by the
// model; it is advisory (reported via Dimensions()) and does not affect the
// HTTP call.
func NewOpenAICompatProvider(baseURL, apiKey, model string, batchSize, dims int) EmbeddingProvider {
	if batchSize <= 0 {
		batchSize = openAICompatDefaultBatch
	}
	if batchSize > openAICompatMaxBatch {
		batchSize = openAICompatMaxBatch
	}
	endpoint := strings.TrimRight(baseURL, "/") + "/embeddings"
	return &openAICompatProvider{
		endpoint:  endpoint,
		apiKey:    apiKey,
		model:     model,
		batchSize: batchSize,
		dims:      dims,
		client:    &http.Client{Timeout: 60 * time.Second},
	}
}

type openAICompatRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

type openAICompatResponse struct {
	Data []openAICompatEmbedding `json:"data"`
}

type openAICompatEmbedding struct {
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

// Embed generates embeddings for the given texts. Chunks into batches of
// batchSize. Empty input returns an empty slice with no API call.
func (p *openAICompatProvider) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if p.endpoint == "" || p.model == "" {
		return nil, fmt.Errorf("openai_compat: endpoint and model must be configured")
	}
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	results := make([][]float32, len(texts))
	for start := 0; start < len(texts); start += p.batchSize {
		end := start + p.batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[start:end]

		embeddings, err := p.callAPI(ctx, batch)
		if err != nil {
			return nil, err
		}
		if len(embeddings) != len(batch) {
			return nil, fmt.Errorf("openai_compat: expected %d embeddings, got %d", len(batch), len(embeddings))
		}
		for i, emb := range embeddings {
			results[start+i] = emb
		}
	}
	return results, nil
}

func (p *openAICompatProvider) callAPI(ctx context.Context, texts []string) ([][]float32, error) {
	bodyBytes, err := json.Marshal(openAICompatRequest{Input: texts, Model: p.model})
	if err != nil {
		return nil, fmt.Errorf("openai_compat: marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("openai_compat: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai_compat: request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai_compat: read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		snippet := string(respBody)
		if len(snippet) > 256 {
			snippet = snippet[:256]
		}
		return nil, fmt.Errorf("openai_compat: HTTP %d: %s", resp.StatusCode, snippet)
	}

	var parsed openAICompatResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, fmt.Errorf("openai_compat: unmarshal response: %w", err)
	}
	sort.Slice(parsed.Data, func(i, j int) bool {
		return parsed.Data[i].Index < parsed.Data[j].Index
	})
	embeddings := make([][]float32, len(parsed.Data))
	for i, d := range parsed.Data {
		embeddings[i] = d.Embedding
	}
	return embeddings, nil
}

// Dimensions returns the configured dimensionality. Returns 0 if unset.
func (p *openAICompatProvider) Dimensions() int {
	return p.dims
}

// Model returns the configured model identifier.
func (p *openAICompatProvider) Model() string {
	return p.model
}
