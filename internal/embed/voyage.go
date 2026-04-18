package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"time"
)

const (
	voyageAPIURL      = "https://api.voyageai.com/v1/embeddings"
	voyageDefaultModel = "voyage-3"
	voyageMaxBatch     = 128
)

// voyageProvider implements EmbeddingProvider using the Voyage AI API.
type voyageProvider struct {
	apiKey    string
	model     string
	batchSize int
	client    *http.Client
}

// NewVoyageProvider creates a Voyage AI embedding provider. If model is empty,
// it defaults to "voyage-3". If batchSize is 0 or exceeds 128, it defaults to
// 128. If apiKey is empty, the returned provider's Embed method will return an
// error without making any network requests.
func NewVoyageProvider(apiKey, model string, batchSize int) EmbeddingProvider {
	if model == "" {
		model = voyageDefaultModel
	}
	if batchSize <= 0 || batchSize > voyageMaxBatch {
		batchSize = voyageMaxBatch
	}
	return &voyageProvider{
		apiKey:    apiKey,
		model:     model,
		batchSize: batchSize,
		client:    &http.Client{Timeout: 30 * time.Second},
	}
}

// voyageRequest is the JSON body sent to the Voyage API.
type voyageRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

// voyageResponse is the JSON response from the Voyage API.
type voyageResponse struct {
	Data []voyageEmbedding `json:"data"`
}

// voyageEmbedding represents a single embedding in the Voyage API response.
type voyageEmbedding struct {
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

// Embed generates embeddings for the given texts via the Voyage AI API.
// Texts are chunked into batches of batchSize. An empty input returns an
// empty slice with no API call.
func (v *voyageProvider) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if v.apiKey == "" {
		return nil, fmt.Errorf("voyage: API key not configured")
	}
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	results := make([][]float32, len(texts))

	for batchStart := 0; batchStart < len(texts); batchStart += v.batchSize {
		batchEnd := batchStart + v.batchSize
		if batchEnd > len(texts) {
			batchEnd = len(texts)
		}
		batch := texts[batchStart:batchEnd]

		embeddings, err := v.callAPI(ctx, batch)
		if err != nil {
			return nil, err
		}

		for i, emb := range embeddings {
			results[batchStart+i] = emb
		}
	}

	return results, nil
}

// callAPI sends a single batch to the Voyage API and returns embeddings
// sorted by index.
func (v *voyageProvider) callAPI(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody := voyageRequest{
		Input: texts,
		Model: v.model,
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("voyage: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, voyageAPIURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("voyage: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+v.apiKey)

	resp, err := v.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("voyage: request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("voyage: read response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		snippet := string(respBody)
		if len(snippet) > 256 {
			snippet = snippet[:256]
		}
		return nil, fmt.Errorf("voyage: HTTP %d: %s", resp.StatusCode, snippet)
	}

	var voyageResp voyageResponse
	if err := json.Unmarshal(respBody, &voyageResp); err != nil {
		return nil, fmt.Errorf("voyage: unmarshal response: %w", err)
	}

	// Sort by index to ensure correct ordering.
	sort.Slice(voyageResp.Data, func(i, j int) bool {
		return voyageResp.Data[i].Index < voyageResp.Data[j].Index
	})

	embeddings := make([][]float32, len(voyageResp.Data))
	for i, d := range voyageResp.Data {
		embeddings[i] = d.Embedding
	}

	return embeddings, nil
}

// Dimensions returns the vector dimensionality for the configured model.
// Returns 1024 for voyage-3 and voyage-code-3, 0 for unknown models.
func (v *voyageProvider) Dimensions() int {
	switch v.model {
	case "voyage-3", "voyage-code-3":
		return 1024
	default:
		return 0
	}
}

// Model returns the configured model identifier.
func (v *voyageProvider) Model() string {
	return v.model
}
