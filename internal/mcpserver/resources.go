package mcpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"

	"personal/reverie/internal/memory"
)

// registerResources adds all resource definitions to the SDK server.
func (s *Server) registerResources(srv *mcpsdk.Server) {
	srv.AddResource(&mcpsdk.Resource{
		URI:         "reverie://status",
		Name:        "Memory System Status",
		Description: "Counts per layer, DB info, decay state, embedding model.",
		MIMEType:    "application/json",
	}, s.handleStatusResource)

	srv.AddResource(&mcpsdk.Resource{
		URI:         "reverie://l1/index",
		Name:        "L1 Cluster Meta-Index",
		Description: "The L1 cluster meta-index — always-resident procedural memory. Lists all clusters with metadata, utility, retention, and item counts. Read this at session start to understand the memory landscape.",
		MIMEType:    "application/json",
	}, s.handleL1IndexResource)

	srv.AddResource(&mcpsdk.Resource{
		URI:         "reverie://l3/recent",
		Name:        "Recent Episodes",
		Description: "Last 10 L3 episodic memories by creation time. Use for 'what did we do last time' queries.",
		MIMEType:    "application/json",
	}, s.handleL3RecentResource)
}

// --- reverie://status ---

// statusResponse is the JSON structure for the status resource.
type statusResponse struct {
	DBPath    string          `json:"db_path"`
	Embedding embeddingStatus `json:"embedding"`
	Decay     decayStatus     `json:"decay"`
	Counts    countStatus     `json:"counts"`
	Disabled  bool            `json:"disabled"`
}

type embeddingStatus struct {
	Provider   string `json:"provider"`
	Model      string `json:"model"`
	Dimensions int    `json:"dimensions"`
}

type decayStatus struct {
	Temperature        float64 `json:"temperature"`
	RetentionThreshold float64 `json:"retention_threshold"`
}

type countStatus struct {
	Facts    int `json:"facts"`
	Episodes int `json:"episodes"`
	Clusters int `json:"clusters"`
}

func (s *Server) handleStatusResource(ctx context.Context, req *mcpsdk.ReadResourceRequest) (*mcpsdk.ReadResourceResult, error) {
	facts, err := s.store.ListFacts(ctx, memory.ListFilter{Limit: 10000})
	if err != nil {
		return nil, fmt.Errorf("status: list facts: %w", err)
	}
	episodes, err := s.store.ListEpisodes(ctx, memory.ListFilter{Limit: 10000})
	if err != nil {
		return nil, fmt.Errorf("status: list episodes: %w", err)
	}
	clusters, err := s.store.ListClusters(ctx)
	if err != nil {
		return nil, fmt.Errorf("status: list clusters: %w", err)
	}

	resp := statusResponse{
		DBPath: s.cfg.Storage.DBPath,
		Embedding: embeddingStatus{
			Provider:   s.cfg.Embedding.Provider,
			Model:      s.cfg.Embedding.Model,
			Dimensions: s.cfg.Embedding.Dimensions,
		},
		Decay: decayStatus{
			Temperature:        s.decayer.Temperature(),
			RetentionThreshold: s.decayer.Threshold(),
		},
		Counts: countStatus{
			Facts:    len(facts),
			Episodes: len(episodes),
			Clusters: len(clusters),
		},
		Disabled: s.cfg.Server.Disabled,
	}

	data, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("status: marshal: %w", err)
	}

	return &mcpsdk.ReadResourceResult{
		Contents: []*mcpsdk.ResourceContents{{
			URI:      req.Params.URI,
			MIMEType: "application/json",
			Text:     string(data),
		}},
	}, nil
}

// --- reverie://l1/index ---

// l1IndexResponse is the JSON structure for the L1 cluster meta-index resource.
type l1IndexResponse struct {
	Clusters []l1ClusterEntry `json:"clusters"`
}

type l1ClusterEntry struct {
	ID         string  `json:"id"`
	Summary    string  `json:"summary"`
	Domain     string  `json:"domain"`
	MetaInstr  string  `json:"meta_instr"`
	ItemCount  int     `json:"item_count"`
	Utility    float64 `json:"utility"`
	Frequency  float64 `json:"frequency"`
	TurnsSince int     `json:"turns_since"`
	Retention  float64 `json:"retention"`
}

func (s *Server) handleL1IndexResource(ctx context.Context, req *mcpsdk.ReadResourceRequest) (*mcpsdk.ReadResourceResult, error) {
	clusters, err := s.store.ListClusters(ctx)
	if err != nil {
		return nil, fmt.Errorf("l1 index: list clusters: %w", err)
	}

	var entries []l1ClusterEntry
	for _, c := range clusters {
		if c.ItemCount == 0 {
			continue
		}
		retention := s.decayer.Retention(c)
		entries = append(entries, l1ClusterEntry{
			ID:         c.ID,
			Summary:    c.Summary,
			Domain:     c.Domain,
			MetaInstr:  c.MetaInstr,
			ItemCount:  c.ItemCount,
			Utility:    c.Utility,
			Frequency:  c.Frequency,
			TurnsSince: c.TurnsSince,
			Retention:  retention,
		})
	}

	// Sort by utility descending so the most-important clusters are first.
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Utility > entries[j].Utility
	})

	resp := l1IndexResponse{Clusters: entries}
	data, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("l1 index: marshal: %w", err)
	}

	return &mcpsdk.ReadResourceResult{
		Contents: []*mcpsdk.ResourceContents{{
			URI:      req.Params.URI,
			MIMEType: "application/json",
			Text:     string(data),
		}},
	}, nil
}

// --- reverie://l3/recent ---

// l3RecentResponse is the JSON structure for the recent episodes resource.
type l3RecentResponse struct {
	Episodes []l3EpisodeEntry `json:"episodes"`
}

type l3EpisodeEntry struct {
	ID         string `json:"id"`
	Situation  string `json:"situation"`
	Action     string `json:"action"`
	Outcome    string `json:"outcome"`
	Preemptive string `json:"preemptive"`
	CreatedAt  string `json:"created_at"`
	ClusterID  string `json:"cluster_id"`
}

func (s *Server) handleL3RecentResource(ctx context.Context, req *mcpsdk.ReadResourceRequest) (*mcpsdk.ReadResourceResult, error) {
	episodes, err := s.store.ListEpisodes(ctx, memory.ListFilter{Limit: 10, Sort: "created"})
	if err != nil {
		return nil, fmt.Errorf("l3 recent: list episodes: %w", err)
	}

	entries := make([]l3EpisodeEntry, len(episodes))
	for i, ep := range episodes {
		entries[i] = l3EpisodeEntry{
			ID:         ep.ID,
			Situation:  ep.Situation,
			Action:     ep.Action,
			Outcome:    ep.Outcome,
			Preemptive: ep.Preemptive,
			CreatedAt:  ep.CreatedAt.Format("2006-01-02T15:04:05Z"),
			ClusterID:  ep.ClusterID,
		}
	}

	resp := l3RecentResponse{Episodes: entries}
	data, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("l3 recent: marshal: %w", err)
	}

	return &mcpsdk.ReadResourceResult{
		Contents: []*mcpsdk.ResourceContents{{
			URI:      req.Params.URI,
			MIMEType: "application/json",
			Text:     string(data),
		}},
	}, nil
}
