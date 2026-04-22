package mcpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"time"

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

	srv.AddResourceTemplate(&mcpsdk.ResourceTemplate{
		URITemplate: "reverie://l1/cluster/{id}{?limit,offset}",
		Name:        "L1 Cluster Detail",
		Description: "Per-cluster detail view — cluster metadata plus a paginated list of members (facts and episodes). Members are ordered by created_at ascending. Query params: limit (default 50, max 200), offset (default 0).",
		MIMEType:    "application/json",
	}, s.handleL1ClusterDetailResource)

	srv.AddResourceTemplate(&mcpsdk.ResourceTemplate{
		URITemplate: "reverie://l1/at_risk{?threshold,limit}",
		Name:        "L1 At-Risk Clusters",
		Description: "Clusters with Ebbinghaus retention below a threshold, most-at-risk first (retention ascending). Use to surface memory that is decaying and may be forgotten. Query params: threshold (float in [0,1], default = configured decay.retention_threshold); limit (default 50, max 500).",
		MIMEType:    "application/json",
	}, s.handleL1AtRiskResource)

	srv.AddResourceTemplate(&mcpsdk.ResourceTemplate{
		URITemplate: "reverie://stats/daily{?from,to}",
		Name:        "Daily Activity Stats",
		Description: "Per-day counters for facts in/out, episodes in/out, and supersedes, maintained by DB triggers. Dates are UTC YYYY-MM-DD. Defaults: from = today-30 days, to = today. Gaps are zero-filled. Max span 365 days.",
		MIMEType:    "application/json",
	}, s.handleStatsDailyResource)
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

// --- reverie://l1/cluster/{id} ---

const (
	clusterDetailDefaultLimit = 50
	clusterDetailMaxLimit     = 200
)

// l1ClusterDetailResponse is the JSON structure for the per-cluster detail
// resource: cluster metadata (same shape as an l1/index entry) plus a
// paginated list of members.
type l1ClusterDetailResponse struct {
	Cluster l1ClusterEntry    `json:"cluster"`
	Members []l1ClusterMember `json:"members"`
	Total   int               `json:"total"`
	Limit   int               `json:"limit"`
	Offset  int               `json:"offset"`
}

type l1ClusterMember struct {
	ID         string   `json:"id"`
	Layer      string   `json:"layer"`
	Subtype    string   `json:"subtype,omitempty"`
	Content    string   `json:"content"`
	Tags       []string `json:"tags,omitempty"`
	CreatedAt  string   `json:"created_at"`
	AccessedAt string   `json:"accessed_at"`
}

// parseClusterDetailURI extracts the cluster id and optional limit/offset from
// a reverie://l1/cluster/{id}[?limit=N&offset=N] URI.
func parseClusterDetailURI(raw string) (id string, limit, offset int, err error) {
	u, perr := url.Parse(raw)
	if perr != nil {
		return "", 0, 0, fmt.Errorf("invalid uri: %w", perr)
	}

	// Path layout: "/cluster/{id}". The host is "l1".
	const prefix = "/cluster/"
	if !strings.HasPrefix(u.Path, prefix) {
		return "", 0, 0, fmt.Errorf("invalid uri path: %q", u.Path)
	}
	id = strings.TrimPrefix(u.Path, prefix)
	id = strings.Trim(id, "/")
	if id == "" {
		return "", 0, 0, fmt.Errorf("cluster id is required")
	}

	limit = clusterDetailDefaultLimit
	if raw := u.Query().Get("limit"); raw != "" {
		v, err := strconv.Atoi(raw)
		if err != nil || v < 0 {
			return "", 0, 0, fmt.Errorf("invalid limit: %q", raw)
		}
		limit = v
	}
	if limit > clusterDetailMaxLimit {
		limit = clusterDetailMaxLimit
	}

	offset = 0
	if raw := u.Query().Get("offset"); raw != "" {
		v, err := strconv.Atoi(raw)
		if err != nil || v < 0 {
			return "", 0, 0, fmt.Errorf("invalid offset: %q", raw)
		}
		offset = v
	}

	return id, limit, offset, nil
}

// truncateContent truncates s to max runes and appends an ellipsis if truncated.
func truncateContent(s string, max int) string {
	runes := []rune(s)
	if len(runes) <= max {
		return s
	}
	return string(runes[:max])
}

// renderEpisodeContent is the canonical single-string rendering of an episode
// used by memory_list / memory_recall / memory_get. Kept local to keep the
// resource handler self-contained.
func renderEpisodeContent(ep memory.Episode) string {
	return memory.Candidate{Episode: &ep}.Content()
}

func (s *Server) handleL1ClusterDetailResource(ctx context.Context, req *mcpsdk.ReadResourceRequest) (*mcpsdk.ReadResourceResult, error) {
	id, limit, offset, err := parseClusterDetailURI(req.Params.URI)
	if err != nil {
		return nil, fmt.Errorf("l1 cluster detail: %w", err)
	}

	cluster, err := s.store.GetCluster(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("l1 cluster detail: get cluster: %w", err)
	}
	if cluster == nil {
		return nil, fmt.Errorf("cluster not found: %s", id)
	}

	factCount, err := s.store.CountFactsByCluster(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("l1 cluster detail: count facts: %w", err)
	}
	epCount, err := s.store.CountEpisodesByCluster(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("l1 cluster detail: count episodes: %w", err)
	}
	total := factCount + epCount

	// We paginate across the union of (facts ∪ episodes) ordered by created_at
	// ASC. To avoid loading everything, walk facts first in the requested
	// window; if the window extends into episodes, pull the remainder from
	// the episodes page. A fully-in-facts or fully-in-episodes request only
	// touches one side.
	members := make([]l1ClusterMember, 0, limit)
	if limit > 0 {
		if offset < factCount {
			take := limit
			if offset+take > factCount {
				take = factCount - offset
			}
			facts, err := s.store.ListFactsByCluster(ctx, id, take, offset)
			if err != nil {
				return nil, fmt.Errorf("l1 cluster detail: list facts: %w", err)
			}
			for _, f := range facts {
				members = append(members, l1ClusterMember{
					ID:         f.ID,
					Layer:      string(memory.TypeL2Semantic),
					Subtype:    f.Subtype,
					Content:    truncateContent(f.Content, 200),
					Tags:       normalizeTagsSlice(f.Tags),
					CreatedAt:  f.CreatedAt.Format("2006-01-02T15:04:05Z"),
					AccessedAt: f.AccessedAt.Format("2006-01-02T15:04:05Z"),
				})
			}
		}

		if len(members) < limit {
			epOffset := offset - factCount
			if epOffset < 0 {
				epOffset = 0
			}
			epLimit := limit - len(members)
			if epLimit > 0 && epOffset < epCount {
				episodes, err := s.store.ListEpisodesByCluster(ctx, id, epLimit, epOffset)
				if err != nil {
					return nil, fmt.Errorf("l1 cluster detail: list episodes: %w", err)
				}
				for _, ep := range episodes {
					members = append(members, l1ClusterMember{
						ID:         ep.ID,
						Layer:      string(memory.TypeL3Episodic),
						Content:    truncateContent(renderEpisodeContent(ep), 200),
						Tags:       normalizeTagsSlice(ep.Tags),
						CreatedAt:  ep.CreatedAt.Format("2006-01-02T15:04:05Z"),
						AccessedAt: ep.AccessedAt.Format("2006-01-02T15:04:05Z"),
					})
				}
			}
		}
	}

	resp := l1ClusterDetailResponse{
		Cluster: l1ClusterEntry{
			ID:         cluster.ID,
			Summary:    cluster.Summary,
			Domain:     cluster.Domain,
			MetaInstr:  cluster.MetaInstr,
			ItemCount:  cluster.ItemCount,
			Utility:    cluster.Utility,
			Frequency:  cluster.Frequency,
			TurnsSince: cluster.TurnsSince,
			Retention:  s.decayer.Retention(*cluster),
		},
		Members: members,
		Total:   total,
		Limit:   limit,
		Offset:  offset,
	}

	data, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("l1 cluster detail: marshal: %w", err)
	}
	return &mcpsdk.ReadResourceResult{
		Contents: []*mcpsdk.ResourceContents{{
			URI:      req.Params.URI,
			MIMEType: "application/json",
			Text:     string(data),
		}},
	}, nil
}

// --- reverie://l1/at_risk ---

const (
	atRiskDefaultLimit = 50
	atRiskMaxLimit     = 500
)

// l1AtRiskResponse is the JSON structure for the at-risk resource. Clusters
// are sorted by retention ascending (most-at-risk first), with ties broken by
// cluster ID for determinism. Total reports the full pre-limit count so
// callers can tell whether pagination truncated the list.
type l1AtRiskResponse struct {
	Threshold float64          `json:"threshold"`
	Clusters  []l1ClusterEntry `json:"clusters"` // ascending by Retention
	Total     int              `json:"total"`    // total non-empty clusters with retention < threshold
}

// parseAtRiskURI extracts the optional threshold (float in [0,1]) and limit
// (int, capped at atRiskMaxLimit) from a reverie://l1/at_risk[?threshold&limit]
// URI. When a query param is absent the corresponding `*Set` return is false
// and the caller supplies the default.
func parseAtRiskURI(raw string) (threshold float64, thresholdSet bool, limit int, limitSet bool, err error) {
	u, perr := url.Parse(raw)
	if perr != nil {
		return 0, false, 0, false, fmt.Errorf("invalid uri: %w", perr)
	}
	q := u.Query()

	if s := q.Get("threshold"); s != "" {
		v, perr := strconv.ParseFloat(s, 64)
		if perr != nil {
			return 0, false, 0, false, fmt.Errorf("invalid threshold: %q", s)
		}
		if v < 0 || v > 1 {
			return 0, false, 0, false, fmt.Errorf("threshold %g out of range [0, 1]", v)
		}
		threshold = v
		thresholdSet = true
	}

	if s := q.Get("limit"); s != "" {
		v, perr := strconv.Atoi(s)
		if perr != nil || v < 0 {
			return 0, false, 0, false, fmt.Errorf("invalid limit: %q", s)
		}
		limit = v
		limitSet = true
	}

	return threshold, thresholdSet, limit, limitSet, nil
}

func (s *Server) handleL1AtRiskResource(ctx context.Context, req *mcpsdk.ReadResourceRequest) (*mcpsdk.ReadResourceResult, error) {
	threshold, thresholdSet, limit, limitSet, err := parseAtRiskURI(req.Params.URI)
	if err != nil {
		return nil, fmt.Errorf("l1 at_risk: %w", err)
	}
	if !thresholdSet {
		threshold = s.decayer.Threshold()
	}
	if !limitSet {
		limit = atRiskDefaultLimit
	}
	if limit > atRiskMaxLimit {
		limit = atRiskMaxLimit
	}

	clusters, err := s.store.ListClusters(ctx)
	if err != nil {
		return nil, fmt.Errorf("l1 at_risk: list clusters: %w", err)
	}

	entries := make([]l1ClusterEntry, 0, len(clusters))
	for _, c := range clusters {
		if c.ItemCount == 0 {
			continue
		}
		retention := s.decayer.Retention(c)
		if retention >= threshold {
			continue
		}
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

	// Most at-risk first. Break ties by id so the output is deterministic
	// across stores (map iteration order in memStore is nondeterministic).
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].Retention != entries[j].Retention {
			return entries[i].Retention < entries[j].Retention
		}
		return entries[i].ID < entries[j].ID
	})

	total := len(entries)
	if len(entries) > limit {
		entries = entries[:limit]
	}

	resp := l1AtRiskResponse{
		Threshold: threshold,
		Clusters:  entries,
		Total:     total,
	}
	data, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("l1 at_risk: marshal: %w", err)
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

// --- reverie://stats/daily ---

const (
	// statsDailyDefaultLookback is the default [from, to] span when the
	// caller omits `from`: 30 days prior to `to` (inclusive on both ends).
	statsDailyDefaultLookback = 30
	// statsDailyMaxSpanDays caps the inclusive span to one year. The table
	// is tiny but a huge range explodes the zero-fill output for no benefit.
	statsDailyMaxSpanDays = 365
	// statsDailyDateFormat is the canonical YYYY-MM-DD representation used
	// for both the daily_stats PK and our API contract. UTC always.
	statsDailyDateFormat = "2006-01-02"
)

// dailyStatsResponse is the JSON structure for the stats/daily resource.
// Days is sorted oldest first and is dense — gaps in the daily_stats table
// are emitted as zero-value rows so clients can graph without interpolating.
type dailyStatsResponse struct {
	From string            `json:"from"`
	To   string            `json:"to"`
	Days []dailyStatsEntry `json:"days"`
}

type dailyStatsEntry struct {
	Date        string `json:"date"`
	FactsIn     int    `json:"facts_in"`
	FactsOut    int    `json:"facts_out"`
	EpisodesIn  int    `json:"episodes_in"`
	EpisodesOut int    `json:"episodes_out"`
	Supersedes  int    `json:"supersedes"`
}

// parseStatsDailyURI extracts `from` and `to` from reverie://stats/daily?from=..&to=..
// Defaults: `to` = today (UTC), `from` = to - 30 days. Values must parse as
// YYYY-MM-DD. Validates `from <= to` and span <= 365 days (inclusive).
func parseStatsDailyURI(raw string, now time.Time) (from, to string, err error) {
	u, perr := url.Parse(raw)
	if perr != nil {
		return "", "", fmt.Errorf("invalid uri: %w", perr)
	}

	today := now.UTC().Format(statsDailyDateFormat)
	q := u.Query()

	to = q.Get("to")
	if to == "" {
		to = today
	}
	toT, err := time.Parse(statsDailyDateFormat, to)
	if err != nil {
		return "", "", fmt.Errorf("invalid to: %q (want YYYY-MM-DD)", to)
	}

	from = q.Get("from")
	if from == "" {
		from = toT.AddDate(0, 0, -statsDailyDefaultLookback).Format(statsDailyDateFormat)
	}
	fromT, err := time.Parse(statsDailyDateFormat, from)
	if err != nil {
		return "", "", fmt.Errorf("invalid from: %q (want YYYY-MM-DD)", from)
	}

	if fromT.After(toT) {
		return "", "", fmt.Errorf("from (%s) must be <= to (%s)", from, to)
	}
	// Inclusive span in days: to - from + 1. The cap is a guardrail on
	// zero-fill output size; the SQL itself would handle larger ranges fine.
	spanDays := int(toT.Sub(fromT).Hours()/24) + 1
	if spanDays > statsDailyMaxSpanDays {
		return "", "", fmt.Errorf("span %d days exceeds max %d", spanDays, statsDailyMaxSpanDays)
	}

	return from, to, nil
}

func (s *Server) handleStatsDailyResource(ctx context.Context, req *mcpsdk.ReadResourceRequest) (*mcpsdk.ReadResourceResult, error) {
	from, to, err := parseStatsDailyURI(req.Params.URI, time.Now())
	if err != nil {
		return nil, fmt.Errorf("stats daily: %w", err)
	}

	rows, err := s.store.ListDailyStats(ctx, from, to)
	if err != nil {
		return nil, fmt.Errorf("stats daily: list: %w", err)
	}

	// Index rows by date for O(1) lookup while expanding the window. The
	// daily_stats table is tiny (one row per day of server activity), so the
	// map is cheap and the expanded output stays dense.
	byDate := make(map[string]memory.DailyStats, len(rows))
	for _, r := range rows {
		byDate[r.Date] = r
	}

	fromT, _ := time.Parse(statsDailyDateFormat, from)
	toT, _ := time.Parse(statsDailyDateFormat, to)

	days := []dailyStatsEntry{}
	for d := fromT; !d.After(toT); d = d.AddDate(0, 0, 1) {
		key := d.Format(statsDailyDateFormat)
		if r, ok := byDate[key]; ok {
			days = append(days, dailyStatsEntry{
				Date:        r.Date,
				FactsIn:     r.FactsIn,
				FactsOut:    r.FactsOut,
				EpisodesIn:  r.EpisodesIn,
				EpisodesOut: r.EpisodesOut,
				Supersedes:  r.Supersedes,
			})
			continue
		}
		days = append(days, dailyStatsEntry{Date: key})
	}

	resp := dailyStatsResponse{From: from, To: to, Days: days}
	data, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("stats daily: marshal: %w", err)
	}
	return &mcpsdk.ReadResourceResult{
		Contents: []*mcpsdk.ResourceContents{{
			URI:      req.Params.URI,
			MIMEType: "application/json",
			Text:     string(data),
		}},
	}, nil
}
