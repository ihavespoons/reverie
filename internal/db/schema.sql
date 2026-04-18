-- Reverie memory system schema.
-- All tables use CREATE TABLE IF NOT EXISTS so this file can be re-executed
-- safely (e.g., on every startup) without migrations.

-- L1 procedural memory clusters.
CREATE TABLE IF NOT EXISTS clusters (
  id TEXT PRIMARY KEY,
  summary TEXT,
  domain TEXT,
  meta_instr TEXT,
  item_count INTEGER DEFAULT 0,
  centroid BLOB,
  utility REAL DEFAULT 0.0,
  frequency REAL DEFAULT 0.0,
  turns_since INTEGER DEFAULT 0,
  last_access TEXT DEFAULT (datetime('now')),
  created_at  TEXT DEFAULT (datetime('now'))
);

-- L2 semantic facts.
CREATE TABLE IF NOT EXISTS facts (
  id TEXT PRIMARY KEY,
  cluster_id TEXT REFERENCES clusters(id),
  content TEXT NOT NULL,
  embedding BLOB,
  content_hash TEXT NOT NULL,
  subtype TEXT,                        -- user|feedback|project|reference
  source TEXT DEFAULT 'inferred',
  confidence REAL DEFAULT 1.0,
  valid_from TEXT DEFAULT (datetime('now')),
  superseded_by TEXT REFERENCES facts(id),
  created_at TEXT DEFAULT (datetime('now')),
  accessed_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_facts_cluster     ON facts(cluster_id);
CREATE INDEX IF NOT EXISTS idx_facts_hash        ON facts(content_hash);
CREATE INDEX IF NOT EXISTS idx_facts_superseded  ON facts(superseded_by);
CREATE INDEX IF NOT EXISTS idx_facts_subtype     ON facts(subtype);

-- L3 episodic memories.
CREATE TABLE IF NOT EXISTS episodes (
  id TEXT PRIMARY KEY,
  cluster_id TEXT REFERENCES clusters(id),
  situation TEXT,
  action TEXT,
  outcome TEXT,
  preemptive TEXT,
  embedding BLOB,
  content_hash TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now')),
  accessed_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_episodes_cluster ON episodes(cluster_id);

-- L2-L3 cross-type links (evidence discovery).
CREATE TABLE IF NOT EXISTS fact_episode_links (
  fact_id TEXT REFERENCES facts(id) ON DELETE CASCADE,
  episode_id TEXT REFERENCES episodes(id) ON DELETE CASCADE,
  link_type TEXT DEFAULT 'evidence',
  PRIMARY KEY (fact_id, episode_id)
);

-- Content-hash-keyed embedding cache (survives model upgrades via model col).
CREATE TABLE IF NOT EXISTS embedding_cache (
  content_hash TEXT,
  model TEXT,
  embedding BLOB NOT NULL,
  created_at TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (content_hash, model)
);

-- Session checkpoint for crash recovery.
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  turn_counter INTEGER DEFAULT 0,
  working_memory TEXT DEFAULT '{}',  -- JSON snapshot
  updated_at TEXT DEFAULT (datetime('now'))
);
