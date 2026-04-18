package memory

import "time"

// testdataFacts returns a set of facts with known embeddings for use in tests.
// The embeddings are small 4-dimensional vectors with known cosine properties.
func testdataFacts() []Fact {
	base := time.Date(2026, 4, 1, 12, 0, 0, 0, time.UTC)

	return []Fact{
		{
			Content:    "The user prefers dark mode in all editors.",
			Subtype:    "user",
			Source:     "conversation",
			Embedding:  []float32{1, 0, 0, 0}, // unit vector along x
			CreatedAt:  base,
			AccessedAt: base,
			ValidFrom:  base,
		},
		{
			Content:    "Always use gofmt before committing Go code.",
			Subtype:    "feedback",
			Source:     "correction",
			Embedding:  []float32{0, 1, 0, 0}, // unit vector along y — orthogonal to first
			CreatedAt:  base.Add(1 * time.Hour),
			AccessedAt: base.Add(1 * time.Hour),
			ValidFrom:  base.Add(1 * time.Hour),
		},
		{
			Content:    "The reverie project uses modernc.org/sqlite for pure-Go SQLite.",
			Subtype:    "project",
			Source:     "inferred",
			Embedding:  []float32{0.6, 0.8, 0, 0}, // cosine with x=0.6, with y=0.8
			CreatedAt:  base.Add(2 * time.Hour),
			AccessedAt: base.Add(2 * time.Hour),
			ValidFrom:  base.Add(2 * time.Hour),
		},
	}
}

// testdataEpisodes returns a set of episodes with known embeddings for use in tests.
func testdataEpisodes() []Episode {
	base := time.Date(2026, 4, 2, 12, 0, 0, 0, time.UTC)

	return []Episode{
		{
			Situation:  "User asked to refactor the config loader.",
			Action:     "Split monolithic Load into sub-functions.",
			Outcome:    "Tests pass. Config loading is now easier to extend.",
			Preemptive: "Always split large functions into testable units.",
			Embedding:  []float32{0.7, 0.7, 0, 0}, // cosine with x ~= 0.707
			CreatedAt:  base,
			AccessedAt: base,
		},
		{
			Situation:  "Deployment failed due to missing env var.",
			Action:     "Added validation at startup with clear error message.",
			Outcome:    "Deployment succeeded after fix. No repeat incidents.",
			Preemptive: "Always validate required env vars at startup.",
			Embedding:  []float32{0, 0, 1, 0}, // unit vector along z
			CreatedAt:  base.Add(1 * time.Hour),
			AccessedAt: base.Add(1 * time.Hour),
		},
		{
			Situation:  "Test suite was flaky due to time-dependent assertions.",
			Action:     "Injected a clock abstraction.",
			Outcome:    "Tests are deterministic now.",
			Preemptive: "Use injected clocks for time-sensitive tests.",
			Embedding:  []float32{0.5, 0, 0, 0.866}, // cosine with x = 0.5
			CreatedAt:  base.Add(2 * time.Hour),
			AccessedAt: base.Add(2 * time.Hour),
		},
	}
}
