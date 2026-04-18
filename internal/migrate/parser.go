// Package migrate reads Claude Code auto-memory markdown files and imports
// them into reverie's persistent store as L2 semantic facts.
package migrate

import (
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

// ParsedMemory holds the parsed contents of a Claude Code auto-memory file.
type ParsedMemory struct {
	Name            string `yaml:"name"`
	Description     string `yaml:"description"`
	Type            string `yaml:"type"`
	OriginSessionID string `yaml:"originSessionId"`
	Body            string // everything after the closing ---
}

// ParseMemoryFile reads and parses a Claude Code auto-memory .md file.
// It expects YAML frontmatter delimited by --- lines at the top of the file,
// followed by a markdown body.
//
// Returns an error if the file cannot be read, has no opening --- delimiter,
// or has no closing --- delimiter.
func ParseMemoryFile(path string) (*ParsedMemory, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read memory file %q: %w", path, err)
	}

	content := string(data)

	// The file must start with "---" (possibly preceded by whitespace/BOM).
	trimmed := strings.TrimSpace(content)
	if !strings.HasPrefix(trimmed, "---") {
		return nil, fmt.Errorf("parse memory file %q: no opening --- delimiter", path)
	}

	// Find the opening and closing --- delimiters.
	// The first --- is at the start; find the second ---.
	// We work on the trimmed content that starts with "---".
	afterOpener := trimmed[3:] // skip the opening "---"

	// Skip the rest of the opening line (e.g., trailing whitespace or newline).
	if idx := strings.Index(afterOpener, "\n"); idx >= 0 {
		afterOpener = afterOpener[idx+1:]
	} else {
		// No newline after opening --- means no closing delimiter.
		return nil, fmt.Errorf("parse memory file %q: no closing --- delimiter", path)
	}

	// Find the closing "---" on its own line.
	closerIdx := -1
	lines := strings.SplitAfter(afterOpener, "\n")
	pos := 0
	for _, line := range lines {
		if strings.TrimSpace(line) == "---" {
			closerIdx = pos
			break
		}
		pos += len(line)
	}

	if closerIdx < 0 {
		return nil, fmt.Errorf("parse memory file %q: no closing --- delimiter", path)
	}

	yamlBlock := afterOpener[:closerIdx]
	body := afterOpener[closerIdx:]
	// Skip the closing "---" line itself.
	if idx := strings.Index(body, "\n"); idx >= 0 {
		body = body[idx+1:]
	} else {
		body = ""
	}

	// Trim leading/trailing whitespace from the body but preserve internal formatting.
	body = strings.TrimSpace(body)

	parsed := &ParsedMemory{}
	if err := yaml.Unmarshal([]byte(yamlBlock), parsed); err != nil {
		return nil, fmt.Errorf("parse memory file %q: unmarshal frontmatter: %w", path, err)
	}

	parsed.Body = body
	return parsed, nil
}
