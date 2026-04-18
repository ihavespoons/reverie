package migrate

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseMemoryFile(t *testing.T) {
	tests := []struct {
		name        string
		content     string
		wantName    string
		wantDesc    string
		wantType    string
		wantSession string
		wantBody    string
		wantErr     bool
	}{
		{
			name: "valid file with all fields",
			content: `---
name: GitHub username
description: The user's GitHub username is ihavespoons
type: user
originSessionId: abc-123
---
The user's GitHub username is **ihavespoons**.
`,
			wantName:    "GitHub username",
			wantDesc:    "The user's GitHub username is ihavespoons",
			wantType:    "user",
			wantSession: "abc-123",
			wantBody:    "The user's GitHub username is **ihavespoons**.",
		},
		{
			name: "missing originSessionId",
			content: `---
name: Project goals
description: Building something
type: project
---
We are building a test framework.
`,
			wantName:    "Project goals",
			wantDesc:    "Building something",
			wantType:    "project",
			wantSession: "",
			wantBody:    "We are building a test framework.",
		},
		{
			name:    "no opening delimiter",
			content: "Just some markdown without frontmatter.\n",
			wantErr: true,
		},
		{
			name: "no closing delimiter",
			content: `---
name: Broken
type: user
`,
			wantErr: true,
		},
		{
			name: "empty body",
			content: `---
name: Empty
description: Nothing after frontmatter
type: feedback
---
`,
			wantName: "Empty",
			wantDesc: "Nothing after frontmatter",
			wantType: "feedback",
			wantBody: "",
		},
		{
			name: "body with multiple paragraphs",
			content: `---
name: Multi
description: Multi paragraph body
type: reference
---
First paragraph.

Second paragraph with **bold**.

Third paragraph.
`,
			wantName: "Multi",
			wantDesc: "Multi paragraph body",
			wantType: "reference",
			wantBody: "First paragraph.\n\nSecond paragraph with **bold**.\n\nThird paragraph.",
		},
		{
			name: "missing type field",
			content: `---
name: No type
description: Has no type
---
Some content here.
`,
			wantName: "No type",
			wantDesc: "Has no type",
			wantType: "",
			wantBody: "Some content here.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			path := filepath.Join(tmpDir, "test.md")
			if err := os.WriteFile(path, []byte(tt.content), 0o644); err != nil {
				t.Fatalf("write temp file: %v", err)
			}

			parsed, err := ParseMemoryFile(path)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if parsed.Name != tt.wantName {
				t.Errorf("Name = %q, want %q", parsed.Name, tt.wantName)
			}
			if parsed.Description != tt.wantDesc {
				t.Errorf("Description = %q, want %q", parsed.Description, tt.wantDesc)
			}
			if parsed.Type != tt.wantType {
				t.Errorf("Type = %q, want %q", parsed.Type, tt.wantType)
			}
			if parsed.OriginSessionID != tt.wantSession {
				t.Errorf("OriginSessionID = %q, want %q", parsed.OriginSessionID, tt.wantSession)
			}
			if parsed.Body != tt.wantBody {
				t.Errorf("Body = %q, want %q", parsed.Body, tt.wantBody)
			}
		})
	}
}

func TestParseMemoryFile_RealTestdata(t *testing.T) {
	parsed, err := ParseMemoryFile("testdata/memory/user_context.md")
	if err != nil {
		t.Fatalf("parse testdata user_context.md: %v", err)
	}

	if parsed.Name != "Test user context" {
		t.Errorf("Name = %q, want %q", parsed.Name, "Test user context")
	}
	if parsed.Type != "user" {
		t.Errorf("Type = %q, want %q", parsed.Type, "user")
	}
	if parsed.OriginSessionID != "test-session-1" {
		t.Errorf("OriginSessionID = %q, want %q", parsed.OriginSessionID, "test-session-1")
	}
	if parsed.Body != "The test user works at TestCorp as a senior engineer." {
		t.Errorf("Body = %q", parsed.Body)
	}
}

func TestParseMemoryFile_NonexistentFile(t *testing.T) {
	_, err := ParseMemoryFile("/nonexistent/path/file.md")
	if err == nil {
		t.Fatal("expected error for nonexistent file, got nil")
	}
}
