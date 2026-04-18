package mcpserver

import (
	"context"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// registerPrompts adds all prompt definitions to the SDK server.
func (s *Server) registerPrompts(srv *mcpsdk.Server) {
	srv.AddPrompt(&mcpsdk.Prompt{
		Name:        "session_start",
		Description: "Bootstrap memory context for a new session. Read the L1 cluster index and recall relevant memories.",
		Arguments: []*mcpsdk.PromptArgument{
			{
				Name:        "project_hint",
				Description: "Project name or path to scope the initial recall query.",
				Required:    false,
			},
		},
	}, s.handleSessionStartPrompt)

	srv.AddPrompt(&mcpsdk.Prompt{
		Name:        "session_end",
		Description: "Wrap up a session: trigger decay tick and optionally write an L3 episode summarizing significant work.",
	}, s.handleSessionEndPrompt)
}

func (s *Server) handleSessionStartPrompt(_ context.Context, req *mcpsdk.GetPromptRequest) (*mcpsdk.GetPromptResult, error) {
	hint := req.Params.Arguments["project_hint"]

	query := "recent context and user preferences"
	if hint != "" {
		query = hint + " project context, conventions, and recent decisions"
	}

	text := `Memory bootstrap for this session.

1. Read the reverie://l1/index resource to see the cluster landscape.
2. Call memory_recall with query: "` + query + `"
3. After using recalled memories in your response, call memory_reinforce with their IDs.

Clusters with high utility are your strongest context — prioritize them.`

	return &mcpsdk.GetPromptResult{
		Description: "Session start recall",
		Messages: []*mcpsdk.PromptMessage{
			{
				Role:    "user",
				Content: &mcpsdk.TextContent{Text: text},
			},
		},
	}, nil
}

func (s *Server) handleSessionEndPrompt(_ context.Context, _ *mcpsdk.GetPromptRequest) (*mcpsdk.GetPromptResult, error) {
	text := `Session wrap-up.

1. Call memory_decay_tick with session_end=true to advance the decay clock for all clusters.
2. If significant work was done this session (architecture decisions, bug fixes, new patterns learned), write an L3 episode summarizing it:
   - Call memory_write with type="project" (or "feedback" for process lessons) and an episode payload:
     - situation: what triggered the work
     - action: what was done
     - outcome: what happened as a result
     - preemptive: actionable lesson for next time
3. Skip the episode if this was a trivial session (quick question, no lasting decisions).`

	return &mcpsdk.GetPromptResult{
		Description: "Session end consolidation",
		Messages: []*mcpsdk.PromptMessage{
			{
				Role:    "user",
				Content: &mcpsdk.TextContent{Text: text},
			},
		},
	}, nil
}
