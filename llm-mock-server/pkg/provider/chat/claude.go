package chat

import (
	"encoding/json"
	"net/http"
	"time"

	"llm-mock-server/pkg/log"
	"llm-mock-server/pkg/utils"

	"github.com/gin-gonic/gin"
)

const (
	claudeDomain       = "api.anthropic.com"
	claudeMessagesPath = "/v1/messages"
	// claudeMockId is an Anthropic-style message id. ai-proxy passes it through as the OpenAI response id.
	claudeMockId    = "msg_llm-mock"
	claudeMockModel = "claude-3-5-sonnet-20241022"
	// claudeMockRequestId mirrors the request id the real API returns in every error body and the
	// "request-id" response header.
	claudeMockRequestId = "req_llm-mock"
)

type claudeProvider struct{}

func (p *claudeProvider) ShouldHandleRequest(ctx *gin.Context) bool {
	context, err := getRequestContext(ctx)
	if err != nil {
		log.Errorf("get request context failed: %v", err)
		return false
	}
	return context.Host == claudeDomain && context.Path == claudeMessagesPath
}

func (p *claudeProvider) HandleChatCompletions(ctx *gin.Context) {
	// The real API requires the anthropic-version header (ai-proxy always injects it).
	if ctx.GetHeader("anthropic-version") == "" {
		p.sendErrorResponse(ctx, http.StatusBadRequest, "invalid_request_error", "anthropic-version header is required")
		return
	}
	// The real Anthropic API authenticates with the "x-api-key" header; the error body mirrors it.
	if ctx.GetHeader("x-api-key") == "" {
		p.sendErrorResponse(ctx, http.StatusUnauthorized, "authentication_error", "invalid x-api-key")
		return
	}

	var req claudeMessagesRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, simpleErrorResponse{Error: err.Error()})
		return
	}
	response := lastClaudeUserText(&req)

	// A sentinel prompt makes the mock return the upstream auth error, letting the e2e verify that
	// ai-proxy surfaces an upstream 401 to the client instead of masking it.
	if response == "__force_auth_error__" {
		p.sendErrorResponse(ctx, http.StatusUnauthorized, "authentication_error", "invalid x-api-key")
		return
	}

	// When the request carries tools, reply with a tool_use block so the tool-call conversion path
	// is exercised (the mock always calls the first tool with a fixed argument).
	if len(req.Tools) > 0 && req.Stream {
		p.handleToolUseStreamResponse(ctx, req.Tools[0].Name)
		return
	}

	if req.Stream {
		p.handleStreamResponse(ctx, response)
	} else {
		p.handleNonStreamResponse(ctx, response)
	}
}

// sendErrorResponse writes an Anthropic-style error response: the request-id header plus a body
// carrying the top-level type/request_id and the nested error object.
func (p *claudeProvider) sendErrorResponse(ctx *gin.Context, status int, errType, message string) {
	ctx.Header("request-id", claudeMockRequestId)
	ctx.JSON(status, claudeErrorResponse{
		Type:      "error",
		RequestID: claudeMockRequestId,
		Error: claudeErrorDetail{
			Type:    errType,
			Message: message,
		},
	})
}

// handleToolUseStreamResponse emits an Anthropic tool_use streaming sequence: message_start,
// content_block_start (tool_use), input_json_delta chunks, content_block_stop, message_delta
// (stop_reason tool_use), message_stop.
func (p *claudeProvider) handleToolUseStreamResponse(ctx *gin.Context, toolName string) {
	utils.SetEventStreamHeaders(ctx)

	start := createClaudeMessageStartEvent()
	if !writeNamedJSONSSE(ctx, start.Type, start) {
		return
	}

	toolUseStart := createClaudeToolUseStartEvent(toolName)
	if !writeNamedJSONSSE(ctx, toolUseStart.Type, toolUseStart) {
		return
	}

	// The tool arguments arrive as partial_json fragments that ai-proxy concatenates.
	for _, frag := range []string{`{"location": `, `"Beijing"}`} {
		event := createClaudeInputJSONDeltaEvent(frag)
		if !writeNamedJSONSSE(ctx, event.Type, event) {
			return
		}
	}

	blockStop := createClaudeContentBlockStopEvent()
	if !writeNamedJSONSSE(ctx, blockStop.Type, blockStop) {
		return
	}

	messageDelta := createClaudeMessageDeltaEvent("tool_use")
	if !writeNamedJSONSSE(ctx, messageDelta.Type, messageDelta) {
		return
	}

	messageStop := createClaudeMessageStopEvent()
	writeNamedJSONSSE(ctx, messageStop.Type, messageStop)
}

func (p *claudeProvider) handleNonStreamResponse(ctx *gin.Context, response string) {
	ctx.JSON(http.StatusOK, createClaudeMessagesResponse(response))
}

func (p *claudeProvider) handleStreamResponse(ctx *gin.Context, response string) {
	utils.SetEventStreamHeaders(ctx)

	start := createClaudeMessageStartEvent()
	if !writeNamedJSONSSE(ctx, start.Type, start) {
		return
	}

	textStart := createClaudeTextStartEvent()
	if !writeNamedJSONSSE(ctx, textStart.Type, textStart) {
		return
	}

	// One text_delta per rune, mirroring the byte-by-byte streaming of the other provider mocks.
	for _, r := range response {
		event := createClaudeTextDeltaEvent(string(r))
		if !writeNamedJSONSSE(ctx, event.Type, event) {
			return
		}
		select {
		case <-ctx.Request.Context().Done():
			return
		case <-time.After(50 * time.Millisecond):
		}
	}

	blockStop := createClaudeContentBlockStopEvent()
	if !writeNamedJSONSSE(ctx, blockStop.Type, blockStop) {
		return
	}

	// message_delta.usage.output_tokens is the cumulative total for the whole message,
	// matching the real Anthropic API.
	messageDelta := createClaudeMessageDeltaEvent("end_turn")
	if !writeNamedJSONSSE(ctx, messageDelta.Type, messageDelta) {
		return
	}

	messageStop := createClaudeMessageStopEvent()
	writeNamedJSONSSE(ctx, messageStop.Type, messageStop)
}

func createClaudeMessagesResponse(response string) claudeMessagesResponse {
	return claudeMessagesResponse{
		ID:           claudeMockId,
		Type:         "message",
		Role:         roleAssistant,
		Model:        claudeMockModel,
		Content:      []claudeContentBlock{{Type: "text", Text: response}},
		StopReason:   ptr("end_turn"),
		StopSequence: nil,
		Usage:        createClaudeUsage(completionMockUsage.CompletionTokens),
	}
}

func createClaudeMessageStartEvent() claudeMessageStartEvent {
	return claudeMessageStartEvent{
		Type: "message_start",
		Message: claudeMessagesResponse{
			ID:           claudeMockId,
			Type:         "message",
			Role:         roleAssistant,
			Model:        claudeMockModel,
			Content:      []claudeContentBlock{},
			StopReason:   nil,
			StopSequence: nil,
			// The real API reports a small initial output_tokens here; the cumulative total arrives in message_delta.
			Usage: createClaudeUsage(1),
		},
	}
}

func createClaudeToolUseStartEvent(toolName string) claudeContentBlockStartEvent {
	return claudeContentBlockStartEvent{
		Type:  "content_block_start",
		Index: 0,
		ContentBlock: claudeContentBlock{
			Type:  "tool_use",
			ID:    "toolu_llm-mock",
			Name:  toolName,
			Input: map[string]any{},
		},
	}
}

func createClaudeTextStartEvent() claudeContentBlockStartEvent {
	return claudeContentBlockStartEvent{
		Type:  "content_block_start",
		Index: 0,
		ContentBlock: claudeContentBlock{
			Type: "text",
			Text: "",
		},
	}
}

func createClaudeInputJSONDeltaEvent(partialJSON string) claudeContentBlockDeltaEvent {
	return claudeContentBlockDeltaEvent{
		Type:  "content_block_delta",
		Index: 0,
		Delta: claudeContentBlockDelta{
			Type:        "input_json_delta",
			PartialJSON: partialJSON,
		},
	}
}

func createClaudeTextDeltaEvent(text string) claudeContentBlockDeltaEvent {
	return claudeContentBlockDeltaEvent{
		Type:  "content_block_delta",
		Index: 0,
		Delta: claudeContentBlockDelta{
			Type: "text_delta",
			Text: text,
		},
	}
}

func createClaudeContentBlockStopEvent() claudeContentBlockStopEvent {
	return claudeContentBlockStopEvent{
		Type:  "content_block_stop",
		Index: 0,
	}
}

func createClaudeMessageDeltaEvent(stopReason string) claudeMessageDeltaEvent {
	return claudeMessageDeltaEvent{
		Type: "message_delta",
		Delta: claudeStopDelta{
			StopReason:   stopReason,
			StopSequence: nil,
		},
		Usage: createClaudeUsage(completionMockUsage.CompletionTokens),
	}
}

func createClaudeMessageStopEvent() claudeMessageStopEvent {
	return claudeMessageStopEvent{Type: "message_stop"}
}

func createClaudeUsage(outputTokens int) claudeUsage {
	return claudeUsage{
		InputTokens:  completionMockUsage.PromptTokens,
		OutputTokens: outputTokens,
	}
}

// lastClaudeUserText returns the text of the last message, handling both string and
// content-block-array content forms.
func lastClaudeUserText(req *claudeMessagesRequest) string {
	if len(req.Messages) == 0 {
		return ""
	}
	last := req.Messages[len(req.Messages)-1]
	var s string
	if err := json.Unmarshal(last.Content, &s); err == nil {
		return s
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(last.Content, &blocks); err == nil {
		text := ""
		for _, b := range blocks {
			if b.Type == "text" {
				text += b.Text
			}
		}
		return text
	}
	return ""
}

// claudeMessagesRequest is the Anthropic /v1/messages request shape. ai-proxy sends this
// after converting the client's OpenAI-format request.
type claudeMessagesRequest struct {
	Model    string          `json:"model"`
	Messages []claudeMessage `json:"messages"`
	System   json.RawMessage `json:"system,omitempty"`
	Stream   bool            `json:"stream,omitempty"`
	Tools    []claudeTool    `json:"tools,omitempty"`
}

type claudeTool struct {
	Name string `json:"name"`
}

type claudeMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type claudeErrorResponse struct {
	Type      string            `json:"type"`
	RequestID string            `json:"request_id"`
	Error     claudeErrorDetail `json:"error"`
}

type claudeErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type claudeMessagesResponse struct {
	ID           string               `json:"id"`
	Type         string               `json:"type"`
	Role         string               `json:"role"`
	Model        string               `json:"model"`
	Content      []claudeContentBlock `json:"content"`
	StopReason   *string              `json:"stop_reason"`
	StopSequence *string              `json:"stop_sequence"`
	Usage        claudeUsage          `json:"usage"`
}

type claudeContentBlock struct {
	Type  string         `json:"type"`
	Text  string         `json:"text,omitempty"`
	ID    string         `json:"id,omitempty"`
	Name  string         `json:"name,omitempty"`
	Input map[string]any `json:"input,omitempty"`
}

type claudeUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type claudeMessageStartEvent struct {
	Type    string                 `json:"type"`
	Message claudeMessagesResponse `json:"message"`
}

type claudeContentBlockStartEvent struct {
	Type         string             `json:"type"`
	Index        int                `json:"index"`
	ContentBlock claudeContentBlock `json:"content_block"`
}

type claudeContentBlockDeltaEvent struct {
	Type  string                  `json:"type"`
	Index int                     `json:"index"`
	Delta claudeContentBlockDelta `json:"delta"`
}

type claudeContentBlockDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
}

type claudeContentBlockStopEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
}

type claudeMessageDeltaEvent struct {
	Type  string          `json:"type"`
	Delta claudeStopDelta `json:"delta"`
	Usage claudeUsage     `json:"usage"`
}

type claudeStopDelta struct {
	StopReason   string  `json:"stop_reason"`
	StopSequence *string `json:"stop_sequence"`
}

type claudeMessageStopEvent struct {
	Type string `json:"type"`
}
