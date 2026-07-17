package chat

import (
	"net/http"
	"time"

	"llm-mock-server/pkg/log"
	"llm-mock-server/pkg/utils"

	"github.com/gin-gonic/gin"
)

const (
	cohereDomain   = "api.cohere.com"
	cohereChatPath = "/v1/chat"
)

// cohereRequest is the Cohere v1 /v1/chat request shape. ai-proxy sends this after converting
// the client's OpenAI-format request (it maps the first message to the top-level "message").
type cohereRequest struct {
	Message string `json:"message"`
	Stream  bool   `json:"stream"`
}

type cohereProvider struct{}

func (p *cohereProvider) ShouldHandleRequest(ctx *gin.Context) bool {
	context, err := getRequestContext(ctx)
	if err != nil {
		log.Errorf("get request context failed: %v", err)
		return false
	}
	return context.Host == cohereDomain && context.Path == cohereChatPath
}

func (p *cohereProvider) HandleChatCompletions(ctx *gin.Context) {
	// The real Cohere API requires "Authorization: Bearer <api key>"; ai-proxy always injects it.
	if ctx.GetHeader("Authorization") == "" {
		p.sendErrorResponse(ctx, http.StatusUnauthorized, "invalid api token")
		return
	}

	var req cohereRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		p.sendErrorResponse(ctx, http.StatusBadRequest, err.Error())
		return
	}

	// This native Cohere response is what ai-proxy converts to OpenAI shape.
	if req.Stream {
		p.handleStreamResponse(ctx, req.Message)
	} else {
		p.handleNonStreamResponse(ctx, req.Message)
	}
}

func (p *cohereProvider) sendErrorResponse(ctx *gin.Context, statusCode int, message string) {
	ctx.JSON(statusCode, cohereErrorResponse{Message: message})
}

func (p *cohereProvider) handleNonStreamResponse(ctx *gin.Context, response string) {
	ctx.JSON(http.StatusOK, createCohereChatResponse(response))
}

func (p *cohereProvider) handleStreamResponse(ctx *gin.Context, response string) {
	// ai-proxy requests streaming with Accept: text/event-stream, so Cohere replies with SSE frames
	// (event: <event_type> + data: <json>).
	utils.SetEventStreamHeaders(ctx)

	send := func(payload cohereStreamEvent) bool {
		return writeNamedJSONSSE(ctx, payload.EventType, payload)
	}

	if !send(cohereStreamEvent{
		EventType:    "stream-start",
		GenerationId: completionMockId,
		IsFinished:   false,
	}) {
		return
	}
	for _, r := range response {
		if !send(cohereStreamEvent{
			EventType:  "text-generation",
			Text:       string(r),
			IsFinished: false,
		}) {
			return
		}
		select {
		case <-ctx.Request.Context().Done():
			return
		case <-time.After(50 * time.Millisecond):
		}
	}
	send(cohereStreamEvent{
		EventType:    "stream-end",
		IsFinished:   true,
		FinishReason: "COMPLETE",
		Response:     createCohereStreamEndResponse(response),
	})
}

// createCohereMeta builds the Cohere v1 "meta" object, which carries api_version plus tokens and billed_units.
func createCohereMeta() cohereMeta {
	counts := cohereTokenCounts{
		InputTokens:  completionMockUsage.PromptTokens,
		OutputTokens: completionMockUsage.CompletionTokens,
	}
	return cohereMeta{
		APIVersion:  cohereAPIVersion{Version: "1"},
		Tokens:      counts,
		BilledUnits: counts,
	}
}

func createCohereChatResponse(response string) cohereChatResponse {
	return cohereChatResponse{
		ResponseId:   completionMockId,
		Text:         response,
		GenerationId: completionMockId,
		ChatHistory: []cohereChatHistoryItem{
			{Role: "USER", Message: response},
			{Role: "CHATBOT", Message: response},
		},
		FinishReason: "COMPLETE",
		Meta:         createCohereMeta(),
	}
}

func createCohereStreamEndResponse(response string) *cohereChatResponse {
	completion := createCohereChatResponse(response)
	completion.ChatHistory = nil
	return &completion
}

type cohereErrorResponse struct {
	Message string `json:"message"`
}

type cohereMeta struct {
	APIVersion  cohereAPIVersion  `json:"api_version"`
	Tokens      cohereTokenCounts `json:"tokens"`
	BilledUnits cohereTokenCounts `json:"billed_units"`
}

type cohereAPIVersion struct {
	Version string `json:"version"`
}

type cohereTokenCounts struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type cohereChatResponse struct {
	ResponseId   string                  `json:"response_id"`
	Text         string                  `json:"text"`
	GenerationId string                  `json:"generation_id"`
	ChatHistory  []cohereChatHistoryItem `json:"chat_history,omitempty"`
	FinishReason string                  `json:"finish_reason"`
	Meta         cohereMeta              `json:"meta"`
}

type cohereChatHistoryItem struct {
	Role    string `json:"role"`
	Message string `json:"message"`
}

type cohereStreamEvent struct {
	EventType    string              `json:"event_type"`
	GenerationId string              `json:"generation_id,omitempty"`
	IsFinished   bool                `json:"is_finished"`
	Text         string              `json:"text,omitempty"`
	FinishReason string              `json:"finish_reason,omitempty"`
	Response     *cohereChatResponse `json:"response,omitempty"`
}
