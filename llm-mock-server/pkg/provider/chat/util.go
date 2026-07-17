package chat

import (
	"encoding/json"
	"net/http"

	"llm-mock-server/pkg/utils"

	"github.com/gin-gonic/gin"
)

type simpleErrorResponse struct {
	Error string `json:"error"`
}

func prompt2Response(prompt string) string {
	return prompt
}

func ptr[T any](v T) *T {
	return &v
}

func renderJSONStreamEvent(ctx *gin.Context, payload any) bool {
	data, err := json.Marshal(payload)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, simpleErrorResponse{Error: "Failed to marshal response"})
		return false
	}
	ctx.Render(-1, streamEvent{Data: "data: " + string(data)})
	ctx.Writer.Flush()
	return true
}

func writeNamedJSONSSE(ctx *gin.Context, eventType string, payload any) bool {
	data, err := json.Marshal(payload)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, simpleErrorResponse{Error: "Failed to marshal response"})
		return false
	}

	select {
	case <-ctx.Request.Context().Done():
		return false
	default:
	}

	if _, err := ctx.Writer.Write([]byte("event: " + eventType + "\ndata: " + string(data) + "\n\n")); err != nil {
		return false
	}
	ctx.Writer.Flush()
	return true
}

// bindAndValidateChatRequest binds and validates the request body, writing a 400 response on failure.
func bindAndValidateChatRequest(ctx *gin.Context, chatRequest *chatCompletionRequest) bool {
	if err := ctx.ShouldBindJSON(chatRequest); err != nil {
		ctx.JSON(http.StatusBadRequest, simpleErrorResponse{Error: err.Error()})
		return false
	}
	if err := utils.Validate.Struct(chatRequest); err != nil {
		ctx.JSON(http.StatusBadRequest, simpleErrorResponse{Error: err.Error()})
		return false
	}
	return true
}

func lastStringPrompt(chatRequest *chatCompletionRequest) string {
	last := chatRequest.Messages[len(chatRequest.Messages)-1]
	if last.IsStringContent() {
		return last.StringContent()
	}
	return ""
}
