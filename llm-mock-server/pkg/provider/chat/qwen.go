package chat

import (
	"fmt"
	"net/http"
	"time"

	"llm-mock-server/pkg/utils"

	"github.com/gin-gonic/gin"
	"github.com/go-playground/validator/v10"
)

const (
	qwenDomain              = "dashscope.aliyuncs.com"
	qwenChatCompletionPath  = "/api/v1/services/aigc/text-generation/generation"
	qwenResultFormatMessage = "message"
)

type qwenProvider struct{}

func (p *qwenProvider) ShouldHandleRequest(ctx *gin.Context) bool {
	context, _ := getRequestContext(ctx)
	return context.Host == qwenDomain && context.Path == qwenChatCompletionPath
}

func (p *qwenProvider) HandleChatCompletions(ctx *gin.Context) {
	authHeader := ctx.GetHeader("Authorization")
	if authHeader == "" {
		p.sendErrorResponse(ctx, http.StatusUnauthorized, "InvalidApiKey", "No API-key provided.")
		return
	}

	var chatRequest qwenTextGenRequest
	if err := ctx.ShouldBindJSON(&chatRequest); err != nil {
		p.sendErrorResponse(ctx, http.StatusBadRequest, "InvalidParameter", fmt.Sprintf("invalid params: %v", err.Error()))
		return
	}

	if err := utils.Validate.Struct(chatRequest); err != nil {
		validationErrors := err.(validator.ValidationErrors)
		for _, fieldError := range validationErrors {
			p.sendErrorResponse(ctx, http.StatusBadRequest, "InvalidParameter", fmt.Sprintf("invalid params: %v", fieldError.Error()))
			return
		}
	}

	prompt := ""
	messages := chatRequest.Input.Messages
	if messages[len(messages)-1].IsStringContent() {
		prompt = messages[len(messages)-1].StringContent()
	}
	response := prompt2Response(prompt)

	if p.isStreamRequest(ctx) {
		p.handleStreamResponse(ctx, chatRequest, response)
	} else {
		p.handleNonStreamResponse(ctx, chatRequest, response)
	}
}

func (p *qwenProvider) sendErrorResponse(ctx *gin.Context, statusCode int, errorCode, errorMsg string) {
	ctx.JSON(statusCode, qwenErrorResp{
		Code:      errorCode,
		Message:   errorMsg,
		RequestId: completionMockId,
	})
}

func (p *qwenProvider) handleStreamResponse(ctx *gin.Context, chatRequest qwenTextGenRequest, response string) {
	utils.SetEventStreamHeaders(ctx)

	accumulated := ""
	incrementalOutput := chatRequest.Parameters.IncrementalOutput

	for _, r := range []rune(response) {
		select {
		case <-ctx.Request.Context().Done():
			return
		default:
		}

		piece := string(r)
		if incrementalOutput {
			accumulated = piece
		} else {
			accumulated += piece
		}

		chunk := createQwenStreamResponse(chatRequest, accumulated, "")
		if !renderJSONStreamEvent(ctx, chunk) {
			return
		}

		select {
		case <-ctx.Request.Context().Done():
			return
		case <-time.After(50 * time.Millisecond):
		}
	}

	renderJSONStreamEvent(ctx, createQwenStreamResponse(chatRequest, "", stopReason))
}

func (p *qwenProvider) handleNonStreamResponse(ctx *gin.Context, chatRequest qwenTextGenRequest, response string) {
	ctx.JSON(http.StatusOK, createQwenTextGenResponse(chatRequest, response))
}

// isStreamRequest checks if the request is a stream request.
func (p *qwenProvider) isStreamRequest(ctx *gin.Context) bool {
	acceptHeader := ctx.GetHeader("Accept")
	sseHeader := ctx.GetHeader("X-DashScope-SSE")

	return acceptHeader == "text/event-stream" || sseHeader == "enable"
}

func createQwenTextGenResponse(chatRequest qwenTextGenRequest, response string) qwenTextGenResponse {
	return qwenTextGenResponse{
		RequestId: completionMockId,
		Output:    createQwenTextGenOutput(chatRequest, response, stopReason),
		Usage:     createQwenUsage(),
	}
}

func createQwenStreamResponse(chatRequest qwenTextGenRequest, response, finishReason string) qwenTextGenResponse {
	return qwenTextGenResponse{
		RequestId: completionMockId,
		Output:    createQwenTextGenOutput(chatRequest, response, finishReason),
		Usage:     createQwenUsage(),
	}
}

func createQwenTextGenOutput(chatRequest qwenTextGenRequest, response, finishReason string) qwenTextGenOutput {
	if chatRequest.Parameters.ResultFormat == "" || chatRequest.Parameters.ResultFormat == qwenResultFormatMessage {
		return qwenTextGenOutput{
			Choices: []qwenTextGenChoice{
				{
					FinishReason: finishReason,
					Message: qwenMessage{
						Role:    roleAssistant,
						Content: response,
					},
				},
			},
		}
	}

	return qwenTextGenOutput{
		FinishReason: finishReason,
		Text:         response,
	}
}

func createQwenUsage() qwenUsage {
	return qwenUsage{
		InputTokens:  completionMockUsage.PromptTokens,
		OutputTokens: completionMockUsage.CompletionTokens,
		TotalTokens:  completionMockUsage.TotalTokens,
	}
}

type qwenErrorResp struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	RequestId string `json:"request_id"`
}

type qwenTextGenRequest struct {
	Model      string                `json:"model" validate:"required"`
	Input      qwenTextGenInput      `json:"input" validate:"required"`
	Parameters qwenTextGenParameters `json:"parameters,omitempty"`
}

type qwenTextGenInput struct {
	Messages []qwenMessage `json:"messages" validate:"required,min=1"`
}

type qwenTextGenParameters struct {
	ResultFormat      string  `json:"result_format,omitempty"`
	MaxTokens         int     `json:"max_tokens,omitempty"`
	RepetitionPenalty float64 `json:"repetition_penalty,omitempty"`
	N                 int     `json:"n,omitempty"`
	Seed              int     `json:"seed,omitempty"`
	Temperature       float64 `json:"temperature,omitempty"`
	TopP              float64 `json:"top_p,omitempty"`
	IncrementalOutput bool    `json:"incremental_output,omitempty"`
	EnableSearch      bool    `json:"enable_search,omitempty"`
	Tools             []tool  `json:"tools,omitempty"`
}

type qwenTextGenResponse struct {
	RequestId string            `json:"request_id"`
	Output    qwenTextGenOutput `json:"output"`
	Usage     qwenUsage         `json:"usage"`
}

type qwenTextGenOutput struct {
	FinishReason string              `json:"finish_reason,omitempty"`
	Text         string              `json:"text,omitempty"`
	Choices      []qwenTextGenChoice `json:"choices,omitempty"`
}

type qwenTextGenChoice struct {
	FinishReason string      `json:"finish_reason"`
	Message      qwenMessage `json:"message"`
}

type qwenUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type qwenMessage struct {
	Name      string     `json:"name,omitempty"`
	Role      string     `json:"role"`
	Content   any        `json:"content"`
	ToolCalls []toolCall `json:"tool_calls,omitempty"`
}

func (m *qwenMessage) IsStringContent() bool {
	_, ok := m.Content.(string)
	return ok
}

func (m *qwenMessage) StringContent() string {
	content, ok := m.Content.(string)
	if ok {
		return content
	}
	contentList, ok := m.Content.([]any)
	if ok {
		var contentStr string
		for _, contentItem := range contentList {
			contentMap, ok := contentItem.(map[string]any)
			if !ok {
				continue
			}
			if contentMap["type"] == contentTypeText {
				if subStr, ok := contentMap[contentTypeText].(string); ok {
					contentStr += subStr + "\n"
				}
			}
		}
		return contentStr
	}
	return ""
}
