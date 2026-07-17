package chat

import (
	"net/http"
	"time"

	"llm-mock-server/pkg/log"
	"llm-mock-server/pkg/utils"

	"github.com/gin-gonic/gin"
)

const (
	hunyuanDomain = "hunyuan.tencentcloudapi.com"
	hunyuanPath   = "/"
	// hunyuanNote is the disclaimer the real Hunyuan API always returns.
	hunyuanNote = "以上内容为AI生成，不代表开发者立场，请勿删除或修改本标记"
)

// hunyuanRequest is the native Tencent Hunyuan ChatCompletions request shape (capitalized keys).
// ai-proxy sends this after converting the client's OpenAI-format request. Only the text-chat
// subset is modeled; Tools / ToolChoice are not (ai-proxy's hunyuan request does not forward them).
type hunyuanRequest struct {
	Model    string                `json:"Model"`
	Messages []hunyuanInputMessage `json:"Messages"`
	Stream   bool                  `json:"Stream"`
}

type hunyuanProvider struct{}

func (p *hunyuanProvider) ShouldHandleRequest(ctx *gin.Context) bool {
	context, err := getRequestContext(ctx)
	if err != nil {
		log.Errorf("get request context failed: %v", err)
		return false
	}
	return context.Host == hunyuanDomain && context.Path == hunyuanPath
}

func (p *hunyuanProvider) HandleChatCompletions(ctx *gin.Context) {
	// The real API rejects unsigned requests; ai-proxy always sets a TC3 Authorization header.
	if ctx.GetHeader("Authorization") == "" {
		p.sendErrorResponse(ctx, http.StatusUnauthorized, "AuthFailure", "missing Authorization")
		return
	}
	// The native TC3 API is selected by the X-TC-Action / X-TC-Version headers; ai-proxy always
	// injects them (X-TC-Action: ChatCompletions, X-TC-Version: 2023-09-01).
	if ctx.GetHeader("X-TC-Action") != "ChatCompletions" || ctx.GetHeader("X-TC-Version") != "2023-09-01" {
		p.sendErrorResponse(ctx, http.StatusBadRequest, "InvalidAction", "invalid X-TC-Action / X-TC-Version")
		return
	}

	var req hunyuanRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		p.sendErrorResponse(ctx, http.StatusBadRequest, "", err.Error())
		return
	}
	response := lastHunyuanUserText(&req)

	// The real API returns the request id in the X-TC-RequestId response header.
	ctx.Header("X-TC-RequestId", completionMockId)
	if req.Stream {
		p.handleStreamResponse(ctx, response)
	} else {
		p.handleNonStreamResponse(ctx, response)
	}
}

func (p *hunyuanProvider) sendErrorResponse(ctx *gin.Context, statusCode int, code, message string) {
	ctx.JSON(statusCode, hunyuanErrorEnvelope{
		Response: hunyuanErrorResponse{
			Error: &hunyuanError{
				Code:    code,
				Message: message,
			},
		},
	})
}

func (p *hunyuanProvider) handleNonStreamResponse(ctx *gin.Context, response string) {
	ctx.JSON(http.StatusOK, createHunyuanNonStreamResponse(response))
}

func (p *hunyuanProvider) handleStreamResponse(ctx *gin.Context, response string) {
	utils.SetEventStreamHeaders(ctx)

	// Every frame MUST carry a non-empty Choices array: ai-proxy indexes Choices[0]
	// without a bounds check when converting Hunyuan chunks.
	send := func(content, finish string) bool {
		return renderJSONStreamEvent(ctx, createHunyuanStreamResponse(content, finish))
	}

	// One content delta per rune, then a terminal frame with finish_reason "stop" (native
	// Hunyuan emits no [DONE]).
	for _, r := range response {
		if !send(string(r), "") {
			return
		}
		select {
		case <-ctx.Request.Context().Done():
			return
		case <-time.After(50 * time.Millisecond):
		}
	}
	send("", stopReason)
}

func createHunyuanNonStreamResponse(response string) hunyuanResponseEnvelope {
	return hunyuanResponseEnvelope{
		Response: hunyuanResponse{
			RequestId: completionMockId,
			Note:      hunyuanNote,
			Id:        completionMockId,
			Created:   completionMockCreated,
			Choices: []hunyuanChoice{
				{
					Index:        0,
					FinishReason: stopReason,
					Message: &hunyuanMessage{
						Role:    roleAssistant,
						Content: response,
					},
				},
			},
			Usage: createHunyuanUsage(),
		},
	}
}

func createHunyuanStreamResponse(content, finish string) hunyuanStreamResponse {
	return hunyuanStreamResponse{
		Note:    hunyuanNote,
		Id:      completionMockId,
		Created: time.Now().Unix(),
		Choices: []hunyuanChoice{
			{
				Index:        0,
				FinishReason: finish,
				Delta: &hunyuanMessage{
					Role:    roleAssistant,
					Content: content,
				},
			},
		},
		Usage: createHunyuanUsage(),
	}
}

func createHunyuanUsage() hunyuanUsage {
	return hunyuanUsage{
		PromptTokens:     completionMockUsage.PromptTokens,
		CompletionTokens: completionMockUsage.CompletionTokens,
		TotalTokens:      completionMockUsage.TotalTokens,
	}
}

func lastHunyuanUserText(req *hunyuanRequest) string {
	if len(req.Messages) == 0 {
		return ""
	}
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" {
			return req.Messages[i].Content
		}
	}
	return req.Messages[len(req.Messages)-1].Content
}

type hunyuanInputMessage struct {
	Role    string `json:"Role"`
	Content string `json:"Content"`
}

type hunyuanErrorEnvelope struct {
	Response hunyuanErrorResponse `json:"Response"`
}

type hunyuanErrorResponse struct {
	Error *hunyuanError `json:"Error,omitempty"`
}

type hunyuanError struct {
	Code    string `json:"Code,omitempty"`
	Message string `json:"Message,omitempty"`
}

type hunyuanResponseEnvelope struct {
	Response hunyuanResponse `json:"Response"`
}

type hunyuanResponse struct {
	RequestId string          `json:"RequestId"`
	Note      string          `json:"Note"`
	Id        string          `json:"Id"`
	Created   int64           `json:"Created"`
	Choices   []hunyuanChoice `json:"Choices"`
	Usage     hunyuanUsage    `json:"Usage"`
}

type hunyuanStreamResponse struct {
	Note    string          `json:"Note"`
	Id      string          `json:"Id"`
	Created int64           `json:"Created"`
	Choices []hunyuanChoice `json:"Choices"`
	Usage   hunyuanUsage    `json:"Usage"`
}

type hunyuanChoice struct {
	Index        int             `json:"Index"`
	FinishReason string          `json:"FinishReason,omitempty"`
	Message      *hunyuanMessage `json:"Message,omitempty"`
	Delta        *hunyuanMessage `json:"Delta,omitempty"`
}

type hunyuanMessage struct {
	Role    string `json:"Role"`
	Content string `json:"Content"`
}

type hunyuanUsage struct {
	PromptTokens     int `json:"PromptTokens"`
	CompletionTokens int `json:"CompletionTokens"`
	TotalTokens      int `json:"TotalTokens"`
}
