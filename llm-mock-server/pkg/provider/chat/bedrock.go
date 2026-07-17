package chat

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"net/http"
	"strings"
	"time"

	"llm-mock-server/pkg/log"

	"github.com/gin-gonic/gin"
)

const (
	// ai-proxy rewrites Bedrock requests to bedrock-runtime.{region}.amazonaws.com
	// (or bedrock-mantle.{region}.api.aws). Match on the stable fragments.
	bedrockHostFragment      = "bedrock"
	bedrockDomainFragment    = "amazonaws.com"
	bedrockConversePath      = "/converse"
	bedrockConverseStream    = "/converse-stream"
	bedrockStopReasonEndTurn = "end_turn"
)

type bedrockProvider struct{}

func (p *bedrockProvider) ShouldHandleRequest(ctx *gin.Context) bool {
	requestCtx, err := getRequestContext(ctx)
	if err != nil {
		log.Errorf("get request context failed: %v", err)
		return false
	}

	host := requestCtx.Host
	path := requestCtx.Path
	if !strings.Contains(host, bedrockHostFragment) || !strings.Contains(host, bedrockDomainFragment) {
		return false
	}
	return strings.HasSuffix(path, bedrockConversePath) || strings.HasSuffix(path, bedrockConverseStream)
}

func (p *bedrockProvider) HandleChatCompletions(ctx *gin.Context) {
	isStreaming := strings.HasSuffix(ctx.Request.URL.Path, bedrockConverseStream)

	var bedrockRequest bedrockConverseRequest
	if err := ctx.ShouldBindJSON(&bedrockRequest); err != nil {
		p.sendErrorResponse(ctx, http.StatusBadRequest, fmt.Sprintf("Invalid request: %v", err.Error()))
		return
	}

	if err := p.validateRequest(&bedrockRequest); err != nil {
		p.sendErrorResponse(ctx, http.StatusBadRequest, fmt.Sprintf("Validation error: %v", err.Error()))
		return
	}

	content := p.generateResponse(&bedrockRequest)

	if isStreaming {
		p.handleStreamResponse(ctx, content)
	} else {
		p.handleNonStreamResponse(ctx, content)
	}
}

func (p *bedrockProvider) sendErrorResponse(ctx *gin.Context, statusCode int, message string) {
	ctx.JSON(statusCode, bedrockErrorResponse{Message: message})
}

func (p *bedrockProvider) validateRequest(req *bedrockConverseRequest) error {
	if len(req.Messages) == 0 {
		return fmt.Errorf("messages are required")
	}
	for i, msg := range req.Messages {
		if len(msg.Content) == 0 {
			return fmt.Errorf("message %d: content is required", i)
		}
		for j, block := range msg.Content {
			if block.Text == "" {
				return fmt.Errorf("message %d, content %d: text is required", i, j)
			}
		}
	}
	return nil
}

func (p *bedrockProvider) generateResponse(req *bedrockConverseRequest) string {
	content := "This is a mock response from Bedrock provider. "
	if len(req.Messages) > 0 {
		lastMsg := req.Messages[len(req.Messages)-1]
		if len(lastMsg.Content) > 0 {
			runes := []rune(lastMsg.Content[len(lastMsg.Content)-1].Text)
			if len(runes) > 50 {
				content += "You said: " + string(runes[:50]) + "..."
			} else {
				content += "You said: " + string(runes)
			}
		}
	}
	return content
}

func (p *bedrockProvider) handleNonStreamResponse(ctx *gin.Context, response string) {
	ctx.JSON(http.StatusOK, createBedrockConverseResponse(response))
}

func (p *bedrockProvider) handleStreamResponse(ctx *gin.Context, response string) {
	ctx.Header("Content-Type", "application/vnd.amazon.eventstream")
	ctx.Header("Cache-Control", "no-cache")
	ctx.Header("Connection", "keep-alive")
	ctx.Header("Access-Control-Allow-Origin", "*")

	words := strings.Fields(response)
	flusher, ok := ctx.Writer.(http.Flusher)

	for i, word := range words {
		deltaPayload, _ := json.Marshal(bedrockContentBlockDeltaEvent{
			ContentBlockIndex: 0,
			Delta:             bedrockDeltaText{Text: word + " "},
		})
		ctx.Writer.Write(encodeBedrockEventStreamMessage("contentBlockDelta", deltaPayload))
		if ok {
			flusher.Flush()
		}
		select {
		case <-ctx.Request.Context().Done():
			return
		default:
		}

		if i == len(words)-1 {
			stopPayload, _ := json.Marshal(bedrockMessageStopEvent{StopReason: bedrockStopReasonEndTurn})
			ctx.Writer.Write(encodeBedrockEventStreamMessage("messageStop", stopPayload))
			if ok {
				flusher.Flush()
			}
		}
		select {
		case <-ctx.Request.Context().Done():
			return
		case <-time.After(50 * time.Millisecond):
		}
	}
}

func createBedrockConverseResponse(response string) bedrockConverseResponse {
	return bedrockConverseResponse{
		Metrics:    bedrockConverseMetrics{LatencyMs: 100},
		Output:     bedrockConverseOutput{Message: bedrockConverseMessage{Role: "assistant", Content: []bedrockContentBlock{{Text: response}}}},
		StopReason: bedrockStopReasonEndTurn,
		Usage: bedrockTokenUsage{
			InputTokens:  completionMockUsage.PromptTokens,
			OutputTokens: completionMockUsage.CompletionTokens,
			TotalTokens:  completionMockUsage.TotalTokens,
		},
	}
}

// encodeBedrockEventStreamMessage builds a single AWS Event Stream message frame:
// [TotalLength:4][HeadersLength:4][PreludeCRC:4][Headers][Payload][MessageCRC:4].
func encodeBedrockEventStreamMessage(eventType string, payload []byte) []byte {
	headers := encodeBedrockEventStreamHeaders(eventType)
	headersLen := len(headers)
	totalLen := uint32(16 + headersLen + len(payload))

	prelude := make([]byte, 8)
	binary.BigEndian.PutUint32(prelude[0:4], totalLen)
	binary.BigEndian.PutUint32(prelude[4:8], uint32(headersLen))
	preludeCRC := crc32.ChecksumIEEE(prelude)
	preludeCrc := make([]byte, 4)
	binary.BigEndian.PutUint32(preludeCrc, preludeCRC)

	msg := make([]byte, 0, int(totalLen))
	msg = append(msg, prelude...)
	msg = append(msg, preludeCrc...)
	msg = append(msg, headers...)
	msg = append(msg, payload...)
	msgCRC := crc32.ChecksumIEEE(msg)
	msgCrc := make([]byte, 4)
	binary.BigEndian.PutUint32(msgCrc, msgCRC)
	msg = append(msg, msgCrc...)
	return msg
}

// encodeBedrockEventStreamHeaders encodes the three headers ai-proxy expects on
// an event frame.
func encodeBedrockEventStreamHeaders(eventType string) []byte {
	var buf bytes.Buffer
	writeHeader := func(name, value string) {
		buf.WriteByte(byte(len(name)))
		buf.WriteString(name)
		buf.WriteByte(7)
		lenBuf := make([]byte, 2)
		binary.BigEndian.PutUint16(lenBuf, uint16(len(value)))
		buf.Write(lenBuf)
		buf.WriteString(value)
	}
	writeHeader(":message-type", "event")
	writeHeader(":event-type", eventType)
	writeHeader(":content-type", "application/json")
	return buf.Bytes()
}

type bedrockConverseRequest struct {
	Messages []bedrockRequestMessage `json:"messages"`
}

type bedrockRequestMessage struct {
	Role    string                  `json:"role"`
	Content []bedrockRequestContent `json:"content"`
}

type bedrockRequestContent struct {
	Text string `json:"text"`
}

type bedrockConverseResponse struct {
	Metrics    bedrockConverseMetrics `json:"metrics"`
	Output     bedrockConverseOutput  `json:"output"`
	StopReason string                 `json:"stopReason"`
	Usage      bedrockTokenUsage      `json:"usage"`
}

type bedrockConverseMetrics struct {
	LatencyMs int `json:"latencyMs"`
}

type bedrockConverseOutput struct {
	Message bedrockConverseMessage `json:"message"`
}

type bedrockConverseMessage struct {
	Content []bedrockContentBlock `json:"content"`
	Role    string                `json:"role"`
}

type bedrockContentBlock struct {
	Text string `json:"text,omitempty"`
}

type bedrockTokenUsage struct {
	InputTokens  int `json:"inputTokens,omitempty"`
	OutputTokens int `json:"outputTokens,omitempty"`
	TotalTokens  int `json:"totalTokens"`
}

type bedrockErrorResponse struct {
	Message string `json:"message"`
}

type bedrockContentBlockDeltaEvent struct {
	ContentBlockIndex int              `json:"contentBlockIndex"`
	Delta             bedrockDeltaText `json:"delta"`
}

type bedrockDeltaText struct {
	Text string `json:"text"`
}

type bedrockMessageStopEvent struct {
	StopReason string `json:"stopReason"`
}
