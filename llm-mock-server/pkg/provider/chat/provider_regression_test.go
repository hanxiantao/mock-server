package chat

import (
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestProviderRegressionNonStreamingResponses(t *testing.T) {
	router := newRegressionRouter()

	testCases := []struct {
		name    string
		path    string
		host    string
		headers map[string]string
		body    string
		assert  func(*testing.T, map[string]any)
	}{
		{
			name: "minimax non-stream",
			path: "/v1/text/chatcompletion_pro",
			host: minimaxDomain,
			headers: map[string]string{
				"Authorization": "Bearer test",
			},
			body: `{
				"model":"abab6.5s-chat",
				"messages":[{"sender_type":"USER","sender_name":"user","text":"hello"}],
				"bot_setting":[{"bot_name":"assistant","content":"You are assistant"}],
				"reply_constraints":{"sender_type":"BOT","sender_name":"assistant"}
			}`,
			assert: func(t *testing.T, body map[string]any) {
				assertString(t, body["reply"], "hello")
				baseResp := mustMap(t, body["base_resp"])
				assertFloat64(t, baseResp["status_code"], 0)
			},
		},
		{
			name: "dify non-stream",
			path: difyChatPath,
			host: difyDomain,
			headers: map[string]string{
				"Authorization": "Bearer test",
			},
			body: `{"query":"hello","response_mode":"blocking","user":"tester"}`,
			assert: func(t *testing.T, body map[string]any) {
				assertString(t, body["answer"], "hello")
				metadata := mustMap(t, body["metadata"])
				usage := mustMap(t, metadata["usage"])
				assertFloat64(t, usage["total_tokens"], float64(completionMockUsage.TotalTokens))
			},
		},
		{
			name: "qwen non-stream",
			path: qwenChatCompletionPath,
			host: qwenDomain,
			headers: map[string]string{
				"Authorization": "Bearer test",
			},
			body: `{
				"model":"qwen-max",
				"input":{"messages":[{"role":"user","content":"hello"}]},
				"parameters":{"result_format":"message"}
			}`,
			assert: func(t *testing.T, body map[string]any) {
				assertString(t, body["request_id"], completionMockId)
				output := mustMap(t, body["output"])
				choices := mustSlice(t, output["choices"])
				choice := mustMap(t, choices[0])
				message := mustMap(t, choice["message"])
				assertString(t, message["content"], "hello")
				assertString(t, choice["finish_reason"], stopReason)
			},
		},
		{
			name: "gemini non-stream",
			path: "/v1beta/models/gemini-2.0-flash:generateContent",
			host: geminiDomain,
			headers: map[string]string{
				"x-goog-api-key": "test",
			},
			body: `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`,
			assert: func(t *testing.T, body map[string]any) {
				candidates := mustSlice(t, body["candidates"])
				candidate := mustMap(t, candidates[0])
				content := mustMap(t, candidate["content"])
				parts := mustSlice(t, content["parts"])
				part := mustMap(t, parts[0])
				assertContains(t, part["text"], "Gemini provider")
				assertString(t, candidate["finishReason"], "STOP")
			},
		},
		{
			name: "vertex non-stream",
			path: "/v1/publishers/google/models/gemini-2.0-flash:generateContent",
			host: vertexDomain,
			body: `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`,
			assert: func(t *testing.T, body map[string]any) {
				assertString(t, body["responseId"], completionMockId)
				candidates := mustSlice(t, body["candidates"])
				candidate := mustMap(t, candidates[0])
				content := mustMap(t, candidate["content"])
				parts := mustSlice(t, content["parts"])
				part := mustMap(t, parts[0])
				assertContains(t, part["text"], "Vertex provider")
				assertString(t, candidate["finishReason"], "STOP")
			},
		},
		{
			name: "bedrock non-stream",
			path: "/model/test-model/converse",
			host: "bedrock-runtime.us-west-2.amazonaws.com",
			body: `{"messages":[{"role":"user","content":[{"text":"hi"}]}]}`,
			assert: func(t *testing.T, body map[string]any) {
				assertString(t, body["stopReason"], bedrockStopReasonEndTurn)
				output := mustMap(t, body["output"])
				message := mustMap(t, output["message"])
				content := mustSlice(t, message["content"])
				first := mustMap(t, content[0])
				assertContains(t, first["text"], "Bedrock provider")
			},
		},
		{
			name: "moonshot non-stream",
			path: moonshotChatCompletionPath,
			host: moonshotDomain,
			headers: map[string]string{
				"Authorization": "Bearer test",
			},
			body: `{"model":"moonshot-v1-8k","messages":[{"role":"user","content":"hello"}]}`,
			assert: func(t *testing.T, body map[string]any) {
				assertString(t, body["id"], completionMockId)
				choices := mustSlice(t, body["choices"])
				choice := mustMap(t, choices[0])
				message := mustMap(t, choice["message"])
				assertString(t, message["content"], "hello")
			},
		},
		{
			name: "claude non-stream",
			path: claudeMessagesPath,
			host: claudeDomain,
			headers: map[string]string{
				"anthropic-version": "2023-06-01",
				"x-api-key":         "test",
			},
			body: `{"model":"claude-3-5-sonnet-20241022","messages":[{"role":"user","content":"hello"}]}`,
			assert: func(t *testing.T, body map[string]any) {
				assertString(t, body["id"], claudeMockId)
				assertString(t, body["type"], "message")
				content := mustSlice(t, body["content"])
				block := mustMap(t, content[0])
				assertString(t, block["text"], "hello")
			},
		},
		{
			name: "cohere non-stream",
			path: cohereChatPath,
			host: cohereDomain,
			headers: map[string]string{
				"Authorization": "Bearer test",
			},
			body: `{"message":"hello","stream":false}`,
			assert: func(t *testing.T, body map[string]any) {
				assertString(t, body["response_id"], completionMockId)
				assertString(t, body["text"], "hello")
				meta := mustMap(t, body["meta"])
				apiVersion := mustMap(t, meta["api_version"])
				assertString(t, apiVersion["version"], "1")
			},
		},
		{
			name: "hunyuan non-stream",
			path: hunyuanPath,
			host: hunyuanDomain,
			headers: map[string]string{
				"Authorization": "TC3-HMAC-SHA256 ...",
				"X-TC-Action":   "ChatCompletions",
				"X-TC-Version":  "2023-09-01",
			},
			body: `{"Model":"hunyuan-lite","Messages":[{"Role":"user","Content":"hello"}],"Stream":false}`,
			assert: func(t *testing.T, body map[string]any) {
				response := mustMap(t, body["Response"])
				assertString(t, response["Note"], hunyuanNote)
				choices := mustSlice(t, response["Choices"])
				choice := mustMap(t, choices[0])
				message := mustMap(t, choice["Message"])
				assertString(t, message["Content"], "hello")
			},
		},
		{
			name: "deepl non-stream",
			path: deeplTranslatePath,
			host: deeplHostPro,
			headers: map[string]string{
				"Authorization": "DeepL-Auth-Key test",
			},
			body: `{"text":["hello"],"target_lang":"ZH"}`,
			assert: func(t *testing.T, body map[string]any) {
				translations := mustSlice(t, body["translations"])
				first := mustMap(t, translations[0])
				assertString(t, first["text"], "hello")
				assertString(t, first["detected_source_language"], deeplDetectedSource)
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			resp := performJSONRequest(t, router, tc.path, tc.host, tc.body, tc.headers)
			if resp.Code != http.StatusOK {
				t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
			}
			body := decodeJSONBody(t, resp.Body.Bytes())
			tc.assert(t, body)
		})
	}
}

func TestProviderRegressionQwenStreamResponse(t *testing.T) {
	router := newRegressionRouter()
	resp := performJSONRequest(t, router, qwenChatCompletionPath, qwenDomain, `{
		"model":"qwen-max",
		"input":{"messages":[{"role":"user","content":"ok"}]},
		"parameters":{"result_format":"message","incremental_output":true}
	}`, map[string]string{
		"Authorization":   "Bearer test",
		"Accept":          "text/event-stream",
		"X-DashScope-SSE": "enable",
	})

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}

	frames := parseDataFrames(resp.Body.String())
	if len(frames) < 2 {
		t.Fatalf("expected multiple stream frames, got %d: %q", len(frames), resp.Body.String())
	}

	first := decodeJSONBody(t, []byte(frames[0]))
	firstOutput := mustMap(t, first["output"])
	firstChoices := mustSlice(t, firstOutput["choices"])
	firstChoice := mustMap(t, firstChoices[0])
	firstMessage := mustMap(t, firstChoice["message"])
	assertString(t, firstMessage["content"], "o")

	last := decodeJSONBody(t, []byte(frames[len(frames)-1]))
	lastOutput := mustMap(t, last["output"])
	lastChoices := mustSlice(t, lastOutput["choices"])
	lastChoice := mustMap(t, lastChoices[0])
	assertString(t, lastChoice["finish_reason"], stopReason)
}

func TestProviderRegressionClaudeToolUseStreamResponse(t *testing.T) {
	router := newRegressionRouter()
	resp := performJSONRequest(t, router, claudeMessagesPath, claudeDomain, `{
		"model":"claude-3-5-sonnet-20241022",
		"stream":true,
		"messages":[{"role":"user","content":"hello"}],
		"tools":[{"name":"get_weather"}]
	}`, map[string]string{
		"anthropic-version": "2023-06-01",
		"x-api-key":         "test",
	})

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}

	frames := parseNamedSSEFrames(resp.Body.String())
	if len(frames) < 6 {
		t.Fatalf("expected multiple named SSE frames, got %d: %q", len(frames), resp.Body.String())
	}

	if frames[0].Event != "message_start" {
		t.Fatalf("expected first event message_start, got %q", frames[0].Event)
	}
	start := decodeJSONBody(t, []byte(frames[0].Data))
	message := mustMap(t, start["message"])
	assertString(t, message["id"], claudeMockId)

	if frames[1].Event != "content_block_start" {
		t.Fatalf("expected second event content_block_start, got %q", frames[1].Event)
	}
	startBlock := decodeJSONBody(t, []byte(frames[1].Data))
	contentBlock := mustMap(t, startBlock["content_block"])
	assertString(t, contentBlock["type"], "tool_use")
	assertString(t, contentBlock["name"], "get_weather")

	foundMessageStop := false
	for _, frame := range frames {
		if frame.Event == "message_stop" {
			foundMessageStop = true
			break
		}
	}
	if !foundMessageStop {
		t.Fatalf("expected message_stop event in stream: %q", resp.Body.String())
	}
}

func TestProviderRegressionStreamingResponses(t *testing.T) {
	router := newRegressionRouter()

	testCases := []struct {
		name    string
		path    string
		host    string
		headers map[string]string
		body    string
		assert  func(*testing.T, *closeNotifyRecorder)
	}{
		{
			name: "minimax stream",
			path: minimaxChatCompletionProPath,
			host: minimaxDomain,
			headers: map[string]string{
				"Authorization": "Bearer test",
			},
			body: `{
				"model":"abab6.5s-chat",
				"stream":true,
				"messages":[{"sender_type":"USER","sender_name":"user","text":"h"}],
				"bot_setting":[{"bot_name":"assistant","content":"You are assistant"}],
				"reply_constraints":{"sender_type":"BOT","sender_name":"assistant"}
			}`,
			assert: func(t *testing.T, resp *closeNotifyRecorder) {
				assertHeaderContains(t, resp.Header().Get("Content-Type"), "text/event-stream")
				frames := parseDataFrames(resp.Body.String())
				if len(frames) < 2 {
					t.Fatalf("expected multiple stream frames, got %d: %q", len(frames), resp.Body.String())
				}
				first := decodeJSONBody(t, []byte(frames[0]))
				firstChoices := mustSlice(t, first["choices"])
				firstChoice := mustMap(t, firstChoices[0])
				firstMessages := mustSlice(t, firstChoice["messages"])
				firstMessage := mustMap(t, firstMessages[0])
				assertString(t, firstMessage["text"], "h")

				last := decodeJSONBody(t, []byte(frames[len(frames)-1]))
				assertString(t, last["reply"], "h")
				lastChoices := mustSlice(t, last["choices"])
				lastChoice := mustMap(t, lastChoices[0])
				assertString(t, lastChoice["finish_reason"], stopReason)
			},
		},
		{
			name: "dify stream",
			path: difyChatPath,
			host: difyDomain,
			headers: map[string]string{
				"Authorization": "Bearer test",
			},
			body: `{"query":"h","response_mode":"streaming","user":"tester"}`,
			assert: func(t *testing.T, resp *closeNotifyRecorder) {
				assertHeaderContains(t, resp.Header().Get("Content-Type"), "text/event-stream")
				frames := parseDataFrames(resp.Body.String())
				if len(frames) < 2 {
					t.Fatalf("expected multiple stream frames, got %d: %q", len(frames), resp.Body.String())
				}
				first := decodeJSONBody(t, []byte(frames[0]))
				assertString(t, first["event"], "agent_thought")
				assertString(t, first["answer"], "h")

				last := decodeJSONBody(t, []byte(frames[len(frames)-1]))
				assertString(t, last["event"], "message_end")
				assertString(t, last["answer"], "h")
				metadata := mustMap(t, last["metadata"])
				usage := mustMap(t, metadata["usage"])
				assertFloat64(t, usage["total_tokens"], float64(completionMockUsage.TotalTokens))
			},
		},
		{
			name: "gemini stream",
			path: "/v1beta/models/gemini-2.0-flash:streamGenerateContent",
			host: geminiDomain,
			headers: map[string]string{
				"x-goog-api-key": "test",
			},
			body: `{"contents":[{"role":"user","parts":[{"text":"h"}]}]}`,
			assert: func(t *testing.T, resp *closeNotifyRecorder) {
				assertHeaderContains(t, resp.Header().Get("Content-Type"), "text/event-stream")
				frames := parseDataFrames(resp.Body.String())
				if len(frames) < 2 {
					t.Fatalf("expected multiple stream frames, got %d: %q", len(frames), resp.Body.String())
				}
				first := decodeJSONBody(t, []byte(frames[0]))
				firstCandidates := mustSlice(t, first["candidates"])
				firstCandidate := mustMap(t, firstCandidates[0])
				firstContent := mustMap(t, firstCandidate["content"])
				firstParts := mustSlice(t, firstContent["parts"])
				firstPart := mustMap(t, firstParts[0])
				assertContains(t, firstPart["text"], "This")

				last := decodeJSONBody(t, []byte(frames[len(frames)-1]))
				lastCandidates := mustSlice(t, last["candidates"])
				lastCandidate := mustMap(t, lastCandidates[0])
				assertString(t, lastCandidate["finishReason"], "STOP")
			},
		},
		{
			name: "vertex stream standard path",
			path: "/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-2.0-flash:streamGenerateContent",
			host: vertexDomain,
			body: `{"contents":[{"role":"user","parts":[{"text":"h"}]}]}`,
			assert: func(t *testing.T, resp *closeNotifyRecorder) {
				assertHeaderContains(t, resp.Header().Get("Content-Type"), "text/event-stream")
				frames := parseDataFrames(resp.Body.String())
				if len(frames) < 2 {
					t.Fatalf("expected multiple stream frames, got %d: %q", len(frames), resp.Body.String())
				}
				first := decodeJSONBody(t, []byte(frames[0]))
				assertString(t, first["responseId"], completionMockId)
				last := decodeJSONBody(t, []byte(frames[len(frames)-1]))
				lastCandidates := mustSlice(t, last["candidates"])
				lastCandidate := mustMap(t, lastCandidates[0])
				assertString(t, lastCandidate["finishReason"], "STOP")
			},
		},
		{
			name: "moonshot stream",
			path: moonshotChatCompletionPath,
			host: moonshotDomain,
			headers: map[string]string{
				"Authorization": "Bearer test",
			},
			body: `{"model":"moonshot-v1-8k","stream":true,"messages":[{"role":"user","content":"h"}]}`,
			assert: func(t *testing.T, resp *closeNotifyRecorder) {
				assertHeaderContains(t, resp.Header().Get("Content-Type"), "text/event-stream")
				frames := parseDataFrames(resp.Body.String())
				if len(frames) < 4 {
					t.Fatalf("expected moonshot stream frames plus [DONE], got %d: %q", len(frames), resp.Body.String())
				}
				first := decodeJSONBody(t, []byte(frames[0]))
				firstChoices := mustSlice(t, first["choices"])
				firstChoice := mustMap(t, firstChoices[0])
				firstDelta := mustMap(t, firstChoice["delta"])
				assertString(t, firstDelta["role"], roleAssistant)
				assertString(t, firstDelta["content"], "")

				lastJSON := decodeJSONBody(t, []byte(frames[len(frames)-2]))
				lastChoices := mustSlice(t, lastJSON["choices"])
				lastChoice := mustMap(t, lastChoices[0])
				assertString(t, lastChoice["finish_reason"], stopReason)
				usage := mustMap(t, lastChoice["usage"])
				assertFloat64(t, usage["total_tokens"], float64(completionMockUsage.TotalTokens))
				assertString(t, anyToString(t, frames[len(frames)-1]), "[DONE]")
			},
		},
		{
			name: "hunyuan stream",
			path: hunyuanPath,
			host: hunyuanDomain,
			headers: map[string]string{
				"Authorization": "TC3-HMAC-SHA256 ...",
				"X-TC-Action":   "ChatCompletions",
				"X-TC-Version":  "2023-09-01",
			},
			body: `{"Model":"hunyuan-lite","Messages":[{"Role":"user","Content":"h"}],"Stream":true}`,
			assert: func(t *testing.T, resp *closeNotifyRecorder) {
				assertHeaderContains(t, resp.Header().Get("Content-Type"), "text/event-stream")
				assertString(t, resp.Header().Get("X-TC-RequestId"), completionMockId)
				frames := parseDataFrames(resp.Body.String())
				if len(frames) < 2 {
					t.Fatalf("expected multiple stream frames, got %d: %q", len(frames), resp.Body.String())
				}
				first := decodeJSONBody(t, []byte(frames[0]))
				firstChoices := mustSlice(t, first["Choices"])
				firstChoice := mustMap(t, firstChoices[0])
				firstDelta := mustMap(t, firstChoice["Delta"])
				assertString(t, firstDelta["Content"], "h")

				last := decodeJSONBody(t, []byte(frames[len(frames)-1]))
				lastChoices := mustSlice(t, last["Choices"])
				lastChoice := mustMap(t, lastChoices[0])
				assertString(t, lastChoice["FinishReason"], stopReason)
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			resp := performJSONRequest(t, router, tc.path, tc.host, tc.body, tc.headers)
			if resp.Code != http.StatusOK {
				t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
			}
			tc.assert(t, resp)
		})
	}
}

func TestProviderRegressionClaudeTextStreamResponse(t *testing.T) {
	router := newRegressionRouter()
	resp := performJSONRequest(t, router, claudeMessagesPath, claudeDomain, `{
		"model":"claude-3-5-sonnet-20241022",
		"stream":true,
		"messages":[{"role":"user","content":"h"}]
	}`, map[string]string{
		"anthropic-version": "2023-06-01",
		"x-api-key":         "test",
	})

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}
	assertHeaderContains(t, resp.Header().Get("Content-Type"), "text/event-stream")

	frames := parseNamedSSEFrames(resp.Body.String())
	if len(frames) < 5 {
		t.Fatalf("expected multiple named SSE frames, got %d: %q", len(frames), resp.Body.String())
	}

	if frames[0].Event != "message_start" {
		t.Fatalf("expected first event message_start, got %q", frames[0].Event)
	}
	if frames[1].Event != "content_block_start" {
		t.Fatalf("expected second event content_block_start, got %q", frames[1].Event)
	}

	startBlock := decodeJSONBody(t, []byte(frames[1].Data))
	contentBlock := mustMap(t, startBlock["content_block"])
	assertString(t, contentBlock["type"], "text")

	foundTextDelta := false
	foundMessageDelta := false
	foundMessageStop := false
	for _, frame := range frames {
		switch frame.Event {
		case "content_block_delta":
			payload := decodeJSONBody(t, []byte(frame.Data))
			delta := mustMap(t, payload["delta"])
			if delta["type"] == "text_delta" {
				foundTextDelta = true
			}
		case "message_delta":
			payload := decodeJSONBody(t, []byte(frame.Data))
			delta := mustMap(t, payload["delta"])
			assertString(t, delta["stop_reason"], "end_turn")
			foundMessageDelta = true
		case "message_stop":
			foundMessageStop = true
		}
	}

	if !foundTextDelta {
		t.Fatalf("expected a text_delta frame in stream: %q", resp.Body.String())
	}
	if !foundMessageDelta {
		t.Fatalf("expected a message_delta frame in stream: %q", resp.Body.String())
	}
	if !foundMessageStop {
		t.Fatalf("expected a message_stop frame in stream: %q", resp.Body.String())
	}
}

func TestProviderRegressionCohereStreamResponse(t *testing.T) {
	router := newRegressionRouter()
	resp := performJSONRequest(t, router, cohereChatPath, cohereDomain, `{"message":"h","stream":true}`, map[string]string{
		"Authorization": "Bearer test",
		"Accept":        "text/event-stream",
	})

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", resp.Code, resp.Body.String())
	}
	assertHeaderContains(t, resp.Header().Get("Content-Type"), "text/event-stream")

	frames := parseNamedSSEFrames(resp.Body.String())
	if len(frames) < 3 {
		t.Fatalf("expected cohere start/text/end frames, got %d: %q", len(frames), resp.Body.String())
	}

	assertString(t, frames[0].Event, "stream-start")
	start := decodeJSONBody(t, []byte(frames[0].Data))
	assertString(t, start["event_type"], "stream-start")

	foundText := false
	foundEnd := false
	for _, frame := range frames {
		switch frame.Event {
		case "text-generation":
			payload := decodeJSONBody(t, []byte(frame.Data))
			assertString(t, payload["text"], "h")
			foundText = true
		case "stream-end":
			payload := decodeJSONBody(t, []byte(frame.Data))
			assertString(t, payload["finish_reason"], "COMPLETE")
			response := mustMap(t, payload["response"])
			assertString(t, response["text"], "h")
			foundEnd = true
		}
	}

	if !foundText {
		t.Fatalf("expected a text-generation frame in stream: %q", resp.Body.String())
	}
	if !foundEnd {
		t.Fatalf("expected a stream-end frame in stream: %q", resp.Body.String())
	}
}

func TestProviderRegressionBedrockStreamResponse(t *testing.T) {
	router := newRegressionRouter()
	resp := performJSONRequest(t, router, "/model/test-model/converse-stream", "bedrock-runtime.us-west-2.amazonaws.com", `{
		"messages":[{"role":"user","content":[{"text":"h"}]}]
	}`, nil)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp.Code)
	}
	assertHeaderContains(t, resp.Header().Get("Content-Type"), "application/vnd.amazon.eventstream")

	frames := parseBedrockEventStreamFrames(t, resp.Body.Bytes())
	if len(frames) < 2 {
		t.Fatalf("expected multiple bedrock event stream frames, got %d", len(frames))
	}

	assertString(t, frames[0].EventType, "contentBlockDelta")
	firstDelta := mustMap(t, frames[0].Payload["delta"])
	assertContains(t, firstDelta["text"], "This")

	last := frames[len(frames)-1]
	assertString(t, last.EventType, "messageStop")
	assertString(t, last.Payload["stopReason"], bedrockStopReasonEndTurn)
}

func newRegressionRouter() *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	SetupRoutes(router, "")
	return router
}

type closeNotifyRecorder struct {
	*httptest.ResponseRecorder
	closeCh chan bool
}

func newCloseNotifyRecorder() *closeNotifyRecorder {
	return &closeNotifyRecorder{
		ResponseRecorder: httptest.NewRecorder(),
		closeCh:          make(chan bool, 1),
	}
}

func (r *closeNotifyRecorder) CloseNotify() <-chan bool {
	return r.closeCh
}

func performJSONRequest(t *testing.T, router http.Handler, path, host, body string, headers map[string]string) *closeNotifyRecorder {
	t.Helper()

	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Host = host
	req.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		req.Header.Set(key, value)
	}

	resp := newCloseNotifyRecorder()
	router.ServeHTTP(resp, req)
	return resp
}

func decodeJSONBody(t *testing.T, body []byte) map[string]any {
	t.Helper()

	var decoded map[string]any
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("failed to decode JSON body %q: %v", string(body), err)
	}
	return decoded
}

func mustMap(t *testing.T, value any) map[string]any {
	t.Helper()

	decoded, ok := value.(map[string]any)
	if !ok {
		t.Fatalf("expected map[string]any, got %T", value)
	}
	return decoded
}

func mustSlice(t *testing.T, value any) []any {
	t.Helper()

	decoded, ok := value.([]any)
	if !ok {
		t.Fatalf("expected []any, got %T", value)
	}
	return decoded
}

func assertString(t *testing.T, value any, want string) {
	t.Helper()

	got, ok := value.(string)
	if !ok {
		t.Fatalf("expected string %q, got %T (%v)", want, value, value)
	}
	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}

func assertContains(t *testing.T, value any, wantSubstring string) {
	t.Helper()

	got, ok := value.(string)
	if !ok {
		t.Fatalf("expected string containing %q, got %T (%v)", wantSubstring, value, value)
	}
	if !strings.Contains(got, wantSubstring) {
		t.Fatalf("expected %q to contain %q", got, wantSubstring)
	}
}

func assertFloat64(t *testing.T, value any, want float64) {
	t.Helper()

	got, ok := value.(float64)
	if !ok {
		t.Fatalf("expected float64 %v, got %T (%v)", want, value, value)
	}
	if got != want {
		t.Fatalf("expected %v, got %v", want, got)
	}
}

func assertHeaderContains(t *testing.T, headerValue, wantSubstring string) {
	t.Helper()
	if !strings.Contains(headerValue, wantSubstring) {
		t.Fatalf("expected header %q to contain %q", headerValue, wantSubstring)
	}
}

func anyToString(t *testing.T, value any) string {
	t.Helper()
	got, ok := value.(string)
	if !ok {
		t.Fatalf("expected string, got %T (%v)", value, value)
	}
	return got
}

func parseDataFrames(body string) []string {
	parts := strings.Split(body, "\n\n")
	frames := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if !strings.HasPrefix(part, "data: ") {
			continue
		}
		frames = append(frames, strings.TrimPrefix(part, "data: "))
	}
	return frames
}

type namedSSEFrame struct {
	Event string
	Data  string
}

func parseNamedSSEFrames(body string) []namedSSEFrame {
	parts := strings.Split(body, "\n\n")
	frames := make([]namedSSEFrame, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		lines := strings.Split(part, "\n")
		frame := namedSSEFrame{}
		for _, line := range lines {
			switch {
			case strings.HasPrefix(line, "event: "):
				frame.Event = strings.TrimPrefix(line, "event: ")
			case strings.HasPrefix(line, "data: "):
				frame.Data = strings.TrimPrefix(line, "data: ")
			}
		}
		if frame.Event != "" || frame.Data != "" {
			frames = append(frames, frame)
		}
	}
	return frames
}

type bedrockEventStreamFrame struct {
	EventType string
	Payload   map[string]any
}

func parseBedrockEventStreamFrames(t *testing.T, body []byte) []bedrockEventStreamFrame {
	t.Helper()

	offset := 0
	frames := make([]bedrockEventStreamFrame, 0)
	for offset < len(body) {
		if len(body)-offset < 16 {
			t.Fatalf("bedrock event stream truncated at offset %d", offset)
		}

		totalLen := int(binary.BigEndian.Uint32(body[offset : offset+4]))
		headersLen := int(binary.BigEndian.Uint32(body[offset+4 : offset+8]))
		if totalLen <= 16 || offset+totalLen > len(body) {
			t.Fatalf("invalid bedrock event stream frame length %d at offset %d", totalLen, offset)
		}

		headersStart := offset + 12
		headersEnd := headersStart + headersLen
		payloadStart := headersEnd
		payloadEnd := offset + totalLen - 4
		if headersEnd > payloadEnd {
			t.Fatalf("invalid bedrock event stream header length %d at offset %d", headersLen, offset)
		}

		headers := parseBedrockEventStreamHeaders(t, body[headersStart:headersEnd])
		payload := decodeJSONBody(t, body[payloadStart:payloadEnd])
		frames = append(frames, bedrockEventStreamFrame{
			EventType: headers[":event-type"],
			Payload:   payload,
		})
		offset += totalLen
	}

	return frames
}

func parseBedrockEventStreamHeaders(t *testing.T, data []byte) map[string]string {
	t.Helper()

	headers := make(map[string]string)
	for idx := 0; idx < len(data); {
		nameLen := int(data[idx])
		idx++
		if idx+nameLen > len(data) {
			t.Fatalf("invalid bedrock header name length %d", nameLen)
		}
		name := string(data[idx : idx+nameLen])
		idx += nameLen
		if idx >= len(data) {
			t.Fatalf("missing bedrock header type for %q", name)
		}
		valueType := data[idx]
		idx++
		if valueType != 7 {
			t.Fatalf("unexpected bedrock header type %d for %q", valueType, name)
		}
		if idx+2 > len(data) {
			t.Fatalf("missing bedrock header value length for %q", name)
		}
		valueLen := int(binary.BigEndian.Uint16(data[idx : idx+2]))
		idx += 2
		if idx+valueLen > len(data) {
			t.Fatalf("invalid bedrock header value length %d for %q", valueLen, name)
		}
		headers[name] = string(data[idx : idx+valueLen])
		idx += valueLen
	}
	return headers
}
