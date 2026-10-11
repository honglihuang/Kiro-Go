package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"kiro-go/config"
	accountpool "kiro-go/pool"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

const replayElisionMarker = "[Earlier turns of this conversation were omitted to fit the model's input limit.]"

func replayFixtureText(tag string, approxTokens int) string {
	unit := tag + " lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod. "
	return strings.Repeat(unit, approxTokens/8+1)
}

func storeOversizedReplayChain(t *testing.T, turns, tokensPerSide int) *ResponsesObject {
	t.Helper()
	if err := config.Init(filepath.Join(t.TempDir(), "config.json")); err != nil {
		t.Fatalf("config.Init: %v", err)
	}

	var newest *ResponsesObject
	previousID := ""
	for i := 0; i < turns; i++ {
		input, err := json.Marshal(fmt.Sprintf("turn %d user: %s", i, replayFixtureText(fmt.Sprintf("t%d", i), tokensPerSide)))
		if err != nil {
			t.Fatalf("marshal turn %d: %v", i, err)
		}
		response := &ResponsesObject{
			ID:                 fmt.Sprintf("resp_oversized_%d", i),
			Object:             "response",
			Status:             "completed",
			Model:              "gpt-5.6-terra",
			StoredInput:        input,
			PreviousResponseID: previousID,
			Output: []ResponseOutputItem{{
				Type: "message",
				Role: "assistant",
				Content: []ResponseContentPart{{
					Type: "output_text",
					Text: fmt.Sprintf("turn %d assistant: %s", i, replayFixtureText("answer", tokensPerSide)),
				}},
			}},
		}
		if err := saveResponse(response); err != nil {
			t.Fatalf("save turn %d: %v", i, err)
		}
		previousID = response.ID
		newest = response
	}
	return newest
}

func replayTokenEstimateForTest(messages []OpenAIMessage) int {
	total := 0
	for _, message := range messages {
		total += estimateOpenAIContentTokens(message.Content)
		total += estimateApproxTokens(message.ToolCallID)
		for _, call := range message.ToolCalls {
			total += estimateApproxTokens(call.Function.Name)
			total += estimateApproxTokens(call.Function.Arguments)
		}
	}
	return total
}

// A compacting Codex client keeps previous_response_id while replacing its
// current input with a short summary. Replaying the full pre-compaction chain
// defeats compaction and eventually triggers CONTENT_LENGTH_EXCEEDS_THRESHOLD.
func TestResponsesReplayHistoryLeavesRoomForCompactedInput(t *testing.T) {
	newest := storeOversizedReplayChain(t, 32, 5_000)
	replayed := expandPreviousResponseHistory(newest, "gpt-5.6-terra")
	budget := getContextWindowSize("gpt-5.6-terra") / 2
	got := replayTokenEstimateForTest(replayed)
	if got > budget {
		t.Fatalf("replayed history uses %d tokens, over %d-token budget", got, budget)
	}

	var joined strings.Builder
	for _, message := range replayed {
		if text, ok := message.Content.(string); ok {
			joined.WriteString(text)
			joined.WriteByte('\n')
		}
	}
	transcript := joined.String()
	if !strings.Contains(transcript, "turn 31 user") {
		t.Fatal("newest turn was dropped")
	}
	if strings.Contains(transcript, "turn 0 user") {
		t.Fatal("oldest turn must be dropped when replay exceeds its budget")
	}
	if !strings.Contains(transcript, replayElisionMarker) {
		t.Fatal("history elision must be disclosed to the model")
	}
}

// A deterministic request-size rejection cannot succeed on another endpoint.
// Retrying it only multiplies latency and upstream load.
func TestCallKiroAPIDoesNotFallbackOnContentLengthRejection(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"message":"Input is too long.","reason":"CONTENT_LENGTH_EXCEEDS_THRESHOLD"}`))
	}))
	defer server.Close()

	oldResolver := resolveKiroEndpoints
	resolveKiroEndpoints = func(*config.Account) []kiroEndpoint {
		return []kiroEndpoint{
			{URL: server.URL, Origin: "AI_EDITOR", Name: "Kiro IDE"},
			{URL: server.URL, Origin: "AI_EDITOR", Name: "CodeWhisperer"},
			{URL: server.URL, Origin: "AI_EDITOR", Name: "AmazonQ"},
		}
	}
	t.Cleanup(func() { resolveKiroEndpoints = oldResolver })

	oldClient := kiroHttpStore.Load()
	kiroHttpStore.Store(&http.Client{Timeout: time.Second, Transport: &http.Transport{}})
	t.Cleanup(func() { kiroHttpStore.Store(oldClient) })

	account := &config.Account{
		ID:          "content-length-account",
		AccessToken: "token",
		ProfileArn:  "arn:aws:codewhisperer:profile/test",
	}
	err := CallKiroAPIContext(context.Background(), account, integrityTestPayload(), &KiroStreamCallback{})
	if err == nil || !strings.Contains(err.Error(), "CONTENT_LENGTH_EXCEEDS_THRESHOLD") {
		t.Fatalf("expected content-length error, got %v", err)
	}
	if hits.Load() != 1 {
		t.Fatalf("content-length rejection must not fallback, hits=%d", hits.Load())
	}
}

// The outer account loop must also stop after the first deterministic 400.
func TestResponsesDoesNotRotateAccountsOnContentLengthRejection(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"message":"Input is too long.","reason":"CONTENT_LENGTH_EXCEEDS_THRESHOLD"}`))
	}))
	defer server.Close()

	if err := config.Init(filepath.Join(t.TempDir(), "config.json")); err != nil {
		t.Fatalf("config.Init: %v", err)
	}
	for i := 0; i < maxAccountRetryAttempts; i++ {
		if err := config.AddAccount(config.Account{
			ID:          fmt.Sprintf("oversize-account-%d", i),
			Enabled:     true,
			AccessToken: "token",
			ProfileArn:  "arn:aws:codewhisperer:profile/test",
		}); err != nil {
			t.Fatalf("add account %d: %v", i, err)
		}
	}
	if err := config.UpdatePreferredEndpoint("kiro"); err != nil {
		t.Fatalf("set preferred endpoint: %v", err)
	}
	if err := config.UpdateEndpointFallback(false); err != nil {
		t.Fatalf("disable fallback: %v", err)
	}
	t.Cleanup(swapKiroEndpointsForTest(t, server))

	pool := accountpool.GetPool()
	pool.Reload()
	handler := &Handler{pool: pool, promptCache: newPromptCacheTracker(defaultPromptCacheTTL)}
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model":"gpt-5.6-terra",
		"input":"compacted summary",
		"store":false
	}`))
	rec := httptest.NewRecorder()
	handler.handleOpenAIResponses(rec, req)

	if hits.Load() != 1 {
		t.Fatalf("content-length rejection must not rotate accounts, hits=%d", hits.Load())
	}
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d, want 400; body=%s", rec.Code, rec.Body.String())
	}

	streamReq := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model":"gpt-5.6-terra",
		"input":"compacted summary",
		"stream":true,
		"store":false
	}`))
	streamRec := httptest.NewRecorder()
	handler.handleOpenAIResponses(streamRec, streamReq)

	if hits.Load() != 2 {
		t.Fatalf("streaming rejection must use one additional attempt, total hits=%d", hits.Load())
	}
	streamBody := streamRec.Body.String()
	if !strings.Contains(streamBody, "response.failed") || !strings.Contains(streamBody, `"type":"invalid_request_error"`) {
		t.Fatalf("stream must report a non-retryable invalid request, body=%s", streamBody)
	}
	if strings.Count(streamBody, "event: response.in_progress") != 1 {
		t.Fatalf("stream must not rotate accounts, body=%s", streamBody)
	}
}
