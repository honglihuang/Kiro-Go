package proxy

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const kiroContentLengthReason = "CONTENT_LENGTH_EXCEEDS_THRESHOLD"

// kiroHTTPError preserves structured upstream status and reason fields so
// deterministic request errors can stop both endpoint and account retries.
type kiroHTTPError struct {
	StatusCode int
	Endpoint   string
	Message    string
	Reason     string
	Body       string
}

func newKiroHTTPError(statusCode int, endpoint string, body []byte) *kiroHTTPError {
	parsed := struct {
		Message string `json:"message"`
		Reason  string `json:"reason"`
	}{}
	_ = json.Unmarshal(body, &parsed)
	return &kiroHTTPError{
		StatusCode: statusCode,
		Endpoint:   endpoint,
		Message:    parsed.Message,
		Reason:     parsed.Reason,
		Body:       string(body),
	}
}

func (e *kiroHTTPError) Error() string {
	return fmt.Sprintf("HTTP %d from %s: %s", e.StatusCode, e.Endpoint, e.Body)
}

func isKiroInputTooLongError(err error) bool {
	if err == nil {
		return false
	}
	var upstreamErr *kiroHTTPError
	if errors.As(err, &upstreamErr) {
		return upstreamErr.StatusCode == 400 && strings.EqualFold(strings.TrimSpace(upstreamErr.Reason), kiroContentLengthReason)
	}
	return strings.Contains(strings.ToUpper(err.Error()), kiroContentLengthReason)
}
