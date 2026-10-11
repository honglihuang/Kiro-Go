package proxy

import "strings"

// maxResponsesHistoryDepth caps how far back we walk the previous_response_id
// chain when expanding history. The cap prevents pathological loops in
// corrupted/cyclic stores from running forever; legitimate chains rarely go
// this deep within the 30-day TTL.
const maxResponsesHistoryDepth = 64

// replayHistoryWindowShare bounds replayed ancestor history as a fraction of the
// model's context window, reserving the remainder for the client's current input
// and its tool schemas.
//
// A depth cap alone is not a size cap: 64 ancestors of full input and output can
// exceed any window on their own. This matters when a client compacts while its
// new request still points at the pre-compaction response. Replaying every turn
// the client discarded would refill the context and trigger compaction again.
const replayHistoryWindowShare = 0.5

// replayTruncationNote marks where replayed ancestor turns were elided.
const replayTruncationNote = "[Earlier turns of this conversation were omitted to fit the model's input limit.]"

func maxReplayHistoryTokens(model string) int {
	return int(float64(getContextWindowSize(model)) * replayHistoryWindowShare)
}

// expandPreviousResponseHistory rebuilds the conversation history that led up
// to prev. It walks the previous_response_id chain backwards and emits messages
// oldest to newest. Replayed history is bounded by a share of the model window;
// the current request input is authoritative and is never trimmed here.
func expandPreviousResponseHistory(prev *ResponsesObject, model string) []OpenAIMessage {
	if prev == nil {
		return nil
	}

	chain := collectAncestorChain(prev)
	groups := make([][]OpenAIMessage, 0, len(chain))
	for _, node := range chain {
		group := make([]OpenAIMessage, 0, 4)
		if node.Instructions != "" {
			group = append(group, OpenAIMessage{
				Role:    "system",
				Content: node.Instructions,
			})
		}
		if prior, err := parseResponsesInput(node.StoredInput); err == nil {
			group = append(group, prior...)
		}
		group = append(group, outputToMessages(node.Output)...)
		groups = append(groups, group)
	}

	return flattenReplayGroups(groups, maxReplayHistoryTokens(model))
}

// flattenReplayGroups drops complete oldest turns until the newest suffix fits
// within budget. The newest turn is retained even if it alone exceeds budget;
// downstream payload truncation remains the final byte-level safety net.
func flattenReplayGroups(groups [][]OpenAIMessage, budget int) []OpenAIMessage {
	keepFrom := 0
	if budget > 0 {
		total := 0
		keepFrom = len(groups)
		for i := len(groups) - 1; i >= 0; i-- {
			total += estimateReplayGroupTokens(groups[i])
			if total > budget && i < len(groups)-1 {
				break
			}
			keepFrom = i
		}
	}

	messages := make([]OpenAIMessage, 0)
	if keepFrom > 0 {
		messages = append(messages, OpenAIMessage{
			Role:    "user",
			Content: replayTruncationNote,
		})
	}
	for _, group := range groups[keepFrom:] {
		messages = append(messages, group...)
	}
	if keepFrom > 0 {
		messages = dropOrphanedLeadingToolMessages(messages)
	}
	return messages
}

// dropOrphanedLeadingToolMessages removes tool results whose assistant call was
// in an older turn removed by replay trimming. Kiro rejects unpaired results.
func dropOrphanedLeadingToolMessages(messages []OpenAIMessage) []OpenAIMessage {
	seenCalls := make(map[string]bool)
	out := make([]OpenAIMessage, 0, len(messages))
	sawAssistantCall := false

	for _, message := range messages {
		if message.Role == "assistant" && len(message.ToolCalls) > 0 {
			sawAssistantCall = true
			for _, call := range message.ToolCalls {
				if call.ID != "" {
					seenCalls[call.ID] = true
				}
			}
		}
		if message.Role == "tool" && !sawAssistantCall && !seenCalls[message.ToolCallID] {
			continue
		}
		out = append(out, message)
	}
	return out
}

func estimateReplayGroupTokens(group []OpenAIMessage) int {
	total := 0
	for _, message := range group {
		total += estimateOpenAIContentTokens(message.Content)
		total += estimateApproxTokens(message.ToolCallID)
		for _, call := range message.ToolCalls {
			total += estimateApproxTokens(call.Function.Name)
			total += estimateApproxTokens(call.Function.Arguments)
		}
	}
	return total
}

// collectAncestorChain walks previous_response_id backwards, returning the
// chain in oldest-first order: [root, ..., parent, prev]. The walker is
// bounded by maxResponsesHistoryDepth and a visited-set to short-circuit
// any cycle in the stored data.
func collectAncestorChain(prev *ResponsesObject) []*ResponsesObject {
	stack := []*ResponsesObject{prev}
	visited := map[string]bool{prev.ID: true}

	cursor := prev
	for depth := 0; depth < maxResponsesHistoryDepth; depth++ {
		if cursor.PreviousResponseID == "" {
			break
		}
		if visited[cursor.PreviousResponseID] {
			break
		}
		ancestor, err := loadResponse(cursor.PreviousResponseID)
		if err != nil || ancestor == nil {
			break
		}
		visited[ancestor.ID] = true
		stack = append(stack, ancestor)
		cursor = ancestor
	}

	for i, j := 0, len(stack)-1; i < j; i, j = i+1, j-1 {
		stack[i], stack[j] = stack[j], stack[i]
	}
	return stack
}

func outputToMessages(items []ResponseOutputItem) []OpenAIMessage {
	if len(items) == 0 {
		return nil
	}
	out := make([]OpenAIMessage, 0, len(items))
	for _, item := range items {
		switch item.Type {
		case "message":
			text := joinTextParts(item.Content)
			role := item.Role
			if role == "" {
				role = "assistant"
			}
			if text == "" && role == "assistant" {
				continue
			}
			out = append(out, OpenAIMessage{Role: role, Content: text})
		case "function_call":
			tc := ToolCall{ID: item.CallID, Type: "function"}
			if tc.ID == "" {
				tc.ID = item.ID
			}
			tc.Function.Name = item.Name
			tc.Function.Arguments = item.Arguments
			out = append(out, OpenAIMessage{
				Role:      "assistant",
				Content:   "",
				ToolCalls: []ToolCall{tc},
			})
		}
	}
	return out
}

func joinTextParts(parts []ResponseContentPart) string {
	if len(parts) == 0 {
		return ""
	}
	var out strings.Builder
	for _, part := range parts {
		if part.Type == "output_text" || part.Type == "text" || part.Type == "input_text" {
			out.WriteString(part.Text)
		}
	}
	return out.String()
}
