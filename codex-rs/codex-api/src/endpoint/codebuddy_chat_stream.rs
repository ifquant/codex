use super::codebuddy_chat::Tool;
use super::codebuddy_chat::invalid;
use crate::common::IncompleteKind;
use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::telemetry::SseTelemetry;
use codex_client::StreamResponse;
use codex_protocol::ResponseUsageMetadata;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio::time::timeout;

#[derive(Default)]
struct Call {
    id: String,
    names: BTreeSet<String>,
    arguments: String,
}

#[derive(Default)]
struct Completion {
    id: String,
    text: String,
    reasoning: String,
    calls: BTreeMap<u64, Call>,
    finish: Option<String>,
    failure: Option<String>,
    message: Option<Value>,
    has_delta_content: bool,
    usage: Option<TokenUsage>,
    raw_usage: Option<Value>,
    bytes: usize,
}

fn item(value: Value) -> Result<ResponseItem, ApiError> {
    serde_json::from_value(value).map_err(|e| invalid(format!("invalid completion item: {e}")))
}

impl Completion {
    fn update(
        &mut self,
        chunk: Value,
        tools: &BTreeMap<String, Tool>,
    ) -> Result<Vec<ResponseEvent>, ApiError> {
        if let Some(usage) = chunk.get("usage").filter(|usage| usage.is_object()) {
            self.raw_usage = Some(usage.clone());
            for counter in [
                &usage["prompt_tokens"],
                &usage["completion_tokens"],
                &usage["total_tokens"],
                &usage["prompt_tokens_details"]["cached_tokens"],
                &usage["prompt_cache_hit_tokens"],
                &usage["cache_write_input_tokens"],
                &usage["completion_tokens_details"]["reasoning_tokens"],
                &usage["completion_thinking_tokens"],
            ] {
                if !counter.is_null() && counter.as_i64().is_none_or(|value| value < 0) {
                    return Err(invalid("usage counters must be non-negative integers"));
                }
            }
            // The host counters require all three totals; absent usage is not measured zero.
            self.usage = match (
                usage["prompt_tokens"].as_i64(),
                usage["completion_tokens"].as_i64(),
                usage["total_tokens"].as_i64(),
            ) {
                (Some(input), Some(output), Some(total))
                    if input >= 0 && output >= 0 && total >= 0 =>
                {
                    Some(TokenUsage {
                        input_tokens: input,
                        output_tokens: output,
                        total_tokens: total,
                        cached_input_tokens: usage["prompt_tokens_details"]["cached_tokens"]
                            .as_i64()
                            .or_else(|| usage["prompt_cache_hit_tokens"].as_i64())
                            .unwrap_or(0),
                        cache_write_input_tokens: usage["cache_write_input_tokens"]
                            .as_i64()
                            .unwrap_or(0),
                        reasoning_output_tokens:
                            usage["completion_tokens_details"]["reasoning_tokens"]
                                .as_i64()
                                .or_else(|| usage["completion_thinking_tokens"].as_i64())
                                .unwrap_or(0),
                        ..Default::default()
                    })
                }
                _ => None,
            };
        }

        if chunk["error"]["code"] == "context_length_exceeded" {
            return Err(ApiError::ContextWindowExceeded);
        }
        if !chunk["error"].is_null() {
            self.id = chunk["id"].as_str().unwrap_or(&self.id).to_owned();
            self.failure = Some(format!("CodeBuddy stream error: {chunk}"));
            self.finish = Some("error".into());
            return std::mem::take(self).finish(tools);
        }
        let mut events = Vec::new();
        if !self.id.is_empty()
            && !chunk["id"].is_null()
            && chunk["id"].as_str() != Some(self.id.as_str())
        {
            return Err(invalid("completion ID changed mid-stream"));
        }
        if self.id.is_empty() {
            self.id = chunk["id"]
                .as_str()
                .filter(|id| !id.is_empty())
                .ok_or_else(|| invalid("stream lacks completion ID"))?
                .to_owned();
            events.push(ResponseEvent::Created {
                response_id: Some(self.id.clone()),
            });
            if let Some(model) = chunk["model"].as_str() {
                events.push(ResponseEvent::ServerModel(model.to_owned()));
            }
        }
        for choice in chunk["choices"]
            .as_array()
            .ok_or_else(|| invalid("stream lacks choices"))?
        {
            if choice["index"].as_u64() != Some(0) {
                return Err(invalid("multiple completion choices are unsupported"));
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                self.finish = Some(reason.to_owned());
            }
            let mut delta = choice["delta"].clone();
            if let Some(message) = choice.get("message").filter(|value| !value.is_null()) {
                if !message.is_object() || self.has_delta_content {
                    return Err(invalid(
                        "conflicting complete message and incremental content",
                    ));
                }
                if delta.as_object().is_some_and(|fields| !fields.is_empty()) {
                    return Err(invalid("choice contains both message and delta"));
                }
                if let Some(previous) = &self.message {
                    if previous != message {
                        return Err(invalid("complete message changed mid-stream"));
                    }
                    continue;
                }
                self.message = Some(message.clone());
                delta = message.clone();
                // A complete message lists the entire batch; positions identify its calls.
                if let Some(calls) = delta["tool_calls"].as_array_mut() {
                    for (index, call) in calls.iter_mut().enumerate() {
                        if !call.is_object() {
                            return Err(invalid("invalid complete tool call"));
                        }
                        call["index"] = json!(index);
                    }
                }
            } else {
                let has_content = ["content", "reasoning_content", "refusal", "tool_calls"]
                    .iter()
                    .any(|key| {
                        let value = &delta[key];
                        value.as_str().is_some_and(|s| !s.is_empty())
                            || value.as_array().is_some_and(|a| !a.is_empty())
                    });
                if has_content && self.message.is_some() {
                    return Err(invalid("incremental content follows a complete message"));
                }
                self.has_delta_content |= has_content;
            }
            for field in ["content", "reasoning_content", "refusal"] {
                if !delta[field].is_null() && !delta[field].is_string() {
                    return Err(invalid("message text fields must be strings"));
                }
            }
            if !delta["tool_calls"].is_null() && !delta["tool_calls"].is_array() {
                return Err(invalid("tool_calls must be an array"));
            }
            if delta["refusal"].as_str().is_some_and(|s| !s.is_empty()) {
                return Err(invalid("upstream refused this request"));
            }
            if let Some(part) = delta["reasoning_content"].as_str() {
                self.reasoning.push_str(part);
            }
            if let Some(part) = delta["content"].as_str().filter(|part| !part.is_empty()) {
                if self.text.is_empty() {
                    events.push(ResponseEvent::OutputItemAdded(item(json!({"type":"message","id":format!("{}-message",self.id),"role":"assistant","content":[]}))?));
                }
                self.text.push_str(part);
                events.push(ResponseEvent::OutputTextDelta(part.to_owned()));
            }
            for delta in delta["tool_calls"].as_array().into_iter().flatten() {
                for field in ["name", "arguments"] {
                    let value = &delta["function"][field];
                    if !value.is_null() && !value.is_string() {
                        return Err(invalid("tool name and argument fragments must be strings"));
                    }
                }
                let id = delta.get("id").filter(|value| !value.is_null());
                let id = id
                    .map(|value| {
                        value
                            .as_str()
                            .filter(|id| !id.is_empty())
                            .ok_or_else(|| invalid("invalid tool call ID"))
                    })
                    .transpose()?;
                let index = if let Some(value) = delta.get("index") {
                    value
                        .as_u64()
                        .filter(|i| *i < 256)
                        .ok_or_else(|| invalid("invalid tool call index"))?
                } else {
                    self.calls
                        .iter()
                        .find(|(_, call)| id == Some(call.id.as_str()))
                        .map(|(index, _)| *index)
                        .ok_or_else(|| invalid("tool fragment lacks a known ID or index"))?
                };
                if let Some(id) = id
                    && self
                        .calls
                        .iter()
                        .any(|(other, call)| *other != index && call.id == id)
                {
                    return Err(invalid("tool call ID belongs to a different index"));
                }
                let call = self.calls.entry(index).or_default();
                if let Some(id) = id {
                    if !call.id.is_empty() && call.id != id {
                        return Err(invalid("tool call ID changed mid-stream"));
                    }
                    call.id = id.to_owned();
                }
                if let Some(part) = delta["function"]["name"]
                    .as_str()
                    .filter(|part| !part.is_empty())
                {
                    if call.names.is_empty() {
                        call.names.insert(String::new());
                    }
                    let mut candidates = BTreeSet::new();
                    for name in std::mem::take(&mut call.names) {
                        let appended = format!("{name}{part}");
                        if tools
                            .range(appended.clone()..)
                            .next()
                            .is_some_and(|(name, _)| name.starts_with(&appended))
                        {
                            candidates.insert(appended);
                        }
                        // A repeated complete advertised name may be metadata, not a fragment.
                        // Retain both interpretations until the advertised table resolves them.
                        if name == part && tools.contains_key(&name) {
                            candidates.insert(name);
                        }
                    }
                    if candidates.is_empty() {
                        return Err(invalid("unadvertised tool name fragments"));
                    }
                    call.names = candidates;
                }
                if let Some(arguments) = delta["function"]["arguments"].as_str() {
                    call.arguments.push_str(arguments);
                }
            }
        }
        Ok(events)
    }

    fn finish(self, tools: &BTreeMap<String, Tool>) -> Result<Vec<ResponseEvent>, ApiError> {
        match self.finish.as_deref() {
            Some("stop") if self.calls.is_empty() => {}
            Some("length") => {}
            Some("error") if self.failure.is_some() => {}
            Some("tool_calls") if !self.calls.is_empty() => {}
            _ => {
                return Err(invalid(
                    "incomplete or unsupported finish reason; no tool calls were executed",
                ));
            }
        }
        let mut events = Vec::new();
        if !self.reasoning.is_empty() {
            let reasoning = item(
                json!({"type":"reasoning","id":format!("{}-reasoning",self.id),"summary":[],"content":[{"type":"reasoning_text","text":self.reasoning}]}),
            )?;
            events.push(ResponseEvent::OutputItemAdded(reasoning.clone()));
            events.push(ResponseEvent::OutputItemDone(reasoning));
        }
        if self.finish.as_deref() == Some("length") || self.failure.is_some() {
            if !self.text.is_empty() {
                events.push(ResponseEvent::OutputItemDone(item(json!({"type":"message","id":format!("{}-message",self.id),"role":"assistant","content":[{"type":"output_text","text":self.text}]}))?));
            }
            let kind = if self.failure.is_some() {
                IncompleteKind::ProviderError
            } else {
                IncompleteKind::OutputLimit
            };
            let mut reason = self.failure.unwrap_or_else(|| "CodeBuddy generation reached its output limit (finish_reason=length); no tool calls were executed".into());
            // Incomplete responses have no completed-response metadata event. Keep the source
            // counters in the durable error too, including missing fields and provider extensions.
            if let Some(usage) = self.raw_usage {
                reason.push_str(&format!("; CodeBuddy usage: {usage}"));
            }
            events.push(ResponseEvent::Incomplete {
                response_id: self.id,
                token_usage: self.usage,
                reason,
                kind,
            });
            return Ok(events);
        }
        let has_calls = !self.calls.is_empty();
        let mut call_ids = std::collections::BTreeSet::new();
        for call in self.calls.into_values() {
            if call.id.is_empty() || !call_ids.insert(call.id.clone()) {
                return Err(invalid("missing or duplicate tool call ID"));
            }
            let mut matching = call.names.iter().filter_map(|name| tools.get(name));
            let tool = matching
                .next()
                .ok_or_else(|| invalid("upstream requested an unadvertised tool"))?;
            if matching.next().is_some() {
                return Err(invalid("ambiguous repeated tool name"));
            }
            let arguments: Value = serde_json::from_str(&call.arguments)
                .map_err(|_| invalid("invalid tool argument JSON"))?;
            let value = if tool.custom {
                let input = arguments["input"]
                    .as_str()
                    .ok_or_else(|| invalid("custom tool requires a string input"))?;
                json!({"type":"custom_tool_call","id":call.id,"call_id":call.id,"name":tool.name,"namespace":tool.namespace,"input":input})
            } else {
                json!({"type":"function_call","id":call.id,"call_id":call.id,"name":tool.name,"namespace":tool.namespace,"arguments":call.arguments,"encrypted_function_args":[]})
            };
            let value = item(value)?;
            events.push(ResponseEvent::OutputItemAdded(value.clone()));
            events.push(ResponseEvent::OutputItemDone(value));
        }
        if !self.text.is_empty() {
            events.push(ResponseEvent::OutputItemDone(item(json!({"type":"message","id":format!("{}-message",self.id),"role":"assistant","content":[{"type":"output_text","text":self.text}]}))?));
        }
        events.push(ResponseEvent::Completed {
            response_id: self.id,
            token_usage: self.usage,
            usage_metadata: self.raw_usage.map(|usage| ResponseUsageMetadata {
                amount: None,
                metadata: Some(json!({"provider":"codebuddy","usage":usage})),
            }),
            end_turn: Some(!has_calls),
        });
        Ok(events)
    }
}

pub(super) fn spawn(
    response: StreamResponse,
    tools: BTreeMap<String, Tool>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) -> ResponseStream {
    let upstream_request_id = response
        .headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let (tx, rx_event) = mpsc::channel(128);
    tokio::spawn(async move {
        let result = async {
            let mut stream = response.bytes.eventsource();
            let mut completion = Completion::default();
            loop {
                let start = Instant::now();
                let next = tokio::select! {
                    _ = tx.closed() => return Ok(()),
                    next = timeout(idle_timeout, stream.next()) => next,
                };
                if let Some(telemetry) = &telemetry {
                    telemetry.on_sse_poll(&next, start.elapsed());
                }
                let event = next
                    .map_err(|_| ApiError::Stream("CodeBuddy stream idle timeout".into()))?
                    .ok_or_else(|| {
                        ApiError::Stream("CodeBuddy stream ended without [DONE]".into())
                    })?
                    .map_err(|_| ApiError::Stream("invalid CodeBuddy event stream".into()))?;
                if event.data == "[DONE]" {
                    for event in completion.finish(&tools)? {
                        if tx.send(Ok(event)).await.is_err() {
                            return Ok(());
                        }
                    }
                    return Ok(());
                }
                completion.bytes = completion.bytes.saturating_add(event.data.len());
                if completion.bytes > 16 * 1024 * 1024 {
                    return Err(invalid("completion exceeds 16 MiB adapter limit"));
                }
                let chunk = serde_json::from_str(&event.data)
                    .map_err(|_| invalid("invalid stream JSON"))?;
                for event in completion.update(chunk, &tools)? {
                    let terminal = matches!(event, ResponseEvent::Incomplete { .. });
                    if tx.send(Ok(event)).await.is_err() || terminal {
                        return Ok(());
                    }
                }
            }
        }
        .await;
        if let Err(error) = result {
            let _ = tx.send(Err(error)).await;
        }
    });
    ResponseStream {
        rx_event,
        upstream_request_id,
    }
}
