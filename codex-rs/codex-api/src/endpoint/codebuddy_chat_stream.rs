use super::codebuddy_chat::Tool;
use super::codebuddy_chat::invalid;
use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::telemetry::SseTelemetry;
use codex_client::StreamResponse;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio::time::timeout;

#[derive(Default)]
struct Call {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Default)]
struct Completion {
    id: String,
    text: String,
    reasoning: String,
    calls: BTreeMap<u64, Call>,
    finish: Option<String>,
    usage: Option<TokenUsage>,
    bytes: usize,
}

fn item(value: Value) -> Result<ResponseItem, ApiError> {
    serde_json::from_value(value).map_err(|e| invalid(format!("invalid completion item: {e}")))
}

impl Completion {
    fn update(&mut self, chunk: Value) -> Result<Vec<ResponseEvent>, ApiError> {
        if !chunk["error"].is_null() {
            return Err(ApiError::Stream(
                "CodeBuddy returned an error in the completion stream".into(),
            ));
        }
        let mut events = Vec::new();
        if self.id.is_empty() {
            self.id = chunk["id"]
                .as_str()
                .ok_or_else(|| invalid("stream lacks completion ID"))?
                .to_owned();
            events.push(ResponseEvent::Created {
                response_id: Some(self.id.clone()),
            });
            if let Some(model) = chunk["model"].as_str() {
                events.push(ResponseEvent::ServerModel(model.to_owned()));
            }
        }
        if let Some(usage) = chunk.get("usage").filter(|usage| usage.is_object()) {
            self.usage = Some(TokenUsage {
                input_tokens: usage["prompt_tokens"].as_i64().unwrap_or(0),
                cached_input_tokens: usage["prompt_tokens_details"]["cached_tokens"]
                    .as_i64()
                    .or_else(|| usage["prompt_cache_hit_tokens"].as_i64())
                    .unwrap_or(0),
                output_tokens: usage["completion_tokens"].as_i64().unwrap_or(0),
                reasoning_output_tokens: usage["completion_tokens_details"]["reasoning_tokens"]
                    .as_i64()
                    .unwrap_or(0),
                total_tokens: usage["total_tokens"].as_i64().unwrap_or(0),
                ..Default::default()
            });
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
            let delta = &choice["delta"];
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
                let index = delta["index"]
                    .as_u64()
                    .filter(|i| *i < 256)
                    .ok_or_else(|| invalid("invalid tool call index"))?;
                let call = self.calls.entry(index).or_default();
                if let Some(id) = delta["id"].as_str() {
                    if !call.id.is_empty() && call.id != id {
                        return Err(invalid("tool call ID changed mid-stream"));
                    }
                    call.id = id.to_owned();
                }
                if let Some(name) = delta["function"]["name"].as_str() {
                    call.name.push_str(name);
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
        let has_calls = !self.calls.is_empty();
        let mut call_ids = std::collections::BTreeSet::new();
        for call in self.calls.into_values() {
            if call.id.is_empty() || !call_ids.insert(call.id.clone()) {
                return Err(invalid("missing or duplicate tool call ID"));
            }
            let tool = tools
                .get(&call.name)
                .ok_or_else(|| invalid("upstream requested an unadvertised tool"))?;
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
            usage_metadata: None,
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
                for event in completion.update(chunk)? {
                    if tx.send(Ok(event)).await.is_err() {
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
