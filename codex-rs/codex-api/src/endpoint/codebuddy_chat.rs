//! Deliberately text-only CodeBuddy adapter. Unsupported history fails before HTTP.

use super::codebuddy_chat_stream;
use super::session::EndpointSession;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::telemetry::SseTelemetry;
use codex_client::EncodedJsonBody;
use codex_client::HttpTransport;
use http::HeaderMap;
use http::Method;
use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

#[derive(Clone, Debug, Hash, PartialEq)]
pub(super) struct Tool {
    pub name: String,
    pub namespace: Option<String>,
    pub custom: bool,
}

pub(super) fn invalid(message: impl Into<String>) -> ApiError {
    ApiError::InvalidRequest {
        message: format!("CodeBuddy Chat: {}", message.into()),
    }
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, ApiError> {
    value[key]
        .as_str()
        .ok_or_else(|| invalid(format!("missing string field {key}")))
}

fn register(tools: &mut BTreeMap<String, Tool>, tool: Tool) -> Result<String, ApiError> {
    let mut name = match &tool.namespace {
        Some(namespace) => format!("{namespace}__{}", tool.name),
        None => tool.name.clone(),
    };
    if name.is_empty() {
        return Err(invalid("empty tool name"));
    }
    if name.len() > 64
        || !name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
    {
        // Hash only the transport alias. History retains the original identity; collisions fail closed.
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        tool.hash(&mut hasher);
        name = format!("tool_{:016x}", hasher.finish());
    }
    if let Some(previous) = tools.insert(name.clone(), tool.clone())
        && previous != tool
    {
        return Err(invalid("flattened tool names collide"));
    }
    Ok(name)
}

fn text(content: &Value) -> Result<String, ApiError> {
    if let Some(text) = content.as_str() {
        return Ok(text.to_owned());
    }
    let parts = content
        .as_array()
        .ok_or_else(|| invalid("expected text content"))?;
    let mut result = String::new();
    for part in parts {
        match part["type"].as_str() {
            Some("input_text" | "output_text" | "text" | "reasoning_text" | "summary_text") => {
                result.push_str(string(part, "text")?);
            }
            _ => {
                return Err(invalid(
                    "non-text or encrypted content is unsupported; use plaintext task handoffs",
                ));
            }
        }
    }
    Ok(result)
}

pub(super) fn encode(request: Value) -> Result<(Value, BTreeMap<String, Tool>), ApiError> {
    let mut mapping = BTreeMap::new();
    let mut tools = Vec::new();
    let input = request["input"]
        .as_array()
        .ok_or_else(|| invalid("input must be an array"))?;
    let mut specs = request["tools"].as_array().cloned().unwrap_or_default();
    for item in input
        .iter()
        .filter(|item| item["type"] == "additional_tools")
    {
        specs.extend(
            item["tools"]
                .as_array()
                .ok_or_else(|| invalid("additional_tools lacks a tools array"))?
                .iter()
                .cloned(),
        );
    }
    for spec in &specs {
        let (namespace, children) = if spec["type"] == "namespace" {
            (
                Some(string(spec, "name")?.to_owned()),
                spec["tools"]
                    .as_array()
                    .ok_or_else(|| invalid("namespace lacks tools"))?
                    .clone(),
            )
        } else {
            (None, vec![spec.clone()])
        };
        for spec in children {
            let custom = match spec["type"].as_str() {
                Some("function") => false,
                Some("custom") => true,
                _ => return Err(invalid(format!("unsupported tool type: {}", spec["type"]))),
            };
            let name = register(
                &mut mapping,
                Tool {
                    name: string(&spec, "name")?.to_owned(),
                    namespace: namespace.clone(),
                    custom,
                },
            )?;
            let parameters = if custom {
                json!({"type":"object","properties":{"input":{"type":"string"}},"required":["input"],"additionalProperties":false})
            } else {
                spec.get("parameters")
                    .cloned()
                    .unwrap_or_else(|| json!({"type":"object","properties":{}}))
            };
            let definition = json!({"type":"function","function":{"name":name,"description":spec["description"].as_str().unwrap_or_default(),"parameters":parameters}});
            if let Some(previous) = tools
                .iter()
                .find(|tool: &&Value| tool["function"]["name"] == name)
            {
                if previous != &definition {
                    return Err(invalid("conflicting definitions for the same tool"));
                }
            } else {
                tools.push(definition);
            }
        }
    }
    let advertised_tools = mapping.clone();
    let mut messages = Vec::new();
    if let Some(instructions) = request["instructions"].as_str().filter(|s| !s.is_empty()) {
        messages.push(json!({"role":"system","content":instructions}));
    }
    let mut reasoning = String::new();
    for item in request["input"]
        .as_array()
        .ok_or_else(|| invalid("input must be an array"))?
    {
        match item["type"].as_str() {
            // Both tool locations have already been collected into the advertised map.
            Some("additional_tools") => {}
            Some("reasoning") => {
                if !item["encrypted_content"].is_null() {
                    return Err(invalid(
                        "encrypted reasoning cannot cross providers; start without inherited history",
                    ));
                }
                if !item["content"].is_null() {
                    reasoning.push_str(&text(&item["content"])?);
                }
                // Display summaries are not raw reasoning and must not be replayed as such.
            }
            Some("message" | "agent_message") => {
                let role = if item["type"] == "agent_message" {
                    "user"
                } else {
                    string(item, "role")?
                };
                if !matches!(role, "user" | "assistant" | "system" | "developer") {
                    return Err(invalid("unsupported message role"));
                }
                let wire_role = if role == "developer" { "system" } else { role };
                if wire_role != "assistant" && !reasoning.is_empty() {
                    messages.push(json!({
                        "role":"assistant",
                        "content":"",
                        "reasoning_content":std::mem::take(&mut reasoning)
                    }));
                }
                let content = text(&item["content"])?;
                // Responses may emit assistant commentary between a tool-call
                // item and its tool results. Chat Completions requires the
                // results to follow the calling assistant message directly;
                // fold that commentary into the calling message.
                if wire_role == "assistant"
                    && messages
                        .last()
                        .is_some_and(|message| message["tool_calls"].is_array())
                {
                    let message = messages
                        .last_mut()
                        .ok_or_else(|| invalid("missing assistant message"))?;
                    if !content.is_empty() {
                        let existing = message["content"].as_str().unwrap_or_default();
                        message["content"] = if existing.is_empty() {
                            content.to_owned().into()
                        } else {
                            format!("{existing}\n\n{content}").into()
                        };
                    }
                    if !reasoning.is_empty() {
                        message["reasoning_content"] = std::mem::take(&mut reasoning).into();
                    }
                    continue;
                }
                let mut message = json!({"role":wire_role,"content":content});
                if role == "assistant" && !reasoning.is_empty() {
                    message["reasoning_content"] = std::mem::take(&mut reasoning).into();
                }
                messages.push(message);
            }
            Some("function_call" | "custom_tool_call") => {
                if item["encrypted_function_args"]
                    .as_array()
                    .is_some_and(|args| !args.is_empty())
                {
                    return Err(invalid("encrypted tool arguments are unsupported"));
                }
                let custom = item["type"] == "custom_tool_call";
                let name = register(
                    &mut mapping,
                    Tool {
                        name: string(item, "name")?.to_owned(),
                        namespace: item["namespace"].as_str().map(str::to_owned),
                        custom,
                    },
                )?;
                let arguments = if custom {
                    json!({"input":string(item,"input")?}).to_string()
                } else {
                    string(item, "arguments")?.to_owned()
                };
                if messages
                    .last()
                    .is_none_or(|message| message["role"] != "assistant")
                {
                    messages.push(json!({"role":"assistant","content":""}));
                }
                let message = messages
                    .last_mut()
                    .ok_or_else(|| invalid("missing assistant message"))?;
                if !reasoning.is_empty() {
                    message["reasoning_content"] = std::mem::take(&mut reasoning).into();
                }
                if message.get("tool_calls").is_none() {
                    message["tool_calls"] = json!([]);
                }
                message["tool_calls"].as_array_mut().ok_or_else(|| invalid("invalid tool call array"))?.push(json!({"id":string(item,"call_id")?,"type":"function","function":{"name":name,"arguments":arguments}}));
            }
            Some("function_call_output" | "custom_tool_call_output") => {
                if !reasoning.is_empty() {
                    messages.push(json!({
                        "role":"assistant",
                        "content":"",
                        "reasoning_content":std::mem::take(&mut reasoning)
                    }));
                }
                messages.push(json!({"role":"tool","tool_call_id":string(item,"call_id")?,"content":text(&item["output"])?}));
            }
            _ => {
                return Err(invalid(
                    "unsupported history item; start without inherited history",
                ));
            }
        }
    }
    if !reasoning.is_empty() {
        messages.push(json!({
            "role":"assistant",
            "content":"",
            "reasoning_content":reasoning
        }));
    }
    // Validate the final Chat sequence after folding assistant commentary.
    // Never fabricate missing results or silently discard unmatched calls.
    let mut seen = BTreeSet::new();
    let mut pending = BTreeSet::new();
    for message in &messages {
        if message["role"] == "tool" {
            if !pending.remove(string(message, "tool_call_id")?) {
                return Err(invalid("unmatched or duplicate tool result"));
            }
        } else {
            if !pending.is_empty() {
                return Err(invalid("tool results must precede the next message"));
            }
            for call in message["tool_calls"].as_array().into_iter().flatten() {
                let id = string(call, "id")?;
                if id.is_empty() || !seen.insert(id) {
                    return Err(invalid("empty or duplicate historical tool call ID"));
                }
                pending.insert(id);
            }
        }
    }
    if !pending.is_empty() {
        return Err(invalid("missing historical tool results"));
    }
    // Fold only exact repeated evidence in the outbound copy. Keep the first result
    // verbatim so appending another duplicate cannot rewrite the cached prefix.
    // Persisted response items are never rewritten.
    let identities: BTreeMap<String, String> = messages
        .iter()
        .flat_map(|message| message["tool_calls"].as_array().into_iter().flatten())
        .map(|call| {
            (
                call["id"].as_str().unwrap_or_default().to_owned(),
                call["function"].to_string(),
            )
        })
        .collect();
    let mut first_results = BTreeMap::new();
    for message in messages
        .iter_mut()
        .filter(|message| message["role"] == "tool")
    {
        let call_id = string(message, "tool_call_id")?;
        let Some(identity) = identities.get(call_id) else {
            continue;
        };
        let content = string(message, "content")?;
        if content.len() < 1024 {
            continue;
        }
        let target = first_results
            .entry((identity.clone(), content.to_owned()))
            .or_insert_with(|| call_id.to_owned());
        if target == call_id {
            continue;
        }
        let reference = Value::String(format!(
            "[Repeated result: identical to tool_call_id={target} earlier in this context. The original call and result remain in the session transcript.]"
        ));
        if reference.to_string().len() < message["content"].to_string().len() {
            message["content"] = reference;
        }
    }
    if request.get("text").is_some_and(|t| !t["format"].is_null()) {
        return Err(invalid("structured response formats are not supported yet"));
    }
    let tool_choice = match request.get("tool_choice") {
        None => "auto".to_owned(),
        Some(Value::String(choice)) if matches!(choice.as_str(), "auto" | "none" | "required") => {
            choice.clone()
        }
        Some(Value::Null) => "auto".to_owned(),
        Some(_) => {
            return Err(invalid(
                "named tool_choice is not supported by CodeBuddy; use auto, none, or required",
            ));
        }
    };
    let effort = request["reasoning"]["effort"].as_str().unwrap_or("high");
    if !matches!(effort, "high" | "max") {
        return Err(invalid(
            "this adapter currently supports only high or max reasoning",
        ));
    }
    let mut body = json!({"model":string(&request,"model")?,"messages":messages,"tools":tools,"tool_choice":tool_choice,"stream":true,"stream_options":{"include_usage":true},"reasoning_effort":effort,"thinking":{"type":"enabled"}});
    let mut output_budget = None;
    for key in ["max_output_tokens", "max_completion_tokens", "max_tokens"] {
        if let Some(value) = request.get(key) {
            let value = value
                .as_u64()
                .filter(|value| *value > 0)
                .ok_or_else(|| invalid(format!("{key} must be a positive integer")))?;
            if output_budget.is_some_and(|previous| previous != value) {
                return Err(invalid("conflicting output token budgets"));
            }
            output_budget = Some(value);
        }
    }
    if let Some(value) = output_budget {
        body["max_tokens"] = value.into();
    }
    Ok((body, advertised_tools))
}

pub(super) async fn stream<T: HttpTransport>(
    session: &EndpointSession<T>,
    request: Value,
    headers: HeaderMap,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) -> Result<ResponseStream, ApiError> {
    let (body, tools) = encode(request)?;
    let body = EncodedJsonBody::encode(&body).map_err(|e| invalid(e.to_string()))?;
    let response = session
        .stream_encoded_json_with(
            Method::POST,
            "/chat/completions",
            headers,
            Some(body),
            |request| {
                request.headers.insert(
                    http::header::ACCEPT,
                    http::HeaderValue::from_static("text/event-stream"),
                );
            },
        )
        .await
        .map_err(|error| {
            if let ApiError::Transport(codex_client::TransportError::Http {
                status,
                body: Some(body),
                ..
            }) = &error
                && *status == http::StatusCode::BAD_REQUEST
                && serde_json::from_str::<Value>(body)
                    .is_ok_and(|body| body["error"]["code"] == "context_length_exceeded")
            {
                return ApiError::ContextWindowExceeded;
            }
            error
        })?;
    Ok(codebuddy_chat_stream::spawn(
        response,
        tools,
        session.provider().stream_idle_timeout,
        telemetry,
    ))
}

#[cfg(test)]
#[path = "codebuddy_chat_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "codebuddy_chat_noise_tests.rs"]
mod noise_tests;
