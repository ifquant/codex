use super::*;
use crate::common::ResponseEvent;
use bytes::Bytes;
use codex_client::StreamResponse;
use codex_protocol::models::ResponseItem;
use futures::StreamExt;
use http::StatusCode;
use pretty_assertions::assert_eq;
use std::time::Duration;

fn request() -> Value {
    json!({"model":"deepseek-v4.1-flash","instructions":"Be concise.","input":[],"reasoning":{"effort":"high"},"tools":[{"type":"namespace","name":"functions","tools":[{"type":"function","name":"exec_command","parameters":{"type":"object"}},{"type":"custom","name":"apply_patch"}]}]})
}

fn stream(data: String, tools: BTreeMap<String, Tool>) -> ResponseStream {
    // Deliberately split inside SSE framing and JSON tokens.
    let chunks = data
        .as_bytes()
        .chunks(7)
        .map(Bytes::copy_from_slice)
        .map(Ok)
        .collect::<Vec<_>>();
    codebuddy_chat_stream::spawn(
        StreamResponse {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            bytes: Box::pin(futures::stream::iter(chunks)),
        },
        tools,
        Duration::from_secs(1),
        None,
    )
}

#[tokio::test]
async fn streams_parallel_tools_and_replays_reasoning_and_custom_input() {
    let (body, tools) = encode(request()).unwrap();
    assert_eq!(
        body["messages"][0],
        json!({"role":"system","content":"You are a coding assistant. Follow the user's request and use available tools when needed."})
    );
    assert!(
        body["messages"][1]["content"]
            .as_str()
            .unwrap()
            .contains("Be concise.")
    );
    assert!(
        body["messages"][1]["content"]
            .as_str()
            .unwrap()
            .contains("<codex-instructions>")
    );
    assert_eq!(body["reasoning_effort"], "high");
    assert_eq!(body["thinking"], json!({"type":"enabled"}));
    assert_eq!(
        body["tools"][0]["function"]["name"],
        "functions__exec_command"
    );
    let chunks = [
        json!({"id":"reply-1","model":"deepseek-v4.1-flash","choices":[{"index":0,"delta":{"reasoning_content":"Use tools."}}]}),
        json!({"id":"reply-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"patch","function":{"name":"functions__apply_patch","arguments":"{\"input\":"}},{"index":0,"id":"exec","function":{"name":"functions__exec_command","arguments":"{\"cmd\":"}}]}}]}),
        json!({"id":"reply-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"pwd\"}"}},{"index":1,"function":{"arguments":"\"patch text\"}"}}]},"finish_reason":"tool_calls"}]}),
        json!({"id":"reply-1","choices":[],"usage":{"prompt_tokens":20,"prompt_cache_hit_tokens":5,"completion_tokens":8,"completion_tokens_details":{"reasoning_tokens":3},"total_tokens":28}}),
    ];
    let data = chunks
        .iter()
        .map(|c| format!("data: {c}\n\n"))
        .collect::<String>()
        + "data: [DONE]\n\n";
    let events = stream(data, tools).collect::<Vec<_>>().await;
    let mut history = Vec::new();
    for event in events {
        match event.unwrap() {
            ResponseEvent::OutputItemDone(item) => {
                history.push(serde_json::to_value(item).unwrap())
            }
            ResponseEvent::Completed {
                token_usage: Some(usage),
                end_turn,
                ..
            } => {
                assert_eq!(
                    (
                        usage.input_tokens,
                        usage.cached_input_tokens,
                        usage.output_tokens,
                        usage.reasoning_output_tokens,
                        usage.total_tokens
                    ),
                    (20, 5, 8, 3, 28)
                );
                assert_eq!(end_turn, Some(false));
            }
            _ => {}
        }
    }
    assert_eq!(history.len(), 3);
    assert_eq!(history[1]["namespace"], "functions");
    assert_eq!(history[1]["arguments"], r#"{"cmd":"pwd"}"#);
    assert_eq!(history[2]["type"], "custom_tool_call");
    assert_eq!(history[2]["input"], "patch text");
    history.push(json!({"type":"function_call_output","call_id":"exec","output":"/workspace"}));
    history.push(json!({"type":"custom_tool_call_output","call_id":"patch","output":"Done"}));
    history.push(json!({"type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"input_text","text":"continue"}]}));
    let mut followup = request();
    followup["input"] = history.into();
    let (body, _) = encode(followup).unwrap();
    assert_eq!(body["messages"][1]["reasoning_content"], "Use tools.");
    assert_eq!(
        body["messages"][1]["tool_calls"].as_array().unwrap().len(),
        2
    );
    assert_eq!(
        body["messages"][3],
        json!({"role":"tool","tool_call_id":"patch","content":"Done"})
    );
    assert!(
        body["messages"][4]["content"]
            .as_str()
            .unwrap()
            .ends_with("continue")
    );
}

#[test]
fn rejects_unsupported_history_effort_and_tool_collisions() {
    for input in [
        json!({"type":"agent_message","content":[{"type":"encrypted_content","encrypted_content":"opaque"}]}),
        json!({"type":"message","role":"user","content":[{"type":"input_image","image_url":"https://example.com/image"}]}),
        json!({"type":"reasoning","encrypted_content":"opaque","summary":[]}),
        json!({"type":"compaction","encrypted_content":"opaque"}),
    ] {
        let mut value = request();
        value["input"] = json!([input]);
        assert!(matches!(
            encode(value),
            Err(ApiError::InvalidRequest { .. })
        ));
    }
    let mut value = request();
    for effort in ["high", "max"] {
        value["reasoning"]["effort"] = effort.into();
        assert_eq!(encode(value.clone()).unwrap().0["reasoning_effort"], effort);
    }
    value["reasoning"]["effort"] = "medium".into();
    assert!(encode(value).is_err());
    let mut value = request();
    value["tools"]
        .as_array_mut()
        .unwrap()
        .push(json!({"type":"function","name":"functions__exec_command"}));
    assert!(encode(value).is_err());
}

#[test]
fn preserves_supported_tool_choices_and_rejects_named_choices() {
    for choice in ["auto", "none", "required"] {
        let mut value = request();
        value["tool_choice"] = choice.into();
        assert_eq!(encode(value).unwrap().0["tool_choice"], choice);
    }
    let mut value = request();
    value["tool_choice"] = json!({"type":"function","function":{"name":"exec_command"}});
    assert!(encode(value).is_err());
}

#[tokio::test]
async fn rejects_truncated_streams_and_incomplete_tool_calls() {
    for suffix in ["", "data: [DONE]\n\n"] {
        let data = format!(
            "data: {{\"id\":\"r\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"c\",\"function\":{{\"name\":\"functions__exec_command\",\"arguments\":\"{{\"}}}}]}}}}]}}\n\n{suffix}"
        );
        let (_, tools) = encode(request()).unwrap();
        let events = stream(data, tools).collect::<Vec<_>>().await;
        assert!(events.iter().any(Result::is_err));
        assert!(!events.iter().any(|e| matches!(
            e,
            Ok(
                ResponseEvent::OutputItemDone(ResponseItem::FunctionCall { .. })
                    | ResponseEvent::Completed { .. }
            )
        )));
    }
}

#[tokio::test]
async fn streams_text_and_completes_only_after_done() {
    let data = "data: {\"id\":\"text\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\ndata: {\"id\":\"text\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    let events = stream(data.into(), BTreeMap::new())
        .collect::<Vec<_>>()
        .await;
    assert!(
        events
            .iter()
            .any(|e| matches!(e,Ok(ResponseEvent::OutputTextDelta(text)) if text=="hello"))
    );
    assert!(matches!(
        events.last(),
        Some(Ok(ResponseEvent::Completed {
            end_turn: Some(true),
            ..
        }))
    ));
}

#[tokio::test]
async fn rejects_unadvertised_and_malformed_completed_calls() {
    let mut value = request();
    value["input"] = json!([
        {"type":"function_call","call_id":"old","name":"removed_tool","arguments":"{}"},
        {"type":"function_call_output","call_id":"old","output":"old result"}
    ]);
    let (_, tools) = encode(value).unwrap();
    for (name, arguments) in [
        ("removed_tool", "{}"),
        ("functions__exec_command", "{"),
        ("functions__apply_patch", "{}"),
    ] {
        let chunk = json!({"id":"bad","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call","function":{"name":name,"arguments":arguments}}]},"finish_reason":"tool_calls"}]});
        let events = stream(format!("data: {chunk}\n\ndata: [DONE]\n\n"), tools.clone())
            .collect::<Vec<_>>()
            .await;
        assert!(events.iter().any(Result::is_err));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Ok(ResponseEvent::OutputItemDone(_))))
        );
    }
}

#[test]
fn long_tool_names_roundtrip_without_becoming_executable_from_history() {
    let long_name = "long-tool-name-".repeat(8);
    let mut value = request();
    value["tools"] = json!([{"type":"function","name":long_name}]);
    value["input"] =
        json!([{"type":"function_call","name":long_name,"call_id":"c","arguments":"{}"}]);
    let (body, tools) = encode(value).unwrap();
    let alias = body["tools"][0]["function"]["name"].as_str().unwrap();
    assert!(alias.len() <= 64);
    assert_eq!(tools[alias].name, long_name);
    let assistant = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "assistant")
        .unwrap();
    assert_eq!(assistant["tool_calls"][0]["function"]["name"], alias);
}
