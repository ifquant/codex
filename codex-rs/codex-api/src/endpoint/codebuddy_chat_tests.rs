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
        json!({"role":"system","content":"Be concise."})
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
    assert_eq!(body["messages"][4]["content"], "continue");
}

#[test]
fn preserves_full_instructions_and_model_switch_messages() {
    let mut value = request();
    let instructions = format!(
        "You are a coding agent. {}",
        "Preserve permissions. ".repeat(400)
    );
    value["instructions"] = instructions.clone().into();
    value["input"] = json!([
        {"type":"message","role":"system","content":"<model_switch>CodeBuddy worker</model_switch>"},
        {"type":"message","role":"developer","content":"Preserve tool and permission rules."}
    ]);
    assert_eq!(
        encode(value).unwrap().0["messages"],
        json!([
            {"role":"system","content":instructions},
            {"role":"system","content":"<model_switch>CodeBuddy worker</model_switch>"},
            {"role":"system","content":"Preserve tool and permission rules."}
        ])
    );
}

#[test]
fn folds_assistant_commentary_before_tool_results() {
    let mut value = request();
    value["input"] = json!([
        {"type":"message","role":"user","content":"Run pwd"},
        {"type":"function_call","call_id":"call_1","name":"exec_command","arguments":"{}"},
        {"type":"message","role":"assistant","content":"I will inspect the workspace first."},
        {"type":"function_call_output","call_id":"call_1","output":"/tmp"}
    ]);
    let (body, _) = encode(value).unwrap();
    assert_eq!(body["messages"][2]["role"], "assistant");
    assert_eq!(
        body["messages"][2]["content"],
        "I will inspect the workspace first."
    );
    assert_eq!(
        body["messages"][3],
        json!({
            "role":"tool", "tool_call_id":"call_1", "content":"/tmp"
        })
    );
}

#[test]
fn preserves_orphan_reasoning_before_parent_message() {
    let mut value = request();
    value["input"] = json!([
        {"type":"message","role":"user","content":"Inspect the workspace"},
        {"type":"reasoning","content":"I was interrupted while planning.","encrypted_content":null},
        {"type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"input_text","text":"Continue."}]}
    ]);
    let (body, _) = encode(value).unwrap();
    assert_eq!(body["messages"][2]["role"], "assistant");
    assert_eq!(
        body["messages"][2]["reasoning_content"],
        "I was interrupted while planning."
    );
    assert_eq!(body["messages"][3]["role"], "user");
}

#[test]
fn collects_lite_tools_with_descriptions_and_rejects_conflicting_duplicates() {
    let mut value = request();
    let mut specs = value["tools"].take();
    specs[0]["tools"][0]["description"] = "Tool usage rules. ".repeat(5000).into();
    value["input"] = json!([{"type":"additional_tools","tools":specs}]);
    let (body, mapping) = encode(value.clone()).unwrap();
    assert_eq!(mapping.len(), 2);
    assert_eq!(
        body["tools"][0]["function"]["description"],
        specs[0]["tools"][0]["description"]
    );
    value["tools"] = specs;
    assert_eq!(encode(value.clone()).unwrap().0, body);
    value["input"][0]["tools"][0]["tools"][0]["parameters"] =
        json!({"type":"object","required":["cmd"]});
    assert!(encode(value).is_err());
}

#[test]
fn does_not_replay_display_summary_as_raw_reasoning() {
    let mut value = request();
    value["input"] = json!([
        {"type":"reasoning","summary":[{"type":"summary_text","text":"Display summary"}]},
        {"type":"message","role":"assistant","content":"Answer"}
    ]);
    assert_eq!(
        encode(value).unwrap().0["messages"],
        json!([
            {"role":"system","content":"Be concise."},
            {"role":"assistant","content":"Answer"}
        ])
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
async fn reports_length_as_incomplete_for_partial_text() {
    let data = "data: {\"id\":\"text\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\ndata: {\"id\":\"text\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\ndata: [DONE]\n\n";
    let events = stream(data.into(), BTreeMap::new())
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().any(|event| matches!(
        event,
        Ok(ResponseEvent::OutputTextDelta(text)) if text == "partial"
    )));
    assert!(matches!(
        events.last(),
        Some(Ok(ResponseEvent::Incomplete { .. }))
    ));
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
    value["input"] = json!([{"type":"function_call","name":long_name,"call_id":"c","arguments":"{}"}, {"type":"function_call_output","call_id":"c","output":"Done"}]);
    let (body, tools) = encode(value).unwrap();
    let alias = body["tools"][0]["function"]["name"].as_str().unwrap();
    assert!(alias.len() <= 64);
    assert_eq!(tools[alias].name, long_name);
    assert_eq!(
        body["messages"][1]["tool_calls"][0]["function"]["name"],
        alias
    );
}

#[tokio::test]
async fn length_preserves_reasoning_usage_but_never_dispatches_tools() {
    let (_, tools) = encode(request()).unwrap();
    let chunk = json!({"id":"limited","choices":[{"index":0,"delta":{"reasoning_content":"Partial plan","tool_calls":[{"index":0,"id":"call","function":{"name":"functions__exec_command","arguments":"{}"}}]},"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":32000,"completion_tokens_details":{"reasoning_tokens":32000},"total_tokens":32010}});
    let events = stream(format!("data: {chunk}\n\ndata: [DONE]\n\n"), tools)
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().any(|event| matches!(
        event,
        Ok(ResponseEvent::OutputItemDone(
            ResponseItem::Reasoning { .. }
        ))
    )));
    assert!(!events.iter().any(|event| matches!(
        event,
        Ok(ResponseEvent::Completed { .. }
            | ResponseEvent::OutputItemDone(
                ResponseItem::FunctionCall { .. } | ResponseItem::CustomToolCall { .. }
            ))
    )));
    assert!(
        matches!(events.last(), Some(Ok(ResponseEvent::Incomplete { token_usage: Some(usage), .. })) if usage.output_tokens == 32000 && usage.reasoning_output_tokens == 32000)
    );
}

#[tokio::test]
async fn complete_message_snapshots_are_preserved_once_and_conflicts_rejected() {
    let message =
        json!({"role":"assistant","reasoning_content":"Plan once.","content":"Answer once."});
    for conflicting in [false, true] {
        let second = if conflicting {
            json!({"role":"assistant","content":"Changed"})
        } else {
            message.clone()
        };
        let chunks = [
            json!({"id":"snapshot","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning_content":"","tool_calls":[]}}]}),
            json!({"id":"snapshot","choices":[{"index":0,"message":message}]}),
            json!({"id":"snapshot","choices":[{"index":0,"message":second,"finish_reason":"stop"}]}),
        ];
        let data = chunks
            .iter()
            .map(|c| format!("data: {c}\n\n"))
            .collect::<String>()
            + "data: [DONE]\n\n";
        let events = stream(data, BTreeMap::new()).collect::<Vec<_>>().await;
        if conflicting {
            assert!(events.iter().any(Result::is_err));
            assert!(
                !events
                    .iter()
                    .any(|e| matches!(e, Ok(ResponseEvent::Completed { .. })))
            );
        } else {
            let items = events
                .into_iter()
                .map(Result::unwrap)
                .filter_map(|e| match e {
                    ResponseEvent::OutputItemDone(item) => {
                        Some(serde_json::to_value(item).unwrap())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(items.len(), 2);
            assert_eq!(items[0]["content"][0]["text"], "Plan once.");
            assert_eq!(items[1]["content"][0]["text"], "Answer once.");
        }
    }
}

#[tokio::test]
async fn complete_tool_message_dispatches_once_and_mixed_content_is_rejected() {
    let message = json!({"role":"assistant","content":null,"reasoning_content":"Use tool.",
        "tool_calls":[{"id":"exec","type":"function","function":{"name":"functions__exec_command","arguments":"{}"}}]});
    for mixed in [false, true] {
        let (_, tools) = encode(request()).unwrap();
        let second = if mixed {
            json!({"index":0,"message":message,"delta":{"content":"Conflicting"},"finish_reason":"tool_calls"})
        } else {
            json!({"index":0,"message":message,"finish_reason":"tool_calls"})
        };
        let chunks = [
            json!({"id":"snapshot","choices":[{"index":0,"message":message}]}),
            json!({"id":"snapshot","choices":[second]}),
        ];
        let data = chunks
            .iter()
            .map(|c| format!("data: {c}\n\n"))
            .collect::<String>()
            + "data: [DONE]\n\n";
        let events = stream(data, tools).collect::<Vec<_>>().await;
        let calls = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Ok(ResponseEvent::OutputItemDone(
                        ResponseItem::FunctionCall { .. }
                    ))
                )
            })
            .count();
        if mixed {
            assert!(events.iter().any(Result::is_err));
            assert_eq!(calls, 0);
        } else {
            assert!(events.iter().all(Result::is_ok));
            assert_eq!(calls, 1);
        }
    }
}

#[test]
fn preserves_explicit_output_budgets_and_rejects_invalid_or_conflicting_values() {
    assert!(encode(request()).unwrap().0.get("max_tokens").is_none());
    for key in ["max_output_tokens", "max_completion_tokens", "max_tokens"] {
        let mut value = request();
        value[key] = json!(128000);
        assert_eq!(
            encode(value.clone()).unwrap().0["max_tokens"],
            json!(128000)
        );
        for invalid_value in [
            Value::Null,
            json!(0),
            json!(-1),
            json!(1.5),
            json!("128000"),
        ] {
            value[key] = invalid_value;
            assert!(encode(value.clone()).is_err());
        }
    }
    let mut value = request();
    value["max_output_tokens"] = json!(256);
    value["max_tokens"] = json!(256);
    assert_eq!(encode(value.clone()).unwrap().0["max_tokens"], json!(256));
    value["max_tokens"] = json!(512);
    assert!(encode(value).is_err());
}

#[tokio::test]
async fn tool_fragments_use_exact_advertised_names_and_stable_identity() {
    let mut request = request();
    request["tools"] = json!([
        {"type":"function","name":"run","parameters":{"type":"object"}},
        {"type":"function","name":"runrun","parameters":{"type":"object"}},
        {"type":"function","name":"read","parameters":{"type":"object"}}
    ]);
    let (_, tools) = encode(request).unwrap();
    let cases = [
        (
            vec![
                json!({"index":0,"id":"a","function":{"name":"re","arguments":"{"}}),
                json!({"id":"a","function":{"name":"ad","arguments":"}"}}),
            ],
            true,
        ),
        (
            vec![
                json!({"index":0,"id":"a","function":{"arguments":"{"}}),
                json!({"index":0,"function":{"name":"read","arguments":"}"}}),
                json!({"index":0,"function":{"name":"read"}}),
            ],
            true,
        ),
        (
            vec![
                json!({"index":0,"id":"a","function":{"name":"run","arguments":"{}"}}),
                json!({"index":0,"function":{"name":"run"}}),
            ],
            false,
        ),
        (
            vec![
                json!({"index":0,"id":"a","function":{"name":"read","arguments":"{"}}),
                json!({"index":1,"id":"b","function":{"name":"run","arguments":"{}"}}),
                json!({"id":"a","function":{"arguments":"}"}}),
            ],
            true,
        ),
        (
            vec![
                json!({"index":0,"id":"a","function":{"name":"read","arguments":"{}"}}),
                json!({"index":1,"id":"a","function":{"name":"run","arguments":"{}"}}),
            ],
            false,
        ),
        (
            vec![
                json!({"index":0,"id":"a","function":{"name":"read","arguments":"{"}}),
                json!({"index":1,"id":"b","function":{"name":"run","arguments":"{}"}}),
                json!({"id":"a","function":{"arguments":""}}),
                json!({"function":{"arguments":"}"}}),
            ],
            false,
        ),
    ];
    for (parts, valid) in cases {
        let mut data = parts
            .into_iter()
            .map(|part| {
                format!(
                    "data: {}\n\n",
                    json!({"id":"reply","choices":[{"index":0,"delta":{"tool_calls":[part]}}]})
                )
            })
            .collect::<String>();
        data.push_str(&format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({"id":"reply","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]})
        ));
        let events = stream(data, tools.clone()).collect::<Vec<_>>().await;
        assert_eq!(events.iter().all(Result::is_ok), valid, "{events:?}");
        let calls = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Ok(ResponseEvent::OutputItemDone(
                        ResponseItem::FunctionCall { .. }
                    ))
                )
            })
            .count();
        assert_eq!(calls > 0, valid, "{events:?}");
    }
}

#[tokio::test]
async fn stream_business_error_preserves_details_usage_and_stops_without_done() {
    let data = format!(
        "data: {}\n\ndata: {}\n\n",
        json!({"id":"failed","choices":[{"index":0,"delta":{"reasoning_content":"Partial work"}}]}),
        json!({"error":{"code":"provider_limit","message":"Limit reached"},"request_id":"trace-123","usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}})
    );
    let events = stream(data, BTreeMap::new()).collect::<Vec<_>>().await;
    assert!(events.iter().all(Result::is_ok), "{events:?}");
    assert!(events.iter().any(|e| matches!(
        e,
        Ok(ResponseEvent::OutputItemDone(
            ResponseItem::Reasoning { .. }
        ))
    )));
    let Some(Ok(ResponseEvent::Incomplete {
        reason,
        token_usage: Some(usage),
        ..
    })) = events.last()
    else {
        panic!("{events:?}")
    };
    assert!(
        reason.contains("provider_limit")
            && reason.contains("Limit reached")
            && reason.contains("trace-123")
    );
    assert_eq!(usage.total_tokens, 15);
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Ok(ResponseEvent::Completed { .. })))
    );
    let events = stream(
        "data: {\"error\":{\"message\":\"failed before ID\"}}\n\n".into(),
        BTreeMap::new(),
    )
    .collect::<Vec<_>>()
    .await;
    assert!(matches!(
        events.as_slice(),
        [Ok(ResponseEvent::Incomplete { .. })]
    ));
}

#[tokio::test]
async fn usage_distinguishes_absence_from_zero_and_retains_source_counters() {
    for invalid_count in [json!(-1), json!("10"), json!(1.5)] {
        let chunk = json!({"id":"usage","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30,"prompt_cache_hit_tokens":invalid_count}});
        let events = stream(
            format!("data: {chunk}\n\ndata: [DONE]\n\n"),
            BTreeMap::new(),
        )
        .collect::<Vec<_>>()
        .await;
        assert!(events.iter().any(Result::is_err));
    }
    for usage in [
        Value::Null,
        json!({}),
        json!({"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}),
        json!({"prompt_tokens":20,"completion_tokens":10,"total_tokens":30,"prompt_tokens_details":{"cached_tokens":5},"prompt_cache_hit_tokens":99,"completion_thinking_tokens":7,"credit":1.2}),
    ] {
        let chunk = json!({"id":"usage","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":usage});
        let events = stream(
            format!("data: {chunk}\n\ndata: [DONE]\n\n"),
            BTreeMap::new(),
        )
        .collect::<Vec<_>>()
        .await;
        let Some(Ok(ResponseEvent::Completed {
            token_usage,
            usage_metadata,
            ..
        })) = events.last()
        else {
            panic!("{events:?}")
        };
        assert_eq!(token_usage.is_some(), usage.get("total_tokens").is_some());
        assert_eq!(
            usage_metadata
                .as_ref()
                .and_then(|m| m.metadata.as_ref())
                .map(|m| &m["usage"]),
            if usage.is_null() { None } else { Some(&usage) }
        );
        if usage["total_tokens"] == 30 {
            let u = token_usage.as_ref().unwrap();
            assert_eq!((u.cached_input_tokens, u.reasoning_output_tokens), (5, 7));
        }
    }
}

#[tokio::test]
async fn rejects_cross_completion_fragments_and_empty_streams() {
    for data in [
        "data: {}\n\n",
        "data: [DONE]\n\n",
        "data: {\"id\":\"a\",\"choices\":[]}\n\ndata: {\"id\":\"b\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"mixed\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
    ] {
        let events = stream(data.into(), BTreeMap::new())
            .collect::<Vec<_>>()
            .await;
        assert!(events.iter().any(Result::is_err), "{events:?}");
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Ok(ResponseEvent::Completed { .. })))
        );
    }
}

#[test]
fn validates_complete_history_tool_batches_without_fabricating_results() {
    let a = json!({"type":"function_call","call_id":"a","name":"exec_command","arguments":"{}"});
    let b = json!({"type":"function_call","call_id":"b","name":"exec_command","arguments":"{}"});
    let ra = json!({"type":"function_call_output","call_id":"a","output":"A"});
    let rb = json!({"type":"function_call_output","call_id":"b","output":"B"});
    let notice = json!({"type":"agent_message","content":"Continue"});
    for (input, valid) in [
        (json!([a, b, rb, ra, notice]), true),
        (json!([a, b, ra]), false),
        (json!([a, b, ra, notice, rb]), false),
        (json!([a, a, ra]), false),
        (json!([a, ra, ra]), false),
        (json!([ra]), false),
    ] {
        let mut value = request();
        value["input"] = input;
        assert_eq!(encode(value).is_ok(), valid);
    }
}
