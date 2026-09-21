//! Local compaction exercised through the real CodeBuddy Chat adapter.
use codex_core::TurnInputRequest;
use codex_core::compact::SUMMARIZATION_PROMPT;
use codex_core::compact::SUMMARY_PREFIX;
use codex_model_provider_info::WireApi;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use wiremock::Mock;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

fn chat(delta: Value, finish: &str) -> ResponseTemplate {
    let chunk = json!({"id":"chat-compaction", "choices":[{
        "index":0,"delta":delta,"finish_reason":finish}],
        "usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}});
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(format!("data: {chunk}\n\ndata: [DONE]\n\n"))
}

#[test_case::test_case("success")]
#[test_case::test_case("empty")]
#[test_case::test_case("whitespace")]
#[test_case::test_case("partial_error")]
#[test_case::test_case("server_error")]
#[test_case::test_case("cancel")]
#[test_case::test_case("overflow")]
#[test_case::test_case("overflow_sse")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn codebuddy_compaction_checkpoint(scenario: &'static str) -> anyhow::Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let started = Arc::new(tokio::sync::Notify::new());
    let notify = started.clone();
    let compactions = Arc::new(AtomicUsize::new(0));
    let counter = compactions.clone();
    Mock::given(method("POST")).and(path("/chat/completions"))
        .respond_with(move |request: &wiremock::Request| {
            let body: Value = request.body_json().expect("valid Chat request JSON");
            let messages = body["messages"].as_array().expect("Chat messages");
            let compact = messages.last().expect("last Chat message")["content"].as_str()
                .is_some_and(|text| text.contains("CONTEXT CHECKPOINT COMPACTION"));
            if compact {
                let attempt = counter.fetch_add(1, Ordering::SeqCst);
                return match scenario {
                    "empty" => chat(json!({}), "stop"),
                    "whitespace" => chat(json!({"content":"   \n"}), "stop"),
                    "partial_error" => ResponseTemplate::new(200)
                        .insert_header("content-type", "text/event-stream")
                        .set_body_string("data: {\"id\":\"partial\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"UNCOMMITTED_SUMMARY\"},\"finish_reason\":null}]}\n\ndata: {\"error\":{\"code\":\"failed\",\"message\":\"summary failed\"}}\n\n"),
                    "server_error" => ResponseTemplate::new(500).set_body_json(json!({"error":{"message":"summary failed"}})),
                    "cancel" => {
                        notify.notify_one();
                        chat(json!({"content":"UNCOMMITTED_SUMMARY"}), "stop")
                            .set_delay(std::time::Duration::from_secs(30))
                    }
                    "overflow_sse" if attempt == 0 => ResponseTemplate::new(200)
                        .insert_header("content-type", "text/event-stream")
                        .set_body_string("data: {\"error\":{\"code\":\"context_length_exceeded\",\"message\":\"context window exceeded\"}}\n\n"),
                    "overflow" if attempt == 0 => ResponseTemplate::new(400)
                        .set_body_json(json!({"error":{"code":"context_length_exceeded","message":"context window exceeded"}})),
                    _ => chat(json!({"content":"Progress: inspected. Constraints: keep provider. Next: verify. Evidence: checkpoint.txt"}), "stop"),
                };
            }
            if messages.last().expect("last Chat message")["role"] == "tool" {
                chat(json!({"content":"OLD_ASSISTANT_EVIDENCE"}), "stop")
            } else {
                let name = body["tools"].as_array().expect("advertised tools").iter()
                    .find_map(|tool| tool["function"]["name"].as_str()
                        .filter(|name| name.ends_with("update_plan"))).expect("update_plan tool");
                chat(json!({"tool_calls":[{"index":0,"id":format!("plan-{}", messages.len()),
                    "function":{"name":name,"arguments":json!({"plan":[{"step":"Preserve checkpoint.txt", "status":"completed"}]}).to_string()}}]}), "tool_calls")
            }
        }).mount(&server).await;
    let base_url = server.uri();
    let configure = move |config: &mut codex_core::config::Config| {
        config.update_plan_enabled = true;
        config.compact_prompt = Some(SUMMARIZATION_PROMPT.into());
        config.model_provider.base_url = Some(base_url.clone());
        config.model_provider.wire_api = WireApi::CodebuddyChat;
        config.model_provider.supports_websockets = false;
        config.model_provider.request_max_retries = Some(0);
        config.model_provider.stream_max_retries = Some(0);
        config.model = Some("deepseek-v4.1-flash".into());
        config.model_reasoning_effort = Some(ReasoningEffort::High);
    };
    let mut builder = test_codex().with_config(configure.clone());
    let initial = builder.build_with_auto_env(&server).await?;
    initial
        .codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "Keep the provider fixed; inspect checkpoint.txt".into(),
            text_elements: vec![],
        }]))
        .await?;
    wait_for_event(&initial.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    initial.codex.submit(Op::Compact).await?;
    if scenario == "cancel" {
        tokio::time::timeout(std::time::Duration::from_secs(10), started.notified()).await?;
        initial.codex.submit(Op::Interrupt).await?;
        wait_for_event(&initial.codex, |event| {
            matches!(event, EventMsg::TurnAborted(_))
        })
        .await;
    } else {
        wait_for_event(&initial.codex, |event| {
            matches!(event, EventMsg::TurnComplete(_))
        })
        .await;
    }
    let success = matches!(scenario, "success" | "overflow" | "overflow_sse");
    // Exercise in-memory continuation before shutdown, then the persisted checkpoint below.
    initial
        .codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "Continue immediately after compaction".into(),
            text_elements: vec![],
        }]))
        .await?;
    let completed = wait_for_event(&initial.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    let EventMsg::TurnComplete(completed) = completed else {
        unreachable!()
    };
    assert_eq!(completed.error, None);

    let resumed = test_codex()
        .with_config(configure)
        .restart(&server, &initial)
        .await?;
    let rollout = std::fs::read_to_string(
        initial
            .session_configured
            .rollout_path
            .as_ref()
            .expect("materialized rollout"),
    )?;
    let checkpoints: Vec<Value> = rollout
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|item| item["type"] == "compacted")
        .collect();
    assert_eq!(checkpoints.len(), usize::from(success), "{scenario}");
    assert!(!rollout.contains("UNCOMMITTED_SUMMARY"));
    if success {
        let checkpoint = checkpoints[0].to_string();
        assert!(checkpoint.contains("Preserve checkpoint.txt"));
        assert!(checkpoint.contains("Plan updated"));
        assert!(checkpoint.contains("Original transcript:"));
        assert!(
            checkpoint.contains(
                initial
                    .session_configured
                    .rollout_path
                    .as_ref()
                    .expect("materialized rollout")
                    .to_str()
                    .expect("UTF-8 test path")
            )
        );
    }
    resumed
        .codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "Continue verification".into(),
            text_elements: vec![],
        }]))
        .await?;
    let completed = wait_for_event(&resumed.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    let EventMsg::TurnComplete(completed) = completed else {
        unreachable!()
    };
    assert_eq!(completed.error, None);
    let requests = server.received_requests().await.expect("recorded requests");
    let body: Value = requests.last().expect("continuation request").body_json()?;
    assert_eq!(body["model"], "deepseek-v4.1-flash");
    assert_eq!(body["reasoning_effort"], "high");
    let messages = body["messages"].as_array().expect("Chat messages");
    assert!(messages.iter().any(|m| {
        m["content"]
            .as_str()
            .is_some_and(|s| s.contains("Keep the provider fixed"))
    }));
    assert_eq!(
        messages.iter().any(|m| m["content"]
            .as_str()
            .is_some_and(|s| s.contains(SUMMARY_PREFIX))),
        success
    );
    if !success {
        assert!(
            messages
                .iter()
                .any(|m| m["content"] == "OLD_ASSISTANT_EVIDENCE")
        );
    }
    for (index, message) in messages.iter().enumerate() {
        if message["role"] == "tool" {
            assert!(
                messages[..index]
                    .iter()
                    .any(|m| m["tool_calls"].as_array().is_some_and(|calls| calls
                        .iter()
                        .any(|call| call["id"] == message["tool_call_id"])))
            );
        }
    }
    assert_eq!(
        compactions.load(Ordering::SeqCst),
        if scenario.starts_with("overflow") {
            2
        } else {
            1
        }
    );
    resumed.codex.shutdown_and_wait().await?;
    Ok(())
}
