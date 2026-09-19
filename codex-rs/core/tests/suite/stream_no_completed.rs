//! Verifies that the agent retries when the SSE stream terminates before
//! delivering a `response.completed` event.

use codex_core::TurnInputRequest;
use codex_model_provider_info::ModelProviderInfo;
use codex_model_provider_info::WireApi;
use codex_protocol::protocol::EventMsg;
use codex_protocol::user_input::UserInput;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::streaming_sse::StreamingSseChunk;
use core_test_support::streaming_sse::start_streaming_sse_server;
use core_test_support::test_codex::TestCodex;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use std::net::TcpListener;
use wiremock::MockServer;

fn sse_incomplete() -> String {
    responses::sse(vec![serde_json::json!({
        "type": "response.output_item.done",
    })])
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retries_on_early_close() {
    skip_if_no_network!();

    let incomplete_sse = sse_incomplete();
    let completed_sse = responses::sse_completed("resp_ok");

    let (server, _) = start_streaming_sse_server(vec![
        vec![StreamingSseChunk {
            gate: None,
            body: incomplete_sse,
        }],
        vec![StreamingSseChunk {
            gate: None,
            body: completed_sse,
        }],
    ])
    .await;

    // Configure retry behavior explicitly to avoid mutating process-wide
    // environment variables.

    let model_provider = ModelProviderInfo {
        name: "openai".into(),
        base_url: Some(format!("{}/v1", server.uri())),
        // Environment variable that should exist in the test environment.
        // ModelClient will return an error if the environment variable for the
        // provider is not set.
        env_key: Some("PATH".into()),
        env_key_instructions: None,
        experimental_bearer_token: None,
        auth: None,
        aws: None,
        wire_api: WireApi::Responses,
        query_params: None,
        http_headers: None,
        env_http_headers: None,
        // exercise retry path: first attempt yields incomplete stream, so allow 1 retry
        request_max_retries: Some(0),
        stream_max_retries: Some(1),
        stream_idle_timeout_ms: Some(2000),
        websocket_connect_timeout_ms: None,
        requires_openai_auth: false,
        supports_websockets: false,
        supports_standalone_web_search: false,
    };

    let TestCodex { codex, .. } = test_codex()
        .with_config(move |config| {
            config.model_provider = model_provider;
        })
        .build_with_streaming_server(&server)
        .await
        .unwrap();

    codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "hello".into(),
            text_elements: Vec::new(),
        }]))
        .await
        .unwrap();

    // Wait until TurnComplete (should succeed after retry).
    wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await;

    let requests = server.requests().await;
    assert_eq!(
        requests.len(),
        2,
        "expected retry after incomplete SSE stream"
    );

    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connection_failure_pauses_retry_budget_until_provider_is_reachable() -> anyhow::Result<()>
{
    skip_if_no_network!(Ok(()));

    let bootstrap_server = responses::start_mock_server().await;
    let unavailable_listener = TcpListener::bind("127.0.0.1:0")?;
    let unavailable_address = unavailable_listener.local_addr()?;
    drop(unavailable_listener);

    let TestCodex { codex, .. } = test_codex()
        .with_config(move |config| {
            config.model_provider.base_url = Some(format!("http://{unavailable_address}/v1"));
            config.model_provider.request_max_retries = Some(0);
            config.model_provider.stream_max_retries = Some(1);
            config.model_provider.supports_websockets = false;
        })
        .build_with_auto_env(&bootstrap_server)
        .await?;

    codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "recover after the network returns".into(),
            text_elements: Vec::new(),
        }]))
        .await?;

    let EventMsg::StreamError(connection_error) =
        wait_for_event(&codex, |event| matches!(event, EventMsg::StreamError(_))).await
    else {
        unreachable!("predicate guarantees a stream error event");
    };
    assert_eq!(
        connection_error.message,
        "Reconnecting... waiting for network"
    );

    let recovered_server = MockServer::builder()
        .listener(TcpListener::bind(unavailable_address)?)
        .start()
        .await;
    let response_mock = responses::mount_sse_sequence(
        &recovered_server,
        vec![sse_incomplete(), responses::sse_completed("resp_recovered")],
    )
    .await;

    let EventMsg::StreamError(stream_error) =
        wait_for_event(&codex, |event| matches!(event, EventMsg::StreamError(_))).await
    else {
        unreachable!("predicate guarantees a stream error event");
    };
    assert_eq!(stream_error.message, "Reconnecting... 1/1");

    let EventMsg::TurnComplete(completed) =
        wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await
    else {
        unreachable!("predicate guarantees a turn complete event");
    };

    assert_eq!(completed.error, None);
    assert_eq!(response_mock.requests().len(), 2);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn codebuddy_length_stops_after_two_continuations() -> anyhow::Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let chunk = serde_json::json!({
        "id":"limited",
        "choices":[{"index":0,"delta":{"reasoning_content":"Partial plan"},"finish_reason":"length"}],
        "usage":{"prompt_tokens":10,"completion_tokens":32000,
            "completion_tokens_details":{"reasoning_tokens":32000},"total_tokens":32010}
    });
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/chat/completions"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!("data: {chunk}\n\ndata: [DONE]\n\n")),
        )
        .mount(&server)
        .await;
    let base_url = server.uri();
    let TestCodex { codex, .. } = test_codex()
        .with_config(move |config| {
            config.model_provider.base_url = Some(base_url);
            config.model_provider.wire_api = WireApi::CodebuddyChat;
            config.model = Some("deepseek-v4.1-flash".into());
            config.model_provider.supports_websockets = false;
            config.model_provider.stream_max_retries = Some(2);
            config.model_reasoning_effort =
                Some(codex_protocol::openai_models::ReasoningEffort::High);
        })
        .build_with_auto_env(&server)
        .await?;
    codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "Complete this task".into(),
            text_elements: Vec::new(),
        }]))
        .await?;
    let mut output_tokens = None;
    loop {
        match codex.next_event().await?.msg {
            EventMsg::TokenCount(event) => {
                if let Some(info) = event.info {
                    output_tokens = Some(info.total_token_usage.output_tokens);
                }
            }
            EventMsg::TurnComplete(event) => {
                let error = event.error.expect("length must not complete successfully");
                assert!(
                    error.message.contains("finish_reason=length"),
                    "unexpected failure: {error:?}"
                );
                break;
            }
            _ => {}
        }
    }
    assert_eq!(output_tokens, Some(96000));
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 3);
    for (index, request) in requests.iter().enumerate() {
        let body: serde_json::Value = request.body_json()?;
        assert_eq!(body["reasoning_effort"], "high");
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(
            messages
                .iter()
                .filter(|m| m["content"]
                    .as_str()
                    .is_some_and(|s| s.contains("<output_limit_recovery>")))
                .count(),
            index
        );
        assert_eq!(
            messages
                .iter()
                .filter(|m| m["reasoning_content"] == "Partial plan")
                .count(),
            index
        );
    }
    Ok(())
}

#[test_case::test_case("interleaved", 5, 96010; "tool_success_does_not_reset_recovery_cap")]
#[test_case::test_case("tools", 3, 32010; "recovers_with_complete_tool_call")]
#[test_case::test_case("success", 2, 32005; "continues_after_length")]
#[test_case::test_case("provider_error", 1, 7; "business_error_is_terminal")]
#[test_case::test_case("cancel", 2, 32000; "cancel_stops_continuation")]
#[test_case::test_case("budget", 1, 32000; "budget_prevents_continuation")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn codebuddy_bounded_recovery(
    scenario: &'static str,
    expected_requests: usize,
    expected_output_tokens: i64,
) -> anyhow::Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let continuation_started = std::sync::Arc::new(tokio::sync::Notify::new());
    let notify = continuation_started.clone();
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/chat/completions"))
        .respond_with(move |request: &wiremock::Request| {
            let attempt = calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let chunk = if (scenario == "tools" && attempt < 2) || scenario == "interleaved" {
                let body: serde_json::Value = request.body_json().unwrap();
                let name = body["tools"].as_array().unwrap().iter()
                    .find_map(|tool| tool["function"]["name"].as_str().filter(|name| name.ends_with("update_plan"))).unwrap();
                let arguments = if attempt % 2 == 0 { "{".to_string() } else {
                    serde_json::json!({"plan":[{"step":"Recovered tool step", "status":"completed"}]}).to_string()
                };
                let finish = if attempt % 2 == 0 { "length" } else { "tool_calls" };
                let tokens = if attempt % 2 == 0 { 32000 } else { 5 };
                serde_json::json!({"id":format!("tool-{attempt}"),"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":format!("call-{attempt}"),"function":{"name":name,"arguments":arguments}}]},"finish_reason":finish}],"usage":{"prompt_tokens":10,"completion_tokens":tokens,"total_tokens":10+tokens}})
            } else if scenario == "provider_error" {
                serde_json::json!({"error":{"code":"blocked","message":"provider refused"},"usage":{"prompt_tokens":10,"completion_tokens":7,"total_tokens":17}})
            } else if attempt == 0 {
                serde_json::json!({"id":"limited","choices":[{"index":0,"delta":{"content":"Partial answer","reasoning_content":"Partial plan"},"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":32000,"total_tokens":32010}})
            } else {
                serde_json::json!({"id":"finished","choices":[{"index":0,"delta":{"content":"Finished"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25}})
            };
            let response = wiremock::ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!("data: {chunk}\n\ndata: [DONE]\n\n"));
            if scenario == "cancel" && attempt > 0 {
                notify.notify_one();
                response.set_delay(std::time::Duration::from_secs(10))
            } else {
                response
            }
        })
        .mount(&server)
        .await;
    let base_url = server.uri();
    let TestCodex { codex, .. } = test_codex()
        .with_config(move |config| {
            config.update_plan_enabled = true;
            config.model_provider.base_url = Some(base_url);
            config.model_provider.wire_api = WireApi::CodebuddyChat;
            config.model = Some("deepseek-v4.1-flash".into());
            config.model_provider.supports_websockets = false;
            config.model_provider.stream_max_retries = Some(2);
            config.model_reasoning_effort =
                Some(codex_protocol::openai_models::ReasoningEffort::High);
            if scenario == "budget" {
                config.rollout_budget = Some(codex_core::config::RolloutBudgetConfig {
                    limit_tokens: 100,
                    reminder_at_remaining_tokens: vec![],
                    sampling_token_weight: 1.0,
                    prefill_token_weight: 1.0,
                });
            }
        })
        .build_with_auto_env(&server)
        .await?;
    codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "Complete this task".into(),
            text_elements: Vec::new(),
        }]))
        .await?;
    if scenario == "cancel" {
        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            continuation_started.notified(),
        )
        .await?;
        codex
            .submit(codex_protocol::protocol::Op::Interrupt)
            .await?;
    }
    let mut output_tokens = None;
    let mut plan_updates = 0;
    loop {
        match codex.next_event().await?.msg {
            EventMsg::TokenCount(event) => {
                if let Some(info) = event.info {
                    output_tokens = Some(info.total_token_usage.output_tokens);
                }
            }
            EventMsg::PlanUpdate(_) => plan_updates += 1,
            EventMsg::TurnAborted(_) if scenario == "cancel" => break,
            EventMsg::TurnComplete(event) => {
                if scenario == "success" || scenario == "tools" {
                    assert_eq!(event.error, None);
                } else {
                    let error = event.error.expect("terminal failure");
                    if scenario == "budget" {
                        assert_eq!(
                            error.codex_error_info,
                            Some(codex_protocol::protocol::CodexErrorInfo::SessionBudgetExceeded)
                        );
                    } else if scenario == "interleaved" {
                        assert!(error.message.contains("recovery exhausted"));
                    } else {
                        assert!(error.message.contains("provider refused"));
                    }
                }
                break;
            }
            _ => {}
        }
    }
    assert_eq!(
        plan_updates,
        if scenario == "interleaved" {
            2
        } else {
            usize::from(scenario == "tools")
        }
    );
    assert_eq!(output_tokens, Some(expected_output_tokens));
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), expected_requests);
    if scenario == "success" {
        let first: serde_json::Value = requests[0].body_json()?;
        let second: serde_json::Value = requests[1].body_json()?;
        for key in ["model", "reasoning_effort", "tools", "max_tokens"] {
            assert_eq!(&first[key], &second[key]);
        }
        let messages = second["messages"].as_array().unwrap();
        assert!(
            messages
                .iter()
                .any(|m| m["content"] == "Partial answer"
                    && m["reasoning_content"] == "Partial plan")
        );
        assert!(messages.iter().any(|m| {
            m["content"]
                .as_str()
                .is_some_and(|s| s.contains("<output_limit_recovery>"))
        }));
    }
    Ok(())
}
