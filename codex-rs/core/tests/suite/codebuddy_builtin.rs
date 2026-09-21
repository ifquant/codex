use codex_core::TurnInputRequest;
use codex_features::Feature;
use codex_login::CodexAuth;
use codex_model_provider_info::CODEBUDDY_PROVIDER_ID;
use codex_models_manager::bundled_models_response;
use codex_models_manager::manager::RefreshStrategy;
use codex_protocol::openai_models::ModelVisibility;
use codex_protocol::openai_models::ModelsResponse;
use codex_protocol::protocol::EventMsg;
use codex_protocol::user_input::UserInput;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;
use wiremock::Mock;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

#[test_case::test_case(false; "v1")]
#[test_case::test_case(true; "v2")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn codebuddy_worker_routes_without_profile_or_role_config(
    use_v2: bool,
) -> anyhow::Result<()> {
    skip_if_no_network!(Ok(()));
    let parent = responses::start_mock_server().await;
    let child = responses::start_mock_server().await;
    let mut parent_model = bundled_models_response()?
        .models
        .into_iter()
        .find(|model| model.slug == "gpt-5.5")
        .unwrap();
    parent_model.visibility = ModelVisibility::List;
    parent_model.supports_search_tool = false;
    let parent_catalog = vec![parent_model];
    let models_mock = responses::mount_models_once(
        &parent,
        ModelsResponse {
            models: parent_catalog.clone(),
        },
    )
    .await;
    let received = Arc::new(Notify::new());
    let notify = received.clone();
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(move |_request: &wiremock::Request| {
            if use_v2 {
                let body: serde_json::Value = _request.body_json().unwrap();
                let messages = body["messages"].as_array().unwrap();
                if !messages.iter().any(|message| message["role"] == "tool") {
                    let names = body["tools"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|tool| tool["function"]["name"].as_str().unwrap())
                        .collect::<Vec<_>>();
                    let send_message = names
                        .iter()
                        .find(|name| name.ends_with("send_message"))
                        .unwrap();
                    assert!(!names.iter().any(|name| name.ends_with("spawn_agent")));
                    let chunk = json!({"id":"child", "choices":[{"index":0,
                        "delta":{"tool_calls":[{"index":0,"id":"progress","type":"function",
                            "function":{"name":send_message,"arguments":
                                "{\"target\":\"/root\",\"message\":\"worker progress\"}"}}]},
                        "finish_reason":"tool_calls"}]});
                    return ResponseTemplate::new(200)
                        .insert_header("content-type", "text/event-stream")
                        .set_body_string(format!("data: {chunk}\n\ndata: [DONE]\n\n"));
                }
                let result = messages
                    .iter()
                    .find(|message| message["role"] == "tool")
                    .unwrap();
                assert_eq!(result["tool_call_id"], "progress");
                assert_eq!(result["content"], "");
            }
            notify.notify_one();
            let chunk = json!({"id":"child", "choices":[{"index":0,
                "delta":{"content":"worker done"}, "finish_reason":"stop"}]});
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!("data: {chunk}\n\ndata: [DONE]\n\n"))
        })
        .expect(if use_v2 { 2 } else { 1 })
        .mount(&child)
        .await;
    let namespace = if use_v2 {
        "external_agents"
    } else {
        "multi_agent_v1"
    };
    let mut spawn_args = json!({"agent_type":"codebuddy_worker", "message":"Reply worker done."});
    if use_v2 {
        spawn_args["task_name"] = json!("worker");
        spawn_args["fork_turns"] = json!("none");
    }
    let mut parent_responses = vec![
        responses::sse(vec![
            responses::ev_response_created("parent-1"),
            responses::ev_function_call_with_namespace(
                "spawn",
                namespace,
                "spawn_agent",
                &spawn_args.to_string(),
            ),
            responses::ev_completed("parent-1"),
        ]),
        responses::sse(vec![
            responses::ev_response_created("parent-2"),
            responses::ev_assistant_message("parent-message", "Delegated."),
            responses::ev_completed("parent-2"),
        ]),
    ];
    if use_v2 {
        // The same task name must remain available after rejecting an encrypted spawn.
        parent_responses.insert(
            0,
            responses::sse(vec![
                responses::ev_response_created("encrypted-parent"),
                responses::ev_function_call_with_namespace(
                    "encrypted-spawn",
                    "collaboration",
                    "spawn_agent",
                    &spawn_args.to_string(),
                ),
                responses::ev_completed("encrypted-parent"),
            ]),
        );
    }
    let parent_mock = responses::mount_sse_sequence(&parent, parent_responses).await;
    let child_url = child.uri();
    let test = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_model("gpt-5.5")
        .with_config(move |config| {
            assert!(!config.agent_roles.contains_key("codebuddy_worker"));
            config.features.enable(Feature::Collab).unwrap();
            config
                .features
                .enable(Feature::ApiKeyModelDiscovery)
                .unwrap();
            if use_v2 {
                config.features.enable(Feature::MultiAgentV2).unwrap();
            } else {
                config.features.disable(Feature::MultiAgentV2).unwrap();
            }
            let provider = config
                .model_providers
                .get_mut(CODEBUDDY_PROVIDER_ID)
                .unwrap();
            // Route only the built-in endpoint/auth to the local fixture.
            provider.base_url = Some(child_url);
            provider.env_key = None;
        })
        .build_with_auto_env(&parent)
        .await?;
    // Reproduce an authoritative remote refresh which excludes the bundled worker model.
    test.thread_manager
        .get_models_manager()
        .raw_model_catalog(
            RefreshStrategy::Online,
            codex_core::test_support::default_http_client_factory(),
        )
        .await;
    assert_eq!(models_mock.requests().len(), 1);
    assert_eq!(
        test.thread_manager
            .get_models_manager()
            .get_remote_models()
            .await,
        parent_catalog
    );
    test.codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "Delegate to codebuddy_worker.".into(),
            text_elements: vec![],
        }]))
        .await?;
    let EventMsg::TurnComplete(completed) = wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await
    else {
        unreachable!();
    };
    assert_eq!(completed.error, None);
    tokio::time::timeout(Duration::from_secs(10), received.notified()).await?;
    if use_v2 {
        let parent_requests = parent_mock.requests();
        let rejection = parent_requests[1].function_call_output("encrypted-spawn");
        let error = rejection["output"].as_str().unwrap();
        assert!(error.contains("No child was created"), "{error}");
        assert!(error.contains("external_agents.spawn_agent"), "{error}");
        assert!(error.contains("fork_turns=\"none\""), "{error}");
        // Only the parent and the successful plaintext child exist.
        assert_eq!(test.thread_manager.list_thread_ids().await.len(), 2);
    }
    let requests = child.received_requests().await.unwrap();
    let body: serde_json::Value = requests[0].body_json()?;
    assert_eq!(
        (body["model"].clone(), body["reasoning_effort"].clone()),
        (json!("deepseek-v4.1-flash"), json!("high"))
    );
    assert!(
        body["messages"].as_array().unwrap().iter().any(|message| {
            message["role"] == "system"
                && message["content"].as_str().is_some_and(|content| {
                    content.contains(
                        "You are Codex, a coding agent based on DeepSeek V4.1 Flash (CodeBuddy).",
                    )
                })
        }),
        "worker must retain its provider's model instructions after the parent's catalog refresh"
    );
    let parent_body = parent_mock.requests()[0].body_json();
    let tools = parent_body["tools"].as_array().unwrap();
    let namespace_spec = tools.iter().find(|tool| tool["name"] == namespace).unwrap();
    let spawn = namespace_spec["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tool| tool["name"] == "spawn_agent")
        .unwrap();
    let properties = &spawn["parameters"]["properties"];
    assert!(
        properties["agent_type"]["description"]
            .as_str()
            .unwrap()
            .contains("codebuddy_worker")
    );
    assert!(properties["message"]["encrypted"].is_null());
    Ok(())
}
