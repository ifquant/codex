#![allow(clippy::expect_used)]
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use codex_api::AuthProvider;
use codex_api::Compression;
use codex_api::Provider;
use codex_api::ResponseEvent;
use codex_api::ResponsesClient;
use codex_client::HttpTransport;
use codex_client::Request;
use codex_client::Response;
use codex_client::StreamResponse;
use codex_client::TransportError;
use codex_protocol::models::ResponseItem;
use futures::StreamExt;
use http::HeaderMap;
use http::StatusCode;
use pretty_assertions::assert_eq;
use serde_json::Value;

#[derive(Clone)]
struct FixtureSseTransport {
    body: String,
}

impl FixtureSseTransport {
    fn new(body: String) -> Self {
        Self { body }
    }
}

impl HttpTransport for FixtureSseTransport {
    async fn execute(&self, _req: Request) -> Result<Response, TransportError> {
        Err(TransportError::Build("execute should not run".to_string()))
    }

    async fn stream(&self, _req: Request) -> Result<StreamResponse, TransportError> {
        let stream = futures::stream::iter(vec![Ok::<Bytes, TransportError>(Bytes::from(
            self.body.clone(),
        ))]);
        Ok(StreamResponse {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            bytes: Box::pin(stream),
        })
    }
}

#[derive(Clone, Default)]
struct NoAuth;

impl AuthProvider for NoAuth {
    fn add_auth_headers(&self, _headers: &mut HeaderMap) {}
}

fn provider(name: &str) -> Provider {
    Provider {
        name: name.to_string(),
        base_url: "https://example.com/v1".to_string(),
        query_params: None,
        headers: HeaderMap::new(),
        retry: codex_api::RetryConfig {
            max_attempts: 1,
            base_delay: Duration::from_millis(1),
            retry_429: false,
            retry_5xx: false,
            retry_transport: true,
        },
        stream_idle_timeout: Duration::from_millis(50),
    }
}

fn build_responses_body(events: Vec<Value>) -> String {
    let mut body = String::new();
    for e in events {
        let kind = e
            .get("type")
            .and_then(|v| v.as_str())
            .expect("SSE fixture event should have a type");
        if e.as_object().map(|o| o.len() == 1).unwrap_or(false) {
            body.push_str(&format!("event: {kind}\n\n"));
        } else {
            body.push_str(&format!("event: {kind}\ndata: {e}\n\n"));
        }
    }
    body
}

#[tokio::test]
async fn responses_stream_parses_items_and_completed_end_to_end() -> Result<()> {
    let item1 = serde_json::json!({
        "type": "response.output_item.done",
        "item": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello"}]
        }
    });

    let item2 = serde_json::json!({
        "type": "response.output_item.done",
        "item": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "World"}]
        }
    });

    let completed = serde_json::json!({
        "type": "response.completed",
        "response": {
            "id": "resp1",
            "usage_metadata": { "amount": "0.12345678901234567890" },
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "extra": { "label": "example", "items": [0, null, true] },
                "codex_rollout_budget_units": 2.5
            }
        }
    });

    let expected_metadata = completed["response"]["usage"].clone();
    let body = build_responses_body(vec![item1, item2, completed]);
    let transport = FixtureSseTransport::new(body);
    let client = ResponsesClient::new(transport, provider("openai"), Arc::new(NoAuth));

    let mut stream = client
        .stream(
            serde_json::json!({"echo": true}),
            HeaderMap::new(),
            Compression::None,
            /*turn_state*/ None,
        )
        .await?;

    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev?);
    }

    let events: Vec<ResponseEvent> = events
        .into_iter()
        .filter(|ev| !matches!(ev, ResponseEvent::RateLimits(_)))
        .collect();

    assert_eq!(events.len(), 3);

    match &events[0] {
        ResponseEvent::OutputItemDone(ResponseItem::Message { role, .. }) => {
            assert_eq!(role, "assistant");
        }
        other => panic!("unexpected first event: {other:?}"),
    }

    match &events[1] {
        ResponseEvent::OutputItemDone(ResponseItem::Message { role, .. }) => {
            assert_eq!(role, "assistant");
        }
        other => panic!("unexpected second event: {other:?}"),
    }

    match &events[2] {
        ResponseEvent::Completed {
            response_id,
            token_usage,
            usage_metadata,
            end_turn,
        } => {
            assert_eq!(response_id, "resp1");
            assert_eq!(
                usage_metadata,
                &Some(codex_protocol::ResponseUsageMetadata {
                    amount: Some("0.12345678901234567890".to_string()),
                    metadata: Some(expected_metadata),
                })
            );
            assert_eq!(
                token_usage.as_ref().map(|usage| usage.total_tokens),
                Some(15)
            );
            assert_eq!(
                token_usage
                    .as_ref()
                    .and_then(|usage| usage.codex_rollout_budget_units.as_ref())
                    .and_then(serde_json::Number::as_f64),
                Some(2.5)
            );
            assert!(end_turn.is_none());
        }
        other => panic!("unexpected third event: {other:?}"),
    }

    Ok(())
}

#[tokio::test]
async fn codebuddy_active_http_stream_survives_idle_window_and_cancellation_closes_socket()
-> Result<()> {
    use codex_api::ResponsesEndpoint;
    use codex_client::ReqwestTransport;
    use codex_http_client::HttpClientBuilder;
    use tokio::io::AsyncBufReadExt;
    use tokio::io::AsyncReadExt;
    use tokio::io::AsyncWriteExt;
    use tokio::io::BufReader;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await?;
        let mut reader = BufReader::new(socket);
        let mut line = String::new();
        reader.read_line(&mut line).await?;
        assert!(line.starts_with("POST /chat/completions "));
        let mut content_length = None;
        loop {
            line.clear();
            reader.read_line(&mut line).await?;
            if line == "\r\n" {
                break;
            }
            if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                content_length = Some(value.trim().parse::<usize>()?);
            }
        }
        let mut body = vec![0; content_length.expect("request must have a content length")];
        reader.read_exact(&mut body).await?;
        let body: Value = serde_json::from_slice(&body)?;
        assert_eq!(body["reasoning_effort"], "max");
        let mut socket = reader.into_inner();
        socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n").await?;
        for _ in 0..22 {
            let data = "data: {\"id\":\"live\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}\n\n";
            socket
                .write_all(format!("{:x}\r\n{data}\r\n", data.len()).as_bytes())
                .await?;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let mut byte = [0];
        assert_eq!(
            socket.read(&mut byte).await?,
            0,
            "cancel must close the real HTTP socket"
        );
        Ok::<_, anyhow::Error>(())
    });
    let transport = ReqwestTransport::from_http_client(HttpClientBuilder::new().build_direct()?);
    let mut provider = provider("codebuddy");
    provider.base_url = format!("http://{address}");
    provider.stream_idle_timeout = Duration::from_secs(2);
    let client = ResponsesClient::new(transport, provider, Arc::new(NoAuth))
        .with_endpoint(ResponsesEndpoint::CodebuddyChat);
    let mut stream = client.stream(
        serde_json::json!({"model":"deepseek-v4.1-flash","reasoning":{"effort":"max"},"input":[],"tools":[]}),
        HeaderMap::new(), Compression::None, /*turn_state*/ None,
    ).await?;
    let started = tokio::time::Instant::now();
    let mut deltas = 0;
    while let Some(event) = stream.next().await {
        if matches!(event?, ResponseEvent::OutputTextDelta(_)) {
            deltas += 1;
        }
        if deltas == 22 {
            break;
        }
    }
    assert_eq!(deltas, 22);
    assert!(started.elapsed() > Duration::from_secs(2));
    drop(stream);
    // The socket must close before the two-second SSE idle timeout can do it.
    tokio::time::timeout(Duration::from_secs(1), server).await???;
    Ok(())
}
