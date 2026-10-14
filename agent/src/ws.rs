use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message as AxumMessage, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use axum::routing::get;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{Mutex, mpsc, watch};

use crate::attachments::{PromptInput, build_run_prompt, save_attachment, select_attachments_to_save};
use crate::r#loop::{AgentConfig, AgentEvent, Loopback, Message, run_agent};
use crate::tools::ToolCtx;

const MALFORMED_PAYLOAD_ERR: &str = "Unknown message type or malformed payload";

#[derive(Clone)]
struct WsState {
    port: u16,
    sandbox_root: PathBuf,
    client: reqwest::Client,
}

struct ActiveRun {
    abort_tx: watch::Sender<bool>,
    steer_tx: mpsc::Sender<String>,
    task: tokio::task::JoinHandle<()>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StartMessage {
    model: Option<String>,
    history: Option<Vec<Message>>,
    prompt: Option<PromptInput>,
    #[serde(alias = "message_id")]
    message_id: Option<String>,
    thinking: Option<bool>,
    effort: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SteerMessage {
    prompt: String,
}

pub fn agent_router(port: u16) -> Router {
    agent_router_with_root(port, std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

pub fn agent_router_with_root(port: u16, sandbox_root: PathBuf) -> Router {
    let state = Arc::new(WsState {
        port,
        sandbox_root,
        client: reqwest::Client::new(),
    });
    Router::new().route("/api/chat", get(ws_handler)).with_state(state)
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<WsState>>) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn send_frame(tx: &mpsc::Sender<AxumMessage>, val: serde_json::Value) {
    let _ = tx.send(AxumMessage::Text(val.to_string().into())).await;
}

async fn send_error(tx: &mpsc::Sender<AxumMessage>, msg: &str) {
    send_frame(
        tx,
        json!({
            "type": "error",
            "message": msg
        }),
    )
    .await;
}

#[allow(clippy::too_many_lines)]
async fn handle_start(
    state: &Arc<WsState>,
    data: StartMessage,
    ws_out_tx: &mpsc::Sender<AxumMessage>,
    saved_history: &Arc<Mutex<Vec<Message>>>,
) -> Option<ActiveRun> {
    let model = match data.model.as_deref() {
        Some(m) if !m.trim().is_empty() => m.trim().to_string(),
        _ => "gemini-3-flash".to_string(),
    };

    let thinking = data.thinking.unwrap_or(false);
    let effort = match data.effort.as_deref() {
        Some(e) if !e.trim().is_empty() => e.trim().to_string(),
        _ => "medium".to_string(),
    };

    let cfg = AgentConfig {
        model,
        thinking,
        effort,
        max_turns: 25,
    };

    let client_message_id = match data.message_id {
        Some(id) if !id.trim().is_empty() => id,
        _ => format!(
            "msg_agent_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_millis())
        ),
    };

    let mut prompt_input = data.prompt.unwrap_or_default();
    let to_save = select_attachments_to_save(&prompt_input)
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();

    let history = match data.history {
        Some(h) => h,
        None => saved_history.lock().await.clone(),
    };

    let mut save_err: Option<String> = None;
    for att in to_save {
        match save_attachment(&state.sandbox_root, &att).await {
            Ok(Some(saved_name)) => {
                if let Some(ref mut single) = prompt_input.attachment
                    && single.data == att.data
                {
                    single.name = Some(saved_name.clone());
                }
                for multi in &mut prompt_input.attachments {
                    if multi.data == att.data {
                        multi.name = Some(saved_name.clone());
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                save_err = Some(e.to_string());
                break;
            }
        }
    }

    if !prompt_input.attachments.is_empty() {
        prompt_input.attachment = prompt_input.attachments.first().cloned();
    }

    if let Some(err) = save_err {
        send_frame(
            ws_out_tx,
            json!({
                "type": "history",
                "messages": history,
            }),
        )
        .await;
        send_error(ws_out_tx, &err).await;
        return None;
    }

    let parts = build_run_prompt(&prompt_input);
    send_frame(
        ws_out_tx,
        json!({
            "type": "status",
            "running": true,
            "status": "Thinking..."
        }),
    )
    .await;

    let (events_tx, mut events_rx) = mpsc::channel::<AgentEvent>(100);
    let (abort_tx, abort_rx) = watch::channel(false);
    let (steer_tx, steer_rx) = mpsc::channel::<String>(10);

    let ctx = ToolCtx {
        sandbox_root: state.sandbox_root.clone(),
        http: state.client.clone(),
        ..Default::default()
    };
    let lb = Loopback {
        base_url: format!("http://127.0.0.1:{}", state.port),
        client: state.client.clone(),
    };

    let run_out_tx = ws_out_tx.clone();
    let mid = client_message_id;
    let saved_history_clone = Arc::clone(saved_history);

    let task = tokio::spawn(async move {
        let agent_task =
            tokio::spawn(
                async move { run_agent(&cfg, history, parts, &lb, &ctx, events_tx, abort_rx, steer_rx).await },
            );

        let mut started = false;
        let mut turn_num: u32 = 0;
        let mut total_content = String::new();
        let mut total_reasoning = String::new();

        while let Some(event) = events_rx.recv().await {
            if !started {
                started = true;
                send_frame(
                    &run_out_tx,
                    json!({
                        "type": "run_start",
                        "messageId": mid,
                        "isBackground": false,
                    }),
                )
                .await;
            }

            match event {
                AgentEvent::Delta { content, reasoning } => {
                    total_content.push_str(&content);
                    total_reasoning.push_str(&reasoning);
                    send_frame(
                        &run_out_tx,
                        json!({
                            "type": "delta",
                            "messageId": mid,
                            "content_delta": content,
                            "content": total_content,
                            "reasoning_delta": reasoning,
                            "reasoning": total_reasoning,
                        }),
                    )
                    .await;
                }
                AgentEvent::ToolStart { id, name, input } => {
                    send_frame(
                        &run_out_tx,
                        json!({
                            "type": "tool_start",
                            "messageId": mid,
                            "tool_call_id": id,
                            "name": name,
                            "input": input,
                        }),
                    )
                    .await;
                }
                AgentEvent::ToolEnd {
                    id,
                    name,
                    output,
                    error,
                    duration_ms,
                } => {
                    send_frame(
                        &run_out_tx,
                        json!({
                            "type": "tool_end",
                            "messageId": mid,
                            "tool_call_id": id,
                            "name": name,
                            "duration_ms": duration_ms,
                            "output": output,
                            "error": error,
                        }),
                    )
                    .await;
                }
                AgentEvent::TurnEnd { terminal } => {
                    turn_num += 1;
                    send_frame(
                        &run_out_tx,
                        json!({
                            "type": "turn_end",
                            "messageId": mid,
                            "turn": turn_num,
                            "terminal": terminal,
                        }),
                    )
                    .await;
                }
            }
        }

        match agent_task.await {
            Ok(final_history) => {
                *saved_history_clone.lock().await = final_history.clone();
                send_frame(
                    &run_out_tx,
                    json!({
                        "type": "history",
                        "messages": final_history,
                    }),
                )
                .await;
                send_frame(
                    &run_out_tx,
                    json!({
                        "type": "status",
                        "running": false,
                        "status": "Idle",
                    }),
                )
                .await;
            }
            Err(e) => {
                if !e.is_cancelled() {
                    send_frame(
                        &run_out_tx,
                        json!({
                            "type": "history",
                            "messages": saved_history_clone.lock().await.clone(),
                        }),
                    )
                    .await;
                    send_error(&run_out_tx, &e.to_string()).await;
                }
            }
        }
    });

    Some(ActiveRun {
        abort_tx,
        steer_tx,
        task,
    })
}

#[allow(clippy::too_many_lines)]
async fn handle_socket(socket: WebSocket, state: Arc<WsState>) {
    let (mut ws_sender, mut ws_receiver) = socket.split();
    let (ws_out_tx, mut ws_out_rx) = mpsc::channel::<AxumMessage>(128);

    let writer_task = tokio::spawn(async move {
        while let Some(msg) = ws_out_rx.recv().await {
            if ws_sender.send(msg).await.is_err() {
                break;
            }
        }
    });

    let saved_history = Arc::new(Mutex::new(Vec::<Message>::new()));
    let mut active_run: Option<ActiveRun> = None;

    while let Some(msg_res) = ws_receiver.next().await {
        let Ok(msg) = msg_res else { break };

        let text = match msg {
            AxumMessage::Text(t) => t,
            AxumMessage::Close(_) => break,
            AxumMessage::Ping(_) | AxumMessage::Pong(_) => continue,
            AxumMessage::Binary(_) => {
                send_error(&ws_out_tx, MALFORMED_PAYLOAD_ERR).await;
                continue;
            }
        };

        let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&text) else {
            send_error(&ws_out_tx, MALFORMED_PAYLOAD_ERR).await;
            continue;
        };

        let Some(msg_type) = json_val.get("type").and_then(serde_json::Value::as_str) else {
            send_error(&ws_out_tx, MALFORMED_PAYLOAD_ERR).await;
            continue;
        };

        match msg_type {
            "start" => {
                if let Some(active) = active_run.take() {
                    let _ = active.abort_tx.send(true);
                    active.task.abort();
                }

                let Ok(start_data) = serde_json::from_value::<StartMessage>(json_val) else {
                    send_error(&ws_out_tx, MALFORMED_PAYLOAD_ERR).await;
                    continue;
                };

                active_run = handle_start(&state, start_data, &ws_out_tx, &saved_history).await;
            }
            "steer" => {
                let Ok(steer_data) = serde_json::from_value::<SteerMessage>(json_val) else {
                    send_error(&ws_out_tx, MALFORMED_PAYLOAD_ERR).await;
                    continue;
                };
                if let Some(ref active) = active_run {
                    let _ = active.steer_tx.send(steer_data.prompt).await;
                }
            }
            "stop" => {
                if let Some(active) = active_run.take() {
                    let _ = active.abort_tx.send(true);
                }
                send_frame(
                    &ws_out_tx,
                    json!({
                        "type": "status",
                        "running": false,
                        "status": "Idle"
                    }),
                )
                .await;
            }
            "reset" => {
                if let Some(active) = active_run.take() {
                    let _ = active.abort_tx.send(true);
                    active.task.abort();
                }
                saved_history.lock().await.clear();
                send_frame(
                    &ws_out_tx,
                    json!({
                        "type": "status",
                        "running": false,
                        "status": "Idle"
                    }),
                )
                .await;
            }
            _ => {
                send_error(&ws_out_tx, MALFORMED_PAYLOAD_ERR).await;
            }
        }
    }

    if let Some(active) = active_run.take() {
        let _ = active.abort_tx.send(true);
        active.task.abort();
    }
    writer_task.abort();
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    async fn connect_test_ws() -> (
        tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
        u16,
    ) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = agent_router(port);
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let (socket, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/api/chat"))
            .await
            .unwrap();
        (socket, port)
    }

    async fn start_model_capture() -> (u16, tokio::sync::mpsc::Receiver<String>) {
        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(move |axum::Json(body): axum::Json<serde_json::Value>| {
                let tx = tx.clone();
                async move {
                    if let Some(m) = body.get("model").and_then(|v| v.as_str()) {
                        let _ = tx.send(m.to_string()).await;
                    }
                    axum::response::Response::builder()
                        .status(axum::http::StatusCode::OK)
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from("data: [DONE]\n\n"))
                        .unwrap()
                }
            }),
        );
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (port, rx)
    }

    async fn drive_start_without_model(loopback: u16) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let ws_port = listener.local_addr().unwrap().port();
        let app = agent_router(loopback);
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{ws_port}/api/chat"))
            .await
            .unwrap();
        socket
            .send(Message::Text(
                serde_json::json!({
                    "type": "start",
                    "prompt": { "text": "hello" }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
        tokio::spawn(async move {
            let mut s = socket;
            while let Some(Ok(_)) = s.next().await {}
        });
    }

    #[tokio::test]
    async fn test_unknown_input_yields_error_frame() {
        let (mut socket, _) = connect_test_ws().await;
        socket.send(Message::Text(r#"{"type":"bogus"}"#.into())).await.unwrap();
        let frame = socket.next().await.unwrap().unwrap().into_text().unwrap();
        assert!(frame.contains(r#""type":"error""#));
    }

    #[tokio::test]
    async fn test_missing_model_defaults_to_gemini_flash() {
        let (loopback, mut seen_model) = start_model_capture().await;
        drive_start_without_model(loopback).await;
        assert_eq!(seen_model.recv().await, Some("gemini-3-flash".to_string()));
    }

    #[tokio::test]
    async fn test_start_and_complete_run() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let loopback_port = listener.local_addr().unwrap().port();
        let mock_upstream = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(|| async {
                let stream_body = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"}}]}\n\n\
                     data: {\"choices\":[{\"delta\":{\"content\":\"world!\"},\"finish_reason\":\"stop\"}]}\n\n\
                     data: [DONE]\n\n";
                axum::response::Response::builder()
                    .status(axum::http::StatusCode::OK)
                    .header("content-type", "text/event-stream")
                    .body(axum::body::Body::from(stream_body))
                    .unwrap()
            }),
        );
        tokio::spawn(async move {
            axum::serve(listener, mock_upstream).await.unwrap();
        });

        let ws_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let ws_port = ws_listener.local_addr().unwrap().port();
        let app = agent_router(loopback_port);
        tokio::spawn(async move {
            axum::serve(ws_listener, app).await.unwrap();
        });

        let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{ws_port}/api/chat"))
            .await
            .unwrap();

        socket
            .send(Message::Text(
                serde_json::json!({
                    "type": "start",
                    "model": "gemini-3.8-flash",
                    "prompt": { "text": "Hi" },
                    "messageId": "msg_test_1"
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();

        // 1. status Thinking...
        let f1 = socket.next().await.unwrap().unwrap().into_text().unwrap();
        let v1: serde_json::Value = serde_json::from_str(&f1).unwrap();
        assert_eq!(v1["type"], "status");
        assert_eq!(v1["running"], true);

        // 2. run_start
        let f2 = socket.next().await.unwrap().unwrap().into_text().unwrap();
        let v2: serde_json::Value = serde_json::from_str(&f2).unwrap();
        assert_eq!(v2["type"], "run_start");
        assert_eq!(v2["messageId"], "msg_test_1");

        // 3. delta "Hello "
        let f3 = socket.next().await.unwrap().unwrap().into_text().unwrap();
        let v3: serde_json::Value = serde_json::from_str(&f3).unwrap();
        assert_eq!(v3["type"], "delta");
        assert_eq!(v3["content_delta"], "Hello ");
        assert_eq!(v3["content"], "Hello ");

        // 4. delta "world!"
        let f4 = socket.next().await.unwrap().unwrap().into_text().unwrap();
        let v4: serde_json::Value = serde_json::from_str(&f4).unwrap();
        assert_eq!(v4["type"], "delta");
        assert_eq!(v4["content_delta"], "world!");
        assert_eq!(v4["content"], "Hello world!");

        // 5. turn_end
        let f5 = socket.next().await.unwrap().unwrap().into_text().unwrap();
        let v5: serde_json::Value = serde_json::from_str(&f5).unwrap();
        assert_eq!(v5["type"], "turn_end");
        assert_eq!(v5["turn"], 1);
        assert_eq!(v5["terminal"], true);

        // 6. history
        let f6 = socket.next().await.unwrap().unwrap().into_text().unwrap();
        let v6: serde_json::Value = serde_json::from_str(&f6).unwrap();
        assert_eq!(v6["type"], "history");
        assert!(v6["messages"].as_array().is_some());

        // 7. status Idle
        let f7 = socket.next().await.unwrap().unwrap().into_text().unwrap();
        let v7: serde_json::Value = serde_json::from_str(&f7).unwrap();
        assert_eq!(v7["type"], "status");
        assert_eq!(v7["running"], false);
    }

    #[tokio::test]
    async fn test_stop_and_reset() {
        let (mut socket, _) = connect_test_ws().await;

        socket.send(Message::Text(r#"{"type":"stop"}"#.into())).await.unwrap();
        let f1 = socket.next().await.unwrap().unwrap().into_text().unwrap();
        let v1: serde_json::Value = serde_json::from_str(&f1).unwrap();
        assert_eq!(v1["type"], "status");
        assert_eq!(v1["running"], false);

        socket.send(Message::Text(r#"{"type":"reset"}"#.into())).await.unwrap();
        let f2 = socket.next().await.unwrap().unwrap().into_text().unwrap();
        let v2: serde_json::Value = serde_json::from_str(&f2).unwrap();
        assert_eq!(v2["type"], "status");
        assert_eq!(v2["running"], false);
    }

    #[tokio::test]
    async fn test_malformed_json_yields_error() {
        let (mut socket, _) = connect_test_ws().await;
        socket.send(Message::Text("not valid json".into())).await.unwrap();
        let frame = socket.next().await.unwrap().unwrap().into_text().unwrap();
        assert!(frame.contains(r#""type":"error""#));
    }

    #[tokio::test]
    async fn test_attachment_saved_and_injected() {
        let temp_dir = tempfile::tempdir().unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::channel(10);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let loopback_port = listener.local_addr().unwrap().port();
        let mock_upstream = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(move |axum::Json(body): axum::Json<serde_json::Value>| {
                let tx = tx.clone();
                async move {
                    let _ = tx.send(body).await;
                    let stream_body =
                        "data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\n\
                         data: [DONE]\n\n";
                    axum::response::Response::builder()
                        .status(axum::http::StatusCode::OK)
                        .header("content-type", "text/event-stream")
                        .body(axum::body::Body::from(stream_body))
                        .unwrap()
                }
            }),
        );
        tokio::spawn(async move {
            axum::serve(listener, mock_upstream).await.unwrap();
        });

        let ws_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let ws_port = ws_listener.local_addr().unwrap().port();
        let app = agent_router_with_root(loopback_port, temp_dir.path().to_path_buf());
        tokio::spawn(async move {
            axum::serve(ws_listener, app).await.unwrap();
        });

        let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{ws_port}/api/chat"))
            .await
            .unwrap();

        // data for "hello attachment" in base64 is "aGVsbG8gYXR0YWNobWVudA=="
        socket
            .send(Message::Text(
                serde_json::json!({
                    "type": "start",
                    "model": "gemini-3.8-flash",
                    "prompt": {
                        "text": "Read the file",
                        "attachment": {
                            "name": "notes.txt",
                            "mimeType": "text/plain",
                            "data": "aGVsbG8gYXR0YWNobWVudA=="
                        }
                    }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();

        let req_body = rx.recv().await.unwrap();
        let req_str = req_body.to_string();
        assert!(req_str.contains("Attached file \\\"notes.txt\\\" has been saved"));
        let saved_file = temp_dir.path().join("notes.txt");
        assert!(saved_file.exists());
        assert_eq!(std::fs::read_to_string(saved_file).unwrap(), "hello attachment");

        // Drain websocket frames
        while let Some(Ok(msg)) = socket.next().await {
            if let Ok(text) = msg.into_text()
                && text.contains(r#""type":"status""#)
                && text.contains(r#""running":false"#)
            {
                break;
            }
        }
    }
}
