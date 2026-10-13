use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use serde::Deserialize;

use crate::app::AppState;
use crate::config::get_port;
use crate::services::anthropic::exchange_claude_code;
use crate::services::auth::exchange_google_code;
use crate::services::codex::{exchange_codex_code, get_stored_codex_verifier};

#[derive(Debug, Deserialize)]
pub struct OauthCallbackQuery {
    pub code: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct StatefulOauthCallbackQuery {
    pub code: Option<String>,
    pub state: Option<String>,
    pub error: Option<String>,
}

pub async fn oauth_callback_handler(
    State(state): State<AppState>,
    Query(query): Query<OauthCallbackQuery>,
) -> Response {
    if let Some(err) = query.error {
        return (
            StatusCode::BAD_REQUEST,
            Html(format!("<h1>Authentication Failed</h1><p>{err}</p>")),
        )
            .into_response();
    }

    let Some(code) = query.code else {
        return (StatusCode::BAD_REQUEST, Html("<h1>Missing authorization code</h1>")).into_response();
    };

    match exchange_google_code(&code, Some(&state.http_client)).await {
        Ok(account_id) => {
            let html = format!(
                "<html><body><h1>Authentication Successful</h1><p>Logged in as {account_id}. You can close this tab.</p></body></html>"
            );
            Html(html).into_response()
        }
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Html(format!("<h1>Token Exchange Failed</h1><p>{err}</p>")),
        )
            .into_response(),
    }
}

pub async fn claude_callback_handler(
    State(state): State<AppState>,
    Query(query): Query<StatefulOauthCallbackQuery>,
) -> Response {
    if let Some(err) = query.error {
        return (
            StatusCode::BAD_REQUEST,
            Html(format!("<h1>Authentication Failed</h1><p>{err}</p>")),
        )
            .into_response();
    }

    let Some(code) = query.code else {
        return (StatusCode::BAD_REQUEST, Html("<h1>Missing authorization code</h1>")).into_response();
    };

    match exchange_claude_code(&code, query.state.as_deref(), Some(&state.http_client)).await {
        Ok(()) => {
            let html =
                "<html><body><h1>Claude OAuth Login Successful!</h1><p>You can close this tab.</p></body></html>";
            Html(html).into_response()
        }
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Html(format!("<h1>Claude Token Exchange Failed</h1><p>{err}</p>")),
        )
            .into_response(),
    }
}

pub async fn codex_callback_handler(
    State(state): State<AppState>,
    Query(query): Query<StatefulOauthCallbackQuery>,
) -> Response {
    if let Some(err) = query.error {
        return (
            StatusCode::BAD_REQUEST,
            Html(format!("<h1>Authentication Failed</h1><p>{err}</p>")),
        )
            .into_response();
    }

    let Some(code) = query.code else {
        return (StatusCode::BAD_REQUEST, Html("<h1>Missing authorization code</h1>")).into_response();
    };

    let Some(verifier) = get_stored_codex_verifier(query.state.as_deref()) else {
        return (
            StatusCode::BAD_REQUEST,
            Html("<h1>Session expired or code verifier not found</h1>"),
        )
            .into_response();
    };

    let port = get_port();
    match exchange_codex_code(&code, &verifier, port, Some(&state.http_client)).await {
        Ok(()) => {
            let html = "<html><body><h1>Codex OAuth Login Successful!</h1><p>You can close this tab.</p></body></html>";
            Html(html).into_response()
        }
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Html(format!("<h1>Codex Token Exchange Failed</h1><p>{err}</p>")),
        )
            .into_response(),
    }
}
