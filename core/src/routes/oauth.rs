use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use serde::Deserialize;

use crate::app::AppState;
use crate::services::auth::exchange_google_code;

#[derive(Debug, Deserialize)]
pub struct OauthCallbackQuery {
    pub code: Option<String>,
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
