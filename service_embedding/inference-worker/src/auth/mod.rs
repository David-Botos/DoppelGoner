// inference_worker/src/auth/mod.rs
use axum::{
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

use crate::api::routes::AppState;

pub struct ApiKeyAuth;

impl ApiKeyAuth {
    pub async fn check_api_key<B>(
        State(state): State<Arc<AppState>>,
        req: Request<B>,
        next: Next<B>,
    ) -> Result<Response, StatusCode> {
        // Skip auth for health check endpoint
        if req.uri().path() == "/api/health" || req.uri().path() == "/metrics" {
            return Ok(next.run(req).await);
        }

        // Get API key from header
        let api_key = req
            .headers()
            .get("X-API-Key")
            .and_then(|value| value.to_str().ok());

        // Check if API key is valid
        match api_key {
            Some(key) if key == state.config.api_key => Ok(next.run(req).await),
            _ => {
                tracing::warn!("Unauthorized access attempt: missing or invalid API key");
                Err(StatusCode::UNAUTHORIZED)
            }
        }
    }
}
