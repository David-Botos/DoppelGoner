// orchestrator/src/api/response.rs

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

/// Standardized API response format
#[derive(Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message.into()),
        }
    }
}

/// Custom response type that handles both successful and error responses
pub enum ApiResult<T: Serialize> {
    Success(T),
    Error(StatusCode, String),
}

impl<T: Serialize> IntoResponse for ApiResult<T> {
    fn into_response(self) -> Response {
        match self {
            ApiResult::Success(data) => {
                (StatusCode::OK, Json(ApiResponse::success(data))).into_response()
            }
            ApiResult::Error(status, message) => {
                (status, Json(ApiResponse::<T>::error(message))).into_response()
            }
        }
    }
}

// Helper functions to create API results
pub fn api_success<T: Serialize>(data: T) -> ApiResult<T> {
    ApiResult::Success(data)
}

pub fn api_error<T: Serialize>(status: StatusCode, message: impl Into<String>) -> ApiResult<T> {
    ApiResult::Error(status, message.into())
}
