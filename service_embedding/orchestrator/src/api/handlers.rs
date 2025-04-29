// orchestrator/src/api/handlers.rs

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;

use crate::api::response::{api_error, api_success, ApiResponse, ApiResult};
use crate::orchestrator::{OrchestratorService, PipelineStats, WorkerManager, WorkerManagerStatus};
use crate::types::types::Worker;

// Application state
pub struct AppState {
    pub orchestrator: Arc<OrchestratorService>,
    pub worker_manager: Arc<WorkerManager>,
}

// Query parameters
#[derive(Debug, Deserialize)]
pub struct PipelineParams {
    pub model_id: Option<String>,
    pub batch_size: Option<i32>,
}

// Status response
#[derive(Serialize)]
pub struct StatusResponse {
    pub worker_status: String,
    pub stats: PipelineStats,
    pub active_workers: usize,
    pub queue_depth: i32,
}

// Create API routes for the orchestrator
pub fn create_api_routes(app_state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/status", get(get_status))
        .route("/api/pipeline/run", post(run_pipeline))
        .route("/api/pipeline/pause", post(pause_pipeline))
        .route("/api/pipeline/resume", post(resume_pipeline))
        .route("/api/stats", get(get_stats))
        .with_state(app_state)
}

// Get current orchestrator status
async fn get_status(State(state): State<Arc<AppState>>) -> Json<ApiResponse<StatusResponse>> {
    let status = state.worker_manager.get_status().await;
    let stats = match state.worker_manager.get_stats().await {
        Ok(stats) => stats,
        Err(_) => PipelineStats::default(),
    };

    // Get active workers
    let workers = state
        .orchestrator
        .get_all_workers()
        .await
        .unwrap_or_default();
    let active_workers = workers
        .iter()
        .filter(|w| w.status != crate::types::types::WorkerStatus::Offline)
        .count();

    // Estimate queue depth (jobs waiting to be processed)
    let queue_depth = workers.iter().map(|w| w.active_jobs).sum();

    let status_str = match status {
        WorkerManagerStatus::Running => "running",
        WorkerManagerStatus::Paused => "paused",
        WorkerManagerStatus::ShuttingDown => "shutting_down",
        WorkerManagerStatus::ShutDown => "shut_down",
    };

    let response = StatusResponse {
        worker_status: status_str.to_string(),
        stats,
        active_workers,
        queue_depth,
    };

    Json(ApiResponse::success(response))
}

// Run the pipeline once
async fn run_pipeline(
    State(state): State<Arc<AppState>>,
    Query(params): Query<PipelineParams>,
) -> ApiResult<PipelineStats> {
    // Override model_id if provided
    if let Some(model_id) = params.model_id {
        // In a full implementation, we'd update the orchestrator config here
        info!("Using model_id override: {}", model_id);
    }

    // Override batch_size if provided
    if let Some(batch_size) = params.batch_size {
        // In a full implementation, we'd update the orchestrator config here
        info!("Using batch_size override: {}", batch_size);
    }

    // Run the pipeline
    match state.worker_manager.run_once().await {
        Ok(stats) => api_success(stats),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

// Pause the pipeline
async fn pause_pipeline(State(state): State<Arc<AppState>>) -> ApiResult<String> {
    match state.worker_manager.pause().await {
        Ok(_) => api_success("Pipeline paused".to_string()),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

// Resume the pipeline
async fn resume_pipeline(State(state): State<Arc<AppState>>) -> ApiResult<String> {
    match state.worker_manager.resume().await {
        Ok(_) => api_success("Pipeline resumed".to_string()),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

// Get current stats
async fn get_stats(State(state): State<Arc<AppState>>) -> ApiResult<PipelineStats> {
    match state.worker_manager.get_stats().await {
        Ok(stats) => api_success(stats),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}
