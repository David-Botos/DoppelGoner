// orchestrator/src/discovery/mod.rs
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post, put},
    Json, Router,
};
use serde::Deserialize;
use std::sync::Arc;
use tracing::{error, info};

use crate::client::worker::WorkerClient;
use crate::db;
use crate::types::types::{Worker, WorkerCapabilities, WorkerStatus, WorkerType};

// Application state
pub struct AppState {
    pub worker_client: WorkerClient,
    pub pool: sqlx::PgPool,
}

// Worker registration request
#[derive(Deserialize)]
pub struct WorkerRegistrationRequest {
    pub worker_id: String,
    pub hostname: String,
    pub ip_address: Option<String>,
    pub worker_type: String,
    pub capabilities: WorkerCapabilities,
    pub status: String,
}

// Worker status update request
#[derive(Deserialize)]
pub struct WorkerStatusUpdateRequest {
    pub status: String,
    pub timestamp: u64,
}

// Create routes for worker discovery and management
pub fn create_worker_routes(app_state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/workers", get(get_workers))
        .route("/api/workers/register", post(register_worker))
        .route("/api/workers/:id", get(get_worker))
        .route("/api/workers/:id/status", put(update_worker_status))
        .route("/api/workers/:id/reset", post(reset_worker_circuit_breaker))
        .with_state(app_state)
}

// Get all workers
async fn get_workers(State(state): State<Arc<AppState>>) -> Json<Vec<Worker>> {
    let workers = state.worker_client.get_all_workers().await;
    Json(workers)
}

// Get a specific worker
async fn get_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<String>,
) -> Result<Json<Worker>, StatusCode> {
    let worker = state.worker_client.registry.get_worker(&worker_id).await;

    match worker {
        Some(worker) => Ok(Json(worker)),
        None => Err(StatusCode::NOT_FOUND),
    }
}

// Register a worker
async fn register_worker(
    State(state): State<Arc<AppState>>,
    Json(registration): Json<WorkerRegistrationRequest>,
) -> Result<StatusCode, StatusCode> {
    // Parse worker status
    let status = match registration.status.as_str() {
        "online" => WorkerStatus::Online,
        "offline" => WorkerStatus::Offline,
        "busy" => WorkerStatus::Busy,
        _ => return Err(StatusCode::BAD_REQUEST),
    };

    // Parse worker type
    let worker_type = match registration.worker_type.as_str() {
        "inference" => WorkerType::Inference,
        "orchestrator" => WorkerType::Orchestrator,
        _ => return Err(StatusCode::BAD_REQUEST),
    };

    // Get original IP from request headers if available
    // This helps with container networking where the hostname might be different
    // from what the worker thinks it is

    // Create worker record
    let worker = Worker {
        id: registration.worker_id,
        hostname: registration.hostname,
        ip_address: registration.ip_address, // Use provided IP or container hostname
        worker_type,
        capabilities: registration.capabilities,
        status,
        last_heartbeat: std::time::SystemTime::now(),
        current_batch_size: None,
        current_load: None,
        active_jobs: 0,
        created_at: std::time::SystemTime::now(),
    };

    // Register with database
    match db::register_worker(
        &state.pool,
        &worker.id,
        &worker.hostname,
        worker.ip_address.as_deref(),
        worker.worker_type,
        &worker.capabilities,
    )
    .await
    {
        Ok(_) => {
            // Register with in-memory registry
            state.worker_client.registry.register_worker(worker).await;
            Ok(StatusCode::OK)
        }
        Err(e) => {
            error!("Failed to register worker in database: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

// Update worker status
async fn update_worker_status(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<String>,
    Json(update): Json<WorkerStatusUpdateRequest>,
) -> Result<StatusCode, StatusCode> {
    // Parse status
    let status = match update.status.as_str() {
        "online" => WorkerStatus::Online,
        "offline" => WorkerStatus::Offline,
        "busy" => WorkerStatus::Busy,
        _ => return Err(StatusCode::BAD_REQUEST),
    };

    // Update status in database
    match db::update_worker_status(&state.pool, &worker_id, status).await {
        Ok(_) => {
            // Update status in memory
            if let Err(_) = state
                .worker_client
                .registry
                .update_worker_status(&worker_id, status)
                .await
            {
                // Worker not in registry, try to fetch from database
                if let Ok(Some(worker)) = db::get_worker_by_id(&state.pool, &worker_id).await {
                    state.worker_client.registry.register_worker(worker).await;
                }
            }

            Ok(StatusCode::OK)
        }
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

// Reset circuit breaker for a worker
async fn reset_worker_circuit_breaker(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<String>,
) -> impl IntoResponse {
    state
        .worker_client
        .registry
        .reset_circuit_breaker(&worker_id)
        .await;
    info!("Reset circuit breaker for worker: {}", worker_id);
    StatusCode::OK
}
