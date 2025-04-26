// orchestrator/src/discovery/mod.rs
use std::sync::Arc;
use std::time::{SystemTime, Duration};
use tokio::task::JoinHandle;
use tracing::{info, warn, error};
use axum::{
    routing::{get, post, put},
    Router, Json, extract::{State, Path},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Deserialize;

use crate::client::worker::{WorkerClient, OrchestratorService};
use crate::types::types::{Worker, WorkerStatus, WorkerType, WorkerCapabilities};

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
async fn get_workers(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<Worker>> {
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
    
    // Create worker record
    let worker = Worker {
        id: registration.worker_id,
        hostname: registration.hostname,
        ip_address: registration.ip_address,
        worker_type: match registration.worker_type.as_str() {
            "inference" => WorkerType::Inference,
            "orchestrator" => WorkerType::Orchestrator,
            _ => return Err(StatusCode::BAD_REQUEST),
        },
        capabilities: registration.capabilities,
        status,
        last_heartbeat: SystemTime::now(),
        current_batch_size: None,
        current_load: None,
        active_jobs: 0,
        created_at: SystemTime::now(),
    };
    
    // Register with database
    match sqlx::query!(
        r#"
        SELECT * FROM embedding.register_worker(
            $1, $2, $3, $4::embedding.worker_type, $5
        )
        "#,
        worker.id,
        worker.hostname,
        worker.ip_address,
        worker.worker_type.to_string(),
        serde_json::to_value(&worker.capabilities).unwrap(),
    )
    .fetch_one(&state.pool)
    .await {
        Ok(_) => {
            // Register with in-memory registry
            state.worker_client.registry.register_worker(worker).await;
            Ok(StatusCode::OK)
        },
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
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
    match sqlx::query!(
        r#"
        UPDATE embedding.workers
        SET 
            status = $1::embedding.worker_status,
            last_heartbeat = NOW()
        WHERE id = $2
        "#,
        status.to_string(),
        worker_id,
    )
    .execute(&state.pool)
    .await {
        Ok(_) => {
            // Update status in memory
            if let Err(_) = state.worker_client.registry.update_worker_status(&worker_id, status).await {
                // Worker not in registry, try to fetch from database
                let worker = sqlx::query_as!(
                    Worker,
                    r#"
                    SELECT 
                        id, 
                        hostname, 
                        ip_address, 
                        worker_type as "worker_type: _", 
                        capabilities as "capabilities: sqlx::types::Json<WorkerCapabilities>", 
                        status as "status: _", 
                        last_heartbeat, 
                        current_batch_size, 
                        current_load, 
                        active_jobs, 
                        created_at
                    FROM embedding.workers
                    WHERE id = $1
                    "#,
                    worker_id
                )
                .fetch_optional(&state.pool)
                .await;
                
                if let Ok(Some(w)) = worker {
                    let worker = Worker {
                        capabilities: w.capabilities.0,
                        ..w
                    };
                    state.worker_client.registry.register_worker(worker).await;
                }
            }
            
            Ok(StatusCode::OK)
        },
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

// Reset circuit breaker for a worker
async fn reset_worker_circuit_breaker(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<String>,
) -> Result<StatusCode, StatusCode> {
    match state.worker_client.reset_worker_circuit_breaker(&worker_id).await {
        Ok(_) => Ok(StatusCode::OK),
        Err(_) => Err(StatusCode::NOT_FOUND),
    }
}

// Start a background task to run the embedding pipeline
pub fn start_embedding_pipeline(
    orchestrator: Arc<OrchestratorService>,
    batch_size: i32,
    model_id: String,
    interval_secs: u64,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
        
        loop {
            interval.tick().await;
            
            // Reset stale jobs
            if let Err(e) = orchestrator.reset_stale_jobs(30).await {
                error!("Failed to reset stale jobs: {}", e);
            }
            
            // Run embedding pipeline
            if let Err(e) = orchestrator.run_embedding_pipeline(batch_size, &model_id).await {
                error!("Embedding pipeline failed: {}", e);
            }
        }
    })
}