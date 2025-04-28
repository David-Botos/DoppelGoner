// inference_worker/src/api/routes.rs
use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use metrics::{counter, gauge, histogram};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::Mutex;
use tower::ServiceBuilder;
use tower_http::trace::TraceLayer;
use uuid::Uuid;

use crate::auth::ApiKeyAuth;
use crate::config::AppConfig;
use crate::inference::engine::InferenceEngine;
use crate::telemetry::gpu_metrics::GPUMetrics;
use crate::types::types::{
    BatchProcessRequest, BatchProcessResponse, BatchStatistics, PipelineMetric, PipelineStage,
    WorkerStatus, WorkerType,
};

// Query parameters for batch endpoints
#[derive(Debug, Deserialize)]
pub struct BatchQueryParams {
    pub priority: Option<i32>,
    pub model_id: Option<String>,
    pub timeout_ms: Option<u64>,
}

// Query parameters for status endpoints
#[derive(Debug, Deserialize)]
pub struct StatusQueryParams {
    pub include_metrics: Option<bool>,
    pub include_gpu_stats: Option<bool>,
}

// Cancel request
#[derive(Debug, Deserialize)]
pub struct CancelRequest {
    pub request_id: Uuid,
    pub reason: Option<String>,
}

// Registration request
#[derive(Debug, Deserialize)]
pub struct RegistrationRequest {
    pub orchestrator_url: String,
    pub api_key: Option<String>,
}

// Registration response
#[derive(Debug, Serialize)]
pub struct RegistrationResponse {
    pub worker_id: String,
    pub status: String,
    pub registered_at: SystemTime,
}

// Application state that will be shared across request handlers
pub struct AppState {
    pub inference_engine: Arc<Mutex<InferenceEngine>>,
    pub config: AppConfig,
    pub worker_id: String,
    pub start_time: SystemTime,
    pub stats: Arc<Mutex<WorkerStats>>,
    pub gpu_metrics: Arc<Mutex<GPUMetrics>>,
    pub orchestrator_url: Arc<Mutex<Option<String>>>,
    pub health_status: Arc<Mutex<bool>>,
}

// Statistics for the worker
pub struct WorkerStats {
    pub total_batches_processed: u64,
    pub total_documents_processed: u64,
    pub batch_processing_times_ms: Vec<f64>,
    pub document_processing_times_ms: Vec<f64>,
    pub peak_gpu_memory_used_mb: f64,
    pub gpu_utilization_samples: Vec<f64>,
    pub errors: u64,
    pub request_timeouts: u64,
    pub circuit_breaks: u64,
    pub max_concurrent_batches: u32,
    pub current_concurrent_batches: u32,
    pub last_metrics_update: SystemTime,
}

impl WorkerStats {
    pub fn new() -> Self {
        WorkerStats {
            total_batches_processed: 0,
            total_documents_processed: 0,
            batch_processing_times_ms: Vec::with_capacity(1000),
            document_processing_times_ms: Vec::with_capacity(1000),
            peak_gpu_memory_used_mb: 0.0,
            gpu_utilization_samples: Vec::with_capacity(1000),
            errors: 0,
            request_timeouts: 0,
            circuit_breaks: 0,
            max_concurrent_batches: 0,
            current_concurrent_batches: 0,
            last_metrics_update: SystemTime::now(),
        }
    }

    // Calculate average batch processing time
    pub fn avg_batch_time(&self) -> f64 {
        if self.batch_processing_times_ms.is_empty() {
            return 0.0;
        }
        self.batch_processing_times_ms.iter().sum::<f64>()
            / self.batch_processing_times_ms.len() as f64
    }

    // Calculate average document processing time
    pub fn avg_document_time(&self) -> f64 {
        if self.document_processing_times_ms.is_empty() {
            return 0.0;
        }
        self.document_processing_times_ms.iter().sum::<f64>()
            / self.document_processing_times_ms.len() as f64
    }

    // Calculate average GPU utilization
    pub fn avg_gpu_utilization(&self) -> f64 {
        if self.gpu_utilization_samples.is_empty() {
            return 0.0;
        }
        self.gpu_utilization_samples.iter().sum::<f64>() / self.gpu_utilization_samples.len() as f64
    }

    // Trim stats vectors if they get too large
    pub fn trim_stats(&mut self) {
        const MAX_SAMPLES: usize = 1000;

        if self.batch_processing_times_ms.len() > MAX_SAMPLES {
            self.batch_processing_times_ms = self
                .batch_processing_times_ms
                .split_off(self.batch_processing_times_ms.len() - MAX_SAMPLES);
        }

        if self.document_processing_times_ms.len() > MAX_SAMPLES {
            self.document_processing_times_ms = self
                .document_processing_times_ms
                .split_off(self.document_processing_times_ms.len() - MAX_SAMPLES);
        }

        if self.gpu_utilization_samples.len() > MAX_SAMPLES {
            self.gpu_utilization_samples = self
                .gpu_utilization_samples
                .split_off(self.gpu_utilization_samples.len() - MAX_SAMPLES);
        }
    }

    // Record batch processing metrics
    pub fn record_batch(
        &mut self,
        document_count: usize,
        processing_time_ms: f64,
        gpu_utilization: f64,
        gpu_memory_mb: f64,
    ) {
        self.total_batches_processed += 1;
        self.total_documents_processed += document_count as u64;
        self.batch_processing_times_ms.push(processing_time_ms);

        // Add individual document times
        let avg_doc_time = processing_time_ms / document_count as f64;
        for _ in 0..document_count {
            self.document_processing_times_ms.push(avg_doc_time);
        }

        self.gpu_utilization_samples.push(gpu_utilization);

        if gpu_memory_mb > self.peak_gpu_memory_used_mb {
            self.peak_gpu_memory_used_mb = gpu_memory_mb;
        }

        // Update last metrics update time
        self.last_metrics_update = SystemTime::now();

        // Keep stats vectors at a reasonable size
        self.trim_stats();

        // Update Prometheus metrics
        gauge!(
            "worker.active_batches",
            self.current_concurrent_batches as f64
        );
        gauge!("worker.gpu_utilization", gpu_utilization);
        gauge!("worker.gpu_memory_mb", gpu_memory_mb);
        histogram!("worker.batch_processing_time_ms", processing_time_ms);
        histogram!("worker.document_processing_time_ms", avg_doc_time);
        counter!("worker.documents_processed", document_count as u64);
    }

    // Record error metrics
    pub fn record_error(&mut self) {
        self.errors += 1;
        counter!("worker.errors", 1);
    }

    // Record timeout metrics
    pub fn record_timeout(&mut self) {
        self.request_timeouts += 1;
        counter!("worker.timeouts", 1);
    }

    // Record circuit break metrics
    pub fn record_circuit_break(&mut self) {
        self.circuit_breaks += 1;
        counter!("worker.circuit_breaks", 1);
    }

    // Update concurrent batch count
    pub fn update_concurrent_batches(&mut self, count: u32) {
        self.current_concurrent_batches = count;
        if count > self.max_concurrent_batches {
            self.max_concurrent_batches = count;
        }
    }
}

// Error types for the API
#[derive(Debug)]
pub enum ApiError {
    InferenceError(String),
    BadRequest(String),
    InternalError(String),
    NotFound(String),
    Timeout(String),
    CircuitOpen(String),
    Unauthorized(String),
}

// Convert API errors to HTTP responses
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            ApiError::InferenceError(message) => (StatusCode::INTERNAL_SERVER_ERROR, message),
            ApiError::BadRequest(message) => (StatusCode::BAD_REQUEST, message),
            ApiError::InternalError(message) => (StatusCode::INTERNAL_SERVER_ERROR, message),
            ApiError::NotFound(message) => (StatusCode::NOT_FOUND, message),
            ApiError::Timeout(message) => (StatusCode::REQUEST_TIMEOUT, message),
            ApiError::CircuitOpen(message) => (StatusCode::SERVICE_UNAVAILABLE, message),
            ApiError::Unauthorized(message) => (StatusCode::UNAUTHORIZED, message),
        };

        let body = Json(json!({
            "error": error_message,
            "status": status.as_u16(),
            "code": status.as_u16(),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }));

        // Log the error
        tracing::error!(
            status = status.as_u16(),
            error = error_message,
            "API error occurred"
        );

        (status, body).into_response()
    }
}

// Create the API router
pub fn create_api_router(app_state: Arc<AppState>) -> Router {
    // Set up middleware stack
    let middleware_stack = ServiceBuilder::new()
        .layer(TraceLayer::new_for_http())
        .layer(middleware::from_fn_with_state(
            app_state.clone(),
            ApiKeyAuth::check_api_key,
        ));

    Router::new()
        // Batch processing
        .route("/api/batches", post(process_batch))
        .route("/api/batches/:request_id", get(get_batch_status))
        .route("/api/batches/:request_id/cancel", post(cancel_batch))
        // Worker status and management
        .route("/api/status", get(worker_status))
        .route("/api/health", get(health_check))
        .route("/api/metrics", get(metrics))
        .route("/api/register", post(register_with_orchestrator))
        // GPU management
        .route("/api/gpu/memory", get(gpu_memory))
        .route("/api/gpu/utilization", get(gpu_utilization))
        // Admin operations
        .route("/api/admin/pause", post(pause_worker))
        .route("/api/admin/resume", post(resume_worker))
        .route("/api/admin/reset-stats", post(reset_stats))
        .route(
            "/api/admin/reset-circuit-breaker",
            post(reset_circuit_breaker),
        )
        // Apply middleware stack
        .layer(middleware_stack)
        .with_state(app_state)
}

async fn process_batch(
    State(state): State<Arc<AppState>>,
    _headers: HeaderMap,
    Query(params): Query<BatchQueryParams>,
    Json(mut request): Json<BatchProcessRequest>,
) -> Result<Json<BatchProcessResponse>, ApiError> {
    // Validate the request
    if request.documents.is_empty() {
        return Err(ApiError::BadRequest("Batch contains no documents".into()));
    }

    // Check if worker is healthy
    if !*state.health_status.lock().await {
        return Err(ApiError::CircuitOpen(
            "Worker is currently unhealthy".into(),
        ));
    }

    // Override model_id if provided in query parameters
    if let Some(model_id) = params.model_id {
        request.model_id = model_id;
    }

    // Set default priority if not provided
    if request.priority.is_none() {
        request.priority = Some(1);
    }

    // Get timeout from query parameters or use default
    let timeout = params.timeout_ms.unwrap_or(60000);

    // Update concurrent batch count
    {
        let mut stats = state.stats.lock().await;
        let stat_count = stats.current_concurrent_batches;
        stats.update_concurrent_batches(stat_count + 1);
        // The stats guard is dropped here when we exit this scope
    }

    // Process with timeout
    let start_time = Instant::now();
    let document_count = request.documents.len();

    // Process the batch using the inference engine
    let results = {
        // Set a timeout for the processing
        let timeout_future = tokio::time::sleep(Duration::from_millis(timeout));
        let processing_future = async {
            let mut engine = state.inference_engine.lock().await;
            engine
                .process_batch(&request.documents, &request.model_id)
                .await
        };

        tokio::select! {
            result = processing_future => {
                match result {
                    Ok(results) => {
                        // Record batch success for batch optimizer
                        if let Some(batch_processor) = &state.inference_engine.lock().await.get_batch_processor() {
                            // Since get_optimizer returns a direct reference, not an Option
                            let optimizer = batch_processor.get_optimizer();
                            let mut optimizer_guard = optimizer.lock().await;
                            optimizer_guard.record_batch_result(true, start_time.elapsed().as_secs_f64() * 1000.0);
                        }

                        Ok(results)
                    },
                    Err(e) => {
                        // Record batch failure for batch optimizer
                        if let Some(batch_processor) = &state.inference_engine.lock().await.get_batch_processor() {
                            // Fixed: Use direct reference instead of Option pattern
                            let optimizer = batch_processor.get_optimizer();
                            let mut optimizer_guard = optimizer.lock().await;
                            optimizer_guard.record_batch_result(false, start_time.elapsed().as_secs_f64() * 1000.0);
                        }

                        // Create a new mutex guard here
                        let mut stats = state.stats.lock().await;
                        stats.record_error();
                        // The stats guard is dropped here
                        Err(ApiError::InferenceError(format!("Inference failed: {}", e)))
                    }
                }
            }
            _ = timeout_future => {
                // Record batch failure (timeout) for batch optimizer
                if let Some(batch_processor) = &state.inference_engine.lock().await.get_batch_processor() {
                    // Fixed: Use direct reference instead of Option pattern
                    let optimizer = batch_processor.get_optimizer();
                    let mut optimizer_guard = optimizer.lock().await;
                    optimizer_guard.record_batch_result(false, timeout as f64);
                }

                // Create a new mutex guard here
                let mut stats = state.stats.lock().await;
                stats.record_timeout();
                // The stats guard is dropped here
                Err(ApiError::Timeout(format!("Processing timeout after {}ms", timeout)))
            }
        }
    }?;

    let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

    // Get GPU metrics
    let mut gpu_metrics = state.gpu_metrics.lock().await;
    let gpu_utilization = gpu_metrics.get_utilization();
    let gpu_memory_used = gpu_metrics.get_memory_used_mb();
    // Drop the gpu_metrics guard
    drop(gpu_metrics);

    // Update stats in a new scope with a fresh mutex guard
    {
        let mut stats = state.stats.lock().await;
        stats.record_batch(
            document_count,
            processing_time,
            gpu_utilization,
            gpu_memory_used,
        );
        let stat_count = stats.current_concurrent_batches;
        stats.update_concurrent_batches(stat_count - 1);
        // The stats guard is dropped here
    }

    // Report metrics to database or telemetry system
    let pipeline_metric = PipelineMetric {
        timestamp: SystemTime::now(),
        worker_id: state.worker_id.clone(),
        stage: PipelineStage::Inference,
        batch_size: Some(document_count as i32),
        items_processed: document_count as i32,
        processing_time_ms: processing_time,
        queue_depth: 0, // TO DO: get from engine
        gpu_memory_used_mb: Some(gpu_memory_used),
        gpu_utilization_pct: Some(gpu_utilization),
        cpu_utilization_pct: None, // TO DO: implement CPU monitoring
    };

    // Report to telemetry in background
    let worker_id = state.worker_id.clone();
    tokio::spawn(async move {
        if let Err(e) = report_metrics(&pipeline_metric, &worker_id).await {
            tracing::error!("Failed to report metrics: {}", e);
        }
    });

    // Prepare response
    let response = BatchProcessResponse {
        results,
        request_id: request.request_id,
        processing_time_ms: processing_time,
        worker_id: state.worker_id.clone(),
        error: None,
    };

    Ok(Json(response))
}

// Get status of a specific batch
async fn get_batch_status(
    State(state): State<Arc<AppState>>,
    Path(request_id): Path<Uuid>,
) -> Result<Json<Value>, ApiError> {
    // TODO: In a real implementation, we would track batches and their status
    // For now, we'll return a simple response
    let engine = state.inference_engine.lock().await;

    // Check if the batch is still being processed
    let is_processing = engine.is_processing_batch(request_id);

    if is_processing {
        Ok(Json(json!({
            "request_id": request_id.to_string(),
            "status": "processing",
            "worker_id": state.worker_id,
            "started_at": SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()
        })))
    } else {
        // If not found in active batches, check if it was completed
        // For now, just return not found
        Err(ApiError::NotFound(format!(
            "Batch with request_id {} not found",
            request_id
        )))
    }
}

// Cancel a batch processing request
async fn cancel_batch(
    State(state): State<Arc<AppState>>,
    Path(request_id): Path<Uuid>,
    Json(cancel_request): Json<CancelRequest>,
) -> Result<Json<Value>, ApiError> {
    // Cancel the batch in the inference engine
    let mut engine = state.inference_engine.lock().await;
    let cancelled = engine.cancel_batch(request_id).await;

    if cancelled {
        Ok(Json(json!({
            "request_id": request_id.to_string(),
            "status": "cancelled",
            "reason": cancel_request.reason,
            "cancelled_at": SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()
        })))
    } else {
        Err(ApiError::NotFound(format!(
            "Batch with request_id {} not found or already completed",
            request_id
        )))
    }
}

// Get worker status
async fn worker_status(
    State(state): State<Arc<AppState>>,
    Query(params): Query<StatusQueryParams>,
) -> Json<Value> {
    // Acquire locks in separate blocks to avoid overlapping mutable borrows

    // First, get the necessary information from stats
    let batch_statistics = {
        let stats = state.stats.lock().await;
        BatchStatistics {
            total_batches_processed: stats.total_batches_processed,
            total_documents_processed: stats.total_documents_processed,
            avg_batch_processing_time_ms: stats.avg_batch_time(),
            avg_document_processing_time_ms: stats.avg_document_time(),
            peak_gpu_memory_used_mb: stats.peak_gpu_memory_used_mb,
            avg_gpu_utilization_pct: stats.avg_gpu_utilization(),
        }
    };

    // Get information from engine in a separate scope
    let (engine_is_busy, engine_capabilities, engine_load, active_jobs, queue_depth, is_in_warmup) = {
        let engine = state.inference_engine.lock().await;
        let is_in_warmup = if let Some(batch_processor) = engine.get_batch_processor() {
            // Fixed: Use direct access since get_optimizer returns a reference not an Option
            let optimizer_mutex = batch_processor.get_optimizer();
            // Use try_lock to avoid waiting
            if let Ok(optimizer) = optimizer_mutex.try_lock() {
                optimizer.is_warmup_phase
            } else {
                false
            }
        } else {
            false
        };

        (
            engine.is_busy(),
            engine.get_capabilities(),
            engine.get_current_load(),
            engine.get_active_jobs() as i32,
            engine.get_queue_depth() as i32,
            is_in_warmup,
        )
    };

    // Get enhanced GPU information
    let gpu_detailed_metrics = if params.include_gpu_stats.unwrap_or(false) {
        let mut gpu_metrics = state.gpu_metrics.lock().await;

        // Get recent metrics history if requested
        let recent_metrics = if params.include_metrics.unwrap_or(false) {
            let samples = gpu_metrics.get_recent_metrics(10);
            let samples_json = samples
                .iter()
                .map(|sample| {
                    json!({
                        "timestamp": sample.timestamp.elapsed().as_secs_f64(),
                        "memory_mb": sample.memory_used_mb,
                        "utilization": sample.utilization,
                        "batch_size": sample.batch_size,
                        "operation": sample.operation,
                    })
                })
                .collect::<Vec<_>>();
            Some(samples_json)
        } else {
            None
        };

        Some(json!({
            "current_utilization": gpu_metrics.get_utilization(),
            "memory_used_mb": gpu_metrics.get_memory_used_mb(),
            "memory_free_mb": gpu_metrics.get_memory_free_mb(),
            "memory_total_mb": gpu_metrics.get_memory_total_mb(),
            "peak_memory_mb": gpu_metrics.get_peak_memory_mb(),
            "peak_utilization": gpu_metrics.get_peak_utilization(),
            "allocation_count": gpu_metrics.get_allocation_count(),
            "deallocation_count": gpu_metrics.get_deallocation_count(),
            "throttling_events": gpu_metrics.get_throttling_events(),
            "metrics_history": recent_metrics
        }))
    } else {
        None
    };

    // Get batch optimization status if requested
    let batch_optimization_status = if params.include_metrics.unwrap_or(false) {
        if let Some(batch_processor) = state.inference_engine.lock().await.get_batch_processor() {
            // Fixed: Direct access to optimizer using the reference
            let optimizer_mutex = batch_processor.get_optimizer();
            if let Ok(optimizer) = optimizer_mutex.try_lock() {
                Some(json!({
                    "current_optimal_batch_size": optimizer.get_optimal_batch_size(),
                    "min_batch_size": optimizer.min_batch_size,
                    "max_batch_size": optimizer.max_batch_size,
                    "is_warmup_phase": optimizer.is_warmup_phase,
                    "warmup_batches_processed": optimizer.warmup_batches_processed,
                    "warmup_target_batches": optimizer.warmup_target_batches,
                    "last_batch_had_error": optimizer.last_batch_had_error,
                    "consecutive_successful_batches": optimizer.consecutive_successful_batches
                }))
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // Calculate uptime
    let uptime = SystemTime::now()
        .duration_since(state.start_time)
        .unwrap_or_default()
        .as_secs();

    // Determine status
    let status = {
        let is_healthy = *state.health_status.lock().await;
        if !is_healthy {
            WorkerStatus::Offline
        } else if engine_is_busy {
            WorkerStatus::Busy
        } else {
            WorkerStatus::Online
        }
    };

    // Create enhanced status response
    let response_json = json!({
        "worker_id": state.worker_id,
        "status": status.to_string(),
        "is_warmup": is_in_warmup,
        "capabilities": engine_capabilities,
        "current_load": engine_load,
        "active_jobs": active_jobs,
        "queue_depth": queue_depth,
        "uptime_seconds": uptime,
        "batch_statistics": {
            "total_batches_processed": batch_statistics.total_batches_processed,
            "total_documents_processed": batch_statistics.total_documents_processed,
            "avg_batch_processing_time_ms": batch_statistics.avg_batch_processing_time_ms,
            "avg_document_processing_time_ms": batch_statistics.avg_document_processing_time_ms,
            "peak_gpu_memory_used_mb": batch_statistics.peak_gpu_memory_used_mb,
            "avg_gpu_utilization_pct": batch_statistics.avg_gpu_utilization_pct
        },
        "gpu_metrics": gpu_detailed_metrics,
        "batch_optimization": batch_optimization_status
    });

    Json(response_json)
}

// Health check endpoint
async fn health_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // Get information in separate scopes to avoid overlapping mutable borrows
    let (is_initialized, engine_is_initialized) = {
        let engine = state.inference_engine.lock().await;
        (engine.is_initialized(), engine.is_initialized())
    };

    let (gpu_memory_free, gpu_utilization) = {
        let mut gpu_metrics = state.gpu_metrics.lock().await;
        (
            gpu_metrics.get_memory_free_mb(),
            gpu_metrics.get_utilization(),
        )
    };

    let is_worker_healthy = *state.health_status.lock().await;
    let is_healthy = is_initialized && is_worker_healthy;

    let status = if is_healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    let json_response = json!({
        "status": if is_healthy { "healthy" } else { "unhealthy" },
        "details": {
            "engine_initialized": engine_is_initialized,
            "worker_healthy": is_worker_healthy,
            "gpu_memory_available_mb": gpu_memory_free,
            "gpu_utilization": gpu_utilization,
        },
        "worker_id": state.worker_id,
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": SystemTime::now()
            .duration_since(state.start_time)
            .unwrap_or_default()
            .as_secs()
    });

    (status, Json(json_response))
}

// Detailed metrics endpoint
async fn metrics(State(state): State<Arc<AppState>>) -> Json<Value> {
    // Get stats information
    let stats_data = {
        let stats = state.stats.lock().await;
        json!({
            "total_batches_processed": stats.total_batches_processed,
            "total_documents_processed": stats.total_documents_processed,
            "avg_batch_processing_time_ms": stats.avg_batch_time(),
            "avg_document_processing_time_ms": stats.avg_document_time(),
            "current_concurrent_batches": stats.current_concurrent_batches,
            "max_concurrent_batches": stats.max_concurrent_batches,
            "errors": stats.errors,
            "timeouts": stats.request_timeouts,
            "circuit_breaks": stats.circuit_breaks,
            "peak_memory_used_mb": stats.peak_gpu_memory_used_mb,
            "avg_utilization": stats.avg_gpu_utilization()
        })
    };

    // Get GPU metrics in a separate scope
    let gpu_data = {
        let mut gpu_metrics = state.gpu_metrics.lock().await;
        json!({
            "current_memory_used_mb": gpu_metrics.get_memory_used_mb(),
            "total_memory_mb": gpu_metrics.get_memory_total_mb(),
            "available_memory_mb": gpu_metrics.get_memory_free_mb(),
            "utilization": gpu_metrics.get_utilization(),
        })
    };

    let metrics_json = json!({
        "worker": {
            "id": state.worker_id,
            "uptime_seconds": SystemTime::now()
                .duration_since(state.start_time)
                .unwrap_or_default()
                .as_secs(),
            "version": env!("CARGO_PKG_VERSION"),
        },
        "performance": stats_data,
        "gpu": {
            "peak_memory_used_mb": stats_data["peak_memory_used_mb"],
            "current_memory_used_mb": gpu_data["current_memory_used_mb"],
            "total_memory_mb": gpu_data["total_memory_mb"],
            "available_memory_mb": gpu_data["available_memory_mb"],
            "utilization": gpu_data["utilization"],
            "avg_utilization": stats_data["avg_utilization"]
        }
    });

    Json(metrics_json)
}

// Register the worker with an orchestrator
async fn register_with_orchestrator(
    State(state): State<Arc<AppState>>,
    Json(registration): Json<RegistrationRequest>,
) -> Result<Json<RegistrationResponse>, ApiError> {
    // Store the orchestrator URL
    {
        let mut orchestrator_url = state.orchestrator_url.lock().await;
        *orchestrator_url = Some(registration.orchestrator_url.clone());
    }

    // Get worker capabilities and status
    let engine = state.inference_engine.lock().await;
    let capabilities = engine.get_capabilities();
    let status = if engine.is_busy() {
        WorkerStatus::Busy
    } else {
        WorkerStatus::Online
    };

    // Create a client to send registration to orchestrator
    let client = reqwest::Client::new();

    // Get hostname
    let hostname = hostname::get()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    // Create registration data
    let registration_data = json!({
        "worker_id": state.worker_id,
        "hostname": hostname,
        "ip_address": std::env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0".to_string()),
        "worker_type": WorkerType::Inference.to_string(),
        "capabilities": capabilities,
        "status": status.to_string(),
    });

    // Send registration to orchestrator
    let registration_url = format!("{}/api/workers/register", registration.orchestrator_url);
    let api_key = registration
        .api_key
        .unwrap_or_else(|| state.config.api_key.clone());

    match client
        .post(&registration_url)
        .header("X-API-Key", api_key)
        .json(&registration_data)
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                tracing::info!(
                    "Successfully registered with orchestrator at {}",
                    registration.orchestrator_url
                );

                // Start heartbeat task
                let state_clone = state.clone();
                let orchestrator_url_clone = registration.orchestrator_url.clone();
                tokio::spawn(async move {
                    send_heartbeats(state_clone, orchestrator_url_clone).await;
                });

                // Return success response
                Ok(Json(RegistrationResponse {
                    worker_id: state.worker_id.clone(),
                    status: "registered".to_string(),
                    registered_at: SystemTime::now(),
                }))
            } else {
                let error = format!(
                    "Failed to register with orchestrator: {} - {}",
                    response.status(),
                    response.text().await.unwrap_or_default()
                );
                tracing::error!("{}", error);
                Err(ApiError::InternalError(error))
            }
        }
        Err(e) => {
            let error = format!("Failed to register with orchestrator: {}", e);
            tracing::error!("{}", error);
            Err(ApiError::InternalError(error))
        }
    }
}

// Get GPU memory details
async fn gpu_memory(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut gpu_metrics = state.gpu_metrics.lock().await;

    let memory_json = json!({
        "total_mb": gpu_metrics.get_memory_total_mb(),
        "used_mb": gpu_metrics.get_memory_used_mb(),
        "free_mb": gpu_metrics.get_memory_free_mb(),
        "utilization_percent": gpu_metrics.get_memory_utilization(),
    });

    Json(memory_json)
}

// Get GPU utilization details
async fn gpu_utilization(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut gpu_metrics = state.gpu_metrics.lock().await;
    let stats = state.stats.lock().await;

    let utilization_json = json!({
        "current_percent": gpu_metrics.get_utilization(),
        "average_percent": stats.avg_gpu_utilization(),
        "compute_utilization": gpu_metrics.get_compute_utilization(),
        "memory_utilization": gpu_metrics.get_memory_utilization(),
    });

    Json(utilization_json)
}

// Pause the worker (admin endpoint)
async fn pause_worker(State(state): State<Arc<AppState>>) -> Result<Json<Value>, ApiError> {
    let mut health_status = state.health_status.lock().await;
    *health_status = false;

    // Notify orchestrator of status change
    let worker_id = state.worker_id.clone();
    let orchestrator_url = state.orchestrator_url.lock().await.clone();

    if let Some(url) = orchestrator_url {
        tokio::spawn(async move {
            if let Err(e) = update_worker_status(&url, &worker_id, WorkerStatus::Offline).await {
                tracing::error!("Failed to update worker status: {}", e);
            }
        });
    }

    Ok(Json(json!({
        "status": "paused",
        "message": "Worker is now paused and will not accept new batches",
        "paused_at": SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()
    })))
}

// Resume the worker (admin endpoint)
async fn resume_worker(State(state): State<Arc<AppState>>) -> Result<Json<Value>, ApiError> {
    let mut health_status = state.health_status.lock().await;
    *health_status = true;

    // Get current status
    let engine = state.inference_engine.lock().await;
    let status = if engine.is_busy() {
        WorkerStatus::Busy
    } else {
        WorkerStatus::Online
    };

    // Notify orchestrator of status change
    let worker_id = state.worker_id.clone();
    let orchestrator_url = state.orchestrator_url.lock().await.clone();

    if let Some(url) = orchestrator_url {
        let status_clone = status;
        tokio::spawn(async move {
            if let Err(e) = update_worker_status(&url, &worker_id, status_clone).await {
                tracing::error!("Failed to update worker status: {}", e);
            }
        });
    }

    Ok(Json(json!({
        "status": "resumed",
        "current_status": status.to_string(),
        "message": "Worker is now active and accepting batches",
        "resumed_at": SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()
    })))
}

// Reset worker statistics (admin endpoint)
async fn reset_stats(State(state): State<Arc<AppState>>) -> Result<Json<Value>, ApiError> {
    let mut stats = state.stats.lock().await;
    *stats = WorkerStats::new();

    Ok(Json(json!({
        "status": "reset",
        "message": "Worker statistics have been reset",
        "reset_at": SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()
    })))
}

// Reset circuit breaker (admin endpoint)
async fn reset_circuit_breaker(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Value>, ApiError> {
    let mut engine = state.inference_engine.lock().await;
    engine.reset_circuit_breaker();

    Ok(Json(json!({
        "status": "reset",
        "message": "Circuit breaker has been reset",
        "reset_at": SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()
    })))
}

// Helper function to send periodic heartbeats to orchestrator
async fn send_heartbeats(state: Arc<AppState>, orchestrator_url: String) {
    let heartbeat_interval = state.config.heartbeat_interval_secs;
    let worker_id = state.worker_id.clone();

    let mut interval = tokio::time::interval(Duration::from_secs(heartbeat_interval));

    loop {
        interval.tick().await;

        // Get current worker status
        let engine = state.inference_engine.lock().await;
        let health_status = *state.health_status.lock().await;

        let status = if !health_status {
            WorkerStatus::Offline
        } else if engine.is_busy() {
            WorkerStatus::Busy
        } else {
            WorkerStatus::Online
        };

        // Send heartbeat to orchestrator
        if let Err(e) = update_worker_status(&orchestrator_url, &worker_id, status).await {
            tracing::error!("Failed to send heartbeat: {}", e);
        }
    }
}

// Helper function to update worker status in orchestrator
async fn update_worker_status(
    orchestrator_url: &str,
    worker_id: &str,
    status: WorkerStatus,
) -> Result<(), anyhow::Error> {
    let client = reqwest::Client::new();
    let url = format!("{}/api/workers/{}/status", orchestrator_url, worker_id);

    let response = client
        .put(&url)
        .json(&json!({
            "status": status.to_string(),
            "timestamp": SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()
        }))
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Failed to update status: {}",
            response.status()
        ));
    }

    Ok(())
}

// Helper function to report metrics to telemetry system
async fn report_metrics(metrics: &PipelineMetric, worker_id: &str) -> Result<(), anyhow::Error> {
    // TODO: In a real implementation, we would send this to a metrics collector
    // For now, just log it
    tracing::info!(
        worker_id = worker_id,
        stage = ?metrics.stage,
        batch_size = metrics.batch_size,
        processing_time_ms = metrics.processing_time_ms,
        "Metrics reported"
    );

    Ok(())
}
