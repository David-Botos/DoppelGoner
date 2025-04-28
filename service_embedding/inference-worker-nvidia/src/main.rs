// inference_worker/src/main.rs
use axum::Server;
use inference_worker::api::routes::{create_api_router, AppState, WorkerStats};
use inference_worker::config::AppConfig;
use inference_worker::inference::engine::InferenceEngine;
use inference_worker::inference::model::get_best_device;
use inference_worker::telemetry::gpu_metrics::GPUMetrics;
use inference_worker::types::types::WorkerCapabilities;
use metrics::{counter, gauge, histogram};
use metrics_exporter_prometheus::PrometheusBuilder;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::Mutex;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;

// Enhanced capability detection function
async fn detect_hardware_capabilities() -> anyhow::Result<WorkerCapabilities> {
    // Get best available device
    let device = get_best_device()?;

    // Initialize GPU metrics to get memory information
    let mut gpu_metrics = GPUMetrics::new();

    // Determine GPU type and memory
    let supports_cuda = device.is_cuda();
    let supports_metal = device.is_metal();

    let mut gpu_type = None;
    let mut gpu_memory_mb = None;

    if supports_cuda {
        gpu_type = Some("NVIDIA GPU".to_string());
        gpu_memory_mb = Some(gpu_metrics.get_memory_total_mb() as u64);
    } else if supports_metal {
        gpu_type = Some("Apple Silicon GPU".to_string());
        gpu_memory_mb = Some(gpu_metrics.get_memory_total_mb() as u64);
    }

    // Determine optimal and max batch sizes based on available memory
    let (optimal_batch_size, max_batch_size) = match gpu_memory_mb {
        Some(mem) if mem >= 24 * 1024 => (64, 128), // High-end GPU (24GB+)
        Some(mem) if mem >= 12 * 1024 => (32, 64),  // Mid-range GPU (12GB+)
        Some(mem) if mem >= 6 * 1024 => (16, 32),   // Entry GPU (6GB+)
        Some(mem) if mem >= 2 * 1024 => (8, 16),    // Low-end GPU (2GB+)
        _ => (4, 8),                                // Very low memory or CPU
    };

    Ok(WorkerCapabilities {
        gpu_type,
        gpu_memory_mb,
        supports_cuda,
        supports_metal,
        cpu_cores: num_cpus::get() as i32,
        optimal_batch_size,
        max_batch_size,
        embedding_dimensions: None, // Will be updated after model loading
    })
}

fn initialize_enhanced_metrics() {
    // Initialize gauges (values that can go up and down)
    gauge!("worker.batch.current_optimal_size", 0.0);
    gauge!("worker.batch.is_warmup_phase", 0.0);
    gauge!("worker.batch.consecutive_successful_batches", 0.0);
    gauge!("worker.gpu.peak_memory_mb", 0.0);
    gauge!("worker.gpu.peak_utilization", 0.0);
    gauge!("worker.gpu.memory_throttling_events", 0.0);
    gauge!("worker.gpu.allocation_count", 0.0);
    gauge!("worker.gpu.deallocation_count", 0.0);

    // Initialize counters (values that only increase)
    counter!("worker.batch.warmup_batches_processed", 0);
    counter!("worker.batch.size_increases", 0);
    counter!("worker.batch.size_decreases", 0);
    counter!("worker.batch.memory_limited_adjustments", 0);

    // Initialize histograms (distribution of values)
    histogram!("worker.batch.processing_time_by_size", 0.0);
    histogram!("worker.gpu.memory_by_batch_size", 0.0);
}

// Add this function to regularly update the Prometheus metrics
fn start_metrics_reporting(app_state: Arc<AppState>) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(5));

        loop {
            interval.tick().await;

            // Update GPU metrics
            {
                let mut gpu_metrics = app_state.gpu_metrics.lock().await;

                // Update standard metrics (already in the code)
                gauge!(
                    "worker.gpu_memory_used_mb",
                    gpu_metrics.get_memory_used_mb()
                );
                gauge!(
                    "worker.gpu_memory_free_mb",
                    gpu_metrics.get_memory_free_mb()
                );
                gauge!("worker.gpu_utilization", gpu_metrics.get_utilization());

                // Update enhanced metrics
                gauge!(
                    "worker.gpu.peak_memory_mb",
                    gpu_metrics.get_peak_memory_mb()
                );
                gauge!(
                    "worker.gpu.peak_utilization",
                    gpu_metrics.get_peak_utilization()
                );
                gauge!(
                    "worker.gpu.memory_throttling_events",
                    gpu_metrics.get_throttling_events() as f64
                );
                gauge!(
                    "worker.gpu.allocation_count",
                    gpu_metrics.get_allocation_count() as f64
                );
                gauge!(
                    "worker.gpu.deallocation_count",
                    gpu_metrics.get_deallocation_count() as f64
                );
            }

            // Update batch optimizer metrics
            if let Some(batch_processor) = app_state
                .inference_engine
                .lock()
                .await
                .get_batch_processor()
            {
                if let Ok(optimizer) = batch_processor.get_optimizer().try_lock() {
                    gauge!(
                        "worker.batch.current_optimal_size",
                        optimizer.get_optimal_batch_size() as f64
                    );
                    gauge!(
                        "worker.batch.is_warmup_phase",
                        if optimizer.is_warmup_phase { 1.0 } else { 0.0 }
                    );
                    gauge!(
                        "worker.batch.consecutive_successful_batches",
                        optimizer.consecutive_successful_batches as f64
                    );
                    counter!(
                        "worker.batch.warmup_batches_processed",
                        optimizer.warmup_batches_processed as u64
                    );
                }
            }

            // Update worker stats metrics
            {
                let stats = app_state.stats.lock().await;
                gauge!(
                    "worker.active_batches",
                    stats.current_concurrent_batches as f64
                );
                gauge!(
                    "worker.max_concurrent_batches",
                    stats.max_concurrent_batches as f64
                );
                counter!(
                    "worker.total_batches_processed",
                    stats.total_batches_processed
                );
                counter!(
                    "worker.total_documents_processed",
                    stats.total_documents_processed
                );
                counter!("worker.errors", stats.errors);
                counter!("worker.timeouts", stats.request_timeouts);
                counter!("worker.circuit_breaks", stats.circuit_breaks);
            }
        }
    });
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Initialize metrics
    let metrics_builder = PrometheusBuilder::new();
    let metrics_handle = metrics_builder
        .install_recorder()
        .expect("Failed to install metrics recorder");

    // Initialize enhanced metrics
    initialize_enhanced_metrics();

    // Initialize configuration
    let config = AppConfig::from_env()?;

    // Create worker ID (hostname + uuid)
    let hostname = hostname::get()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let worker_id = format!("{}-{}", hostname, Uuid::new_v4());

    // Initialize GPU metrics
    let gpu_metrics = Arc::new(Mutex::new(GPUMetrics::new()));

    // Set up capabilities based on hardware detection
    let capabilities = detect_hardware_capabilities().await?;

    // Update metrics with detected capabilities
    if let Some(gpu_mem) = capabilities.gpu_memory_mb {
        gauge!("worker.gpu_memory_total_mb", gpu_mem as f64);
    }
    counter!("worker.initialized", 1);
    gauge!(
        "worker.optimal_batch_size",
        capabilities.optimal_batch_size as f64
    );
    gauge!("worker.max_batch_size", capabilities.max_batch_size as f64);

    // Initialize the inference engine
    let mut engine = InferenceEngine::new(
        config.model_id.clone(),
        config.model_path.clone(),
        capabilities,
        gpu_metrics.clone(),
    );

    // Load model
    tracing::info!("Loading model {}...", config.model_id);
    if let Err(e) = engine.initialize().await {
        tracing::error!("Failed to initialize model: {}", e);
        std::process::exit(1);
    }
    tracing::info!("Model loaded successfully");

    // Create application state
    let app_state = Arc::new(AppState {
        inference_engine: Arc::new(Mutex::new(engine)),
        config,
        worker_id,
        start_time: SystemTime::now(),
        stats: Arc::new(Mutex::new(WorkerStats::new())),
        gpu_metrics: gpu_metrics.clone(),
        orchestrator_url: Arc::new(Mutex::new(None)),
        health_status: Arc::new(Mutex::new(true)),
    });

    // Start enhanced metrics reporting
    start_metrics_reporting(app_state.clone());

    // Start background monitoring task with enhanced logging
    let app_state_clone = app_state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));

        loop {
            interval.tick().await;

            // Update GPU metrics with enhanced logging
            {
                let mut gpu_metrics = app_state_clone.gpu_metrics.lock().await;
                let _ = gpu_metrics.update();

                // Log enhanced metrics
                tracing::info!(
                 "GPU status: {}% utilized, {}MB used, {}MB free, peak: {}MB, throttling events: {}",
                 gpu_metrics.get_utilization().round(),
                 gpu_metrics.get_memory_used_mb().round(),
                 gpu_metrics.get_memory_free_mb().round(),
                 gpu_metrics.get_peak_memory_mb().round(),
                 gpu_metrics.get_throttling_events()
             );

                // Log memory allocation statistics
                tracing::debug!(
                    "Memory operations: {} allocations, {} deallocations",
                    gpu_metrics.get_allocation_count(),
                    gpu_metrics.get_deallocation_count()
                );
            }

            // Update engine load info and log batch optimization status
            {
                let mut engine = app_state_clone.inference_engine.lock().await;
                engine.update_load().await;

                if let Some(batch_processor) = engine.get_batch_processor() {
                    if let Ok(optimizer) = batch_processor.get_optimizer().try_lock() {
                        tracing::info!(
                            "Batch optimizer status: size={}, warmup={}, consecutive_successes={}",
                            optimizer.get_optimal_batch_size(),
                            optimizer.is_warmup_phase,
                            optimizer.consecutive_successful_batches
                        );
                    }
                }
            }
        }
    });

    // Create API router
    let app = create_api_router(app_state.clone());

    // Start metrics server on a different port
    let metrics_addr = SocketAddr::from(([0, 0, 0, 0], app_state.config.listen_port + 1));
    tokio::spawn(async move {
        let metrics_app = axum::Router::new().route(
            "/metrics",
            axum::routing::get(move || async move {
                // Added 'move' keyword here
                let metrics = metrics_handle.render();
                axum::response::Response::builder()
                    .header("content-type", "text/plain")
                    .body(metrics)
                    .unwrap()
            }),
        );

        tracing::info!("Starting metrics server on {}", metrics_addr);
        if let Err(e) = Server::bind(&metrics_addr)
            .serve(metrics_app.into_make_service())
            .await
        {
            tracing::error!("Metrics server error: {}", e);
        }
    });

    // Start main server
    let addr = SocketAddr::from(([0, 0, 0, 0], app_state.config.listen_port));
    tracing::info!("Starting inference worker on {}", addr);

    Server::bind(&addr).serve(app.into_make_service()).await?;

    Ok(())
}
