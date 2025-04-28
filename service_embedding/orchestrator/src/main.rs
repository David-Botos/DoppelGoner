// orchestrator/src/main.rs
// Main application entry point for the orchestrator service

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{routing::get, Router};
use sqlx::postgres::PgPoolOptions;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use dotenv::dotenv;

// Internal modules
mod api;
mod client;
mod db;
mod discovery;
mod orchestrator;
mod tokenizer;
mod types;

// Import from modules
use crate::api::{create_api_routes, AppState};
use crate::client::worker::{LoadBalanceStrategy, WorkerClient, WorkerClientConfig};
use crate::discovery::create_worker_routes;
use crate::orchestrator::{OrchestratorConfig, OrchestratorService, WorkerManager};
use crate::tokenizer::TokenizerConfig;

// Health check endpoint
async fn health_check() -> &'static str {
    "OK"
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting orchestrator service...");

    // Load configuration from environment
    let api_key = std::env::var("API_KEY").unwrap_or_else(|_| "default-api-key".to_string());

    // Get database connection details
    let host = std::env::var("POSTGRES_HOST").unwrap_or("localhost".into());
    let port = std::env::var("POSTGRES_PORT").unwrap_or("5432".into());
    let user = std::env::var("POSTGRES_USER").unwrap_or("postgres".into());
    let pass = std::env::var("POSTGRES_PASSWORD").unwrap_or("".into());
    let db = std::env::var("POSTGRES_DB").unwrap_or("postgres".into());

    let db_url = format!("postgres://{}:{}@{}:{}/{}", user, pass, host, port, db);

    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "3001".to_string())
        .parse::<u16>()?;

    let batch_size = std::env::var("BATCH_SIZE")
        .unwrap_or_else(|_| "10".to_string())
        .parse::<i32>()?;

    let model_id = std::env::var("MODEL_ID").unwrap_or_else(|_| "bge-small-en-v1.5".to_string());

    let pipeline_interval = std::env::var("PIPELINE_INTERVAL_SECS")
        .unwrap_or_else(|_| "60".to_string())
        .parse::<u64>()?;

// Parse worker locations from environment
let default_worker_locations = std::env::var("WORKER_LOCATIONS")
    .unwrap_or_else(|_| "inference-worker-cuda:3000,inference-worker-metal:3000".to_string())
    .split(',')
    .map(|s| s.trim().to_string())
    .collect::<Vec<String>>();

    // Setup database connection
    info!("Connecting to database: {}", db_url);
    let pool = PgPoolOptions::new()
        .max_connections(20)
        .connect(&db_url)
        .await?;

    // Initialize worker client
    let worker_config = WorkerClientConfig {
        api_key: api_key.clone(),
        request_timeout_secs: 60,
        max_concurrent_requests: 10,
        max_retries: 3,
        circuit_breaker_threshold: 5,
        circuit_breaker_reset_secs: 300,
        stale_worker_threshold_secs: 60,
        worker_discovery_interval_secs: 30,
        load_balance_strategy: LoadBalanceStrategy::LeastLoaded,
    };

    info!("Initializing worker client");
    let worker_client = WorkerClient::new(worker_config);

    // Configure tokenizer
    let tokenizer_config = TokenizerConfig {
        max_tokens: 384,
        model_id: model_id.clone(),
        truncation_strategy: "intelligent".to_string(),
        name_weight: 1.5,
        description_weight: 1.0,
        taxonomy_weight: 0.8,
        include_url: true,
        include_email: false,
        language: "en".to_string(),
    };

    // Create orchestrator config
    let orchestrator_config = OrchestratorConfig {
        batch_size,
        model_id,
        tokenizer_config,
        worker_check_interval_secs: 30,
        job_timeout_mins: 10,
        stale_job_threshold_mins: 30,
        max_concurrent_tokenization: 5,
        max_concurrent_inference: 3,
        default_worker_locations,
    };

    // Create orchestrator service
    info!("Creating orchestrator service");
    let orchestrator = Arc::new(OrchestratorService::new(
        worker_client.clone(),
        pool.clone(),
        orchestrator_config,
    ));

    // Register default workers from config
    info!("Registering default workers");
    match orchestrator.register_default_workers().await {
        Ok(workers) => {
            info!("Registered {} default workers", workers.len());
        }
        Err(e) => {
            error!("Failed to register default workers: {}", e);
        }
    }

    // Start worker manager
    info!(
        "Starting worker manager with interval of {}s",
        pipeline_interval
    );
    let worker_manager = Arc::new(WorkerManager::new(orchestrator.clone(), pipeline_interval));

    // Create application state
    let app_state = Arc::new(AppState {
        orchestrator: orchestrator.clone(),
        worker_manager: worker_manager.clone(),
    });

    let legacy_app_state = Arc::new(crate::discovery::AppState {
        worker_client: worker_client.clone(),
        pool: pool.clone(),
    });

    // Create API router
    let api_routes = create_api_routes(app_state);
    let worker_routes = create_worker_routes(legacy_app_state);

    // Combine routes with a simple health check endpoint
    let app = Router::new()
        .route("/health", get(health_check))
        .merge(api_routes)
        .merge(worker_routes);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("Starting orchestrator API server on {}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;

    info!("Orchestrator service shutting down");

    Ok(())
}
