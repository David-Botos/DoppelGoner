// -----------------------------------------------------------------------------
// Main application entry point
// -----------------------------------------------------------------------------

// orchestrator/src/main.rs
use axum::Server;
use std::net::SocketAddr;
use std::sync::Arc;
use sqlx::postgres::PgPoolOptions;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use std::time::Duration;

mod client;
mod types;
mod discovery;
mod tokenizer;

use crate::client::worker::{WorkerClient, WorkerClientConfig, LoadBalanceStrategy, DatabaseWorkerDiscovery, OrchestratorService};
use crate::discovery::{create_worker_routes, AppState, start_embedding_pipeline};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    // Database connection
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost/embeddings".to_string());
    
    let pool = PgPoolOptions::new()
        .max_connections(20)
        .connect(&database_url)
        .await?;
    
    // Initialize worker client
    let worker_config = WorkerClientConfig {
        api_key: std::env::var("API_KEY").unwrap_or_else(|_| "default-api-key".to_string()),
        request_timeout_secs: 60,
        max_concurrent_requests: 10,
        max_retries: 3,
        circuit_breaker_threshold: 5,
        circuit_breaker_reset_secs: 300,
        stale_worker_threshold_secs: 60,
        worker_discovery_interval_secs: 30,
        load_balance_strategy: LoadBalanceStrategy::LeastLoaded,
    };
    
    let worker_client = WorkerClient::new(worker_config);
    
    // Create worker discovery service
    let discovery_service = Arc::new(DatabaseWorkerDiscovery::new(pool.clone()));
    
    // Start worker discovery
    worker_client.start_discovery(discovery_service).await;
    
    // Create orchestrator service
    let orchestrator = Arc::new(OrchestratorService::new(worker_client.clone(), pool.clone()));
    
    // Start embedding pipeline background task
    let batch_size = std::env::var("BATCH_SIZE")
        .unwrap_or_else(|_| "10".to_string())
        .parse::<i32>()?;
    
    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "bge-small-en-v1.5".to_string());
    
    let pipeline_interval = std::env::var("PIPELINE_INTERVAL_SECS")
        .unwrap_or_else(|_| "60".to_string())
        .parse::<u64>()?;
    
    let _pipeline_handle = start_embedding_pipeline(
        orchestrator.clone(),
        batch_size,
        model_id,
        pipeline_interval,
    );
    
    // Create application state
    let app_state = Arc::new(AppState {
        worker_client,
        pool,
    });
    
    // Create API router
    let app = create_worker_routes(app_state);
    
    // Start server
    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "3001".to_string())
        .parse::<u16>()?;
    
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("Starting orchestrator service on {}", addr);
    
    Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;
    
    Ok(())
}