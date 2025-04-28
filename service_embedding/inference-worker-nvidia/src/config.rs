// inference_worker/src/config.rs
use anyhow::Result;
use std::env;

/// Application configuration
pub struct AppConfig {
    // Model configuration
    pub model_id: String,
    pub model_path: String,
    pub max_token_length: usize,

    // Server configuration
    pub listen_addr: String,
    pub listen_port: u16,
    pub api_key: String,
    pub request_timeout_secs: u64,

    // Batch processing
    pub min_batch_size: usize,
    pub max_batch_size: usize,
    pub initial_batch_size: usize,
    pub target_latency_ms: f64,

    // Worker settings
    pub heartbeat_interval_secs: u64,
    pub metrics_retention_days: u32,

    // Circuit breaker settings
    pub circuit_breaker_threshold: usize,
    pub circuit_breaker_reset_secs: u64,

    // GPU settings
    pub gpu_memory_limit_mb: Option<u64>,
    pub gpu_utilization_threshold: f64,

    // Logging
    pub log_level: String,
}

impl AppConfig {
    /// Load configuration from environment variables with sensible defaults
    // inference_worker/src/config.rs
    // Update the from_env method to include safe defaults for GPU memory limits

    pub fn from_env() -> Result<Self> {
        // Create a temporary GPU metrics to detect total memory
        let mut gpu_metrics = crate::telemetry::gpu_metrics::GPUMetrics::new();
        let total_gpu_memory = gpu_metrics.get_memory_total_mb();

        Ok(AppConfig {
            // Model configuration
            model_id: env::var("MODEL_ID").unwrap_or_else(|_| "bge-small-en-v1.5".to_string()),
            model_path: env::var("MODEL_PATH")
                .unwrap_or_else(|_| "./models/bge-small-en-v1.5".to_string()),
            max_token_length: env::var("MAX_TOKEN_LENGTH")
                .unwrap_or_else(|_| "512".to_string())
                .parse()?,

            // Server configuration
            listen_addr: env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0".to_string()),
            listen_port: env::var("LISTEN_PORT")
                .unwrap_or_else(|_| "3000".to_string())
                .parse()?,
            api_key: env::var("API_KEY").unwrap_or_else(|_| "fart".to_string()),
            request_timeout_secs: env::var("REQUEST_TIMEOUT_SECS")
                .unwrap_or_else(|_| "60".to_string())
                .parse()?,

            // Batch processing
            min_batch_size: env::var("MIN_BATCH_SIZE")
                .unwrap_or_else(|_| "1".to_string())
                .parse()?,
            max_batch_size: env::var("MAX_BATCH_SIZE")
                .unwrap_or_else(|_| "64".to_string())
                .parse()?,
            initial_batch_size: env::var("INITIAL_BATCH_SIZE")
                .unwrap_or_else(|_| "16".to_string())
                .parse()?,
            target_latency_ms: env::var("TARGET_LATENCY_MS")
                .unwrap_or_else(|_| "500.0".to_string())
                .parse()?,

            // Worker settings
            heartbeat_interval_secs: env::var("HEARTBEAT_INTERVAL_SECS")
                .unwrap_or_else(|_| "30".to_string())
                .parse()?,
            metrics_retention_days: env::var("METRICS_RETENTION_DAYS")
                .unwrap_or_else(|_| "30".to_string())
                .parse()?,

            // Circuit breaker settings
            circuit_breaker_threshold: env::var("CIRCUIT_BREAKER_THRESHOLD")
                .unwrap_or_else(|_| "5".to_string())
                .parse()?,
            circuit_breaker_reset_secs: env::var("CIRCUIT_BREAKER_RESET_SECS")
                .unwrap_or_else(|_| "300".to_string())
                .parse()?,

            // GPU settings - Use 90% of detected memory as a safe default if not specified
            gpu_memory_limit_mb: match env::var("GPU_MEMORY_LIMIT_MB") {
                Ok(val) => Some(val.parse()?),
                Err(_) => {
                    if total_gpu_memory > 0.0 {
                        Some((total_gpu_memory * 0.9) as u64)
                    } else {
                        None
                    }
                }
            },
            gpu_utilization_threshold: env::var("GPU_UTILIZATION_THRESHOLD")
                .unwrap_or_else(|_| "90.0".to_string())
                .parse()?,

            // Logging
            log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
        })
    }

    /// Get database URL (for future use if needed)
    pub fn get_database_url(&self) -> String {
        env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://postgres:postgres@localhost/embeddings".to_string())
    }

    /// Get orchestrator URL (for future use if needed)
    pub fn get_orchestrator_url(&self) -> Option<String> {
        env::var("ORCHESTRATOR_URL").ok()
    }

    /// Get whether CUDA is enabled
    pub fn is_cuda_enabled(&self) -> bool {
        #[cfg(feature = "cuda")]
        return true;

        #[cfg(not(feature = "cuda"))]
        return false;
    }

    /// Get whether Metal is enabled
    pub fn is_metal_enabled(&self) -> bool {
        #[cfg(feature = "metal")]
        return true;

        #[cfg(not(feature = "metal"))]
        return false;
    }
}
