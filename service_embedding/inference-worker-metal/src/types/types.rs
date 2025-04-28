// inference_worker/src/types.rs
// This file contains all shared types used across the application
// It should match the types defined in the orchestrator

use serde::{Deserialize, Serialize};
use std::time::SystemTime;
use uuid::Uuid;

//
// Database entity types
//

/// Represents a service with its taxonomies for embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceWithTaxonomies {
    pub id: String, // Character(36)
    pub name: String,
    pub description: Option<String>,
    pub short_description: Option<String>,
    pub taxonomies: Vec<TaxonomyTerm>,
    pub organization_id: String, // Character(36)
    pub url: Option<String>,
    pub email: Option<String>,
    pub status: String,
}

/// Represents a taxonomy term
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxonomyTerm {
    pub id: String, // Character(36)
    pub term: String,
    pub description: Option<String>,
    pub taxonomy: Option<String>, // Category of taxonomy
}

//
// Job and processing types
//

/// Status of a embedding job
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    #[serde(rename = "queued")]
    Queued,

    #[serde(rename = "fetched")]
    Fetched,

    #[serde(rename = "tokenized")]
    Tokenized,

    #[serde(rename = "processing")]
    Processing,

    #[serde(rename = "completed")]
    Completed,

    #[serde(rename = "failed")]
    Failed,
}

impl ToString for JobStatus {
    fn to_string(&self) -> String {
        match self {
            JobStatus::Queued => "queued".to_string(),
            JobStatus::Fetched => "fetched".to_string(),
            JobStatus::Tokenized => "tokenized".to_string(),
            JobStatus::Processing => "processing".to_string(),
            JobStatus::Completed => "completed".to_string(),
            JobStatus::Failed => "failed".to_string(),
        }
    }
}

/// Embedding job record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingJob {
    pub id: Uuid,
    pub service_id: String,
    pub status: JobStatus,
    pub worker_id: Option<String>,
    pub batch_id: Option<Uuid>,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub error_message: Option<String>,
    pub retry_count: i32,
    pub last_retry_at: Option<SystemTime>,
    pub input_tokens: Option<i32>,
    pub truncation_strategy: Option<String>,
    pub metadata: serde_json::Value,
}

/// Embedding document prepared for tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingDocument {
    pub service_id: String,
    pub service_name: String,
    pub service_desc: String,
    pub taxonomies: Vec<TaxonomyDocument>,
}

/// Taxonomy info for embedding document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxonomyDocument {
    pub taxonomy_name: String,
    pub taxonomy_desc: Option<String>,
}

/// Document after tokenization, ready for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizedDocument {
    pub service_id: String,
    pub tokenized_text: String,
    pub token_count: usize,
    pub job_id: Uuid,
}

/// Result from embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResult {
    pub service_id: String,
    pub job_id: Uuid,
    pub embedding: Vec<f32>, // Vector representation
    pub processing_time_ms: f64,
    pub model_id: String, // e.g., "bge-small-en-v1.5"
    pub token_count: usize,
}

//
// Worker and infrastructure types
//

/// Pipeline stage identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineStage {
    #[serde(rename = "fetch")]
    Fetch,

    #[serde(rename = "tokenize")]
    Tokenize,

    #[serde(rename = "inference")]
    Inference,

    #[serde(rename = "store")]
    Store,
}

impl ToString for PipelineStage {
    fn to_string(&self) -> String {
        match self {
            PipelineStage::Fetch => "fetch".to_string(),
            PipelineStage::Tokenize => "tokenize".to_string(),
            PipelineStage::Inference => "inference".to_string(),
            PipelineStage::Store => "store".to_string(),
        }
    }
}

/// Worker type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerType {
    #[serde(rename = "orchestrator")]
    Orchestrator,

    #[serde(rename = "inference")]
    Inference,
}

impl ToString for WorkerType {
    fn to_string(&self) -> String {
        match self {
            WorkerType::Orchestrator => "orchestrator".to_string(),
            WorkerType::Inference => "inference".to_string(),
        }
    }
}

/// Worker status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerStatus {
    #[serde(rename = "online")]
    Online,

    #[serde(rename = "offline")]
    Offline,

    #[serde(rename = "busy")]
    Busy,
}

impl ToString for WorkerStatus {
    fn to_string(&self) -> String {
        match self {
            WorkerStatus::Online => "online".to_string(),
            WorkerStatus::Offline => "offline".to_string(),
            WorkerStatus::Busy => "busy".to_string(),
        }
    }
}

/// Compute node (worker) definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Worker {
    pub id: String,
    pub hostname: String,
    pub ip_address: Option<String>,
    pub worker_type: WorkerType,
    pub capabilities: WorkerCapabilities,
    pub status: WorkerStatus,
    pub last_heartbeat: SystemTime,
    pub current_batch_size: Option<i32>,
    pub current_load: Option<f32>,
    pub active_jobs: i32,
    pub created_at: SystemTime,
}

/// Worker capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerCapabilities {
    pub gpu_type: Option<String>,
    pub gpu_memory_mb: Option<u64>,
    pub supports_cuda: bool,
    pub supports_metal: bool,
    pub cpu_cores: i32,
    pub optimal_batch_size: i32,
    pub max_batch_size: i32,
    // Add embedding dimensions field
    pub embedding_dimensions: Option<i32>,
}

/// Performance metrics for a pipeline stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetric {
    pub timestamp: SystemTime,
    pub worker_id: String,
    pub stage: PipelineStage,
    pub batch_size: Option<i32>,
    pub items_processed: i32,
    pub processing_time_ms: f64,
    pub queue_depth: i32,
    pub gpu_memory_used_mb: Option<f64>,
    pub gpu_utilization_pct: Option<f64>,
    pub cpu_utilization_pct: Option<f64>,
}

//
// API request/response types
//

/// Request to process a batch of documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessRequest {
    pub documents: Vec<TokenizedDocument>,
    pub request_id: Uuid,
    pub priority: Option<i32>,
    pub model_id: String,
}

/// Response from batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessResponse {
    pub results: Vec<EmbeddingResult>,
    pub request_id: Uuid,
    pub processing_time_ms: f64,
    pub worker_id: String,
    pub error: Option<String>,
}

/// Worker status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStatusResponse {
    pub worker_id: String,
    pub status: WorkerStatus,
    pub capabilities: WorkerCapabilities,
    pub current_load: f64,
    pub active_jobs: i32,
    pub queue_depth: i32,
    pub uptime_seconds: u64,
    pub batch_statistics: BatchStatistics,
}

/// Batch processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStatistics {
    pub total_batches_processed: u64,
    pub total_documents_processed: u64,
    pub avg_batch_processing_time_ms: f64,
    pub avg_document_processing_time_ms: f64,
    pub peak_gpu_memory_used_mb: f64,
    pub avg_gpu_utilization_pct: f64,
}

/// Configuration for the embedding pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model_id: String,
    pub max_tokens: usize,
    pub fetch_batch_size: usize,
    pub tokenize_batch_size: usize,
    pub inference_batch_size: usize,
    pub max_fetch_concurrency: usize,
    pub max_tokenize_concurrency: usize,
    pub max_storage_concurrency: usize,
    pub worker_heartbeat_interval_secs: u64,
    pub job_timeout_mins: u64,
    pub stale_worker_mins: u64,
    pub metrics_retention_days: u64,
}

/// Error types for the embedding pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineError {
    DatabaseError(String),
    TokenizationError(String),
    InferenceError(String),
    WorkerCommunicationError(String),
    ConfigurationError(String),
    ResourceUnavailableError(String),
    TimeoutError(String),
}

impl ToString for PipelineError {
    fn to_string(&self) -> String {
        match self {
            PipelineError::DatabaseError(msg) => format!("Database error: {}", msg),
            PipelineError::TokenizationError(msg) => format!("Tokenization error: {}", msg),
            PipelineError::InferenceError(msg) => format!("Inference error: {}", msg),
            PipelineError::WorkerCommunicationError(msg) => {
                format!("Worker communication error: {}", msg)
            }
            PipelineError::ConfigurationError(msg) => format!("Configuration error: {}", msg),
            PipelineError::ResourceUnavailableError(msg) => {
                format!("Resource unavailable: {}", msg)
            }
            PipelineError::TimeoutError(msg) => format!("Timeout: {}", msg),
        }
    }
}
