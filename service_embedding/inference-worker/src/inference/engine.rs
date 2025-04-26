// inference_worker/src/inference/engine.rs
use anyhow::Result;
use candle_core::Device;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::inference::batch::{BatchOptimizer, BatchProcessor};
use crate::inference::model::{get_best_device, BgeEmbeddingModel, BgeTokenizer};
use crate::telemetry::gpu_metrics::GPUMetrics;
use crate::types::types::{EmbeddingResult, TokenizedDocument, WorkerCapabilities};

// Model wrapper to hold the loaded model and tokenizer
pub struct ModelWrapper {
    model: BgeEmbeddingModel,
    tokenizer: BgeTokenizer,
    max_token_length: usize,
}

impl ModelWrapper {
    pub fn new(model: BgeEmbeddingModel, tokenizer: BgeTokenizer, max_token_length: usize) -> Self {
        Self {
            model,
            tokenizer,
            max_token_length,
        }
    }
}

// Pending batch information
#[derive(Clone)]
pub struct PendingBatch {
    pub documents: Vec<TokenizedDocument>,
    pub model_id: String,
    pub created_at: Instant,
    pub request_id: Uuid,
    pub priority: Option<i32>,
}

/// InferenceEngine manages the ML model and performs embedding generation
pub struct InferenceEngine {
    // Model state and configuration
    model_id: String,
    model_path: String,
    model: Option<Arc<ModelWrapper>>,
    active_batches: HashMap<Uuid, PendingBatch>,
    batch_queue: Vec<PendingBatch>,
    capabilities: WorkerCapabilities,
    device: Device,
    current_load: f32,
    active_jobs: usize,
    circuit_open: bool,
    consecutive_errors: usize,
    error_threshold: usize,

    // Batch processing
    batch_processor: Option<Arc<BatchProcessor>>,

    // GPU monitoring
    gpu_metrics: Arc<Mutex<GPUMetrics>>,
}

impl InferenceEngine {
    pub fn new(
        model_id: String,
        model_path: String,
        capabilities: WorkerCapabilities,
        gpu_metrics: Arc<Mutex<GPUMetrics>>,
    ) -> Self {
        // Get best available device
        let device = get_best_device().unwrap_or(Device::Cpu);

        // Update capabilities based on available hardware
        let mut updated_capabilities = capabilities;
        updated_capabilities.supports_cuda = device.is_cuda();
        updated_capabilities.supports_metal = device.is_metal();

        Self {
            model_id,
            model_path,
            model: None,
            active_batches: HashMap::new(),
            batch_queue: Vec::new(),
            capabilities: updated_capabilities,
            device,
            current_load: 0.0,
            active_jobs: 0,
            circuit_open: false,
            consecutive_errors: 0,
            error_threshold: 5,
            batch_processor: None,
            gpu_metrics,
        }
    }

    pub fn get_batch_processor(&self) -> Option<&Arc<BatchProcessor>> {
        self.batch_processor.as_ref()
    }

    // Initialize method to pass GPU metrics to the model

    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing BGE model from {}", self.model_path);

        // Load model
        let mut model = BgeEmbeddingModel::load(&self.model_path, &self.device, &self.model_id)?;
        let embedding_size = model.embedding_size();

        // Pass GPU metrics to model
        model.set_gpu_metrics(self.gpu_metrics.clone());

        // Load tokenizer
        let tokenizer = BgeTokenizer::load(&self.model_path)?;

        info!(
            "Model initialized successfully with {} dimensions",
            embedding_size
        );

        // Update capabilities with correct embedding dimensions
        self.capabilities.embedding_dimensions = Some(embedding_size as i32);

        // Create model wrapper
        let model_wrapper = ModelWrapper::new(
            model, tokenizer, 512, // BGE model max token length
        );

        self.model = Some(Arc::new(model_wrapper));

        // Initialize batch optimizer and processor
        let optimizer = BatchOptimizer::new(
            1,                                             // min batch size
            self.capabilities.max_batch_size as usize,     // max batch size
            self.capabilities.optimal_batch_size as usize, // initial optimal size
            500.0,                                         // target latency in ms (500ms)
            self.gpu_metrics.clone(),
        );

        let batch_optimizer = Arc::new(Mutex::new(optimizer));
        self.batch_processor = Some(Arc::new(BatchProcessor::new(batch_optimizer)));

        Ok(())
    }

    /// Check if the engine is initialized
    pub fn is_initialized(&self) -> bool {
        self.model.is_some()
    }

    /// Process a batch of documents
    pub async fn process_batch(
        &mut self,
        documents: &[TokenizedDocument],
        model_id: &str,
    ) -> Result<Vec<EmbeddingResult>> {
        // Check if model is initialized
        if self.model.is_none() {
            return Err(anyhow::anyhow!("Model not initialized"));
        }

        // Check circuit breaker
        if self.circuit_open {
            return Err(anyhow::anyhow!(
                "Circuit breaker is open due to consecutive errors"
            ));
        }

        // Increase active jobs count
        self.active_jobs += 1;

        // Create a pending batch
        let batch = PendingBatch {
            documents: documents.to_vec(),
            model_id: model_id.to_string(),
            created_at: Instant::now(),
            request_id: Uuid::new_v4(),
            priority: None,
        };

        // Track the batch
        self.active_batches.insert(batch.request_id, batch.clone());

        // Process using batch processor if available
        let result = if let Some(processor) = &self.batch_processor {
            let documents = batch.documents.clone();
            let model_wrapper = self.model.as_ref().unwrap().clone();
            let batch_request_id = batch.request_id;

            // Use batch processor to split and process optimally
            processor
                .process_documents(documents, move |batch_chunk| {
                    Self::process_document_chunk(
                        model_wrapper.clone(),
                        batch_chunk,
                        batch_request_id,
                    )
                })
                .await
        } else {
            // Fall back to processing the whole batch at once
            self.perform_inference(&batch).await
        };

        // Remove from active batches
        self.active_batches.remove(&batch.request_id);

        // Handle error tracking for circuit breaker
        match &result {
            Ok(_) => {
                self.consecutive_errors = 0;
                self.circuit_open = false;
            }
            Err(_) => {
                self.consecutive_errors += 1;
                if self.consecutive_errors >= self.error_threshold {
                    self.circuit_open = true;
                    error!(
                        "Circuit breaker opened after {} consecutive errors",
                        self.consecutive_errors
                    );
                }
            }
        }

        // Decrease active jobs count
        self.active_jobs -= 1;

        result
    }

    /// Process a chunk of documents (used by batch processor)
    async fn process_document_chunk(
        model_wrapper: Arc<ModelWrapper>,
        documents: Vec<TokenizedDocument>,
        batch_id: Uuid,
    ) -> Result<Vec<EmbeddingResult>> {
        let start_time = Instant::now();

        // Extract tokenized texts
        let texts: Vec<String> = documents
            .iter()
            .map(|doc| doc.tokenized_text.clone())
            .collect();

        // Tokenize the batch
        let (input_ids, attention_masks) = model_wrapper
            .tokenizer
            .tokenize_batch(&texts, model_wrapper.max_token_length)?;

        // Generate embeddings
        let embeddings = model_wrapper
            .model
            .generate_embeddings(&input_ids, &attention_masks)?;

        // Create results
        let results: Vec<EmbeddingResult> = documents
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

                EmbeddingResult {
                    service_id: doc.service_id.clone(),
                    job_id: doc.job_id,
                    embedding: embeddings[i].clone(),
                    processing_time_ms: processing_time,
                    model_id: model_wrapper.model.model_id().to_string(),
                    token_count: doc.token_count,
                }
            })
            .collect();

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        debug!(
            "Generated {} embeddings in {:.2}ms ({:.2}ms per document)",
            results.len(),
            total_time,
            total_time / results.len() as f64
        );

        Ok(results)
    }

    /// Perform the actual inference on a full batch
    async fn perform_inference(&self, batch: &PendingBatch) -> Result<Vec<EmbeddingResult>> {
        let start_time = Instant::now();
        let model_wrapper = self.model.as_ref().unwrap();

        // Extract tokenized texts
        let texts: Vec<String> = batch
            .documents
            .iter()
            .map(|doc| doc.tokenized_text.clone())
            .collect();

        // Tokenize the batch
        let (input_ids, attention_masks) = model_wrapper
            .tokenizer
            .tokenize_batch(&texts, model_wrapper.max_token_length)?;

        // Generate embeddings
        let embeddings = model_wrapper
            .model
            .generate_embeddings(&input_ids, &attention_masks)?;

        // Create results
        let results: Vec<EmbeddingResult> = batch
            .documents
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

                EmbeddingResult {
                    service_id: doc.service_id.clone(),
                    job_id: doc.job_id,
                    embedding: embeddings[i].clone(),
                    processing_time_ms: processing_time,
                    model_id: model_wrapper.model.model_id().to_string(),
                    token_count: doc.token_count,
                }
            })
            .collect();

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        info!(
            "Generated {} embeddings in {:.2}ms ({:.2}ms per document)",
            results.len(),
            total_time,
            total_time / results.len() as f64
        );

        Ok(results)
    }

    /// Check if a specific batch is being processed
    pub fn is_processing_batch(&self, request_id: Uuid) -> bool {
        self.active_batches.contains_key(&request_id)
    }

    /// Cancel a batch if it's still being processed
    pub async fn cancel_batch(&mut self, request_id: Uuid) -> bool {
        if self.active_batches.contains_key(&request_id) {
            self.active_batches.remove(&request_id);
            true
        } else {
            false
        }
    }

    /// Check if the engine is busy
    pub fn is_busy(&self) -> bool {
        self.active_jobs > 0 || !self.batch_queue.is_empty()
    }

    /// Get worker capabilities
    pub fn get_capabilities(&self) -> WorkerCapabilities {
        self.capabilities.clone()
    }

    /// Get current load
    pub fn get_current_load(&self) -> f64 {
        self.current_load.into()
    }

    /// Get number of active jobs
    pub fn get_active_jobs(&self) -> usize {
        self.active_jobs
    }

    /// Get queue depth
    pub fn get_queue_depth(&self) -> usize {
        self.batch_queue.len()
    }

    /// Reset circuit breaker
    pub fn reset_circuit_breaker(&mut self) {
        self.circuit_open = false;
        self.consecutive_errors = 0;
    }

    /// Update current load based on GPU metrics
    pub async fn update_load(&mut self) {
        let mut gpu_metrics = self.gpu_metrics.lock().await;
        self.current_load = gpu_metrics.get_utilization() as f32 / 100.0;
    }
}
