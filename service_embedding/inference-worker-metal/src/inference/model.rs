// inference_worker/src/inference/model.rs
// Update the BgeEmbeddingModel struct to include GPU metrics

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::bert::{BertModel, Config};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

use crate::telemetry::gpu_metrics::GPUMetrics;

// BGE embedding model implementation
pub struct BgeEmbeddingModel {
    model: BertModel,
    device: Device,
    model_id: String,
    embedding_size: usize,
    // Add GPU metrics field
    gpu_metrics: Option<Arc<Mutex<GPUMetrics>>>,
}

impl BgeEmbeddingModel {
    // Load the model from disk
    pub fn load(model_path: impl AsRef<Path>, device: &Device, model_id: &str) -> Result<Self> {
        let model_path = model_path.as_ref();
        info!("Loading BGE model from {}", model_path.display());

        // Load configuration
        let config_path = model_path.join("config.json");
        let config_data = std::fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&config_data)?;
        debug!("Loaded model config: {:?}", config);

        // Create embedding size based on config
        let embedding_size = config.hidden_size as usize;

        // Load model weights
        let model_file = model_path.join("model.safetensors");

        // Fix: Use the correct approach to load safetensors
        let weights = candle_core::safetensors::load(&model_file, device)?;
        let vb = candle_nn::VarBuilder::from_tensors(weights, candle_core::DType::F32, device);

        // Create the model
        let model = BertModel::load(vb, &config)?;
        info!(
            "Successfully loaded BGE model with {} embedding dimensions",
            embedding_size
        );

        Ok(Self {
            model,
            device: device.clone(),
            model_id: model_id.to_string(),
            embedding_size,
            gpu_metrics: None,
        })
    }

    // New constructor that includes GPU metrics
    pub fn new_with_metrics(
        model: BertModel,
        device: Device,
        model_id: String,
        embedding_size: usize,
        gpu_metrics: Arc<Mutex<GPUMetrics>>,
    ) -> Self {
        Self {
            model,
            device,
            model_id,
            embedding_size,
            gpu_metrics: Some(gpu_metrics),
        }
    }

    // Get embedding size
    pub fn embedding_size(&self) -> usize {
        self.embedding_size
    }

    // Get model_id
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    // Set GPU metrics
    pub fn set_gpu_metrics(&mut self, gpu_metrics: Arc<Mutex<GPUMetrics>>) {
        self.gpu_metrics = Some(gpu_metrics);
    }

    // Helper function to estimate memory needed for a batch
    fn estimate_memory_for_batch(&self, batch_size: usize) -> usize {
        // Estimate memory based on batch size and embedding dimensions
        // This is a simple estimation and can be refined further
        let bytes_per_float = 4; // f32 uses 4 bytes
        let tokens_per_doc = 512; // Max token length

        // Memory for input tensors (input_ids, attention_mask)
        let input_tensors_bytes = 2 * batch_size * tokens_per_doc * 8; // 8 bytes for i64

        // Memory for intermediate activations (estimated)
        let hidden_states_bytes =
            batch_size * tokens_per_doc * self.embedding_size * bytes_per_float * 4;

        // Memory for output embeddings
        let output_embeddings_bytes = batch_size * self.embedding_size * bytes_per_float;

        // Total estimated memory
        input_tensors_bytes + hidden_states_bytes + output_embeddings_bytes
    }

    // Generate embeddings for a batch of tokenized inputs
    pub fn generate_embeddings(
        &self,
        input_ids: &[Vec<i64>],
        attention_mask: &[Vec<i64>],
    ) -> Result<Vec<Vec<f32>>> {
        if input_ids.is_empty() {
            return Err(anyhow!("Empty input batch"));
        }

        debug!(
            "Generating embeddings for batch of size {}",
            input_ids.len()
        );

        // Track memory allocation for GPU metrics
        let batch_size = input_ids.len();
        let estimated_bytes = self.estimate_memory_for_batch(batch_size);

        // Record allocation in GPU metrics if available
        if let Some(gpu_metrics) = &self.gpu_metrics {
            #[cfg(feature = "metal")]
            {
                let metrics = gpu_metrics.clone();
                // Use tokio spawn to avoid blocking on the mutex in sync context
                tokio::task::spawn_blocking(move || {
                    if let Ok(mut metrics_guard) = metrics.try_lock() {
                        metrics_guard.record_allocation(estimated_bytes);
                    }
                });
            }
        }

        // Create a defer-like cleanup for Metal builds to ensure memory is released
        #[cfg(feature = "metal")]
        let gpu_metrics_clone = self.gpu_metrics.clone();
        #[cfg(feature = "metal")]
        let cleanup_bytes = estimated_bytes;

        // Determine the max sequence length in this batch
        let max_len = input_ids.iter().map(|ids| ids.len()).max().unwrap();

        // Create padded tensors
        let mut padded_input_ids = Vec::with_capacity(input_ids.len() * max_len);
        let mut padded_attention_mask = Vec::with_capacity(input_ids.len() * max_len);

        // Pad sequences to max_len
        for (ids, mask) in input_ids.iter().zip(attention_mask.iter()) {
            padded_input_ids.extend_from_slice(ids);
            padded_input_ids.extend(vec![0; max_len - ids.len()]);

            padded_attention_mask.extend_from_slice(mask);
            padded_attention_mask.extend(vec![0; max_len - mask.len()]);
        }

        // Convert to tensors
        let input_ids_tensor =
            Tensor::from_vec(padded_input_ids, (input_ids.len(), max_len), &self.device)?;

        let attention_mask_tensor = Tensor::from_vec(
            padded_attention_mask,
            (attention_mask.len(), max_len),
            &self.device,
        )?;

        // Forward pass through the model
        let output = self.model.forward(
            &input_ids_tensor,
            &attention_mask_tensor,
            None, // token_type_ids
        )?;

        // Get CLS token embedding (first token) for each sequence
        let cls_embeddings = output.narrow(1, 0, 1)?.squeeze(1)?;

        // Normalize embeddings (L2 norm)
        let squared = cls_embeddings.sqr()?;
        let sum = squared.sum_keepdim(1)?;
        let norm = sum.sqrt()?;
        let normalized_embeddings = cls_embeddings.broadcast_div(&norm)?;

        // Convert to Vec<Vec<f32>>
        let embeddings = normalized_embeddings.to_vec2::<f32>()?;

        // Clean up memory tracking for Metal
        #[cfg(feature = "metal")]
        if let Some(gpu_metrics) = gpu_metrics_clone {
            tokio::task::spawn_blocking(move || {
                if let Ok(mut metrics_guard) = gpu_metrics.try_lock() {
                    metrics_guard.record_deallocation(cleanup_bytes);
                }
            });
        }

        debug!("Successfully generated {} embeddings", embeddings.len());

        Ok(embeddings)
    }
}

// Function to determine the best device to use
pub fn get_best_device() -> Result<Device> {
    // Try CUDA first (for NVIDIA GPUs)
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::cuda_if_available(0) {
            if device.is_cuda() {
                info!("Using CUDA device");
                return Ok(device);
            }
        }
    }

    // Try Metal next (for Apple GPUs)
    #[cfg(feature = "metal")]
    {
        // Use the correct method for Metal device initialization
        if let Ok(device) = Device::new_metal(0) {
            info!("Using Metal device");
            return Ok(device);
        }
    }

    // Fall back to CPU
    info!("No GPU available, using CPU");
    Ok(Device::Cpu)
}

// Tokenizer implementation for BGE model
pub struct BgeTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl BgeTokenizer {
    // Load the tokenizer from disk
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();
        let tokenizer_path = model_path.join("tokenizer.json");

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self { tokenizer })
    }

    // Tokenize a batch of texts
    pub fn tokenize_batch(
        &self,
        texts: &[String],
        max_length: usize,
    ) -> Result<(Vec<Vec<i64>>, Vec<Vec<i64>>)> {
        let mut input_ids = Vec::with_capacity(texts.len());
        let mut attention_masks = Vec::with_capacity(texts.len());

        for text in texts {
            // Fix: Convert &String to &str for encode method
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

            // Truncate if necessary
            let ids: Vec<i64> = encoding
                .get_ids()
                .iter()
                .take(max_length)
                .map(|&id| id as i64)
                .collect();

            // Create attention mask (1 for real tokens, 0 for padding)
            let mask: Vec<i64> = vec![1; ids.len()];

            input_ids.push(ids);
            attention_masks.push(mask);
        }

        Ok((input_ids, attention_masks))
    }
}
