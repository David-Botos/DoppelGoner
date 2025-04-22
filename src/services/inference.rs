// src/services/inference.rs

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use log::{debug, info, warn};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

use crate::services::config::{CONFIG_PATH, MODEL_PATH};
use crate::services::types::{BatchMetrics, TokenizedBatch};

/// Manages the loading and inference of the BERT model
pub struct InferenceEngine {
    model: Arc<BertModel>,
    device: Arc<Device>,
    batch_counter: Arc<Mutex<usize>>,
}

impl InferenceEngine {
    /// Creates a new inference engine with the specified model and device settings
    ///
    /// # Arguments
    /// * `model_path` - Path to the model weights (.safetensors file)
    /// * `config_path` - Path to the model configuration (.json file)
    /// * `force_cpu` - Whether to force CPU execution regardless of GPU availability
    ///
    /// # Returns
    /// * `Result<Self>` - The initialized inference engine
    pub async fn new(
        model_path: Option<&str>,
        config_path: Option<&str>,
        force_cpu: bool,
    ) -> Result<Self> {
        // Use default paths if not provided
        let model_path = model_path.unwrap_or(MODEL_PATH);
        let config_path = config_path.unwrap_or(CONFIG_PATH);

        // Initialize device (try Metal first if not forced to CPU)
        let device = if !force_cpu {
            info!("Attempting to initialize Metal device");
            match Device::new_metal(0) {
                Ok(device) => {
                    info!("Successfully initialized Metal device for GPU acceleration");
                    device
                }
                Err(e) => {
                    warn!("Metal not available, falling back to CPU: {}", e);
                    Device::Cpu
                }
            }
        } else {
            info!("Forced CPU execution mode");
            Device::Cpu
        };
        info!("Using device: {:?}", device);

        // Load model weights
        info!("Loading model weights from {}", model_path);
        let weights_start = Instant::now();
        let weights = candle_core::safetensors::load(model_path, &device)
            .with_context(|| format!("Failed to load model weights from {}", model_path))?;

        info!(
            "Model weights loaded successfully in {:.2?}",
            weights_start.elapsed()
        );
        debug!("Model keys: {:?}", weights.keys());

        // Load configuration from JSON file
        info!("Loading model configuration from {}", config_path);
        let config_contents = std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config file: {}", config_path))?;

        let model_config: BertConfig =
            serde_json::from_str(&config_contents).context("Failed to parse config JSON")?;

        info!("Model configuration loaded successfully");

        // Initialize model from weights
        info!("Initializing model from weights");
        let model_init_start = Instant::now();
        let vb = candle_nn::VarBuilder::from_tensors(weights, DType::F32, &device);

        // Load model
        let model = BertModel::load(vb, &model_config).context("Failed to initialize model")?;

        info!(
            "Model initialized successfully in {:.2?}",
            model_init_start.elapsed()
        );

        Ok(Self {
            model: Arc::new(model),
            device: Arc::new(device),
            batch_counter: Arc::new(Mutex::new(0)),
        })
    }

    /// Performs inference on a batch of tokenized inputs
    ///
    /// # Arguments
    /// * `batch` - The tokenized batch to process
    ///
    /// # Returns
    /// * `Result<(Vec<String>, Vec<Vec<f32>>, BatchMetrics)>` - Service IDs, embeddings, and metrics
    pub async fn generate_embeddings(
        &self,
        batch: TokenizedBatch,
    ) -> Result<(Vec<String>, Vec<Vec<f32>>, BatchMetrics)> {
        let batch_id = batch.batch_id.clone();
        let service_ids = batch.service_ids.clone();
        let start_time = Instant::now();

        debug!(
            "{}: Starting embedding generation for {} texts",
            batch_id,
            batch.input_ids.len()
        );

        // Skip empty batches
        if batch.input_ids.is_empty() {
            debug!("{}: Empty batch, returning empty results", batch_id);
            return Ok((
                Vec::new(),
                Vec::new(),
                BatchMetrics {
                    batch_id: batch_id.clone(),
                    fetch_time: std::time::Duration::from_secs(0),
                    tokenize_time: std::time::Duration::from_secs(0),
                    inference_time: std::time::Duration::from_millis(1), // Avoid division by zero
                    db_time: std::time::Duration::from_secs(0),
                    total_time: start_time.elapsed(),
                    num_services: 0,
                    services_per_batch: 0,
                    batches_processed: 0,
                },
            ));
        }

        let inference_start = Instant::now();

        // Prepare tensors for the model (pad sequences to same length)
        let tensor_result = self.prepare_tensors(&batch)?;
        let (input_ids_tensor, attention_mask_tensor) = tensor_result;

        // Run model inference
        debug!("{}: Running model forward pass", batch_id);
        let model_start = Instant::now();

        let embeddings = self
            .model
            .forward(
                &input_ids_tensor,
                &attention_mask_tensor,
                None, // No token type IDs
            )
            .context("Model forward pass failed")?;

        debug!(
            "{}: Model forward pass completed in {:.2?}",
            batch_id,
            model_start.elapsed()
        );

        // Perform mean pooling on token embeddings
        let pooled_embeddings =
            self.mean_pooling(&embeddings, &batch.attention_mask, batch_id.as_str())?;

        let inference_time = inference_start.elapsed();
        debug!(
            "{}: Total embedding generation completed in {:.2?}",
            batch_id, inference_time
        );

        // Return results with metrics
        Ok((
            service_ids,
            pooled_embeddings,
            BatchMetrics {
                batch_id,
                fetch_time: std::time::Duration::from_secs(0), // Set by caller
                tokenize_time: std::time::Duration::from_secs(0), // Set by caller
                inference_time,
                db_time: std::time::Duration::from_secs(0), // Set by caller
                total_time: start_time.elapsed(),
                num_services: batch.input_ids.len(),
                services_per_batch: batch.input_ids.len(), // This is the current batch size
                batches_processed: 1, // This method processes one batch at a time
            },
        ))
    }

    /// Prepares tensors for the model from tokenized inputs
    ///
    /// # Arguments
    /// * `batch` - The tokenized batch to process
    ///
    /// # Returns
    /// * `Result<(Tensor, Tensor, usize)>` - Input IDs tensor, attention mask tensor, and max sequence length
    fn prepare_tensors(&self, batch: &TokenizedBatch) -> Result<(Tensor, Tensor)> {
        let batch_id = &batch.batch_id;
        let batch_size = batch.input_ids.len();

        // Find the maximum sequence length in the batch
        let max_len = batch
            .input_ids
            .iter()
            .map(|ids| ids.len())
            .max()
            .unwrap_or(0);

        debug!(
            "{}: Preparing tensors with max_len={}, batch_size={}",
            batch_id, max_len, batch_size
        );

        // Pad inputs to the same length
        let padding_start = Instant::now();
        let mut input_ids_padded = vec![vec![0_u32; max_len]; batch_size];
        let mut attention_mask_padded = vec![vec![0_u32; max_len]; batch_size];

        for (i, (ids, mask)) in batch
            .input_ids
            .iter()
            .zip(batch.attention_mask.iter())
            .enumerate()
        {
            for (j, &id) in ids.iter().enumerate() {
                input_ids_padded[i][j] = id;
            }
            for (j, &mask_val) in mask.iter().enumerate() {
                attention_mask_padded[i][j] = mask_val;
            }
        }

        debug!(
            "{}: Padding completed in {:.2?}",
            batch_id,
            padding_start.elapsed()
        );

        // Create tensors
        let tensor_start = Instant::now();

        // Create input IDs tensor
        let input_ids_tensor = Tensor::new(input_ids_padded, self.device.as_ref())
            .context("Failed to create input_ids tensor")?;

        let input_ids_tensor = input_ids_tensor
            .reshape((batch_size, max_len))
            .context("Failed to reshape input_ids tensor")?;

        // Create attention mask tensor
        let attention_mask_tensor = Tensor::new(attention_mask_padded, self.device.as_ref())
            .context("Failed to create attention_mask tensor")?;

        let attention_mask_tensor = attention_mask_tensor
            .reshape((batch_size, max_len))
            .context("Failed to reshape attention_mask tensor")?;

        debug!(
            "{}: Tensor creation completed in {:.2?}",
            batch_id,
            tensor_start.elapsed()
        );

        Ok((input_ids_tensor, attention_mask_tensor))
    }

    /// Performs mean pooling on token embeddings to get sentence embeddings
    ///
    /// # Arguments
    /// * `embeddings` - Token embeddings from the model
    /// * `attention_mask` - Attention mask to identify which tokens to include
    /// * `batch_id` - Batch identifier for logging
    ///
    /// # Returns
    /// * `Result<Vec<Vec<f32>>>` - Mean-pooled embeddings, one vector per input
    fn mean_pooling(
        &self,
        embeddings: &Tensor,
        attention_mask: &[Vec<u32>],
        batch_id: &str,
    ) -> Result<Vec<Vec<f32>>> {
        let pooling_start = Instant::now();
        let batch_size = embeddings.dims()[0];
        let embedding_dim = embeddings.dims()[2];

        debug!(
            "{}: Performing mean pooling with batch_size={}, embedding_dim={}",
            batch_id, batch_size, embedding_dim
        );

        let mut pooled_embeddings = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let mask_sum = attention_mask[i].iter().sum::<u32>() as usize;

            if mask_sum == 0 {
                // If mask is all zeros, use zero embedding
                debug!(
                    "{}: Sample {} has zero attention mask, using zero embedding",
                    batch_id, i
                );
                pooled_embeddings.push(vec![0.0; embedding_dim]);
                continue;
            }

            // Extract the embeddings for this sample
            let sample_embedding = embeddings
                .get(i)
                .context(format!("Failed to get embeddings for sample {}", i))?;

            // Sum the embeddings of non-masked tokens
            let mut sum_embedding = vec![0.0; embedding_dim];

            for j in 0..mask_sum {
                // Get each token's embedding vector
                let token_embedding: Vec<f32> = sample_embedding
                    .get(j)
                    .context(format!(
                        "Failed to get token embedding for sample {}, token {}",
                        i, j
                    ))?
                    .to_vec1()
                    .context(format!(
                        "Failed to convert tensor to vector for sample {}, token {}",
                        i, j
                    ))?;

                // Sum the embeddings
                for k in 0..embedding_dim {
                    sum_embedding[k] += token_embedding[k];
                }
            }

            // Average the embeddings
            for k in 0..embedding_dim {
                sum_embedding[k] /= mask_sum as f32;
            }

            pooled_embeddings.push(sum_embedding);
        }

        debug!(
            "{}: Mean pooling completed in {:.2?}",
            batch_id,
            pooling_start.elapsed()
        );

        Ok(pooled_embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::types::TokenizedBatch;

    // These tests would require the model files to be present
    // Instead, we'll just demonstrate the structure of the tests

    #[tokio::test]
    #[ignore] // Ignore this test by default since it requires model files
    async fn test_inference_engine_initialization() {
        let engine = InferenceEngine::new(None, None, true).await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    #[ignore] // Ignore this test by default
    async fn test_generate_embeddings() {
        // This would be a more comprehensive test in a real implementation
        // Create a dummy TokenizedBatch
        let batch = TokenizedBatch {
            service_ids: vec!["test1".to_string(), "test2".to_string()],
            input_ids: vec![vec![101, 2054, 2003, 102], vec![101, 2054, 2003, 102]],
            attention_mask: vec![vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
            batch_id: "test-batch".to_string(),
        };

        // Initialize the engine with force_cpu=true for testing
        let engine = InferenceEngine::new(None, None, true).await.unwrap();

        // Generate embeddings
        let result = engine.generate_embeddings(batch).await;
        assert!(result.is_ok());

        let (ids, embeddings, metrics) = result.unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(embeddings.len(), 2);
        assert!(metrics.inference_time.as_millis() > 0);
    }
}
