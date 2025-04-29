// inference_worker/src/inference/model.rs
// Migrated from Candle to tch-rs for improved layer normalization support

use anyhow::{anyhow, Result};
use std::path::Path;
use std::sync::Arc;
use tch::{Device, Kind, Tensor, nn};
use tch::nn::{Module, VarStore};
use tokenizers::Tokenizer;
use std::collections::HashMap;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::telemetry::gpu_metrics::GPUMetrics;

// Configuration for BERT models like BGE
#[derive(Debug, Clone, serde::Deserialize)]
pub struct BertConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    pub intermediate_size: i64,
    pub hidden_act: String,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub max_position_embeddings: i64,
    pub type_vocab_size: i64,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: Option<i64>,
    pub position_embedding_type: Option<String>,
    pub use_cache: Option<bool>,
    pub classifier_dropout: Option<f64>,
}

// Attention layer implementation for BERT
struct BertAttention {
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    output: nn::Linear,
    layer_norm: nn::LayerNorm,
    num_attention_heads: i64,
    attention_head_size: i64,
}

impl BertAttention {
    fn new(vs: &nn::Path, config: &BertConfig) -> Self {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        
        let query = nn::linear(vs / "query", config.hidden_size, all_head_size, Default::default());
        let key = nn::linear(vs / "key", config.hidden_size, all_head_size, Default::default());
        let value = nn::linear(vs / "value", config.hidden_size, all_head_size, Default::default());
        
        let output = nn::linear(vs / "output" / "dense", all_head_size, config.hidden_size, Default::default());
        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_eps,
            ..Default::default()
        };
        let layer_norm = nn::layer_norm(vs / "output" / "LayerNorm", vec![config.hidden_size], layer_norm_config);
        
        Self {
            query,
            key,
            value,
            output,
            layer_norm,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
        }
    }
    
    fn transpose_for_scores(&self, x: &Tensor) -> Tensor {
        let new_shape = [
            x.size()[0], 
            x.size()[1], 
            self.num_attention_heads, 
            self.attention_head_size
        ];
        x.view(new_shape).permute([0, 2, 1, 3])
    }
}

impl Module for BertAttention {
    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        let batch_size = hidden_states.size()[0];
        let seq_length = hidden_states.size()[1];
        
        let query = self.query.forward(hidden_states);
        let key = self.key.forward(hidden_states);
        let value = self.value.forward(hidden_states);
        
        let query = self.transpose_for_scores(&query);
        let key = self.transpose_for_scores(&key);
        let value = self.transpose_for_scores(&value);
        
        // Calculate attention scores
        let attention_scores = query.matmul(&key.transpose(-1, -2));
        let attention_scores = attention_scores / (self.attention_head_size as f64).sqrt();
        
        // Apply attention mask if provided
        let attention_probs = if let Some(mask) = attention_mask {
            // Expand mask for attention heads
            let mask = mask.view([batch_size, 1, seq_length]).expand([batch_size, self.num_attention_heads, seq_length, seq_length], true);
            // Apply mask (add large negative value to masked positions)
            let masked_scores = attention_scores + mask;
            masked_scores.softmax(-1, Kind::Float)
        } else {
            attention_scores.softmax(-1, Kind::Float)
        };
        
        // Apply attention to values
        let context = attention_probs.matmul(&value);
        let context = context.permute([0, 2, 1, 3]).contiguous();
        let context_shape = [batch_size, seq_length, self.num_attention_heads * self.attention_head_size];
        let context = context.view(context_shape);
        
        // Linear projection and layer norm
        let output = self.output.forward(&context);
        let output = hidden_states + output;  // Residual connection
        self.layer_norm.forward(&output)
    }
}

// Transformer layer implementation for BERT
struct BertLayer {
    attention: BertAttention,
    intermediate: nn::Linear,
    output: nn::Linear,
    layer_norm: nn::LayerNorm,
    activation: String,
}

impl BertLayer {
    fn new(vs: &nn::Path, config: &BertConfig) -> Self {
        let attention = BertAttention::new(vs / "attention", config);
        
        let intermediate = nn::linear(
            vs / "intermediate" / "dense", 
            config.hidden_size, 
            config.intermediate_size, 
            Default::default()
        );
        
        let output = nn::linear(
            vs / "output" / "dense", 
            config.intermediate_size, 
            config.hidden_size, 
            Default::default()
        );
        
        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_eps,
            ..Default::default()
        };
        let layer_norm = nn::layer_norm(
            vs / "output" / "LayerNorm", 
            vec![config.hidden_size], 
            layer_norm_config
        );
        
        Self {
            attention,
            intermediate,
            output,
            layer_norm,
            activation: config.hidden_act.clone(),
        }
    }
    
    fn activate(&self, x: &Tensor) -> Tensor {
        match self.activation.as_str() {
            "gelu" => x.gelu("none"),
            "relu" => x.relu(),
            _ => x.gelu("none"), // Default to gelu
        }
    }
}

impl Module for BertLayer {
    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        let attention_output = self.attention.forward(hidden_states, attention_mask);
        
        // FFN
        let intermediate_output = self.intermediate.forward(&attention_output);
        let intermediate_output = self.activate(&intermediate_output);
        
        // Output
        let layer_output = self.output.forward(&intermediate_output);
        let layer_output = attention_output + layer_output;  // Residual connection
        
        self.layer_norm.forward(&layer_output)
    }
}

// Full BERT model implementation using tch-rs
pub struct BertModel {
    embeddings: nn::Embedding,
    position_ids: Tensor,
    position_embeddings: nn::Embedding,
    token_type_embeddings: nn::Embedding,
    layer_norm: nn::LayerNorm,
    encoder_layers: Vec<BertLayer>,
    config: BertConfig,
    vs: VarStore,
}

impl BertModel {
    pub fn load(model_path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let model_path = model_path.as_ref();
        
        // Load configuration
        let config_path = model_path.join("config.json");
        let config_data = std::fs::read_to_string(&config_path)?;
        let config: BertConfig = serde_json::from_str(&config_data)?;
        
        info!("Loaded BERT config with {} layers, {} hidden size", 
              config.num_hidden_layers, config.hidden_size);
        
        // Initialize VarStore for model parameters
        let mut vs = VarStore::new(device);
        let root = vs.root();
        
        // Create embeddings
        let embeddings_path = root.sub("embeddings");
        let embeddings = nn::embedding(
            &embeddings_path / "word_embeddings",
            config.vocab_size,
            config.hidden_size,
            Default::default()
        );
        
        let position_embeddings = nn::embedding(
            &embeddings_path / "position_embeddings",
            config.max_position_embeddings,
            config.hidden_size,
            Default::default()
        );
        
        let token_type_embeddings = nn::embedding(
            &embeddings_path / "token_type_embeddings",
            config.type_vocab_size,
            config.hidden_size,
            Default::default()
        );
        
        // Position IDs tensor
        let position_ids = Tensor::arange(config.max_position_embeddings, (Kind::Int64, device));
        
        // Layer normalization
        let layer_norm_config = nn::LayerNormConfig {
            eps: config.layer_norm_eps,
            ..Default::default()
        };
        
        let layer_norm = nn::layer_norm(
            &embeddings_path / "LayerNorm",
            vec![config.hidden_size],
            layer_norm_config
        );
        
        // Create transformer layers
        let encoder_path = root.sub("encoder");
        let mut encoder_layers = Vec::new();
        
        for i in 0..config.num_hidden_layers {
            let layer = BertLayer::new(&encoder_path / "layer" / i, &config);
            encoder_layers.push(layer);
        }
        
        // Load weights from safetensors file
        let weights_path = model_path.join("model.safetensors");
        if weights_path.exists() {
            info!("Loading model weights from safetensors");
            vs.load(&weights_path)?;
        } else {
            // Try loading PyTorch format if safetensors not available
            let pytorch_weights = model_path.join("pytorch_model.bin");
            if pytorch_weights.exists() {
                info!("Loading model weights from PyTorch format");
                vs.load(&pytorch_weights)?;
            } else {
                return Err(anyhow!("No model weights found at {:?}", model_path));
            }
        }
        
        Ok(Self {
            embeddings,
            position_ids,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            encoder_layers,
            config,
            vs,
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor, token_type_ids: Option<&Tensor>) -> Tensor {
        let seq_length = input_ids.size()[1];
        
        // Get embeddings
        let inputs_embeds = self.embeddings.forward(input_ids);
        
        // Get position embeddings
        let position_ids = self.position_ids.slice(0, 0, seq_length, 1)
            .expand(input_ids.size().as_slice(), true);
        let position_embeds = self.position_embeddings.forward(&position_ids);
        
        // Get token type embeddings (optional)
        let token_type_embeds = if let Some(token_type_ids) = token_type_ids {
            self.token_type_embeddings.forward(token_type_ids)
        } else {
            // Default to zeros if not provided
            Tensor::zeros_like(&inputs_embeds)
        };
        
        // Sum embeddings and apply layer normalization
        let embeddings = inputs_embeds + position_embeds + token_type_embeds;
        let mut hidden_states = self.layer_norm.forward(&embeddings);
        
        // Process through transformer layers
        let extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2);
        let extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0;
        
        for layer in &self.encoder_layers {
            hidden_states = layer.forward(&hidden_states, Some(&extended_attention_mask));
        }
        
        hidden_states
    }
    
    pub fn embedding_size(&self) -> i64 {
        self.config.hidden_size
    }
}

// BGE embedding model implementation using tch-rs
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
    pub fn load(model_path: impl AsRef<Path>, device_str: &str, model_id: &str) -> Result<Self> {
        let model_path = model_path.as_ref();
        info!("Loading BGE model from {}", model_path.display());

        // Convert string device to tch Device
        let device = match device_str {
            "cuda" => Device::Cuda(0),
            "cpu" => Device::Cpu,
            _ => Device::Cpu,
        };

        // Load the model
        let model = BertModel::load(model_path, device)?;
        let embedding_size = model.embedding_size() as usize;

        info!(
            "Successfully loaded BGE model with {} embedding dimensions",
            embedding_size
        );

        Ok(Self {
            model,
            device,
            model_id: model_id.to_string(),
            embedding_size,
            gpu_metrics: None,
        })
    }

    // Set GPU metrics
    pub fn set_gpu_metrics(&mut self, gpu_metrics: Arc<Mutex<GPUMetrics>>) {
        self.gpu_metrics = Some(gpu_metrics);
    }

    // Get embedding size
    pub fn embedding_size(&self) -> usize {
        self.embedding_size
    }

    // Get model_id
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    // Helper function to estimate memory needed for a batch
    fn estimate_memory_for_batch(&self, batch_size: usize) -> usize {
        // Estimate memory based on batch size and embedding dimensions
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
            let metrics = gpu_metrics.clone();
            // Use tokio spawn to avoid blocking on the mutex in sync context
            tokio::task::spawn_blocking(move || {
                if let Ok(mut metrics_guard) = metrics.try_lock() {
                    metrics_guard.record_allocation(estimated_bytes);
                }
            });
        }

        // Determine the max sequence length in this batch
        let max_len = input_ids.iter().map(|ids| ids.len()).max().unwrap();
        let batch_size = input_ids.len();

        // Create tensors from input data
        let mut input_ids_data = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask_data = Vec::with_capacity(batch_size * max_len);

        // Pad sequences to max_len
        for (ids, mask) in input_ids.iter().zip(attention_mask.iter()) {
            input_ids_data.extend_from_slice(ids);
            input_ids_data.extend(vec![0; max_len - ids.len()]);

            attention_mask_data.extend_from_slice(mask);
            attention_mask_data.extend(vec![0; max_len - mask.len()]);
        }

        // Convert to tensors
        let input_ids_tensor = Tensor::of_slice(&input_ids_data)
            .view([batch_size as i64, max_len as i64])
            .to_device(self.device);

        let attention_mask_tensor = Tensor::of_slice(&attention_mask_data)
            .view([batch_size as i64, max_len as i64])
            .to_device(self.device);

        // No token type IDs for BGE
        let token_type_ids = None;

        // Forward pass
        let outputs = self.model.forward(&input_ids_tensor, &attention_mask_tensor, token_type_ids);

        // Get CLS token embedding (first token) for each sequence
        let cls_embeddings = outputs.select(1, 0);

        // Normalize embeddings (L2 norm)
        let norm = cls_embeddings.pow_tensor_scalar(2.0)
            .sum_dim_intlist([-1], true, Kind::Float)
            .sqrt();
        
        let normalized_embeddings = cls_embeddings / norm;

        // Convert to Vec<Vec<f32>>
        let embeddings = normalized_embeddings.try_into()
            .map_err(|e| anyhow!("Failed to convert tensor to Vec: {:?}", e))?;

        // Clean up memory tracking
        if let Some(gpu_metrics) = &self.gpu_metrics {
            let metrics = gpu_metrics.clone();
            let cleanup_bytes = estimated_bytes;
            tokio::task::spawn_blocking(move || {
                if let Ok(mut metrics_guard) = metrics.try_lock() {
                    metrics_guard.record_deallocation(cleanup_bytes);
                }
            });
        }

        debug!("Successfully generated {} embeddings", embeddings.len());

        Ok(embeddings)
    }
}

// Function to determine the best device to use
pub fn get_best_device() -> Result<String> {
    // Check for CUDA
    if tch::Cuda::is_available() {
        info!("CUDA is available, using GPU");
        return Ok("cuda".to_string());
    }
    
    // Fall back to CPU
    info!("No GPU available, using CPU");
    Ok("cpu".to_string())
}

// Tokenizer implementation for BGE model
pub struct BgeTokenizer {
    tokenizer: Tokenizer,
}

impl BgeTokenizer {
    // Load the tokenizer from disk
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();
        let tokenizer_path = model_path.join("tokenizer.json");

        let tokenizer = Tokenizer::from_file(tokenizer_path)
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