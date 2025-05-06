// src/main.rs
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::ops::softmax;
use candle_transformers::models::bert::{BertModel, Config};
use log::{info, warn, debug};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

mod db;

const BATCH_SIZE: usize = 32;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    info!("Starting organization embedding generation");

    // Load environment variables from .env file
    let env_paths = [".env", ".env.local", "../.env"];
    let mut loaded_env = false;

    for path in env_paths.iter() {
        if Path::new(path).exists() {
            if let Err(e) = db::load_env_from_file(path) {
                warn!("Failed to load environment from {}: {}", path, e);
            } else {
                info!("Loaded environment variables from {}", path);
                loaded_env = true;
                break;
            }
        }
    }

    if !loaded_env {
        info!("No .env file found, using environment variables from system");
    }

    // Connect to database
    let pool = db::connect().await?;

    // Initialize Metal device for M2 acceleration
    let device = Device::new_metal(0).unwrap_or_else(|_| {
        warn!("Failed to initialize Metal device, falling back to CPU");
        Device::Cpu
    });

    // Load BGE model
    let model_path = "./models/bge-small-en-v1.5";
    let (model, tokenizer) = load_model(model_path, &device)?;

    // Process organizations in batches
    generate_embeddings(&pool, &model, &tokenizer, &device).await?;

    info!("Organization embedding generation completed");
    Ok(())
}

fn load_model(model_path: impl AsRef<Path>, device: &Device) -> Result<(BertModel, Tokenizer)> {
    let model_path = model_path.as_ref();
    info!("Loading BGE model from {}", model_path.display());
    
    // Load configuration
    let config_path = model_path.join("config.json");
    let config_data = std::fs::read_to_string(&config_path)?;
    let config: Config = serde_json::from_str(&config_data)?;
    debug!("Loaded model config: {:?}", config);
    
    // Load model weights
    let model_file = model_path.join("model.safetensors");
    let weights = candle_core::safetensors::load(&model_file, device)?;
    let vb = candle_nn::VarBuilder::from_tensors(weights, candle_core::DType::F32, device);
    
    // Create the model
    let model = BertModel::load(vb, &config)?;
    
    // Load tokenizer
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    
    info!("Model and tokenizer loaded successfully");
    Ok((model, tokenizer))
}

async fn generate_embeddings(
    pool: &db::PgPool,
    model: &BertModel,
    tokenizer: &Tokenizer,
    device: &Device,
) -> Result<()> {
    let conn = pool.get().await?;
    
    // Process until no organizations are left without embeddings
    let mut total_processed = 0;
    loop {
        // Get organizations without embeddings, including description
        let rows = conn
            .query(
                "SELECT id, name, description FROM organization WHERE embedding IS NULL LIMIT 1000",
                &[],
            )
            .await?;
        
        if rows.is_empty() {
            info!("No more organizations to process");
            break;
        }
        
        info!("Found {} organizations to process in this batch", rows.len());
        
        let mut organizations = Vec::with_capacity(rows.len());
        for row in rows {
            let id: String = row.get(0);
            let name: String = row.get(1);
            let description: Option<String> = row.get(2);
            organizations.push((id, name, description));
        }
        
        // Process in batches
        for chunk in organizations.chunks(BATCH_SIZE) {
            let start = Instant::now();
            
            // Prepare batch - combine name and description
            let texts: Vec<String> = chunk.iter().map(|(_, name, desc)| {
                if let Some(desc) = desc {
                    format!("{} {}", name, desc)
                } else {
                    name.clone()
                }
            }).collect();
            
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let encodings = tokenizer.encode_batch(text_refs, true)
                .map_err(anyhow::Error::msg)?;
            
            // Create token tensors
            let token_ids: Vec<_> = encodings
                .iter()
                .map(|e| e.get_ids().iter().map(|&x| x as i64).collect::<Vec<_>>())
                .collect();
            
            let attention_masks: Vec<_> = encodings
                .iter()
                .map(|e| e.get_attention_mask().iter().map(|&x| x as i64).collect::<Vec<_>>())
                .collect();
            
            let max_len = token_ids.iter().map(|x| x.len()).max().unwrap_or(0);
            
            // Pad both token_ids and attention_masks
            let padded_ids: Vec<_> = token_ids
                .iter()
                .map(|ids| {
                    let mut padded = vec![0; max_len];
                    padded[..ids.len()].copy_from_slice(ids);
                    padded
                })
                .collect();
            
            let padded_masks: Vec<_> = attention_masks
                .iter()
                .map(|mask| {
                    let mut padded = vec![0; max_len];
                    padded[..mask.len()].copy_from_slice(mask);
                    padded
                })
                .collect();
            
            let input_ids = Tensor::new(padded_ids, device)?;
            let attention_mask = Tensor::new(padded_masks, device)?;
            
            // Create token_type_ids (all zeros for single sequence tasks)
            let token_type_ids = input_ids.zeros_like()?;
            
            // Get embeddings
            let outputs = model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
            
            // Apply attention mask to outputs for mean pooling
            let (batch_size, seq_len, hidden_size) = outputs.dims3()?;
            let expanded_mask = attention_mask
                .unsqueeze(2)?
                .broadcast_as((batch_size, seq_len, hidden_size))?;
            
            // Zero out padded positions
            let masked_outputs = outputs.mul(&expanded_mask.to_dtype(outputs.dtype())?)?;
            
            // Sum and divide by the sum of attention mask to get mean pooling
            let sum_outputs = masked_outputs.sum(1)?;
            let sum_mask = attention_mask.sum(1)?;
            
            // Fix for the shape mismatch error - properly reshape for broadcasting
            let sum_mask_reshaped = sum_mask.reshape(&[batch_size, 1])?.to_dtype(outputs.dtype())?;
            let mean_pooled = sum_outputs.broadcast_div(&sum_mask_reshaped)?;
            
            // Convert to Vec for storage
            let embeddings: Vec<Vec<f32>> = mean_pooled
                .to_device(&Device::Cpu)?
                .to_vec2()?;
            
            // Store embeddings
            for ((id, _, _), embedding) in chunk.iter().zip(embeddings.iter()) {
                let query = "UPDATE organization SET embedding = $1, embedding_updated_at = CURRENT_TIMESTAMP WHERE id = $2";
                let embedded_vec = pgvector::Vector::from(embedding.clone());
                conn.execute(query, &[&embedded_vec, id]).await?;
            }
            
            total_processed += chunk.len();
            info!(
                "Processed {} organizations in {:?} (Total: {})",
                chunk.len(),
                start.elapsed(),
                total_processed
            );
        }
        
        // Check how many remain
        let count_query = "SELECT COUNT(*) FROM organization WHERE embedding IS NULL";
        let count_row = conn.query_one(count_query, &[]).await?;
        let remaining: i64 = count_row.get(0);
        info!("Remaining organizations without embeddings: {}", remaining);
    }
    
    info!("All organizations processed. Total: {}", total_processed);
    Ok(())
}