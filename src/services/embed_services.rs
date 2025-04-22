// src/services/embed_services.rs

use anyhow::Result;
use bb8::Pool;
use bb8_postgres::PostgresConnectionManager;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use futures::{StreamExt, stream};
use log::{debug, error, info, warn};
use pgvector::Vector;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tokio_postgres::NoTls;

const MODEL_PATH: &str = "./models/bge-small-en-v1.5/model.safetensors";
const TOKENIZER_PATH: &str = "./models/bge-small-en-v1.5/tokenizer.json";
const CONFIG_PATH: &str = "./models/bge-small-en-v1.5/config.json";
const BATCH_SIZE: usize = 64; // Adjust based on GPU memory
const CONCURRENT_BATCHES: usize = 4; // Adjust based on CPU cores and memory
// Max token length for BGE small (BERT-based models typically have 512 token limit)
const MAX_TOKEN_LENGTH: usize = 512;

/// A struct to hold taxonomy information
struct TaxonomyInfo {
    term: String,
    description: Option<String>,
}

/// Generate embeddings for services using MiniLM
pub async fn embed_services(pool: &Pool<PostgresConnectionManager<NoTls>>) -> Result<()> {
    let start_time = Instant::now();
    info!("Starting embedding service generation process");

    info!("Loading tokenizer from {}", TOKENIZER_PATH);
    let tokenizer = Tokenizer::from_file(TOKENIZER_PATH).map_err(|e| {
        error!("Failed to load tokenizer: {}", e);
        anyhow::anyhow!("Failed to load tokenizer: {}", e)
    })?;
    info!("Tokenizer loaded successfully");

    // use metal acceleration for mac
    info!("Attempting to initialize Metal device");
    let device = match Device::new_metal(0) {
        Ok(device) => {
            info!("Successfully initialized Metal device for GPU acceleration");
            device
        }
        Err(e) => {
            warn!("Metal not available, falling back to CPU: {}", e);
            Device::Cpu
        }
    };
    info!("Using device: {:?}", device);

    // Load weights and print the keys to understand the structure
    info!("Loading model weights from {}", MODEL_PATH);
    let weights_start = Instant::now();
    let weights = match candle_core::safetensors::load(MODEL_PATH, &device) {
        Ok(w) => {
            info!(
                "Model weights loaded successfully in {:.2?}",
                weights_start.elapsed()
            );
            w
        }
        Err(e) => {
            error!("Failed to load model weights: {}", e);
            return Err(anyhow::anyhow!("Failed to load model weights: {}", e));
        }
    };
    debug!("Model keys: {:?}", weights.keys());

    // Load configuration from the JSON file
    info!("Loading model configuration from {}", CONFIG_PATH);
    let config_contents = match std::fs::read_to_string(CONFIG_PATH) {
        Ok(c) => c,
        Err(e) => {
            error!("Failed to read config file: {}", e);
            return Err(anyhow::anyhow!("Failed to read config file: {}", e));
        }
    };

    let model_config: BertConfig = match serde_json::from_str(&config_contents) {
        Ok(c) => {
            info!("Model configuration loaded successfully");
            c
        }
        Err(e) => {
            error!("Failed to parse config JSON: {}", e);
            return Err(anyhow::anyhow!("Failed to parse config JSON: {}", e));
        }
    };

    // Create var builder from weights
    info!("Initializing model from weights");
    let model_init_start = Instant::now();
    let vb = candle_nn::VarBuilder::from_tensors(weights, DType::F32, &device);

    // Load model
    let model = match BertModel::load(vb, &model_config) {
        Ok(m) => {
            info!(
                "Model initialized successfully in {:.2?}",
                model_init_start.elapsed()
            );
            m
        }
        Err(e) => {
            error!("Failed to initialize model: {}", e);
            return Err(anyhow::anyhow!("Failed to initialize model: {}", e));
        }
    };

    // Wrap the model in Arc for sharing across tasks
    let model = Arc::new(model);
    let tokenizer = Arc::new(tokenizer);
    let device = Arc::new(device);

    // Get total count for progress tracking
    info!("Querying database for services requiring embeddings");
    let db_start = Instant::now();
    let client = match pool.get().await {
        Ok(c) => c,
        Err(e) => {
            error!("Failed to get DB connection: {}", e);
            return Err(anyhow::anyhow!("Failed to get DB connection: {}", e));
        }
    };

    let row = match client
        .query_one("SELECT COUNT(*) FROM service WHERE embedding IS NULL", &[])
        .await
    {
        Ok(r) => r,
        Err(e) => {
            error!("DB query failed: {}", e);
            return Err(anyhow::anyhow!("DB query failed: {}", e));
        }
    };

    let total_count: i64 = row.get(0);
    info!("Database query completed in {:.2?}", db_start.elapsed());
    info!("Found {} services without embeddings", total_count);

    if total_count == 0 {
        info!("No services to embed, skipping");
        return Ok(());
    }

    let processed = Arc::new(Mutex::new(0));
    let overall_start = Instant::now();

    // Process in parallel batches
    info!(
        "Starting parallel embedding generation with {} concurrent batches of size {}",
        CONCURRENT_BATCHES, BATCH_SIZE
    );

    let mut batch_number = 0;
    loop {
        batch_number += 1;
        let batch_start = Instant::now();
        info!("Processing batch group #{}", batch_number);

        // Get batch IDs (we'll fetch actual data in parallel tasks)
        let client = match pool.get().await {
            Ok(c) => c,
            Err(e) => {
                error!(
                    "Failed to get DB connection for batch {}: {}",
                    batch_number, e
                );
                return Err(anyhow::anyhow!("Failed to get DB connection: {}", e));
            }
        };

        let fetch_start = Instant::now();
        let rows = match client
            .query(
                &format!(
                    "SELECT id FROM service WHERE embedding IS NULL LIMIT {}",
                    BATCH_SIZE * CONCURRENT_BATCHES
                ),
                &[],
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                error!(
                    "Failed to fetch service IDs for batch {}: {}",
                    batch_number, e
                );
                return Err(anyhow::anyhow!("Failed to fetch service IDs: {}", e));
            }
        };
        debug!(
            "Fetched {} service IDs in {:.2?}",
            rows.len(),
            fetch_start.elapsed()
        );

        if rows.is_empty() {
            info!("No more services to process, embedding generation complete");
            break;
        }

        // Collect IDs
        let batch_ids: Vec<String> = rows.iter().map(|row| row.get::<_, String>("id")).collect();
        info!(
            "Processing {} services in this batch group",
            batch_ids.len()
        );

        // Process batches in parallel
        let batch_chunks: Vec<Vec<String>> = batch_ids
            .chunks(BATCH_SIZE)
            .map(|chunk| chunk.to_vec())
            .collect();

        info!("Split into {} parallel batches", batch_chunks.len());

        let parallel_start = Instant::now();
        let results = stream::iter(batch_chunks)
            .map(|ids_chunk| {
                let pool = pool.clone();
                let model = model.clone();
                let tokenizer = tokenizer.clone();
                let device = device.clone();
                let processed = processed.clone();
                let batch_id = uuid::Uuid::new_v4().to_string()[..8].to_string(); // For tracking

                async move {
                    debug!(
                        "Starting parallel batch {} with {} services",
                        batch_id,
                        ids_chunk.len()
                    );
                    let result = process_batch(
                        &pool,
                        &ids_chunk,
                        &model,
                        &tokenizer,
                        &device,
                        &processed,
                        total_count,
                        &batch_id,
                    )
                    .await;

                    if let Err(ref e) = result {
                        error!("Batch {} failed: {}", batch_id, e);
                    } else {
                        debug!("Batch {} completed successfully", batch_id);
                    }

                    result
                }
            })
            .buffer_unordered(CONCURRENT_BATCHES)
            .collect::<Vec<Result<()>>>()
            .await;

        info!(
            "All parallel batches in group #{} completed in {:.2?}",
            batch_number,
            parallel_start.elapsed()
        );

        // Check for errors
        let mut error_count = 0;
        for result in results {
            if let Err(e) = result {
                error_count += 1;
                error!("Batch processing error: {}", e);
            }
        }

        if error_count > 0 {
            warn!(
                "{} batches had errors in batch group #{}",
                error_count, batch_number
            );
        }

        info!(
            "Batch group #{} completed in {:.2?}",
            batch_number,
            batch_start.elapsed()
        );
    }

    let total_elapsed = overall_start.elapsed();
    let processed_lock = processed.lock().await;

    info!(
        "Completed embedding generation for all {} services in {:.2?}",
        *processed_lock, total_elapsed
    );

    if *processed_lock > 0 {
        let avg_time = total_elapsed.as_secs_f64() / (*processed_lock as f64);
        info!(
            "Average processing time: {:.3?} per service",
            std::time::Duration::from_secs_f64(avg_time)
        );
    }

    info!(
        "Total embedding service execution time: {:.2?}",
        start_time.elapsed()
    );
    Ok(())
}

/// Fetch taxonomy information for a service
async fn fetch_taxonomy_info(
    client: &tokio_postgres::Client, 
    service_id: &str
) -> Result<Vec<TaxonomyInfo>> {
    debug!("Fetching taxonomy info for service: {}", service_id);
    
    let query = "
        SELECT tt.term, tt.description
        FROM service_taxonomy st
        JOIN taxonomy_term tt ON st.taxonomy_term_id = tt.id
        WHERE st.service_id = $1
    ";
    
    let rows = match client.query(query, &[&service_id]).await {
        Ok(r) => r,
        Err(e) => {
            error!("Failed to fetch taxonomy info: {}", e);
            return Err(anyhow::anyhow!("Failed to fetch taxonomy info: {}", e));
        }
    };
    
    let mut taxonomy_info = Vec::with_capacity(rows.len());
    for row in rows {
        taxonomy_info.push(TaxonomyInfo {
            term: row.get("term"),
            description: row.get("description"),
        });
    }
    
    debug!("Found {} taxonomy terms for service {}", taxonomy_info.len(), service_id);
    Ok(taxonomy_info)
}

/// Generate embeddings for a batch of texts
fn generate_batch_embeddings(
    model: &BertModel,
    input_ids: Vec<Vec<u32>>,
    attention_mask: Vec<Vec<u32>>,
    device: &Device,
    batch_id: &str,
) -> Result<Vec<Vec<f32>>> {
    let embedding_start = Instant::now();
    debug!(
        "Batch {}: Starting embedding generation for {} texts",
        batch_id,
        input_ids.len()
    );

    // Convert input IDs to tensors
    let max_len = input_ids.iter().map(|ids| ids.len()).max().unwrap_or(0);
    let batch_size = input_ids.len();
    debug!(
        "Batch {}: Max sequence length: {}, batch size: {}",
        batch_id, max_len, batch_size
    );

    // Pad inputs to the same length
    let padding_start = Instant::now();
    let mut input_ids_padded = vec![vec![0_u32; max_len]; batch_size];
    let mut attention_mask_padded = vec![vec![0_u32; max_len]; batch_size];

    for (i, (ids, mask)) in input_ids.iter().zip(attention_mask.iter()).enumerate() {
        for (j, &id) in ids.iter().enumerate() {
            input_ids_padded[i][j] = id;
        }
        for (j, &mask_val) in mask.iter().enumerate() {
            attention_mask_padded[i][j] = mask_val;
        }
    }
    debug!(
        "Batch {}: Padding completed in {:.2?}",
        batch_id,
        padding_start.elapsed()
    );

    // Create tensors
    let tensor_start = Instant::now();
    let input_ids_tensor = match Tensor::new(input_ids_padded, device) {
        Ok(t) => t,
        Err(e) => {
            error!(
                "Batch {}: Failed to create input_ids tensor: {}",
                batch_id, e
            );
            return Err(anyhow::anyhow!("Failed to create input_ids tensor: {}", e));
        }
    };

    let input_ids_tensor = match input_ids_tensor.reshape((batch_size, max_len)) {
        Ok(t) => t,
        Err(e) => {
            error!(
                "Batch {}: Failed to reshape input_ids tensor: {}",
                batch_id, e
            );
            return Err(anyhow::anyhow!("Failed to reshape input_ids tensor: {}", e));
        }
    };

    let attention_mask_tensor = match Tensor::new(attention_mask_padded, device) {
        Ok(t) => t,
        Err(e) => {
            error!(
                "Batch {}: Failed to create attention_mask tensor: {}",
                batch_id, e
            );
            return Err(anyhow::anyhow!(
                "Failed to create attention_mask tensor: {}",
                e
            ));
        }
    };

    let attention_mask_tensor = match attention_mask_tensor.reshape((batch_size, max_len)) {
        Ok(t) => t,
        Err(e) => {
            error!(
                "Batch {}: Failed to reshape attention_mask tensor: {}",
                batch_id, e
            );
            return Err(anyhow::anyhow!(
                "Failed to reshape attention_mask tensor: {}",
                e
            ));
        }
    };
    debug!(
        "Batch {}: Tensor creation completed in {:.2?}",
        batch_id,
        tensor_start.elapsed()
    );

    // Generate embeddings
    debug!("Batch {}: Running model forward pass", batch_id);
    let model_start = Instant::now();
    let embeddings = match model.forward(&input_ids_tensor, &attention_mask_tensor, None) {
        Ok(e) => {
            debug!(
                "Batch {}: Model forward pass completed in {:.2?}",
                batch_id,
                model_start.elapsed()
            );
            e
        }
        Err(e) => {
            error!("Batch {}: Model forward pass failed: {}", batch_id, e);
            return Err(anyhow::anyhow!("Model forward pass failed: {}", e));
        }
    };

    // Mean pooling - take average of all token embeddings
    let pooling_start = Instant::now();
    let embedding_dim = embeddings.dims()[2];
    debug!(
        "Batch {}: Performing mean pooling with embedding dimension {}",
        batch_id, embedding_dim
    );
    let mut pooled_embeddings = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let mask_sum = attention_mask[i].iter().sum::<u32>() as usize;
        if mask_sum == 0 {
            // If mask is all zeros, use zero embedding
            debug!(
                "Batch {}: Sample {} has zero attention mask, using zero embedding",
                batch_id, i
            );
            pooled_embeddings.push(vec![0.0; embedding_dim]);
            continue;
        }

        // Extract the embeddings for this sample
        let sample_embedding = match embeddings.get(i) {
            Ok(e) => e,
            Err(e) => {
                error!(
                    "Batch {}: Failed to get embeddings for sample {}: {}",
                    batch_id, i, e
                );
                return Err(anyhow::anyhow!(
                    "Failed to get embeddings for sample {}: {}",
                    i,
                    e
                ));
            }
        };

        // Sum the embeddings of non-masked tokens
        let mut sum_embedding = vec![0.0; embedding_dim];
        for j in 0..mask_sum {
            // Explicitly specify f32 type
            let token_embedding: Vec<f32> = match sample_embedding.get(j) {
                Ok(t) => match t.to_vec1() {
                    Ok(v) => v,
                    Err(e) => {
                        error!(
                            "Batch {}: Failed to convert tensor to vector for sample {}, token {}: {}",
                            batch_id, i, j, e
                        );
                        return Err(anyhow::anyhow!("Failed to convert tensor to vector: {}", e));
                    }
                },
                Err(e) => {
                    error!(
                        "Batch {}: Failed to get token embedding for sample {}, token {}: {}",
                        batch_id, i, j, e
                    );
                    return Err(anyhow::anyhow!("Failed to get token embedding: {}", e));
                }
            };

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
        "Batch {}: Mean pooling completed in {:.2?}",
        batch_id,
        pooling_start.elapsed()
    );
    debug!(
        "Batch {}: Total embedding generation completed in {:.2?}",
        batch_id,
        embedding_start.elapsed()
    );

    Ok(pooled_embeddings)
}

/// Build text for embedding that includes taxonomy information
fn build_text_for_embedding(
    name: &str, 
    description: &str, 
    taxonomies: &[TaxonomyInfo],
    tokenizer: &Tokenizer,
) -> String {
    // Extract taxonomy terms and descriptions
    let taxonomy_terms: Vec<String> = taxonomies.iter()
        .map(|t| t.term.clone())
        .collect();
    
    let taxonomy_descriptions: Vec<String> = taxonomies.iter()
        .filter_map(|t| t.description.clone())
        .collect();
    
    // Use the templated approach with section markers
    let text = format!(
        "[SERVICE] {} [DESCRIPTION] {} [CATEGORIES] {} [CATEGORY_DETAILS] {}",
        name.trim(),
        description.trim(),
        taxonomy_terms.join(", "),
        taxonomy_descriptions.join(" ")
    );
    
    // Check if the text is too long for the model and truncate if necessary
    let encoding = tokenizer.encode(text.clone(), true).unwrap();
    if encoding.get_ids().len() > MAX_TOKEN_LENGTH {
        debug!("Text too long ({} tokens), truncating to {} tokens", 
               encoding.get_ids().len(), MAX_TOKEN_LENGTH);
        
        // Start with the most important parts - service name and taxonomy terms
        let essential_text = format!(
            "[SERVICE] {} [CATEGORIES] {}",
            name.trim(), 
            taxonomy_terms.join(", ")
        );
        
        // Calculate remaining space
        let essential_encoding = tokenizer.encode(essential_text.clone(), true).unwrap();
        let remaining_tokens = MAX_TOKEN_LENGTH - essential_encoding.get_ids().len();
        
        if remaining_tokens <= 0 {
            // Just return the essential parts
            return essential_text;
        }
        
        // Now add description and taxonomy details with even split
        let desc_tokens = remaining_tokens / 2;
        let desc_encoding = tokenizer.encode(description.trim(), true).unwrap();
        let desc_truncated = if desc_encoding.get_ids().len() > desc_tokens {
            // Truncate description text (this is simplified - ideally would truncate at word boundaries)
            let desc_token_ids = desc_encoding.get_ids();
            let truncated_ids = &desc_token_ids[0..desc_tokens];
            tokenizer.decode(truncated_ids, false).unwrap()
        } else {
            description.trim().to_string()
        };
        
        let tax_desc_tokens = remaining_tokens - desc_encoding.get_ids().len().min(desc_tokens);
        let tax_desc_text = taxonomy_descriptions.join(" ");
        let tax_desc_encoding = tokenizer.encode(tax_desc_text.clone(), true).unwrap();
        let tax_desc_truncated = if tax_desc_encoding.get_ids().len() > tax_desc_tokens {
            let tax_desc_token_ids = tax_desc_encoding.get_ids();
            let truncated_ids = &tax_desc_token_ids[0..tax_desc_tokens];
            tokenizer.decode(truncated_ids, false).unwrap()
        } else {
            tax_desc_text
        };
        
        format!(
            "[SERVICE] {} [DESCRIPTION] {} [CATEGORIES] {} [CATEGORY_DETAILS] {}",
            name.trim(),
            desc_truncated.trim(),
            taxonomy_terms.join(", "),
            tax_desc_truncated.trim()
        )
    } else {
        text
    }
}

/// Process a single batch of service IDs
async fn process_batch(
    pool: &Pool<PostgresConnectionManager<NoTls>>,
    ids: &[String],
    model: &Arc<BertModel>,
    tokenizer: &Arc<Tokenizer>,
    device: &Arc<Device>,
    processed_counter: &Arc<Mutex<i64>>,
    total_count: i64,
    batch_id: &str,
) -> Result<()> {
    let batch_start = Instant::now();
    debug!(
        "Batch {}: Starting processing of {} service IDs",
        batch_id,
        ids.len()
    );

    // Get a connection from the pool
    debug!("Batch {}: Requesting database connection", batch_id);
    let db_conn_start = Instant::now();
    let mut client = match pool.get().await {
        Ok(c) => {
            debug!(
                "Batch {}: Got database connection in {:.2?}",
                batch_id,
                db_conn_start.elapsed()
            );
            c
        }
        Err(e) => {
            error!(
                "Batch {}: Failed to get database connection: {}",
                batch_id, e
            );
            return Err(anyhow::anyhow!("Failed to get database connection: {}", e));
        }
    };

    // Fetch service data
    let placeholders: Vec<String> = ids
        .iter()
        .enumerate()
        .map(|(i, _)| format!("${}", i + 1))
        .collect();

    let query = format!(
        "SELECT id, name, description FROM service WHERE id IN ({})",
        placeholders.join(",")
    );

    let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = ids
        .iter()
        .map(|id| id as &(dyn tokio_postgres::types::ToSql + Sync))
        .collect();

    debug!(
        "Batch {}: Fetching service data for {} IDs",
        batch_id,
        ids.len()
    );
    let fetch_start = Instant::now();
    let rows = match client.query(&query, &params[..]).await {
        Ok(r) => {
            debug!(
                "Batch {}: Fetched {} services in {:.2?}",
                batch_id,
                r.len(),
                fetch_start.elapsed()
            );
            r
        }
        Err(e) => {
            error!("Batch {}: Failed to fetch service data: {}", batch_id, e);
            return Err(anyhow::anyhow!("Failed to fetch service data: {}", e));
        }
    };

    // Prepare text for embedding
    let mut batch_ids = Vec::new();
    let mut texts = Vec::new();

    for row in &rows {
        let id = row.get::<_, String>("id");
        let name = row.get::<_, Option<String>>("name").unwrap_or_default();
        let description = row
            .get::<_, Option<String>>("description")
            .unwrap_or_default();

        // Fetch taxonomy information for this service
        let taxonomy_info = match fetch_taxonomy_info(&client, &id).await {
            Ok(info) => info,
            Err(e) => {
                warn!(
                    "Batch {}: Failed to fetch taxonomy info for service {}: {}. Continuing with empty taxonomy.",
                    batch_id, id, e
                );
                Vec::new()
            }
        };

        // Build text for embedding
        let text = build_text_for_embedding(&name, &description, &taxonomy_info, tokenizer);
        
        if !text.trim().is_empty() {
            batch_ids.push(id);
            texts.push(text);
        }
    }

    if texts.is_empty() {
        debug!("Batch {}: No valid texts to process, skipping", batch_id);
        return Ok(());
    }

    debug!(
        "Batch {}: Prepared {} texts for embedding",
        batch_id,
        texts.len()
    );

    // Generate embeddings
    debug!("Batch {}: Tokenizing texts", batch_id);
    let tokenize_start = Instant::now();
    let encodings = match tokenizer.encode_batch(texts, true) {
        Ok(e) => {
            debug!(
                "Batch {}: Tokenization completed in {:.2?}",
                batch_id,
                tokenize_start.elapsed()
            );
            e
        }
        Err(e) => {
            error!("Batch {}: Failed to encode texts: {}", batch_id, e);
            return Err(anyhow::anyhow!("Failed to encode batch: {}", e));
        }
    };

    let input_ids = encodings.iter().map(|e| e.get_ids().to_vec()).collect();
    let attention_mask = encodings
        .iter()
        .map(|e| e.get_attention_mask().to_vec())
        .collect();

    debug!(
        "Batch {}: Generating embeddings for {} texts",
        batch_id,
        encodings.len()
    );
    let embeddings =
        match generate_batch_embeddings(model, input_ids, attention_mask, device, batch_id) {
            Ok(e) => e,
            Err(e) => {
                error!("Batch {}: Failed to generate embeddings: {}", batch_id, e);
                return Err(anyhow::anyhow!("Failed to generate embeddings: {}", e));
            }
        };

    // Update database with embeddings
    debug!(
        "Batch {}: Starting database transaction to update {} embeddings",
        batch_id,
        embeddings.len()
    );
    let db_update_start = Instant::now();
    let transaction = match client.transaction().await {
        Ok(t) => t,
        Err(e) => {
            error!(
                "Batch {}: Failed to start database transaction: {}",
                batch_id, e
            );
            return Err(anyhow::anyhow!(
                "Failed to start database transaction: {}",
                e
            ));
        }
    };

    for (i, (id, embedding)) in batch_ids.iter().zip(embeddings).enumerate() {
        // Convert Vec<f32> to pgvector::Vector
        let pgvector = Vector::from(embedding);

        match transaction
            .execute(
                "UPDATE service SET embedding = $1 WHERE id = $2",
                &[&pgvector, id],
            )
            .await
        {
            Ok(_) => {}
            Err(e) => {
                error!(
                    "Batch {}: Failed to update embedding for service {} ({}/{}): {}",
                    batch_id,
                    id,
                    i + 1,
                    batch_ids.len(),
                    e
                );
                return Err(anyhow::anyhow!("Failed to update embedding: {}", e));
            }
        }
    }

    match transaction.commit().await {
        Ok(_) => {
            debug!(
                "Batch {}: Database transaction committed successfully in {:.2?}",
                batch_id,
                db_update_start.elapsed()
            );
        }
        Err(e) => {
            error!(
                "Batch {}: Failed to commit database transaction: {}",
                batch_id, e
            );
            return Err(anyhow::anyhow!(
                "Failed to commit database transaction: {}",
                e
            ));
        }
    }

    // Update progress counter
    let mut processed = processed_counter.lock().await;
    *processed += batch_ids.len() as i64;
    let progress_percentage = (*processed as f64 / total_count as f64) * 100.0;

    info!(
        "Batch {}: Embedded {}/{} services ({:.1}%) - batch completed in {:.2?}",
        batch_id,
        *processed,
        total_count,
        progress_percentage,
        batch_start.elapsed()
    );

    Ok(())
}