// src/services/embed_services.rs

use anyhow::{Context, Result};
use bb8::Pool;
use bb8_postgres::PostgresConnectionManager;
use log::{debug, error, info, warn};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Semaphore, mpsc};
use tokio_postgres::NoTls;

use crate::services::config::{
    BATCH_SIZE, CONCURRENT_BATCHES, CONFIG_PATH, MODEL_PATH, TOKENIZER_PATH,
};
use crate::services::data_fetcher::DataFetcher;
use crate::services::data_writer::{DatabaseWriter, ProgressTracker};
use crate::services::inference::InferenceEngine;
use crate::services::tokenizer::{TokenizationManager, load_tokenizer};
use crate::services::types::{ServiceData, TokenizedBatch};

/// Configuration for the embedding pipeline
#[derive(Clone)]
pub struct EmbeddingConfig {
    /// Path to the tokenizer JSON file
    pub tokenizer_path: String,
    /// Path to the model weights file
    pub model_path: String,
    /// Path to the model configuration file
    pub config_path: String,
    /// Number of services to process in a single batch
    pub batch_size: usize,
    /// Number of batches to process concurrently
    pub concurrent_batches: usize,
    /// Number of concurrent database fetchers
    pub concurrent_fetchers: usize,
    /// Number of concurrent tokenizers
    pub concurrent_tokenizers: usize,
    /// Number of concurrent inference operations
    pub concurrent_inference: usize,
    /// Number of concurrent database writers
    pub concurrent_writers: usize,
    /// Force CPU execution (even if GPU is available)
    pub force_cpu: bool,
    /// Progress update interval in seconds
    pub progress_update_interval: u64,
    /// If true, create index after embedding generation
    pub create_index: bool,
    pub id_batch_accumulation: usize,
    pub max_meta_batch_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            tokenizer_path: TOKENIZER_PATH.to_string(),
            model_path: MODEL_PATH.to_string(),
            config_path: CONFIG_PATH.to_string(),
            batch_size: BATCH_SIZE,
            concurrent_batches: CONCURRENT_BATCHES,
            concurrent_fetchers: 4,
            concurrent_tokenizers: 8,
            concurrent_inference: 2,
            concurrent_writers: 4,
            force_cpu: false,
            progress_update_interval: 5,
            create_index: true,
            id_batch_accumulation: 5,
            max_meta_batch_size: 500,
        }
    }
}

/// Main function to embed services using a parallel pipeline
///
/// This function orchestrates the entire embedding process:
/// 1. Fetches services from the database
/// 2. Tokenizes the service texts
/// 3. Generates embeddings using the model
/// 4. Saves the embeddings back to the database
///
/// # Arguments
///
/// * `pool` - Database connection pool
///
/// # Returns
///
/// * `Result<()>` - Success or error result
pub async fn embed_services(pool: &Pool<PostgresConnectionManager<NoTls>>) -> Result<()> {
    // Use default configuration
    embed_services_with_config(pool, EmbeddingConfig::default()).await
}

/// Embed services using custom configuration
///
/// # Arguments
///
/// * `pool` - Database connection pool
/// * `config` - Custom configuration for the embedding pipeline
///
/// # Returns
///
/// * `Result<()>` - Success or error result
pub async fn embed_services_with_config(
    pool: &Pool<PostgresConnectionManager<NoTls>>,
    config: EmbeddingConfig,
) -> Result<()> {
    let overall_start = Instant::now();
    info!("Starting embedding service generation process");

    // Initialize components
    let data_fetcher =
        DataFetcher::new(pool.clone(), config.batch_size, config.concurrent_fetchers);

    // Count services needing embeddings
    let total_count = data_fetcher.count_services_needing_embeddings().await?;
    info!(
        "Found {} services without embeddings (embedding_v2)",
        total_count
    );

    // If no services need embeddings, exit early
    if total_count == 0 {
        info!("No services to embed, skipping");
        return Ok(());
    }

    // Initialize tokenizer
    info!("Loading tokenizer from {}", config.tokenizer_path);
    let tokenizer = load_tokenizer(&config.tokenizer_path).context("Failed to load tokenizer")?;

    let tokenization_manager = Arc::new(TokenizationManager::new(tokenizer.clone()));

    // Initialize model
    info!("Initializing inference engine");
    let inference_engine = Arc::new(
        InferenceEngine::new(
            Some(&config.model_path),
            Some(&config.config_path),
            config.force_cpu,
        )
        .await
        .context("Failed to initialize inference engine")?,
    );

    // Initialize database writer
    let db_writer = DatabaseWriter::new(pool.clone(), config.concurrent_writers);

    // Initialize progress tracker
    let progress_tracker = ProgressTracker::new(
        total_count,
        Duration::from_secs(config.progress_update_interval),
    );

    // Run the embedding pipeline
    let result = run_embedding_pipeline(
        data_fetcher,
        tokenization_manager,
        inference_engine,
        db_writer,
        progress_tracker,
        &config,
        total_count,
    )
    .await;

    // Report final results
    match &result {
        Ok(processed_count) => {
            info!(
                "Successfully embedded {} services in {:.2?}",
                processed_count,
                overall_start.elapsed()
            );
        }
        Err(e) => {
            error!("Error during embedding process: {}", e);
        }
    }

    // Create index if specified and the pipeline completed successfully
    if config.create_index && result.is_ok() {
        let db_writer = DatabaseWriter::new(pool.clone(), 1);
        match db_writer.ensure_embedding_index_exists().await {
            Ok(created) => {
                if created {
                    info!("Successfully created embedding_v2 index");
                } else {
                    info!("Embedding_v2 index already exists");
                }
            }
            Err(e) => {
                warn!("Failed to create embedding_v2 index: {}", e);
            }
        }
    }

    // Return the final result
    result.map(|_| ())
}

/// Runs the embedding pipeline with all stages
///
/// This function creates and manages the pipeline with multiple stages:
/// 1. Fetching services from the database
/// 2. Tokenizing service texts
/// 3. Generating embeddings
/// 4. Saving embeddings back to the database
///
/// Each stage operates concurrently with controlled parallelism.
async fn run_embedding_pipeline(
    data_fetcher: DataFetcher,
    tokenization_manager: Arc<TokenizationManager>,
    inference_engine: Arc<InferenceEngine>,
    db_writer: DatabaseWriter,
    progress_tracker: ProgressTracker,
    config: &EmbeddingConfig,
    total_count: i64,
) -> Result<i64> {
    let pipeline_start = Instant::now();
    info!(
        "Starting embedding pipeline with batch_size={}, concurrent_batches={}",
        config.batch_size, config.concurrent_batches
    );

    // Create channels for the pipeline stages
    let (fetch_tx, fetch_rx) = mpsc::channel(config.concurrent_batches);
    let (tokenize_tx, tokenize_rx) = mpsc::channel(config.concurrent_batches);
    let (inference_tx, inference_rx) = mpsc::channel(config.concurrent_batches);
    let (write_tx, write_rx) = mpsc::channel(config.concurrent_batches);

    // Create semaphores to control concurrency at each stage
    let fetch_semaphore = Arc::new(Semaphore::new(config.concurrent_fetchers));
    let tokenize_semaphore = Arc::new(Semaphore::new(config.concurrent_tokenizers));
    let inference_semaphore = Arc::new(Semaphore::new(config.concurrent_inference));
    let write_semaphore = Arc::new(Semaphore::new(config.concurrent_writers));

    // Spawn the pipeline stages - note we're passing config to fetch_stage
    let fetch_handle = spawn_fetch_stage(
        data_fetcher,
        fetch_tx,
        fetch_semaphore.clone(),
        config.clone(), // Clone it here
    );

    let tokenize_handle = spawn_tokenize_stage(
        fetch_rx,
        tokenize_tx,
        tokenization_manager,
        tokenize_semaphore.clone(),
    );
    let inference_handle = spawn_inference_stage(
        tokenize_rx,
        inference_tx,
        inference_engine,
        inference_semaphore.clone(),
    );
    let write_handle = spawn_write_stage(
        inference_rx,
        write_tx,
        db_writer,
        write_semaphore.clone(),
        progress_tracker.clone(),
    );

    // Create a task to monitor the pipeline
    let monitor_handle = spawn_pipeline_monitor(write_rx, progress_tracker.clone(), total_count);

    // Wait for all pipeline stages to complete
    let (fetch_result, tokenize_result, inference_result, write_result, monitor_result) = tokio::join!(
        fetch_handle,
        tokenize_handle,
        inference_handle,
        write_handle,
        monitor_handle
    );

    // Check for errors in each stage
    fetch_result.context("Error in fetch stage")?;
    tokenize_result.context("Error in tokenize stage")?;
    inference_result.context("Error in inference stage")?;
    write_result.context("Error in write stage")?;
    let processed_count = monitor_result.context("Error in monitor stage")?;

    info!(
        "Embedding pipeline completed in {:.2?}",
        pipeline_start.elapsed()
    );

    Ok(processed_count)
}

async fn process_accumulated_ids(
    data_fetcher: &DataFetcher,
    accumulated_ids: &[String],
    tx: &mpsc::Sender<Vec<ServiceData>>,
    semaphore: &Arc<Semaphore>,
    batch_id: String,
) -> Result<()> {
    if accumulated_ids.is_empty() {
        return Ok(());
    }

    // Acquire a permit from the semaphore to control concurrency
    let permit = match semaphore.acquire().await {
        Ok(permit) => permit,
        Err(e) => {
            error!("Failed to acquire fetch semaphore: {}", e);
            return Err(anyhow::anyhow!("Failed to acquire semaphore: {}", e));
        }
    };

    // Fetch data for all accumulated IDs in parallel batches
    match data_fetcher
        .fetch_service_data_in_batches(accumulated_ids)
        .await
    {
        Ok(batches) => {
            debug!(
                "{}: Fetched {} batches with {} total services",
                batch_id,
                batches.len(),
                batches.iter().map(|batch| batch.len()).sum::<usize>()
            );

            // Send each batch to the tokenization stage separately
            for (i, batch) in batches.into_iter().enumerate() {
                if batch.is_empty() {
                    continue;
                }

                debug!(
                    "{}-{}: Sending batch of {} services",
                    batch_id,
                    i,
                    batch.len()
                );

                match tx.send(batch).await {
                    Ok(_) => {
                        debug!(
                            "{}-{}: Successfully sent services to tokenize stage",
                            batch_id, i
                        );
                    }
                    Err(e) => {
                        error!(
                            "{}-{}: Failed to send services to tokenize stage: {}",
                            batch_id, i, e
                        );
                        return Err(anyhow::anyhow!(
                            "Failed to send batch to tokenize stage: {}",
                            e
                        ));
                    }
                }
            }
        }
        Err(e) => {
            error!(
                "{}: Error fetching service data in batches: {}",
                batch_id, e
            );
            return Err(anyhow::anyhow!("Failed to fetch service data: {}", e));
        }
    }

    // Release the permit
    drop(permit);

    Ok(())
}
/// Spawns the fetch stage of the pipeline
///
/// This stage fetches service IDs from the database and retrieves the complete
/// service data for each batch.
async fn spawn_fetch_stage(
    data_fetcher: DataFetcher,
    tx: mpsc::Sender<Vec<ServiceData>>,
    semaphore: Arc<Semaphore>,
    config: EmbeddingConfig,
) -> Result<()> {
    // Start a tokio task for the fetch stage
    tokio::spawn(async move {
        let stage_start = Instant::now();
        debug!("Starting fetch stage with batched fetching");

        let mut total_fetched = 0;
        let mut batch_num = 0;
        let mut accumulated_ids: Vec<String> = Vec::new();

        loop {
            batch_num += 1;
            let batch_id = format!("fetch-{}", batch_num);

            // Fetch a batch of service IDs
            let service_ids = match data_fetcher
                .fetch_services_needing_embeddings(config.batch_size)
                .await
            {
                Ok(ids) => ids,
                Err(e) => {
                    error!("Error fetching service IDs: {}", e);
                    continue;
                }
            };

            // If no more services, process any accumulated IDs and finish
            if service_ids.is_empty() {
                if !accumulated_ids.is_empty() {
                    debug!(
                        "Processing final accumulated batch of {} IDs",
                        accumulated_ids.len()
                    );
                    if let Err(e) = process_accumulated_ids(
                        &data_fetcher,
                        &accumulated_ids,
                        &tx,
                        &semaphore,
                        batch_id,
                    )
                    .await
                    {
                        error!("Error processing final batch: {}", e);
                    }
                }
                debug!("No more services to fetch, fetch stage complete");
                break;
            }

            // Add new IDs to accumulation
            accumulated_ids.extend(service_ids);

            // When we've accumulated enough batches or reached size limit, process them
            if batch_num % config.id_batch_accumulation == 0
                || accumulated_ids.len() >= config.max_meta_batch_size
            {
                debug!(
                    "Processing accumulated batch of {} IDs",
                    accumulated_ids.len()
                );
                total_fetched += accumulated_ids.len();

                let ids_to_process = std::mem::take(&mut accumulated_ids);
                if let Err(e) = process_accumulated_ids(
                    &data_fetcher,
                    &ids_to_process,
                    &tx,
                    &semaphore,
                    batch_id.clone(),
                )
                .await
                {
                    error!("Error processing accumulated batch: {}", e);
                }
            }
        }

        // Signal completion by dropping the sender
        drop(tx);

        info!(
            "Fetch stage completed in {:.2?}, processed {} services in {} batches",
            stage_start.elapsed(),
            total_fetched,
            batch_num - 1
        );

        Ok(()) as Result<()>
    })
    .await?
}

/// Spawns the tokenize stage of the pipeline
///
/// This stage tokenizes the service texts for model input.
async fn spawn_tokenize_stage(
    mut rx: mpsc::Receiver<Vec<ServiceData>>,
    tx: mpsc::Sender<TokenizedBatch>,
    tokenization_manager: Arc<TokenizationManager>,
    semaphore: Arc<Semaphore>,
) -> Result<()> {
    // Start a tokio task for the tokenize stage
    tokio::spawn(async move {
        let stage_start = Instant::now();
        debug!("Starting tokenize stage");

        let mut total_tokenized = 0;
        let mut batch_num = 0;

        while let Some(services) = rx.recv().await {
            batch_num += 1;
            let batch_id = format!("tokenize-{}", batch_num);

            if services.is_empty() {
                debug!("{}: Empty batch received, skipping", batch_id);
                continue;
            }

            let service_count = services.len();
            debug!(
                "{}: Received {} services for tokenization",
                batch_id, service_count
            );

            // Acquire a permit from the semaphore to control concurrency
            let permit = match semaphore.acquire().await {
                Ok(permit) => permit,
                Err(e) => {
                    error!("Failed to acquire tokenize semaphore: {}", e);
                    continue;
                }
            };

            // Tokenize the services
            match tokenization_manager.tokenize_services(services).await {
                Ok(tokenized_batch) => {
                    // Check if we actually have tokens
                    if tokenized_batch.input_ids.is_empty() {
                        debug!("{}: No tokens generated, skipping batch", batch_id);
                        drop(permit);
                        continue;
                    }

                    total_tokenized += tokenized_batch.service_ids.len();
                    debug!(
                        "{}: Tokenized {} services, total so far: {}",
                        batch_id,
                        tokenized_batch.service_ids.len(),
                        total_tokenized
                    );

                    // Forward the tokenized batch to the next stage
                    match tx.send(tokenized_batch).await {
                        Ok(_) => {
                            debug!("{}: Successfully sent tokens to inference stage", batch_id);
                        }
                        Err(e) => {
                            error!(
                                "{}: Failed to send tokens to inference stage: {}",
                                batch_id, e
                            );
                            break;
                        }
                    }
                }
                Err(e) => {
                    error!("{}: Error tokenizing services: {}", batch_id, e);
                }
            }

            // Release the permit
            drop(permit);
        }

        // Signal completion by dropping the sender
        drop(tx);

        info!(
            "Tokenize stage completed in {:.2?}, processed {} services in {} batches",
            stage_start.elapsed(),
            total_tokenized,
            batch_num
        );

        Ok(()) as Result<()>
    })
    .await?
}

/// Spawns the inference stage of the pipeline
///
/// This stage generates embeddings using the model.
async fn spawn_inference_stage(
    mut rx: mpsc::Receiver<TokenizedBatch>,
    tx: mpsc::Sender<(Vec<String>, Vec<Vec<f32>>)>,
    inference_engine: Arc<InferenceEngine>,
    semaphore: Arc<Semaphore>,
) -> Result<()> {
    // Start a tokio task for the inference stage
    tokio::spawn(async move {
        let stage_start = Instant::now();
        debug!("Starting inference stage");

        let mut total_inferred = 0;
        let mut batch_num = 0;

        while let Some(batch) = rx.recv().await {
            batch_num += 1;
            let batch_id = format!("inference-{}", batch_num);

            if batch.service_ids.is_empty() || batch.input_ids.is_empty() {
                debug!("{}: Empty batch received, skipping", batch_id);
                continue;
            }

            let service_count = batch.service_ids.len();
            debug!(
                "{}: Received {} services for inference",
                batch_id, service_count
            );

            // Acquire a permit from the semaphore to control concurrency
            let permit = match semaphore.acquire().await {
                Ok(permit) => permit,
                Err(e) => {
                    error!("Failed to acquire inference semaphore: {}", e);
                    continue;
                }
            };

            // Generate embeddings
            match inference_engine.generate_embeddings(batch).await {
                Ok((service_ids, embeddings, metrics)) => {
                    // Skip if no embeddings were generated
                    if service_ids.is_empty() || embeddings.is_empty() {
                        debug!("{}: No embeddings generated, skipping batch", batch_id);
                        drop(permit);
                        continue;
                    }

                    total_inferred += service_ids.len();
                    debug!(
                        "{}: Generated embeddings for {} services in {:.2?}, total so far: {}",
                        batch_id,
                        service_ids.len(),
                        metrics.inference_time,
                        total_inferred
                    );

                    // Forward the embeddings to the next stage
                    match tx.send((service_ids, embeddings)).await {
                        Ok(_) => {
                            debug!("{}: Successfully sent embeddings to write stage", batch_id);
                        }
                        Err(e) => {
                            error!(
                                "{}: Failed to send embeddings to write stage: {}",
                                batch_id, e
                            );
                            break;
                        }
                    }
                }
                Err(e) => {
                    error!("{}: Error generating embeddings: {}", batch_id, e);
                }
            }

            // Release the permit
            drop(permit);
        }

        // Signal completion by dropping the sender
        drop(tx);

        info!(
            "Inference stage completed in {:.2?}, processed {} services in {} batches",
            stage_start.elapsed(),
            total_inferred,
            batch_num
        );

        Ok(()) as Result<()>
    })
    .await?
}

/// Spawns the write stage of the pipeline
///
/// This stage saves the embeddings back to the database.
async fn spawn_write_stage(
    mut rx: mpsc::Receiver<(Vec<String>, Vec<Vec<f32>>)>,
    tx: mpsc::Sender<usize>,
    db_writer: DatabaseWriter,
    semaphore: Arc<Semaphore>,
    progress_tracker: ProgressTracker,
) -> Result<()> {
    // Start a tokio task for the write stage
    tokio::spawn(async move {
        let stage_start = Instant::now();
        debug!("Starting write stage");

        let mut total_written = 0;
        let mut batch_num = 0;
        let mut batches = Vec::new();

        while let Some((service_ids, embeddings)) = rx.recv().await {
            batch_num += 1;
            let batch_id = format!("write-{}", batch_num);

            if service_ids.is_empty() || embeddings.is_empty() {
                debug!("{}: Empty batch received, skipping", batch_id);
                continue;
            }

            let service_count = service_ids.len();
            debug!(
                "{}: Received {} services for database write",
                batch_id, service_count
            );

            // Add the batch to our collection
            batches.push((service_ids, embeddings));

            // Process in larger chunks for better database efficiency
            if batches.len() >= 10 || batch_num % 20 == 0 {
                // Acquire a permit from the semaphore to control concurrency
                let permit = match semaphore.acquire().await {
                    Ok(permit) => permit,
                    Err(e) => {
                        error!("Failed to acquire write semaphore: {}", e);
                        continue;
                    }
                };

                let batches_to_process = std::mem::take(&mut batches);
                let batch_count = batches_to_process.len();
                let service_count: usize =
                    batches_to_process.iter().map(|(ids, _)| ids.len()).sum();

                debug!(
                    "{}: Writing {} batches with {} services to database",
                    batch_id, batch_count, service_count
                );

                // Save the embeddings
                match db_writer
                    .save_embeddings_in_batches(batches_to_process, &progress_tracker)
                    .await
                {
                    Ok(count) => {
                        total_written += count;
                        debug!(
                            "{}: Wrote {} embeddings, total so far: {}",
                            batch_id, count, total_written
                        );

                        // Forward the count to the monitor
                        if let Err(e) = tx.send(count).await {
                            error!("{}: Failed to send count to monitor: {}", batch_id, e);
                            break;
                        }
                    }
                    Err(e) => {
                        error!("{}: Error saving embeddings: {}", batch_id, e);
                    }
                }

                // Release the permit
                drop(permit);
            }
        }

        // Process any remaining batches
        if !batches.is_empty() {
            debug!("Processing {} remaining batches", batches.len());

            match db_writer
                .save_embeddings_in_batches(batches, &progress_tracker)
                .await
            {
                Ok(count) => {
                    total_written += count;
                    debug!("Wrote {} embeddings, total: {}", count, total_written);

                    // Forward the count to the monitor
                    if let Err(e) = tx.send(count).await {
                        error!("Failed to send final count to monitor: {}", e);
                    }
                }
                Err(e) => {
                    error!("Error saving final embeddings: {}", e);
                }
            }
        }

        // Signal completion by dropping the sender
        drop(tx);

        info!(
            "Write stage completed in {:.2?}, saved {} embeddings in {} batches",
            stage_start.elapsed(),
            total_written,
            batch_num
        );

        // Log final statistics
        if let Err(e) = db_writer.log_embedding_stats(&progress_tracker).await {
            warn!("Failed to log final embedding statistics: {}", e);
        }

        Ok(()) as Result<()>
    })
    .await?
}

/// Spawns a monitor task to track overall pipeline progress
async fn spawn_pipeline_monitor(
    mut rx: mpsc::Receiver<usize>,
    progress_tracker: ProgressTracker,
    total_count: i64,
) -> Result<i64> {
    // Start a tokio task for the monitor
    tokio::spawn(async move {
        let monitor_start = Instant::now();
        debug!("Starting pipeline monitor");

        let mut processed_count = 0;

        // Process counts coming from the write stage
        while let Some(count) = rx.recv().await {
            processed_count += count as i64;

            // Force a progress update every 10,000 services
            if processed_count % 10_000 < count as i64 {
                if let Err(e) = progress_tracker.update(0, "monitor", true).await {
                    warn!("Failed to update progress: {}", e);
                }
            }
        }

        // Ensure we have a final progress update
        if let Err(e) = progress_tracker.update(0, "final", true).await {
            warn!("Failed to update final progress: {}", e);
        }

        let completion_percentage = (processed_count as f64 / total_count as f64) * 100.0;

        info!(
            "Pipeline monitor completed in {:.2?}, tracked {} services ({:.1}%)",
            monitor_start.elapsed(),
            processed_count,
            completion_percentage
        );

        Ok(processed_count)
    })
    .await?
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;

    #[tokio::test]
    #[ignore] // Ignore by default since it requires database and model files
    async fn test_embed_services() {
        // This would be a more comprehensive test in a real implementation

        // Load environment
        db::load_env_from_file(".env").unwrap();

        // Connect to database
        let pool = db::connect().await.unwrap();

        // Create a minimal config for testing
        let test_config = EmbeddingConfig {
            batch_size: 10,
            concurrent_batches: 2,
            concurrent_fetchers: 2,
            concurrent_tokenizers: 2,
            concurrent_inference: 1,
            concurrent_writers: 2,
            force_cpu: true,     // Force CPU for testing
            create_index: false, // Skip index creation for tests
            ..EmbeddingConfig::default()
        };

        // Run the embedding process
        let result = embed_services_with_config(&pool, test_config).await;
        assert!(result.is_ok());
    }
}
