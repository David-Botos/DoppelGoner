// src/services/embed_services.rs (optimized version)

use anyhow::{Context, Result};
use bb8::Pool;
use bb8_postgres::PostgresConnectionManager;
use futures::lock::Mutex;
use log::{debug, error, info, warn};
use std::time::{Duration, Instant};
use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use tokio::sync::{Semaphore, mpsc};
use tokio_postgres::NoTls;

use crate::services::config::{
    BATCH_SIZE, CONCURRENT_BATCHES, CONFIG_PATH, MAX_TOKEN_LENGTH, MODEL_PATH, TOKENIZER_PATH,
};
use crate::services::data_fetcher::DataFetcher;
use crate::services::data_writer::{DatabaseWriter, ProgressTracker};
use crate::services::inference::InferenceEngine;
use crate::services::tokenizer::{TokenizationManager, load_tokenizer};
use crate::services::types::{ServiceData, TokenizedBatch};

/// Configuration for the optimized embedding pipeline
#[derive(Clone)]
pub struct EmbeddingConfig {
    // Existing parameters
    pub tokenizer_path: String,
    pub model_path: String,
    pub config_path: String,
    pub batch_size: usize,
    pub concurrent_batches: usize,
    pub concurrent_fetchers: usize,
    pub concurrent_tokenizers: usize,
    pub concurrent_inference: usize,
    pub concurrent_writers: usize,
    pub force_cpu: bool,
    pub progress_update_interval: u64,
    pub create_index: bool,
    pub id_batch_accumulation: usize,
    pub max_meta_batch_size: usize,

    // Buffer optimization parameters
    pub pre_inference_buffer_capacity: usize, // Size of buffer before inference
    pub post_inference_buffer_capacity: usize, // Size of buffer after inference
    pub db_batch_accumulation_size: usize,    // Number of batches to accumulate before DB write
    pub db_batch_flush_interval_ms: u64,      // Max time to wait before flushing to DB
    pub adaptive_concurrency: bool,           // Enable dynamic concurrency adjustment
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
            concurrent_inference: 6,
            concurrent_writers: 4,
            force_cpu: false,
            progress_update_interval: 5,
            create_index: true,
            id_batch_accumulation: 5,
            max_meta_batch_size: 500,

            // New optimized defaults for M2 MacBook Pro
            pre_inference_buffer_capacity: 16, // Keep GPU fed with work
            post_inference_buffer_capacity: 8, // Prevent memory overflow
            db_batch_accumulation_size: 10,    // Accumulate batches for efficient DB writes
            db_batch_flush_interval_ms: 1000,  // Max 1 second before forced flush
            adaptive_concurrency: true,        // Enable dynamic resource allocation
        }
    }
}

/// M2-optimized embedding service function
pub async fn embed_services_m2_optimized(
    pool: &Pool<PostgresConnectionManager<NoTls>>,
) -> Result<()> {
    // Use optimized configuration
    embed_services_with_optimized_config(pool, EmbeddingConfig::default()).await
}

/// Embed services using optimized configuration
pub async fn embed_services_with_optimized_config(
    pool: &Pool<PostgresConnectionManager<NoTls>>,
    config: EmbeddingConfig,
) -> Result<()> {
    let overall_start = Instant::now();
    info!("Starting optimized embedding service generation process");

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

    // Run the optimized embedding pipeline
    let result = run_optimized_embedding_pipeline(
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

/// Runs the optimized embedding pipeline with fixed progress tracking
async fn run_optimized_embedding_pipeline(
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
        "Starting optimized embedding pipeline with batch_size={}, concurrent_batches={}",
        config.batch_size, config.concurrent_batches
    );

    // Buffer monitoring counters
    let pre_inference_level = Arc::new(AtomicUsize::new(0));
    let post_inference_level = Arc::new(AtomicUsize::new(0));

    // Create channels for the pipeline stages with optimized capacities
    let (fetch_tx, fetch_rx) = mpsc::channel(config.concurrent_batches);
    let (tokenize_tx, tokenize_rx) = mpsc::channel(config.pre_inference_buffer_capacity);
    let (inference_tx, inference_rx) = mpsc::channel(config.post_inference_buffer_capacity);

    // NEW: Use a separate channel for tracking write status
    let (write_tx, write_rx) = mpsc::channel(config.concurrent_batches);

    // NEW: Create a channel for sending progress updates directly from DB writer to monitor
    let (progress_tx, progress_rx) = mpsc::channel(config.concurrent_batches);

    // Create semaphores to control concurrency at each stage
    let fetch_semaphore = Arc::new(Semaphore::new(config.concurrent_fetchers));
    let tokenize_semaphore = Arc::new(Semaphore::new(config.concurrent_tokenizers));
    let inference_semaphore = Arc::new(Semaphore::new(config.concurrent_inference));
    let write_semaphore = Arc::new(Semaphore::new(config.concurrent_writers));

    // Spawn the pipeline stages with improved progress tracking
    let fetch_handle = spawn_optimized_fetch_stage(
        data_fetcher,
        fetch_tx,
        fetch_semaphore.clone(),
        pre_inference_level.clone(),
        config.clone(),
    );

    let tokenize_handle = spawn_optimized_tokenize_stage(
        fetch_rx,
        tokenize_tx,
        tokenization_manager,
        tokenize_semaphore.clone(),
        pre_inference_level.clone(),
        config.clone(),
    );

    let inference_handle = spawn_optimized_inference_stage(
        tokenize_rx,
        inference_tx,
        inference_engine,
        inference_semaphore.clone(),
        pre_inference_level.clone(),
        post_inference_level.clone(),
        config.clone(),
    );

    // MODIFIED: Pass progress_tx instead of updating progress directly
    let write_handle = spawn_optimized_write_stage_fixed(
        inference_rx,
        write_tx,
        progress_tx, // NEW: Separate channel for progress updates
        db_writer,
        write_semaphore.clone(),
        post_inference_level.clone(),
        config.clone(),
    );

    // MODIFIED: Use progress_rx to update the tracker only after confirmed DB writes
    let monitor_handle = spawn_pipeline_monitor_fixed(
        write_rx,
        progress_rx, // NEW: Receive confirmed writes
        progress_tracker.clone(),
        total_count,
    );

    // If adaptive concurrency is enabled, spawn a buffer monitor
    let buffer_monitor_handle: Option<tokio::task::JoinHandle<()>> = if config.adaptive_concurrency
    {
        Some(spawn_buffer_monitor(
            pre_inference_level.clone(),
            post_inference_level.clone(),
            fetch_semaphore.clone(),
            write_semaphore.clone(),
            config.clone(),
        ))
    } else {
        None
    };

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

    // If we had a buffer monitor, abort it to clean up
    if let Some(buffer_monitor) = buffer_monitor_handle {
        buffer_monitor.abort();
    }

    info!(
        "Optimized embedding pipeline completed in {:.2?}",
        pipeline_start.elapsed()
    );

    Ok(processed_count)
}

/// Spawns the optimized fetch stage with buffer awareness
async fn spawn_optimized_fetch_stage(
    data_fetcher: DataFetcher,
    tx: mpsc::Sender<Vec<ServiceData>>,
    semaphore: Arc<Semaphore>,
    pre_inference_level: Arc<AtomicUsize>,
    config: EmbeddingConfig,
) -> Result<()> {
    // Start a tokio task for the fetch stage
    tokio::spawn(async move {
        let stage_start = Instant::now();
        debug!("Starting optimized fetch stage with buffer monitoring");

        let mut total_fetched = 0;
        let mut batch_num = 0;
        let mut accumulated_ids: Vec<String> = Vec::new();

        loop {
            // Check if pre-inference buffer is getting full
            // If buffer is >75% full, pause briefly to let it drain
            let buffer_level = pre_inference_level.load(Ordering::Relaxed);
            if buffer_level > config.pre_inference_buffer_capacity * 3 / 4 {
                debug!(
                    "Pre-inference buffer nearly full ({}/{}), pausing fetch",
                    buffer_level, config.pre_inference_buffer_capacity
                );
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }

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
                        &pre_inference_level,
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
                    &pre_inference_level,
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

// Helper function to process accumulated IDs with buffer awareness
async fn process_accumulated_ids(
    data_fetcher: &DataFetcher,
    accumulated_ids: &[String],
    tx: &mpsc::Sender<Vec<ServiceData>>,
    semaphore: &Arc<Semaphore>,
    pre_inference_level: &Arc<AtomicUsize>,
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

                // Update the pre-inference buffer counter before sending
                pre_inference_level.fetch_add(1, Ordering::Relaxed);

                match tx.send(batch).await {
                    Ok(_) => {
                        debug!(
                            "{}-{}: Successfully sent services to tokenize stage",
                            batch_id, i
                        );
                    }
                    Err(e) => {
                        // If send fails, decrement the counter we just incremented
                        pre_inference_level.fetch_sub(1, Ordering::Relaxed);
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

/// Spawns the optimized tokenize stage with buffer awareness
async fn spawn_optimized_tokenize_stage(
    mut rx: mpsc::Receiver<Vec<ServiceData>>,
    tx: mpsc::Sender<TokenizedBatch>,
    tokenization_manager: Arc<TokenizationManager>,
    semaphore: Arc<Semaphore>,
    pre_inference_level: Arc<AtomicUsize>,
    config: EmbeddingConfig,
) -> Result<()> {
    // Start a tokio task for the tokenize stage
    tokio::spawn(async move {
        let stage_start = Instant::now();
        debug!("Starting optimized tokenize stage with buffer monitoring");

        let mut total_tokenized = 0;
        let mut batch_num = 0;

        while let Some(services) = rx.recv().await {
            // We've received a batch, decrement the pre-inference buffer counter
            pre_inference_level.fetch_sub(1, Ordering::Relaxed);

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

                    // Update pre-inference buffer counter before sending
                    pre_inference_level.fetch_add(1, Ordering::Relaxed);

                    // Forward the tokenized batch to the next stage
                    match tx.send(tokenized_batch).await {
                        Ok(_) => {
                            debug!("{}: Successfully sent tokens to inference stage", batch_id);
                        }
                        Err(e) => {
                            // If send fails, decrement the counter we just incremented
                            pre_inference_level.fetch_sub(1, Ordering::Relaxed);
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

/// Spawns the optimized inference stage with priority handling
async fn spawn_optimized_inference_stage(
    mut rx: mpsc::Receiver<TokenizedBatch>,
    tx: mpsc::Sender<(Vec<String>, Vec<Vec<f32>>)>,
    inference_engine: Arc<InferenceEngine>,
    semaphore: Arc<Semaphore>,
    pre_inference_level: Arc<AtomicUsize>,
    post_inference_level: Arc<AtomicUsize>,
    config: EmbeddingConfig,
) -> Result<()> {
    // Start a tokio task for the inference stage
    tokio::spawn(async move {
        let stage_start = Instant::now();
        debug!("Starting optimized inference stage with buffer monitoring");

        let mut total_inferred = 0;
        let mut batch_num = 0;

        while let Some(batch) = rx.recv().await {
            // We've received a batch, decrement the pre-inference buffer counter
            pre_inference_level.fetch_sub(1, Ordering::Relaxed);

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

            // Check if post-inference buffer is nearly full
            // This prevents memory overflow by pausing inference if the write stage is falling behind
            let post_level = post_inference_level.load(Ordering::Relaxed);
            if post_level >= config.post_inference_buffer_capacity * 3 / 4 {
                debug!(
                    "{}: Post-inference buffer nearly full ({}/{}), waiting before processing",
                    batch_id, post_level, config.post_inference_buffer_capacity
                );

                // Wait for buffer to drain a bit before continuing
                // This is important for M2 with unified memory
                while post_inference_level.load(Ordering::Relaxed)
                    >= config.post_inference_buffer_capacity / 2
                {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }

                debug!("{}: Resuming inference after buffer drained", batch_id);
            }

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

                    // Update post-inference buffer counter before sending
                    post_inference_level.fetch_add(1, Ordering::Relaxed);

                    // Forward the embeddings to the next stage
                    match tx.send((service_ids, embeddings)).await {
                        Ok(_) => {
                            debug!("{}: Successfully sent embeddings to write stage", batch_id);
                        }
                        Err(e) => {
                            // If send fails, decrement the counter we just incremented
                            post_inference_level.fetch_sub(1, Ordering::Relaxed);
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

/// Spawns the optimized write stage with batch accumulation
async fn spawn_optimized_write_stage_fixed(
    mut rx: mpsc::Receiver<(Vec<String>, Vec<Vec<f32>>)>,
    tx: mpsc::Sender<usize>,
    progress_tx: mpsc::Sender<(Vec<String>, usize)>, // NEW: Send service IDs with count
    db_writer: DatabaseWriter,
    semaphore: Arc<Semaphore>,
    post_inference_level: Arc<AtomicUsize>,
    config: EmbeddingConfig,
) -> Result<()> {
    tokio::spawn(async move {
        let stage_start = Instant::now();
        debug!("Starting fixed optimized write stage with batch accumulation");

        let mut total_written = 0;
        let mut batch_num = 0;
        let mut accumulated_batches = Vec::new();
        let mut last_flush_time = Instant::now();

        while let Some((service_ids, embeddings)) = rx.recv().await {
            // We've received a batch, decrement the post-inference buffer counter
            post_inference_level.fetch_sub(1, Ordering::Relaxed);

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

            // Add the batch to our accumulation
            accumulated_batches.push((service_ids, embeddings));
            let accumulated_count = accumulated_batches
                .iter()
                .map(|(ids, _)| ids.len())
                .sum::<usize>();

            // Check if we should flush accumulated batches
            let should_flush = accumulated_batches.len() >= config.db_batch_accumulation_size
                || last_flush_time.elapsed().as_millis()
                    >= config.db_batch_flush_interval_ms as u128
                || accumulated_count >= config.max_meta_batch_size;

            if should_flush && !accumulated_batches.is_empty() {
                debug!(
                    "{}: Flushing {} accumulated batches with {} total services",
                    batch_id,
                    accumulated_batches.len(),
                    accumulated_count
                );

                // Acquire a permit from the semaphore to control concurrency
                let permit = match semaphore.acquire().await {
                    Ok(permit) => permit,
                    Err(e) => {
                        error!("Failed to acquire write semaphore: {}", e);
                        continue;
                    }
                };

                let batches_to_process = std::mem::take(&mut accumulated_batches);
                last_flush_time = Instant::now();

                // Collect all service IDs for progress tracking
                let all_service_ids: Vec<String> = batches_to_process
                    .iter()
                    .flat_map(|(ids, _)| ids.clone())
                    .collect();

                // Save the embeddings
                match db_writer
                    .save_embeddings_in_batches_fixed(batches_to_process)
                    .await
                {
                    Ok(count) => {
                        total_written += count;
                        debug!(
                            "{}: Successfully wrote {} embeddings, total so far: {}",
                            batch_id, count, total_written
                        );

                        // Forward the successful IDs and count to the progress tracker
                        if let Err(e) = progress_tx.send((all_service_ids, count)).await {
                            error!("{}: Failed to send progress update: {}", batch_id, e);
                        }

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
        if !accumulated_batches.is_empty() {
            debug!("Processing {} remaining batches", accumulated_batches.len());

            // Collect all service IDs for progress tracking
            let all_service_ids: Vec<String> = accumulated_batches
                .iter()
                .flat_map(|(ids, _)| ids.clone())
                .collect();

            match db_writer
                .save_embeddings_in_batches_fixed(accumulated_batches)
                .await
            {
                Ok(count) => {
                    total_written += count;
                    debug!("Wrote {} embeddings, total: {}", count, total_written);

                    // Forward the service IDs and count to the progress tracker
                    if let Err(e) = progress_tx.send((all_service_ids, count)).await {
                        error!("Failed to send final progress update: {}", e);
                    }

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

        // Signal completion by dropping the senders
        drop(tx);
        drop(progress_tx);

        info!(
            "Write stage completed in {:.2?}, saved {} embeddings in {} batches",
            stage_start.elapsed(),
            total_written,
            batch_num
        );

        Ok(()) as Result<()>
    })
    .await?
}
/// New function to monitor buffer levels and adjust concurrency
fn spawn_buffer_monitor(
    pre_inference_level: Arc<AtomicUsize>,
    post_inference_level: Arc<AtomicUsize>,
    fetch_semaphore: Arc<Semaphore>,
    write_semaphore: Arc<Semaphore>,
    config: EmbeddingConfig,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        debug!("Starting buffer monitor for adaptive concurrency");

        // Keep track of any additional permits we've issued
        let mut additional_fetch_permits = 0;
        let mut additional_write_permits = 0;

        loop {
            // Get current buffer levels
            let pre_level = pre_inference_level.load(Ordering::Relaxed);
            let post_level = post_inference_level.load(Ordering::Relaxed);

            // Log current buffer state periodically
            debug!(
                "Buffer status: pre-inference={}/{}, post-inference={}/{}",
                pre_level,
                config.pre_inference_buffer_capacity,
                post_level,
                config.post_inference_buffer_capacity
            );

            // Adaptive concurrency control based on buffer levels

            // 1. If pre-inference buffer is getting low, increase fetch concurrency
            if pre_level < config.pre_inference_buffer_capacity / 4 && additional_fetch_permits < 4
            {
                // Add more fetch capacity to fill the buffer
                fetch_semaphore.add_permits(1);
                additional_fetch_permits += 1;
                debug!("Pre-inference buffer low, increased fetch concurrency");
            }
            // 2. If pre-inference buffer is high, reduce fetch concurrency
            else if pre_level > config.pre_inference_buffer_capacity * 3 / 4
                && additional_fetch_permits > 0
            {
                // Reduce fetch capacity to let buffer drain
                // Note: we're assuming permits are not all in use
                additional_fetch_permits -= 1;
                debug!("Pre-inference buffer high, reduced fetch concurrency");
            }

            // 3. If post-inference buffer is getting high, increase write concurrency
            if post_level > config.post_inference_buffer_capacity / 2
                && additional_write_permits < 4
            {
                // Add more write capacity to drain the buffer
                write_semaphore.add_permits(1);
                additional_write_permits += 1;
                debug!("Post-inference buffer high, increased write concurrency");
            }
            // 4. If post-inference buffer is low, reduce write concurrency
            else if post_level < config.post_inference_buffer_capacity / 4
                && additional_write_permits > 0
            {
                // Reduce write capacity
                additional_write_permits -= 1;
                debug!("Post-inference buffer low, reduced write concurrency");
            }

            // Brief sleep before next adjustment
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    })
}

/// Spawns a monitor task to track overall pipeline progress (unchanged from original)
async fn spawn_pipeline_monitor_fixed(
    mut rx: mpsc::Receiver<usize>,
    mut progress_rx: mpsc::Receiver<(Vec<String>, usize)>,
    progress_tracker: ProgressTracker,
    total_count: i64,
) -> Result<i64> {
    tokio::spawn(async move {
        let monitor_start = Instant::now();
        debug!("Starting fixed pipeline monitor");

        let mut processed_count = 0;

        // Create a shared HashSet for tracking service IDs
        let tracked_service_ids = Arc::new(Mutex::new(HashSet::<String>::new()));

        // Clone the progress tracker and tracked_service_ids for the nested task
        let progress_tracker_for_handler = progress_tracker.clone();
        let tracked_ids_for_handler = tracked_service_ids.clone();

        // Create a task to handle progress updates separately
        // The progress_rx is moved into this task automatically
        let progress_handler = tokio::spawn(async move {
            while let Some((service_ids, count)) = progress_rx.recv().await {
                // Update progress only with successfully written services
                let mut duplicate_ids = 0;
                let mut unique_ids = 0;

                // Lock the tracked IDs set
                let mut ids = tracked_ids_for_handler.lock().await;

                for id in &service_ids {
                    if ids.insert(id.clone()) {
                        unique_ids += 1;
                    } else {
                        duplicate_ids += 1;
                    }
                }

                // Release the lock
                drop(ids);

                if duplicate_ids > 0 {
                    warn!(
                        "Detected {} duplicate service IDs in progress update",
                        duplicate_ids
                    );
                }

                // Only update progress with unique IDs
                if unique_ids > 0 {
                    if let Err(e) = progress_tracker_for_handler
                        .update(unique_ids as i64, "db-confirmed", false)
                        .await
                    {
                        warn!("Failed to update progress: {}", e);
                    }
                }
            }

            // progress_rx is implicitly dropped when this task ends
            debug!("Progress handler task completed");
        });

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

        // Wait for progress handler to complete (with timeout for safety)
        let timeout = tokio::time::Duration::from_secs(10);
        match tokio::time::timeout(timeout, progress_handler).await {
            Ok(result) => {
                if let Err(e) = result {
                    warn!("Error in progress handler: {}", e);
                }
            }
            Err(_) => {
                warn!("Timeout waiting for progress handler to complete");
            }
        }

        // Log the total number of unique service IDs
        let unique_count = tracked_service_ids.lock().await.len();
        info!("Total unique service IDs tracked: {}", unique_count);

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
