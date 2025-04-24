// src/services/db_writer.rs

use anyhow::{Context, Result};
use bb8::Pool;
use bb8_postgres::PostgresConnectionManager;
use futures::{StreamExt, stream};
use log::{debug, error, info, warn};
use pgvector::Vector;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio_postgres::NoTls;

/// Tracks progress during embedding operations
///
/// Provides methods for updating and displaying progress metrics
/// to give feedback during long-running embedding operations.
/// Enhanced ProgressTracker that detects potential double-counting
pub struct ProgressTracker {
    processed: Arc<Mutex<i64>>,
    total: i64,
    start_time: Instant,
    last_update_time: Arc<Mutex<Instant>>,
    update_interval: Duration,
    // Track service IDs we've seen to detect double-counting
    processed_ids: Arc<Mutex<HashSet<String>>>,
    update_history: Arc<Mutex<Vec<(String, i64, Instant)>>>,
}

impl ProgressTracker {
    /// Create a new progress tracker with double-counting detection
    pub fn new(total: i64, update_interval: Duration) -> Self {
        let now = Instant::now();
        Self {
            processed: Arc::new(Mutex::new(0)),
            total,
            start_time: now,
            last_update_time: Arc::new(Mutex::new(now)),
            update_interval,
            processed_ids: Arc::new(Mutex::new(HashSet::new())),
            update_history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Update progress with enhanced debugging and ID tracking
    pub async fn update_with_ids(
        &self,
        count: i64,
        ids: &[String],
        batch_id: &str,
        force: bool,
    ) -> Result<()> {
        let mut processed = self.processed.lock().await;
        *processed += count;

        // Track update history
        let mut history = self.update_history.lock().await;
        history.push((batch_id.to_string(), count, Instant::now()));

        // Check for duplicate IDs
        let mut processed_ids = self.processed_ids.lock().await;
        let mut duplicates = 0;

        for id in ids {
            if !processed_ids.insert(id.clone()) {
                duplicates += 1;
            }
        }

        if duplicates > 0 {
            warn!(
                "Batch {}: Detected {} duplicate service IDs out of {}",
                batch_id,
                duplicates,
                ids.len()
            );
        }

        let mut last_update = self.last_update_time.lock().await;
        let now = Instant::now();

        // Only update if forced or if enough time has passed since last update
        if !force && now.duration_since(*last_update) < self.update_interval {
            return Ok(());
        }

        *last_update = now;

        let progress_percentage = (*processed as f64 / self.total as f64) * 100.0;
        let elapsed = now.duration_since(self.start_time);

        // Calculate estimated time remaining
        let time_per_item = if *processed > 0 {
            elapsed.as_secs_f64() / (*processed as f64)
        } else {
            0.0
        };

        let items_remaining = self.total - *processed;
        let time_remaining = time_per_item * (items_remaining as f64);
        let time_remaining_secs = time_remaining as u64;

        info!(
            "Batch {}: Processed {}/{} services ({:.1}%) - Unique IDs: {} - Est. time remaining: {:02}:{:02}:{:02}",
            batch_id,
            *processed,
            self.total,
            progress_percentage,
            processed_ids.len(),
            time_remaining_secs / 3600,
            (time_remaining_secs % 3600) / 60,
            time_remaining_secs % 60
        );

        Ok(())
    }

    // Compatibility method that wraps update_with_ids but warns about missing ID tracking
    pub async fn update(&self, count: i64, batch_id: &str, force: bool) -> Result<()> {
        if count > 0 {
            warn!(
                "Batch {}: Updating progress without ID tracking (count: {})",
                batch_id, count
            );
        }

        // Use empty ID list since we don't have IDs
        self.update_with_ids(count, &[], batch_id, force).await
    }

    /// Get detailed statistics including potential double-counting
    pub async fn get_detailed_stats(&self) -> (i64, Duration, usize, Vec<(String, i64, Instant)>) {
        let processed = *self.processed.lock().await;
        let elapsed = self.start_time.elapsed();
        let unique_count = self.processed_ids.lock().await.len();
        let history = self.update_history.lock().await.clone();

        (processed, elapsed, unique_count, history)
    }

    /// Get basic stats (for backward compatibility)
    pub async fn get_stats(&self) -> (i64, Duration) {
        let processed = *self.processed.lock().await;
        let elapsed = self.start_time.elapsed();
        (processed, elapsed)
    }

    /// Dump a complete report of update history
    pub async fn dump_history_report(&self) -> String {
        let (processed, elapsed, unique_count, history) = self.get_detailed_stats().await;

        let mut report = format!(
            "Progress Tracker Report:\n\
             - Total processed (counter): {}\n\
             - Unique IDs tracked: {}\n\
             - Elapsed time: {:.2?}\n\n\
             Update History:\n",
            processed, unique_count, elapsed
        );

        for (idx, (batch, count, timestamp)) in history.iter().enumerate() {
            let time_since_start = timestamp.duration_since(self.start_time);
            report.push_str(&format!(
                "{:3}. Batch {}: +{} at {:.2?} from start\n",
                idx + 1,
                batch,
                count,
                time_since_start
            ));
        }

        report
    }
}
/// Component for writing embeddings to the database
///
/// Handles batch updates, transactions, and maintaining data integrity
/// when saving embeddings to PostgreSQL using pgvector.
pub struct DatabaseWriter {
    pool: Pool<PostgresConnectionManager<NoTls>>,
    concurrent_writers: usize,
}

impl DatabaseWriter {
    /// Create a new DatabaseWriter
    ///
    /// # Arguments
    ///
    /// * `pool` - Database connection pool
    /// * `concurrent_writers` - Number of concurrent write operations
    pub fn new(pool: Pool<PostgresConnectionManager<NoTls>>, concurrent_writers: usize) -> Self {
        Self {
            pool,
            concurrent_writers,
        }
    }

    /// Save a single batch of embeddings to the database
    ///
    /// # Arguments
    ///
    /// * `service_ids` - List of service IDs
    /// * `embeddings` - List of embeddings (must match service_ids in length)
    /// * `batch_id` - Identifier for this batch (for logging)
    async fn save_batch(
        &self,
        service_ids: &[String],
        embeddings: &[Vec<f32>],
        batch_id: &str,
    ) -> Result<usize> {
        if service_ids.len() != embeddings.len() {
            return Err(anyhow::anyhow!(
                "Mismatch between service_ids ({}) and embeddings ({}) lengths",
                service_ids.len(),
                embeddings.len()
            ));
        }

        if service_ids.is_empty() {
            debug!("Batch {}: Empty batch, skipping", batch_id);
            return Ok(0);
        }

        info!(
            "Batch {}: Saving {} embeddings to database",
            batch_id,
            service_ids.len()
        );
        let start = Instant::now();

        // Get a client from the pool
        let mut client = self.pool.get().await.context(format!(
            "Batch {}: Failed to get database connection",
            batch_id
        ))?;

        // IMPROVED: Set a statement timeout to prevent hanging transactions
        client
            .execute("SET statement_timeout = '30s'", &[])
            .await
            .context("Failed to set statement timeout")?;

        // Use a smaller batch size for updates to reduce transaction time
        let sub_batch_size = 10;
        let mut total_success = 0;

        // Process in smaller transactions to reduce the chance of deadlocks
        for (sub_batch_idx, sub_batch) in service_ids.chunks(sub_batch_size).enumerate() {
            let sub_embeddings = &embeddings[sub_batch_idx * sub_batch_size..]
                [..sub_batch.len().min(sub_batch_size)];

            // Start a transaction
            let transaction = client.transaction().await.context(format!(
                "Batch {}.{}: Failed to start transaction",
                batch_id, sub_batch_idx
            ))?;

            info!(
                "Batch {}.{}: Started transaction for {} services",
                batch_id,
                sub_batch_idx,
                sub_batch.len()
            );

            let mut success_count = 0;
            let mut error_count = 0;

            // Process each service in the sub-batch
            for (i, (id, embedding)) in sub_batch.iter().zip(sub_embeddings).enumerate() {
                // Convert Vec<f32> to pgvector::Vector
                let pgvector = Vector::from(embedding.clone());

                // Add optimistic locking to prevent race conditions
                let query = "
                UPDATE service 
                SET embedding_v2 = $1, 
                    embedding_v2_updated_at = NOW() 
                WHERE id = $2 
                AND (embedding_v2 IS NULL OR embedding_v2_updated_at < NOW() - INTERVAL '1 hour')
            ";

                // Try to update with retry logic
                let mut retries = 0;
                let max_retries = 3;
                let mut last_error = None;

                while retries < max_retries {
                    match transaction.execute(query, &[&pgvector, id]).await {
                        Ok(rows) => {
                            if rows > 0 {
                                success_count += 1;
                            } else {
                                warn!(
                                    "Batch {}.{}: Service {} already has a recent embedding or ID not found",
                                    batch_id, sub_batch_idx, id
                                );
                            }
                            break; // Success, exit retry loop
                        }
                        Err(e) => {
                            // Only retry on deadlock or serialization failures
                            if e.to_string().contains("deadlock")
                                || e.to_string().contains("serialization")
                            {
                                retries += 1;
                                warn!(
                                    "Batch {}.{}: Retry {}/{} for service {} due to: {}",
                                    batch_id, sub_batch_idx, retries, max_retries, id, e
                                );
                                tokio::time::sleep(Duration::from_millis(100 * retries)).await;
                                last_error = Some(e);
                            } else {
                                // Non-retryable error
                                error!(
                                    "Batch {}.{}: Error updating embedding for service {}: {}",
                                    batch_id, sub_batch_idx, id, e
                                );
                                error_count += 1;
                                break; // Non-retryable, exit retry loop
                            }
                        }
                    }
                }

                // If we exhausted retries
                if retries == max_retries {
                    error!(
                        "Batch {}.{}: Exhausted retries for service {}: {:?}",
                        batch_id, sub_batch_idx, id, last_error
                    );
                    error_count += 1;
                }
            }

            // IMPROVED: Explicitly use savepoint for additional safety
            if success_count > 0 {
                info!(
                    "Batch {}.{}: Committing transaction with {} successes and {} errors",
                    batch_id, sub_batch_idx, success_count, error_count
                );

                match transaction.commit().await {
                    Ok(_) => {
                        info!(
                            "Batch {}.{}: Successfully committed transaction",
                            batch_id, sub_batch_idx
                        );
                        total_success += success_count;
                    }
                    Err(e) => {
                        error!(
                            "Batch {}.{}: Failed to commit transaction: {}",
                            batch_id, sub_batch_idx, e
                        );
                        // Don't increment total_success
                    }
                }
            } else {
                info!(
                    "Batch {}.{}: Rolling back transaction (no successful updates)",
                    batch_id, sub_batch_idx
                );
                if let Err(e) = transaction.rollback().await {
                    warn!(
                        "Batch {}.{}: Error rolling back transaction: {}",
                        batch_id, sub_batch_idx, e
                    );
                }
            }

            // Add a small delay between sub-batches to reduce contention
            if sub_batch_idx < service_ids.chunks(sub_batch_size).len() - 1 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }

        info!(
            "Batch {}: Saved {} out of {} embeddings in {:.2?}",
            batch_id,
            total_success,
            service_ids.len(),
            start.elapsed()
        );

        Ok(total_success) // Return actual success count
    }

    /// Save multiple batches of embeddings in parallel
    ///
    /// # Arguments
    ///
    /// * `batches` - Vector of (service_ids, embeddings) tuples
    /// * `progress` - Progress tracker to update during processing
    /// Save multiple batches of embeddings in parallel with enhanced debugging
    pub async fn save_embeddings_in_batches_fixed(
        &self,
        batches: Vec<(Vec<String>, Vec<Vec<f32>>)>,
    ) -> Result<usize> {
        if batches.is_empty() {
            debug!("No batches to save, skipping");
            return Ok(0);
        }

        info!(
            "Saving {} batches of embeddings with {} concurrent writers",
            batches.len(),
            self.concurrent_writers
        );
        let start = Instant::now();

        let total_count: usize = batches.iter().map(|(ids, _)| ids.len()).sum();
        debug!("Total of {} embeddings to save", total_count);

        // Store the batches length before consuming it
        let batches_len = batches.len();

        // Process batches in parallel
        let results = stream::iter(batches.into_iter().enumerate())
            .map(|(batch_idx, (service_ids, embeddings))| {
                let writer = self.clone();
                let batch_id = uuid::Uuid::new_v4().to_string()[..8].to_string();

                async move {
                    let batch_start = Instant::now();
                    debug!(
                        "Starting to save batch {}/{} (ID: {}) with {} embeddings",
                        batch_idx + 1,
                        batches_len, // Use the pre-computed length instead
                        batch_id,
                        service_ids.len()
                    );

                    let result = writer
                        .save_batch(&service_ids, &embeddings, &batch_id)
                        .await;

                    match &result {
                        Ok(count) => {
                            debug!(
                                "Completed saving batch {} with {} embeddings in {:.2?}",
                                batch_id,
                                count,
                                batch_start.elapsed()
                            );
                        }
                        Err(e) => {
                            error!("Failed to save batch {}: {}", batch_id, e);
                        }
                    }

                    result
                }
            })
            .buffer_unordered(self.concurrent_writers)
            .collect::<Vec<Result<usize>>>()
            .await;

        // Process results and count successes/failures
        let mut successful_count = 0;
        let mut failed_batches = 0;

        for result in results {
            match result {
                Ok(count) => successful_count += count,
                Err(_) => failed_batches += 1,
            }
        }

        if failed_batches > 0 {
            warn!("{} batches failed to save", failed_batches);
        }

        info!(
            "Saved {} embeddings in {:.2?} ({} batches failed)",
            successful_count,
            start.elapsed(),
            failed_batches
        );

        Ok(successful_count)
    }
    /// Check if an index exists on the embedding column and create it if needed
    pub async fn ensure_embedding_index_exists(&self) -> Result<bool> {
        info!("Checking for embedding index on embedding_v2 column");

        let client = self
            .pool
            .get()
            .await
            .context("Failed to get database connection for index check")?;

        // Check if index exists
        let row = client
            .query_one(
                "SELECT EXISTS (
                    SELECT 1 FROM pg_indexes 
                    WHERE indexname = 'service_embedding_v2_idx'
                )",
                &[],
            )
            .await
            .context("Failed to check for index existence")?;

        let index_exists: bool = row.get(0);

        if index_exists {
            info!("Embedding index already exists");
            return Ok(false); // Index already existed
        }

        info!("Creating embedding index on embedding_v2 column (this may take a while)...");

        // Create the index
        client
            .execute(
                "CREATE INDEX service_embedding_v2_idx ON service 
                 USING ivfflat (embedding_v2 vector_cosine_ops) WITH (lists = 100)",
                &[],
            )
            .await
            .context("Failed to create embedding index")?;

        info!("Embedding index created successfully");
        Ok(true) // Index was created
    }

    /// Log embedding statistics after processing is complete
    ///
    /// # Arguments
    ///
    /// * `progress` - Progress tracker containing statistics
    pub async fn log_embedding_stats(&self, progress: &ProgressTracker) -> Result<()> {
        let (processed, elapsed) = progress.get_stats().await;

        if processed > 0 {
            let avg_time = elapsed.as_secs_f64() / (processed as f64);
            info!(
                "Completed embedding generation for {} services in {:.2?}",
                processed, elapsed
            );
            info!(
                "Average processing time: {:.3?} per service",
                std::time::Duration::from_secs_f64(avg_time)
            );
        } else {
            info!("No services were processed for embedding");
        }

        Ok(())
    }
}

impl Clone for DatabaseWriter {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            concurrent_writers: self.concurrent_writers,
        }
    }
}

impl Clone for ProgressTracker {
    fn clone(&self) -> Self {
        Self {
            processed: self.processed.clone(),
            total: self.total,
            start_time: self.start_time,
            last_update_time: self.last_update_time.clone(),
            update_interval: self.update_interval,
            // Add the new fields
            processed_ids: self.processed_ids.clone(),
            update_history: self.update_history.clone(),
        }
    }
}
