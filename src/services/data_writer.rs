// src/services/db_writer.rs

use anyhow::{Context, Result};
use bb8::Pool;
use bb8_postgres::PostgresConnectionManager;
use futures::{StreamExt, stream};
use log::{debug, error, info, warn};
use pgvector::Vector;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio_postgres::NoTls;

/// Tracks progress during embedding operations
///
/// Provides methods for updating and displaying progress metrics
/// to give feedback during long-running embedding operations.
pub struct ProgressTracker {
    processed: Arc<Mutex<i64>>,
    total: i64,
    start_time: Instant,
    last_update_time: Arc<Mutex<Instant>>,
    update_interval: Duration,
}

impl ProgressTracker {
    /// Create a new progress tracker
    ///
    /// # Arguments
    ///
    /// * `total` - Total number of services to process
    /// * `update_interval` - Minimum time between progress log messages
    pub fn new(total: i64, update_interval: Duration) -> Self {
        let now = Instant::now();
        Self {
            processed: Arc::new(Mutex::new(0)),
            total,
            start_time: now,
            last_update_time: Arc::new(Mutex::new(now)),
            update_interval,
        }
    }

    /// Update progress and log information if enough time has passed since last update
    ///
    /// # Arguments
    ///
    /// * `count` - Number of new items processed
    /// * `batch_id` - Identifier for the current batch (for logging)
    /// * `force` - Force update even if interval hasn't passed
    pub async fn update(&self, count: i64, batch_id: &str, force: bool) -> Result<()> {
        let mut processed = self.processed.lock().await;
        *processed += count;

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
            "Batch {}: Processed {}/{} services ({:.1}%) - Est. time remaining: {:02}:{:02}:{:02}",
            batch_id,
            *processed,
            self.total,
            progress_percentage,
            time_remaining_secs / 3600,
            (time_remaining_secs % 3600) / 60,
            time_remaining_secs % 60
        );

        Ok(())
    }

    /// Get the current progress statistics
    ///
    /// Returns a tuple containing the number of processed items and elapsed time
    pub async fn get_stats(&self) -> (i64, Duration) {
        let processed = *self.processed.lock().await;
        let elapsed = self.start_time.elapsed();
        (processed, elapsed)
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

        debug!(
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

        // Start a transaction
        let transaction = client
            .transaction()
            .await
            .context(format!("Batch {}: Failed to start transaction", batch_id))?;

        for (i, (id, embedding)) in service_ids.iter().zip(embeddings).enumerate() {
            // Convert Vec<f32> to pgvector::Vector
            let pgvector = Vector::from(embedding.clone());

            // Update the service record with the new embedding
            transaction
                .execute(
                    "UPDATE service SET embedding_v2 = $1, embedding_v2_updated_at = NOW() WHERE id = $2",
                    &[&pgvector, id],
                )
                .await
                .context(format!("Batch {}: Failed to update embedding for service {}", batch_id, id))?;

            // Log progress for large batches
            if (i + 1) % 20 == 0 || i == service_ids.len() - 1 {
                debug!(
                    "Batch {}: Progress {}/{} embeddings",
                    batch_id,
                    i + 1,
                    service_ids.len()
                );
            }
        }

        // Commit the transaction
        transaction
            .commit()
            .await
            .context(format!("Batch {}: Failed to commit transaction", batch_id))?;

        debug!(
            "Batch {}: Saved {} embeddings in {:.2?}",
            batch_id,
            service_ids.len(),
            start.elapsed()
        );

        Ok(service_ids.len())
    }

    /// Save multiple batches of embeddings in parallel
    ///
    /// # Arguments
    ///
    /// * `batches` - Vector of (service_ids, embeddings) tuples
    /// * `progress` - Progress tracker to update during processing
    pub async fn save_embeddings_in_batches(
        &self,
        batches: Vec<(Vec<String>, Vec<Vec<f32>>)>,
        progress: &ProgressTracker,
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
                let progress_tracker = progress.clone();
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
                            // Update progress
                            progress_tracker
                                .update(*count as i64, &batch_id, false)
                                .await
                                .unwrap_or_else(|e| warn!("Failed to update progress: {}", e));

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

        // Final progress update with force=true to ensure the last update is shown
        progress.update(0, "final", true).await?;

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
        }
    }
}
