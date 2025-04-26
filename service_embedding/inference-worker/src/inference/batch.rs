// inference_worker/src/inference/batch.rs
use anyhow::Result;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::telemetry::gpu_metrics::GPUMetrics;
use crate::types::types::TokenizedDocument;

/// BatchOptimizer dynamically adjusts batch sizes based on
/// performance metrics and GPU utilization
pub struct BatchOptimizer {
    // Configuration
    pub min_batch_size: usize,
    pub max_batch_size: usize,
    initial_batch_size: usize,
    target_latency_ms: f64,

    // Current state
    current_optimal_batch_size: usize,
    performance_samples: VecDeque<(usize, f64)>, // (batch_size, latency)
    last_optimization: Instant,
    optimization_interval: Duration,

    // GPU metrics for informed decisions
    gpu_metrics: Arc<Mutex<GPUMetrics>>,

    // progressive batch optimization
    pub is_warmup_phase: bool,
    pub warmup_batches_processed: usize,
    pub warmup_target_batches: usize,
    pub warmup_scale_factor: f64,
    pub last_batch_had_error: bool,
    pub consecutive_successful_batches: usize,
}

impl BatchOptimizer {
    pub fn new(
        min_batch_size: usize,
        max_batch_size: usize,
        initial_batch_size: usize,
        target_latency_ms: f64,
        gpu_metrics: Arc<Mutex<GPUMetrics>>,
    ) -> Self {
        Self {
            min_batch_size,
            max_batch_size,
            initial_batch_size,
            target_latency_ms,

            current_optimal_batch_size: initial_batch_size,
            performance_samples: VecDeque::with_capacity(100),
            last_optimization: Instant::now(),
            optimization_interval: Duration::from_secs(60),

            // Progressive sizing fields
            is_warmup_phase: true,
            warmup_batches_processed: 0,
            warmup_target_batches: 5, // Exit warmup after 5 successful batches
            warmup_scale_factor: 1.2, // Increase batch size by 20% after each successful batch
            last_batch_had_error: false,
            consecutive_successful_batches: 0,

            gpu_metrics,
        }
    }

    /// Record a batch processing result
    pub fn record_batch_performance(&mut self, batch_size: usize, latency_ms: f64) {
        self.performance_samples.push_back((batch_size, latency_ms));

        // Keep the samples queue at a reasonable size
        if self.performance_samples.len() > 100 {
            self.performance_samples.pop_front();
        }

        debug!(
            "Recorded batch performance: size={}, latency={}ms",
            batch_size, latency_ms
        );
    }

    /// Optimize batch size if needed
    pub async fn optimize_if_needed(&mut self) -> Result<usize> {
        // Check if it's time to optimize
        if self.last_optimization.elapsed() < self.optimization_interval {
            return Ok(self.current_optimal_batch_size);
        }

        info!("Performing batch size optimization");
        self.last_optimization = Instant::now();

        // Get GPU metrics - store values instead of keeping the lock
        let (memory_utilization, compute_utilization) = {
            let mut gpu_metrics = self.gpu_metrics.lock().await;
            (
                gpu_metrics.get_memory_utilization(),
                gpu_metrics.get_compute_utilization(),
            )
        }; // The lock is released here

        // If we don't have enough samples, use heuristics based on GPU utilization
        if self.performance_samples.len() < 10 {
            return self
                .optimize_heuristic(memory_utilization, compute_utilization)
                .await;
        }

        // Get average latency for each batch size
        let mut batch_latencies: std::collections::HashMap<usize, Vec<f64>> =
            std::collections::HashMap::new();

        for &(batch_size, latency) in &self.performance_samples {
            batch_latencies
                .entry(batch_size)
                .or_insert_with(Vec::new)
                .push(latency);
        }

        // Calculate average latency for each batch size
        let mut avg_latencies: Vec<(usize, f64)> = batch_latencies
            .iter()
            .map(|(&size, latencies)| {
                let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;
                (size, avg)
            })
            .collect();

        // Sort by batch size
        avg_latencies.sort_by_key(|&(size, _)| size);

        // Find optimal batch size that meets target latency
        let mut optimal_size = self.min_batch_size;

        for &(size, latency) in &avg_latencies {
            if latency <= self.target_latency_ms {
                optimal_size = size;
            } else {
                break;
            }
        }

        // Adjust by GPU utilization
        if memory_utilization > 90.0 {
            // Memory is nearly full, reduce batch size
            optimal_size = (optimal_size * 8 / 10).max(self.min_batch_size);
            warn!(
                "High memory utilization ({}%), reducing batch size to {}",
                memory_utilization, optimal_size
            );
        } else if compute_utilization < 50.0 && memory_utilization < 70.0 {
            // GPU is underutilized, increase batch size
            optimal_size = (optimal_size * 12 / 10).min(self.max_batch_size);
            info!(
                "Low GPU utilization ({}%), increasing batch size to {}",
                compute_utilization, optimal_size
            );
        }

        // Update optimal batch size
        self.current_optimal_batch_size = optimal_size;
        info!("Optimal batch size determined: {}", optimal_size);

        Ok(optimal_size)
    }

    pub fn record_batch_result(&mut self, success: bool, latency_ms: f64) {
        if success {
            self.last_batch_had_error = false;
            self.consecutive_successful_batches += 1;

            // If in warmup phase, process successful batch
            if self.is_warmup_phase {
                self.warmup_batches_processed += 1;

                // If enough successful batches, exit warmup
                if self.warmup_batches_processed >= self.warmup_target_batches {
                    tracing::info!(
                        "Exiting warmup phase after {} successful batches",
                        self.warmup_batches_processed
                    );
                    self.is_warmup_phase = false;
                    // Set to the initial configured batch size
                    self.current_optimal_batch_size = self.initial_batch_size;
                } else {
                    // Gradually increase batch size during warmup
                    let new_size = (self.current_optimal_batch_size as f64
                        * self.warmup_scale_factor)
                        .ceil() as usize;
                    self.current_optimal_batch_size = new_size.min(self.initial_batch_size);
                    tracing::info!(
                        "Warmup phase: increasing batch size to {}",
                        self.current_optimal_batch_size
                    );
                }
            }
        } else {
            // On error, reduce batch size and reset consecutive success counter
            self.last_batch_had_error = true;
            self.consecutive_successful_batches = 0;

            // Reduce batch size by 25% on errors, but not below minimum
            let new_size = (self.current_optimal_batch_size as f64 * 0.75).ceil() as usize;
            self.current_optimal_batch_size = new_size.max(self.min_batch_size);
            tracing::warn!(
                "Batch processing error: reducing batch size to {}",
                self.current_optimal_batch_size
            );
        }

        // Still record performance for optimization
        self.record_batch_performance(self.current_optimal_batch_size, latency_ms);
    }

    async fn optimize_heuristic(
        &mut self,
        memory_utilization: f64,
        compute_utilization: f64,
    ) -> Result<usize> {
        let mut optimal_size = self.current_optimal_batch_size;

        // If in warmup phase, stick with current progressive sizing
        if self.is_warmup_phase {
            return Ok(optimal_size);
        }

        // If we had errors recently, be more conservative
        if self.last_batch_had_error {
            // Just keep the reduced size from record_batch_result
            return Ok(optimal_size);
        }

        // Get GPU metrics asynchronously and store the result
        let mut gpu_metrics = self.gpu_metrics.lock().await;

        // Calculate available memory and memory per document estimate
        let available_memory_mb = gpu_metrics.get_memory_free_mb();
        let embedding_dimensions = 384;

        // Estimate memory per document
        let estimated_memory_per_doc_mb =
            0.5 + (embedding_dimensions as f64 * 4.0 / 1024.0 / 1024.0);

        // Calculate memory-based limit
        let memory_based_limit =
            ((available_memory_mb * 0.8) / estimated_memory_per_doc_mb) as usize;
        debug!(
            "Memory-based batch limit: {} (available: {:.2}MB, per-doc: {:.2}MB)",
            memory_based_limit, available_memory_mb, estimated_memory_per_doc_mb
        );

        // Apply more sophisticated rules based on utilization
        if memory_utilization > 85.0 {
            // High memory utilization - reduce batch size
            optimal_size = (optimal_size * 8 / 10).max(self.min_batch_size);
            warn!(
                "High memory utilization ({}%), reducing batch size to {}",
                memory_utilization, optimal_size
            );
        } else if memory_utilization < 60.0 && compute_utilization < 60.0 {
            // Low utilization - increase batch size
            optimal_size = (optimal_size * 12 / 10).min(self.max_batch_size);
            info!(
                "Low utilization (mem={}%, compute={}%), increasing batch size to {}",
                memory_utilization, compute_utilization, optimal_size
            );
        } else {
            debug!("Keeping current batch size of {}", optimal_size);
        }

        // Cap by memory-based limit
        if optimal_size > memory_based_limit {
            optimal_size = memory_based_limit;
            info!(
                "Capping batch size to {} due to memory constraints",
                optimal_size
            );
        }

        // Cap by memory-based limit and ensure minimum batch size
        optimal_size = optimal_size
            .min(memory_based_limit)
            .max(self.min_batch_size);

        // Update optimal batch size
        self.current_optimal_batch_size = optimal_size;

        Ok(optimal_size)
    }

    /// Get current optimal batch size
    pub fn get_optimal_batch_size(&self) -> usize {
        self.current_optimal_batch_size
    }

    /// Suggest a batch size for the given set of documents
    pub async fn suggest_batch_size(&mut self, documents: &[TokenizedDocument]) -> Result<usize> {
        let total_docs = documents.len();

        // If we have fewer documents than min batch size, use all of them
        if total_docs <= self.min_batch_size {
            return Ok(total_docs);
        }

        // Otherwise, optimize batch size
        let optimal = self.optimize_if_needed().await?;

        // Return the smaller of the optimal size and the total documents
        Ok(optimal.min(total_docs))
    }

    /// Split documents into optimal batches
    pub async fn create_optimal_batches(
        &mut self,
        documents: Vec<TokenizedDocument>,
    ) -> Result<Vec<Vec<TokenizedDocument>>> {
        let total_docs = documents.len();

        // If empty, return empty result
        if total_docs == 0 {
            return Ok(Vec::new());
        }

        // Optimize batch size
        let batch_size = self.optimize_if_needed().await?;

        // Split documents into batches
        let mut batches = Vec::new();
        let mut remaining = documents;

        while !remaining.is_empty() {
            let split_at = remaining.len().min(batch_size);
            let batch = remaining.drain(0..split_at).collect();
            batches.push(batch);
        }

        info!(
            "Split {} documents into {} batches of optimal size ~{}",
            total_docs,
            batches.len(),
            batch_size
        );

        Ok(batches)
    }
}

/// Advanced batch processing strategy with dynamic batching
pub struct BatchProcessor {
    optimizer: Arc<Mutex<BatchOptimizer>>,
}

impl BatchProcessor {
    pub fn new(optimizer: Arc<Mutex<BatchOptimizer>>) -> Self {
        Self { optimizer }
    }

    pub fn get_optimizer(&self) -> &Arc<Mutex<BatchOptimizer>> {
        &self.optimizer
    }

    /// Process documents in optimized batches
    pub async fn process_documents<F, Fut, R>(
        &self,
        documents: Vec<TokenizedDocument>,
        processor: F,
    ) -> Result<Vec<R>>
    where
        F: Fn(Vec<TokenizedDocument>) -> Fut,
        Fut: std::future::Future<Output = Result<Vec<R>>>,
    {
        // Get batches from optimizer
        let batches = {
            let mut optimizer = self.optimizer.lock().await;
            optimizer.create_optimal_batches(documents).await?
        };

        // Process each batch and collect results
        let mut all_results = Vec::new();

        for batch in batches {
            let batch_size = batch.len();
            let start = Instant::now();

            // Process this batch
            let batch_results = processor(batch).await?;

            // Record performance
            let latency = start.elapsed().as_secs_f64() * 1000.0;
            {
                let mut optimizer = self.optimizer.lock().await;
                optimizer.record_batch_performance(batch_size, latency);
            }

            // Add results to overall results
            all_results.extend(batch_results);
        }

        Ok(all_results)
    }
}
