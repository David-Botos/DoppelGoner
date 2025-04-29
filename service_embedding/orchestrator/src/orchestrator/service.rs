// orchestrator/src/orchestrator/service.rs

use anyhow::{anyhow, Result};
use futures::{stream, StreamExt};
use serde_json::Value;
use sqlx::PgPool;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{mpsc, RwLock};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::client::worker::WorkerClient;
use crate::db;
use crate::orchestrator::stats::PipelineStats;
use crate::tokenizer::{ServiceTokenizer, TokenizerConfig};
use crate::types::types::{
    BatchProcessResponse, EmbeddingDocument, EmbeddingJob, EmbeddingResult, JobStatus,
    ServiceWithTaxonomies, TaxonomyDocument, TokenizedDocument, Worker, WorkerCapabilities,
    WorkerStatus, WorkerType,
};

// Configuration for the orchestrator service
#[derive(Clone, Debug)]
pub struct OrchestratorConfig {
    pub batch_size: i32,
    pub model_id: String,
    pub tokenizer_config: TokenizerConfig,
    pub worker_check_interval_secs: u64,
    pub job_timeout_mins: u64,
    pub stale_job_threshold_mins: i32,
    pub max_concurrent_tokenization: usize,
    pub max_concurrent_inference: usize,
    pub default_worker_locations: Vec<String>,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            batch_size: 10,
            model_id: "bge-small-en-v1.5".to_string(),
            tokenizer_config: TokenizerConfig::default(),
            worker_check_interval_secs: 30,
            job_timeout_mins: 10,
            stale_job_threshold_mins: 30,
            max_concurrent_tokenization: 5,
            max_concurrent_inference: 3,
            default_worker_locations: vec![
                "10.0.0.3:3000".to_string(),
                "10.0.0.4:3000".to_string(),
            ],
        }
    }
}

// Orchestrator service for managing embedding jobs
pub struct OrchestratorService {
    config: OrchestratorConfig,
    pub worker_client: WorkerClient,
    pool: PgPool,
    stats: Arc<RwLock<PipelineStats>>,
    tokenizer: ServiceTokenizer,
}

impl OrchestratorService {
    pub fn new(worker_client: WorkerClient, pool: PgPool, config: OrchestratorConfig) -> Self {
        // Try to create the tokenizer, but panic if it fails
        let tokenizer = ServiceTokenizer::with_config(config.tokenizer_config.clone())
            .unwrap_or_else(|e| {
                error!("FATAL: Failed to load tokenizer: {}. Cannot continue.", e);
                // This will stop the program immediately
                panic!("Failed to initialize tokenizer: {}", e);

                // Alternative approach using process::exit
                // std::process::exit(1);
            });

        Self {
            config,
            worker_client,
            pool,
            stats: Arc::new(RwLock::new(PipelineStats::default())),
            tokenizer,
        }
    }

    /// Queue new jobs for services with missing embeddings
    pub async fn queue_new_jobs(&self, batch_limit: i32) -> Result<i64> {
        let count: i64 = sqlx::query_scalar(
            r#"
            WITH inserted_jobs AS (
                INSERT INTO embedding.embedding_jobs (id, service_id)
                SELECT gen_random_uuid(), s.id
                FROM public.service s
                WHERE s.embedding_v2 IS NULL
                AND NOT EXISTS (
                    SELECT 1 
                    FROM embedding.embedding_jobs ej 
                    WHERE ej.service_id = s.id 
                    AND ej.status IN ('queued', 'processing')
                )
                LIMIT $1
                RETURNING 1
            )
            SELECT COUNT(*) FROM inserted_jobs
            "#,
        )
        .bind(batch_limit)
        .fetch_one(&self.pool)
        .await?;
    
        Ok(count)
    }

    // Get current pipeline stats
    pub async fn get_stats(&self) -> PipelineStats {
        self.stats.read().await.clone()
    }

    // Reset pipeline stats
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = PipelineStats::default();
    }

    // Update stats after a pipeline run
    async fn update_stats(&self, new_stats: PipelineStats) {
        let mut stats = self.stats.write().await;
        stats.add(&new_stats);
    }

    // Dynamically determine optimal batch size based on available workers
    pub async fn get_optimal_batch_size(&self) -> i32 {
        let available_workers = self.worker_client.get_all_workers().await;
        let active_workers: Vec<&Worker> = available_workers
            .iter()
            .filter(|w| w.status == WorkerStatus::Online || w.status == WorkerStatus::Busy)
            .collect();

        if active_workers.is_empty() {
            // Default to configured batch size if no workers available
            return self.config.batch_size;
        }

        // Calculate total capacity of active workers
        let total_capacity: i32 = active_workers
            .iter()
            .map(|w| w.capabilities.optimal_batch_size)
            .sum();

        // Calculate total current load
        let total_load: i32 = active_workers
            .iter()
            .filter_map(|w| {
                w.current_load
                    .map(|load| (load * w.capabilities.optimal_batch_size as f32) as i32)
            })
            .sum();

        // Calculate available capacity
        let available_capacity = std::cmp::max(1, total_capacity - total_load);

        // Use the min of available capacity and configured batch size
        std::cmp::min(available_capacity, self.config.batch_size)
    }

    // Claim a batch of jobs from the database
    pub async fn claim_jobs(&self, batch_size: i32) -> Result<Vec<EmbeddingJob>> {
        // Generate worker ID for identification
        let worker_id = format!("orchestrator-{}", Uuid::new_v4());

        db::claim_jobs(&self.pool, &worker_id, batch_size).await
    }

    // Fetch service data for a batch of jobs
    pub async fn fetch_service_data(
        &self,
        jobs: &[EmbeddingJob],
    ) -> Result<Vec<(EmbeddingJob, ServiceWithTaxonomies)>> {
        let mut result = Vec::with_capacity(jobs.len());

        // Process jobs in parallel with bounded concurrency
        let (tx, mut rx) = mpsc::channel(jobs.len());

        let mut handles = Vec::new();
        for job in jobs {
            let job_clone = job.clone();
            let tx = tx.clone();
            let pool = self.pool.clone();

            let handle = tokio::spawn(async move {
                let job_id = job_clone.id;

                // Fetch service data
                match db::fetch_service_with_taxonomies(&pool, &job_clone.service_id).await {
                    Ok(service_data) => {
                        // Convert to ServiceWithTaxonomies
                        match convert_to_service_with_taxonomies(service_data) {
                            Ok(service) => {
                                let _ = tx.send(Ok((job_clone, service))).await;
                            }
                            Err(e) => {
                                error!("Failed to convert service data for job {}: {}", job_id, e);
                                let _ = tx.send(Err((job_clone, e))).await;
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to fetch service data for job {}: {}", job_id, e);
                        let _ = tx
                            .send(Err((
                                job_clone,
                                anyhow!("Failed to fetch service data: {}", e),
                            )))
                            .await;
                    }
                }
            });

            handles.push(handle);
        }

        // Drop the original sender so the channel can close when all senders are dropped
        drop(tx);

        // Process results as they come in
        while let Some(res) = rx.recv().await {
            match res {
                Ok((job, service)) => {
                    result.push((job, service));
                }
                Err((job, e)) => {
                    // Update job status to failed
                    let _ = self
                        .update_job_status(
                            &job.id,
                            JobStatus::Failed,
                            Some(format!("Service data fetch error: {}", e)),
                            None,
                            None,
                        )
                        .await;
                }
            }
        }

        // Wait for all tasks to complete
        for handle in handles {
            let _ = handle.await;
        }

        Ok(result)
    }

    // Tokenize services for embedding with parallel processing
    pub async fn tokenize_services(
        &self,
        services_data: &[(EmbeddingJob, ServiceWithTaxonomies)],
    ) -> Result<Vec<(EmbeddingJob, TokenizedDocument)>> {
        let mut tokenized_docs = Vec::with_capacity(services_data.len());

        // Process in parallel with bounded concurrency
        let concurrency = self.config.max_concurrent_tokenization;

        let (tx, mut rx) = mpsc::channel(services_data.len());

        let mut handles = Vec::new();
        for (job, service) in services_data {
            let job_clone = job.clone();
            let service_clone = service.clone();
            let tx = tx.clone();
            let tokenizer = self.tokenizer.clone();
            let pool = self.pool.clone();

            let handle = tokio::spawn(async move {
                let job_id = job_clone.id;

                // Create embedding document with combined text
                let embedding_doc = EmbeddingDocument {
                    service_id: service_clone.id.clone(),
                    service_name: service_clone.name.clone(),
                    service_desc: service_clone.description.clone().unwrap_or_default(),
                    taxonomies: service_clone
                        .taxonomies
                        .iter()
                        .map(|t| TaxonomyDocument {
                            taxonomy_name: t.term.clone(),
                            taxonomy_desc: t.description.clone(),
                        })
                        .collect(),
                };

                // Tokenize document
                match tokenizer.tokenize(&embedding_doc) {
                    Ok((tokenized_text, token_count)) => {
                        // Create tokenized document
                        let tokenized_doc = TokenizedDocument {
                            service_id: service_clone.id.clone(),
                            tokenized_text,
                            token_count,
                            job_id: job_clone.id,
                        };

                        // Update job status to tokenized
                        match db::update_job_status(
                            &pool,
                            &job_id,
                            JobStatus::Tokenized,
                            None,
                            Some(token_count as i32),
                            None,
                        )
                        .await
                        {
                            Ok(_) => {
                                let _ = tx.send(Ok((job_clone, tokenized_doc))).await;
                            }
                            Err(e) => {
                                error!("Failed to update job status for {}: {}", job_id, e);
                                let _ = tx
                                    .send(Err((
                                        job_clone,
                                        anyhow!("Failed to update job status: {}", e),
                                    )))
                                    .await;
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to tokenize service {}: {}", job_clone.service_id, e);

                        // Update job status to failed
                        let _ = db::update_job_status(
                            &pool,
                            &job_id,
                            JobStatus::Failed,
                            Some(&format!("Tokenization error: {}", e)),
                            None,
                            None,
                        )
                        .await;

                        let _ = tx
                            .send(Err((job_clone, anyhow!("Tokenization error: {}", e))))
                            .await;
                    }
                }
            });

            handles.push(handle);

            // Limit concurrency
            if handles.len() >= concurrency {
                let _ = futures::future::join_all(handles.drain(..1)).await;
            }
        }

        // Drop the original sender
        drop(tx);

        // Process results as they come in
        while let Some(res) = rx.recv().await {
            match res {
                Ok((job, tokenized_doc)) => {
                    tokenized_docs.push((job, tokenized_doc));
                }
                Err((job, e)) => {
                    // Job status already updated to failed in the tokenization task
                    error!("Tokenization failed for job {}: {}", job.id, e);
                }
            }
        }

        // Wait for all remaining tasks to complete
        for handle in handles {
            let _ = handle.await;
        }

        Ok(tokenized_docs)
    }

    // Process a batch of tokenized documents through inference workers
    pub async fn process_tokenized_batch(
        &self,
        tokenized_data: &[(EmbeddingJob, TokenizedDocument)],
    ) -> Result<Vec<(EmbeddingJob, EmbeddingResult)>> {
        if tokenized_data.is_empty() {
            return Ok(Vec::new());
        }

        // Extract jobs and tokenized documents
        let jobs: Vec<EmbeddingJob> = tokenized_data.iter().map(|(job, _)| job.clone()).collect();
        let tokenized_docs: Vec<TokenizedDocument> =
            tokenized_data.iter().map(|(_, doc)| doc.clone()).collect();

        // Update job status to processing
        for job in &jobs {
            let _ = self
                .update_job_status(&job.id, JobStatus::Processing, None, None, None)
                .await;
        }

        // Process in smaller sub-batches if needed for optimal worker utilization
        let available_workers = self.worker_client.get_all_workers().await;
        let mut job_results: Vec<(EmbeddingJob, EmbeddingResult)> =
            Vec::with_capacity(tokenized_docs.len());

        // Get optimal sub-batch size
        let optimal_batch_size = available_workers
            .iter()
            .filter(|w| w.status == WorkerStatus::Online)
            .map(|w| w.capabilities.optimal_batch_size as usize)
            .min()
            .unwrap_or(10);

        // Split into sub-batches if needed
        let mut sub_batches = Vec::new();
        for chunk in tokenized_docs.chunks(optimal_batch_size) {
            sub_batches.push(chunk.to_vec());
        }

        // Process sub-batches with limited concurrency
        let concurrency = self.config.max_concurrent_inference;
        let results = stream::iter(sub_batches)
            .map(|batch| {
                let client = self.worker_client.clone();
                let model_id = self.config.model_id.clone();
                async move { client.process_batch(batch, &model_id, None).await }
            })
            .buffer_unordered(concurrency)
            .collect::<Vec<Result<BatchProcessResponse>>>()
            .await;

        // Create job_id to job mapping for easy lookup
        let mut job_map: std::collections::HashMap<Uuid, EmbeddingJob> =
            jobs.into_iter().map(|job| (job.id, job)).collect();

        // Process results from all sub-batches
        for result in results {
            match result {
                Ok(batch_response) => {
                    for embedding_result in batch_response.results {
                        if let Some(job) = job_map.remove(&embedding_result.job_id) {
                            job_results.push((job, embedding_result));
                        }
                    }
                }
                Err(e) => {
                    // Mark remaining jobs as failed
                    for (_, job) in job_map.iter() {
                        let _ = self
                            .update_job_status(
                                &job.id,
                                JobStatus::Failed,
                                Some(format!("Inference failed: {}", e)),
                                None,
                                None,
                            )
                            .await;
                    }

                    return Err(anyhow!("Inference processing failed: {}", e));
                }
            }
        }

        // Mark any remaining jobs as failed (this shouldn't happen if workers behave correctly)
        for (_, job) in job_map {
            let _ = self
                .update_job_status(
                    &job.id,
                    JobStatus::Failed,
                    Some("Job not processed by any worker".to_string()),
                    None,
                    None,
                )
                .await;
        }

        Ok(job_results)
    }

    // Store embedding results in the database with parallel processing
    // Store embedding results in the database with parallel processing
    pub async fn store_embeddings(
        &self,
        results: &[(EmbeddingJob, EmbeddingResult)],
    ) -> Result<()> {
        if results.is_empty() {
            return Ok(());
        }

        let concurrency = 10; // Database writes can be done in parallel
        let (tx, mut rx) = mpsc::channel(results.len());

        let mut handles = Vec::new();
        for (job, result) in results {
            let job_clone = job.clone();
            let result_clone = result.clone();
            let tx = tx.clone();
            let pool = self.pool.clone();

            let handle = tokio::spawn(async move {
                let job_id = job_clone.id;

                // Update service with embedding - using query instead of query! macro
                match sqlx::query(
                    r#"
                UPDATE public.service
                SET 
                    embedding_v2 = $1::float4[]::vector(384),
                    embedding_v2_updated_at = NOW()
                WHERE id = $2
                "#,
                )
                .bind(&result_clone.embedding)
                .bind(&result_clone.service_id)
                .execute(&pool)
                .await
                {
                    Ok(_) => {
                        // Update job status to completed
                        let metadata = serde_json::json!({
                            "model_id": result_clone.model_id,
                            "processing_time_ms": result_clone.processing_time_ms,
                            "embedding_dimensions": result_clone.embedding.len()
                        });

                        match db::update_job_status(
                            &pool,
                            &job_id,
                            JobStatus::Completed,
                            None,
                            Some(result_clone.token_count as i32),
                            Some(&metadata),
                        )
                        .await
                        {
                            Ok(_) => {
                                let _ = tx.send(Ok(job_id)).await;
                            }
                            Err(e) => {
                                error!("Failed to update job status for {}: {}", job_id, e);
                                let _ = tx
                                    .send(Err((
                                        job_id,
                                        anyhow!("Failed to update job status: {}", e),
                                    )))
                                    .await;
                            }
                        }
                    }
                    Err(e) => {
                        error!(
                            "Failed to store embedding for service {}: {}",
                            result_clone.service_id, e
                        );

                        // Update job status to failed
                        let error_msg = format!("Embedding storage error: {}", e);
                        let _ = db::update_job_status(
                            &pool,
                            &job_id,
                            JobStatus::Failed,
                            Some(&error_msg),
                            None,
                            None,
                        )
                        .await;

                        let _ = tx.send(Err((job_id, anyhow!(error_msg)))).await;
                    }
                }
            });

            handles.push(handle);

            // Limit concurrency
            if handles.len() >= concurrency {
                let _ = futures::future::join_all(handles.drain(..1)).await;
            }
        }

        // Drop the original sender
        drop(tx);

        // Process results for error tracking
        let mut errors = 0;
        while let Some(res) = rx.recv().await {
            if let Err(_) = res {
                errors += 1;
            }
        }

        // Wait for all remaining tasks to complete
        for handle in handles {
            let _ = handle.await;
        }

        if errors > 0 {
            warn!("{} embeddings failed to store", errors);
        }

        Ok(())
    }

    // Update job status
    pub async fn update_job_status(
        &self,
        job_id: &Uuid,
        status: JobStatus,
        error_message: Option<String>,
        input_tokens: Option<i32>,
        metadata: Option<Value>,
    ) -> Result<EmbeddingJob> {
        db::update_job_status(
            &self.pool,
            job_id,
            status,
            error_message.as_deref(),
            input_tokens,
            metadata.as_ref(),
        )
        .await
    }

    // Run a full embedding pipeline with improved resource utilization
    pub async fn run_embedding_pipeline(&self) -> Result<PipelineStats> {
        let start_time = Instant::now();
        let mut stats = PipelineStats::default();

        // 1. Queue new jobs for services needing embeddings
        let queue_batch_size = self.config.batch_size * 2; // Queue more than we'll process
        let newly_queued = self.queue_new_jobs(queue_batch_size).await?;

        info!(
            "Queued {} new jobs for services needing embeddings",
            newly_queued
        );

        // 2. Determine optimal batch size based on available workers
        let batch_size = self.get_optimal_batch_size().await;

        // 3. Claim jobs
        let fetch_start = Instant::now();
        let jobs = self.claim_jobs(batch_size).await?;

        if jobs.is_empty() {
            // No jobs to process
            return Ok(stats);
        }

        stats.jobs_processed = jobs.len() as i64;
        info!("Claimed {} jobs for processing", jobs.len());

        // 4. Fetch service data
        let services_data = self.fetch_service_data(&jobs).await?;
        stats.services_fetched = services_data.len();
        stats.fetch_time_ms = fetch_start.elapsed().as_millis() as u64;

        if services_data.is_empty() {
            info!("No valid services found for the claimed jobs");
            return Ok(stats);
        }

        // 5. Tokenize services
        let tokenize_start = Instant::now();
        let tokenized_data = self.tokenize_services(&services_data).await?;
        stats.documents_tokenized = tokenized_data.len() as i64;
        stats.tokenize_time_ms = tokenize_start.elapsed().as_millis() as u64;

        if tokenized_data.is_empty() {
            info!("No documents were successfully tokenized");
            return Ok(stats);
        }

        // 6. Process through inference workers
        let inference_start = Instant::now();
        let results = match self.process_tokenized_batch(&tokenized_data).await {
            Ok(results) => results,
            Err(e) => {
                // We already marked jobs as failed in process_tokenized_batch
                error!("Inference pipeline failed: {}", e);
                stats.errors = stats.documents_tokenized; // All documents failed
                stats.inference_time_ms = inference_start.elapsed().as_millis() as u64;
                stats.total_time_ms = start_time.elapsed().as_millis() as u64;
                return Ok(stats);
            }
        };

        stats.embeddings_generated = results.len() as i64;
        stats.inference_time_ms = inference_start.elapsed().as_millis() as u64;

        // 7. Store embeddings
        let storage_start = Instant::now();
        if let Err(e) = self.store_embeddings(&results).await {
            error!("Failed to store embeddings: {}", e);
            stats.errors = stats.embeddings_generated; // All embeddings failed to store
        }

        stats.storage_time_ms = storage_start.elapsed().as_millis() as u64;
        stats.total_time_ms = start_time.elapsed().as_millis() as u64;

        info!(
            "Pipeline completed: processed {} jobs, tokenized {} documents, generated {} embeddings in {}ms",
            stats.jobs_processed, stats.documents_tokenized, stats.embeddings_generated, stats.total_time_ms
        );

        // Update global stats
        self.update_stats(stats.clone()).await;

        Ok(stats)
    }

    // Reset stale jobs that haven't been completed
    pub async fn reset_stale_jobs(&self) -> Result<Vec<EmbeddingJob>> {
        let threshold_mins = self.config.stale_job_threshold_mins;

        let jobs = db::reset_stale_jobs(&self.pool, threshold_mins).await?;

        if !jobs.is_empty() {
            info!("Reset {} stale jobs", jobs.len());
        }

        Ok(jobs)
    }

    // Get all workers from the registry
    pub async fn get_all_workers(&self) -> Result<Vec<Worker>> {
        Ok(self.worker_client.get_all_workers().await)
    }

    // Register fixed inference workers if they're not already registered
    pub async fn register_default_workers(&self) -> Result<Vec<Worker>> {
        let mut registered_workers = Vec::new();

        // Check which workers are already registered
        let existing_workers = self.worker_client.get_all_workers().await;
        let existing_hostnames: std::collections::HashSet<String> = existing_workers
            .iter()
            .map(|w| w.hostname.clone())
            .collect();

        // Register workers that aren't already registered
        for location in &self.config.default_worker_locations {
            // Handle both formats: "hostname" and "hostname:port"
            let (hostname, port_str) = if location.contains(':') {
                let parts: Vec<&str> = location.split(':').collect();
                if parts.len() != 2 {
                    warn!("Invalid worker location format: {}", location);
                    continue;
                }
                (parts[0].to_string(), parts[1].to_string())
            } else {
                // Default port if not specified
                (location.clone(), "3000".to_string())
            };

            // Skip if already registered
            if existing_hostnames.contains(&hostname) {
                info!("Worker {} already registered", hostname);
                continue;
            }

            // Store full address for API calls
            let full_address = if hostname.contains(':') {
                hostname.clone()
            } else {
                format!("{}:{}", hostname, port_str)
            };

            // Try to connect to worker and get its status
            let client = reqwest::Client::new();
            let url = format!("http://{}/api/status", full_address);

            match client
                .get(&url)
                .header("X-API-Key", &self.worker_client.config.api_key)
                .timeout(Duration::from_secs(5))
                .send()
                .await
            {
                Ok(response) => {
                    if response.status().is_success() {
                        // Register worker
                        let worker_id = format!("inference-{}", Uuid::new_v4());

                        let worker = Worker {
                            id: worker_id.clone(),
                            hostname: hostname.clone(),
                            ip_address: Some(full_address.clone()), // Store the full address including port
                            worker_type: WorkerType::Inference,
                            capabilities: WorkerCapabilities {
                                gpu_type: Some("Unknown".to_string()),
                                gpu_memory_mb: Some(8192), // Default assumption
                                supports_cuda: true,
                                supports_metal: false,
                                cpu_cores: 4,
                                optimal_batch_size: 24,
                                max_batch_size: 48,
                                embedding_dimensions: Some(384),
                            },
                            status: WorkerStatus::Online,
                            last_heartbeat: SystemTime::now(),
                            current_batch_size: None,
                            current_load: Some(0.0),
                            active_jobs: 0,
                            created_at: SystemTime::now(),
                        };

                        // Register with database
                        if let Err(e) = db::register_worker(
                            &self.pool,
                            &worker.id,
                            &worker.hostname,
                            worker.ip_address.as_deref(),
                            worker.worker_type,
                            &worker.capabilities,
                        )
                        .await
                        {
                            error!("Failed to register worker in database: {}", e);
                            continue;
                        }

                        // Register with registry
                        self.worker_client
                            .registry
                            .register_worker(worker.clone())
                            .await;
                        registered_workers.push(worker);
                        info!("Registered worker at {}", full_address);
                    } else {
                        warn!(
                            "Worker at {} returned status {}",
                            full_address,
                            response.status()
                        );
                    }
                }
                Err(e) => {
                    warn!("Failed to connect to worker at {}: {}", full_address, e);
                }
            }
        }

        Ok(registered_workers)
    }
}

// Helper to convert database rows to ServiceWithTaxonomies
fn convert_to_service_with_taxonomies(
    rows: Vec<crate::client::worker::ServiceWithTaxonomiesRow>,
) -> Result<ServiceWithTaxonomies> {
    if rows.is_empty() {
        return Err(anyhow!("No service data found"));
    }

    let first_row = &rows[0];

    let taxonomies = rows
        .iter()
        .filter_map(|row| {
            // Skip rows with null taxonomy
            if row.taxonomy_id.is_none() {
                return None;
            }

            Some(crate::types::types::TaxonomyTerm {
                id: row.taxonomy_id.clone().unwrap(),
                term: row.taxonomy_term.clone().unwrap_or_default(),
                description: row.taxonomy_description.clone(),
                taxonomy: row.taxonomy_category.clone(),
            })
        })
        .collect();

    let service = ServiceWithTaxonomies {
        id: first_row.service_id.clone(),
        name: first_row.service_name.clone().unwrap_or_default(),
        description: first_row.service_description.clone(),
        short_description: first_row.service_short_description.clone(),
        taxonomies,
        organization_id: first_row.organization_id.clone(),
        url: first_row.service_url.clone(),
        email: first_row.service_email.clone(),
        status: first_row.service_status.clone().unwrap_or_default(),
    };

    Ok(service)
}
