// orchestrator/src/client/worker.rs

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::{stream, StreamExt};
use metrics::{counter, histogram};
use rand::seq::SliceRandom;
use rand::thread_rng;
use reqwest::Client;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::types::types::{
    BatchProcessRequest, BatchProcessResponse, TokenizedDocument, Worker, WorkerCapabilities,
    WorkerStatus, WorkerType,
};

// Constants for client behavior
const MAX_RETRIES: usize = 3;
const BASE_RETRY_DELAY_MS: u64 = 100;
const CIRCUIT_BREAKER_THRESHOLD: usize = 5;
const CIRCUIT_BREAKER_RESET_SECS: u64 = 300;
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 60;

// Worker client configuration
#[derive(Clone, Debug)]
pub struct WorkerClientConfig {
    pub api_key: String,
    pub request_timeout_secs: u64,
    pub max_concurrent_requests: usize,
    pub max_retries: usize,
    pub circuit_breaker_threshold: usize,
    pub circuit_breaker_reset_secs: u64,
    pub stale_worker_threshold_secs: u64,
    pub worker_discovery_interval_secs: u64,
    pub load_balance_strategy: LoadBalanceStrategy,
}

impl Default for WorkerClientConfig {
    fn default() -> Self {
        Self {
            api_key: "default-api-key".to_string(),
            request_timeout_secs: DEFAULT_REQUEST_TIMEOUT_SECS,
            max_concurrent_requests: 10,
            max_retries: MAX_RETRIES,
            circuit_breaker_threshold: CIRCUIT_BREAKER_THRESHOLD,
            circuit_breaker_reset_secs: CIRCUIT_BREAKER_RESET_SECS,
            stale_worker_threshold_secs: 60,
            worker_discovery_interval_secs: 30,
            load_balance_strategy: LoadBalanceStrategy::LeastLoaded,
        }
    }
}

// Strategy for load balancing
#[derive(Clone, Debug, PartialEq)]
pub enum LoadBalanceStrategy {
    RoundRobin,
    LeastLoaded,
    FastestResponse,
    OptimalBatchSize,
    Random,
}

// Circuit breaker state for each worker
#[derive(Debug)]
struct CircuitBreakerState {
    failure_count: usize,
    last_failure: Instant,
    is_open: bool,
    threshold: usize,
    reset_duration: Duration,
}

impl CircuitBreakerState {
    fn new(threshold: usize, reset_duration: Duration) -> Self {
        Self {
            failure_count: 0,
            last_failure: Instant::now(),
            is_open: false,
            threshold,
            reset_duration,
        }
    }

    fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure = Instant::now();

        if self.failure_count >= self.threshold {
            self.is_open = true;
        }
    }

    fn record_success(&mut self) {
        self.failure_count = 0;
        self.is_open = false;
    }

    fn can_request(&self) -> bool {
        if !self.is_open {
            return true;
        }

        // Check if circuit should auto-reset
        if self.last_failure.elapsed() >= self.reset_duration {
            return true;
        }

        false
    }

    fn reset(&mut self) {
        self.failure_count = 0;
        self.is_open = false;
    }
}

// Worker registry to track available workers
#[derive(Clone)]
pub struct WorkerRegistry {
    workers: Arc<RwLock<HashMap<String, Worker>>>,
    circuit_breakers: Arc<Mutex<HashMap<String, CircuitBreakerState>>>,
    response_times: Arc<Mutex<HashMap<String, Vec<f64>>>>,
    worker_order: Arc<Mutex<Vec<String>>>, // For round-robin
    current_index: Arc<Mutex<usize>>,      // For round-robin
    config: WorkerClientConfig,
}

impl WorkerRegistry {
    pub fn new(config: WorkerClientConfig) -> Self {
        Self {
            workers: Arc::new(RwLock::new(HashMap::new())),
            circuit_breakers: Arc::new(Mutex::new(HashMap::new())),
            response_times: Arc::new(Mutex::new(HashMap::new())),
            worker_order: Arc::new(Mutex::new(Vec::new())),
            current_index: Arc::new(Mutex::new(0)),
            config,
        }
    }

    // Add a worker to the registry
    pub async fn register_worker(&self, worker: Worker) {
        let worker_id = worker.id.clone();

        // Add worker to registry
        {
            let mut workers = self.workers.write().await;
            workers.insert(worker_id.clone(), worker);
        }

        // Add circuit breaker state
        {
            let mut breakers = self.circuit_breakers.lock().await;
            if !breakers.contains_key(&worker_id) {
                breakers.insert(
                    worker_id.clone(),
                    CircuitBreakerState::new(
                        self.config.circuit_breaker_threshold,
                        Duration::from_secs(self.config.circuit_breaker_reset_secs),
                    ),
                );
            }
        }

        // Add to worker order for round-robin
        {
            let mut order = self.worker_order.lock().await;
            if !order.contains(&worker_id) {
                order.push(worker_id.clone());
            }
        }

        // Initialize response times tracking
        {
            let mut times = self.response_times.lock().await;
            if !times.contains_key(&worker_id) {
                times.insert(worker_id.clone(), Vec::new());
            }
        }

        info!("Worker registered: {}", worker_id);
    }

    // Remove a worker from the registry
    pub async fn deregister_worker(&self, worker_id: &str) {
        {
            let mut workers = self.workers.write().await;
            workers.remove(worker_id);
        }

        {
            let mut breakers = self.circuit_breakers.lock().await;
            breakers.remove(worker_id);
        }

        {
            let mut order = self.worker_order.lock().await;
            if let Some(pos) = order.iter().position(|id| id == worker_id) {
                order.remove(pos);
            }
        }

        {
            let mut times = self.response_times.lock().await;
            times.remove(worker_id);
        }

        info!("Worker deregistered: {}", worker_id);
    }

    // Update worker status
    pub async fn update_worker_status(&self, worker_id: &str, status: WorkerStatus) -> Result<()> {
        let mut workers = self.workers.write().await;

        if let Some(worker) = workers.get_mut(worker_id) {
            worker.status = status;
            worker.last_heartbeat = SystemTime::now();
            Ok(())
        } else {
            Err(anyhow!("Worker not found: {}", worker_id))
        }
    }

    // Get a list of available workers
    pub async fn get_available_workers(&self) -> Vec<Worker> {
        let workers = self.workers.read().await;
        let breakers = self.circuit_breakers.lock().await;

        workers
            .values()
            .filter(|w| w.status == WorkerStatus::Online || w.status == WorkerStatus::Busy)
            .filter(|w| {
                if let Some(breaker) = breakers.get(&w.id) {
                    breaker.can_request()
                } else {
                    true
                }
            })
            .cloned()
            .collect()
    }

    // Get worker by ID
    pub async fn get_worker(&self, worker_id: &str) -> Option<Worker> {
        let workers = self.workers.read().await;
        workers.get(worker_id).cloned()
    }

    // Find a worker for a batch using the configured load balancing strategy
    pub async fn find_worker_for_batch(&self, batch: &[TokenizedDocument]) -> Result<Worker> {
        let available_workers = self.get_available_workers().await;

        if available_workers.is_empty() {
            return Err(anyhow!("No available workers"));
        }

        // Filter workers by capability to process the batch
        let batch_size = batch.len();
        let capable_workers: Vec<Worker> = available_workers
            .into_iter()
            .filter(|w| w.capabilities.max_batch_size >= batch_size as i32)
            .collect();

        if capable_workers.is_empty() {
            return Err(anyhow!(
                "No workers capable of processing batch size {}",
                batch_size
            ));
        }

        // Apply load balancing strategy
        match self.config.load_balance_strategy {
            LoadBalanceStrategy::RoundRobin => self.get_next_round_robin(&capable_workers).await,
            LoadBalanceStrategy::LeastLoaded => self.get_least_loaded(&capable_workers).await,
            LoadBalanceStrategy::FastestResponse => {
                self.get_fastest_response(&capable_workers).await
            }
            LoadBalanceStrategy::OptimalBatchSize => {
                self.get_optimal_batch_size(&capable_workers, batch_size)
                    .await
            }
            LoadBalanceStrategy::Random => self.get_random(&capable_workers).await,
        }
    }

    // Record successful response time for a worker
    pub async fn record_response_time(&self, worker_id: &str, response_time_ms: f64) {
        let mut times = self.response_times.lock().await;

        if let Some(worker_times) = times.get_mut(worker_id) {
            worker_times.push(response_time_ms);

            // Keep only recent response times (last 50)
            if worker_times.len() > 50 {
                *worker_times = worker_times.split_off(worker_times.len() - 50);
            }
        }

        // Update metrics
        histogram!("worker.response_time_ms", response_time_ms, "worker_id" => worker_id.to_string());
    }

    // Record success for circuit breaker
    pub async fn record_success(&self, worker_id: &str) {
        let mut breakers = self.circuit_breakers.lock().await;

        if let Some(breaker) = breakers.get_mut(worker_id) {
            breaker.record_success();
        }
    }

    // Record failure for circuit breaker
    pub async fn record_failure(&self, worker_id: &str) {
        let mut breakers = self.circuit_breakers.lock().await;

        if let Some(breaker) = breakers.get_mut(worker_id) {
            breaker.record_failure();

            // Update metrics
            counter!("worker.failures", 1, "worker_id" => worker_id.to_string());

            if breaker.is_open {
                counter!("worker.circuit_breaks", 1, "worker_id" => worker_id.to_string());
                info!("Circuit breaker opened for worker: {}", worker_id);
            }
        }
    }

    // Reset circuit breaker for worker
    pub async fn reset_circuit_breaker(&self, worker_id: &str) {
        let mut breakers = self.circuit_breakers.lock().await;

        if let Some(breaker) = breakers.get_mut(worker_id) {
            breaker.reset();
            info!("Circuit breaker reset for worker: {}", worker_id);
        }
    }

    // Clean up stale workers
    pub async fn clean_stale_workers(&self) -> Vec<String> {
        let now = SystemTime::now();
        let threshold = Duration::from_secs(self.config.stale_worker_threshold_secs);
        let mut workers_to_remove = Vec::new();

        // Identify stale workers
        {
            let workers = self.workers.read().await;

            for (id, worker) in workers.iter() {
                if let Ok(elapsed) = now.duration_since(worker.last_heartbeat) {
                    if elapsed > threshold {
                        workers_to_remove.push(id.clone());
                    }
                }
            }
        }

        // Remove stale workers
        for worker_id in &workers_to_remove {
            self.deregister_worker(worker_id).await;
            info!("Removed stale worker: {}", worker_id);
        }

        workers_to_remove
    }

    // Round-robin worker selection
    async fn get_next_round_robin(&self, workers: &[Worker]) -> Result<Worker> {
        let mut index = self.current_index.lock().await;
        let order = self.worker_order.lock().await;

        // Find the next available worker in our order
        let worker_ids: HashSet<String> = workers.iter().map(|w| w.id.clone()).collect();
        let valid_ordered_ids: Vec<&String> =
            order.iter().filter(|id| worker_ids.contains(*id)).collect();

        if valid_ordered_ids.is_empty() {
            return Err(anyhow!("No workers available in round-robin sequence"));
        }

        // Increment index and wrap around
        *index = (*index + 1) % valid_ordered_ids.len();

        // Find the worker in our original list
        let selected_id = valid_ordered_ids[*index];
        workers
            .iter()
            .find(|w| w.id == *selected_id)
            .cloned()
            .ok_or_else(|| anyhow!("Worker not found in available list"))
    }

    // Least loaded worker selection
    async fn get_least_loaded(&self, workers: &[Worker]) -> Result<Worker> {
        workers
            .iter()
            .min_by(|a, b| {
                a.current_load
                    .partial_cmp(&b.current_load)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .ok_or_else(|| anyhow!("No workers available for least loaded selection"))
    }

    // Fastest response worker selection
    async fn get_fastest_response(&self, workers: &[Worker]) -> Result<Worker> {
        let times = self.response_times.lock().await;

        // Calculate average response time for each worker
        let mut avg_times: Vec<(String, f64)> = Vec::new();

        for worker in workers {
            if let Some(response_times) = times.get(&worker.id) {
                if !response_times.is_empty() {
                    let avg = response_times.iter().sum::<f64>() / response_times.len() as f64;
                    avg_times.push((worker.id.clone(), avg));
                } else {
                    // If no response times, assume fast (give new workers a chance)
                    avg_times.push((worker.id.clone(), 0.0));
                }
            } else {
                // If no entry, assume fast (give new workers a chance)
                avg_times.push((worker.id.clone(), 0.0));
            }
        }

        // Find worker with lowest average response time
        if avg_times.is_empty() {
            return self.get_random(workers).await;
        }

        let fastest = avg_times
            .iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id.clone())
            .unwrap();

        workers
            .iter()
            .find(|w| w.id == fastest)
            .cloned()
            .ok_or_else(|| anyhow!("Worker not found for fastest response selection"))
    }

    // Optimal batch size worker selection
    async fn get_optimal_batch_size(
        &self,
        workers: &[Worker],
        batch_size: usize,
    ) -> Result<Worker> {
        // Find worker with optimal batch size closest to our batch size
        workers
            .iter()
            .min_by_key(|w| {
                (w.capabilities.optimal_batch_size as isize - batch_size as isize).abs()
            })
            .cloned()
            .ok_or_else(|| anyhow!("No workers available for optimal batch size selection"))
    }

    // Random worker selection
    async fn get_random(&self, workers: &[Worker]) -> Result<Worker> {
        let mut rng = thread_rng();
        workers
            .choose(&mut rng)
            .cloned()
            .ok_or_else(|| anyhow!("No workers available for random selection"))
    }
}

// Worker client to communicate with inference workers
#[derive(Clone)]
pub struct WorkerClient {
    http_client: Client,
    pub registry: WorkerRegistry,
    pub config: WorkerClientConfig,
}

impl WorkerClient {
    pub fn new(config: WorkerClientConfig) -> Self {
        // Create HTTP client with timeout
        let http_client = Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            http_client,
            registry: WorkerRegistry::new(config.clone()),
            config,
        }
    }

    // Process a batch of documents using available workers
    pub async fn process_batch(
        &self,
        documents: Vec<TokenizedDocument>,
        model_id: &str,
        priority: Option<i32>,
    ) -> Result<BatchProcessResponse> {
        if documents.is_empty() {
            return Err(anyhow!("Cannot process empty batch"));
        }

        // Create request structure
        let request = BatchProcessRequest {
            documents,
            request_id: Uuid::new_v4(),
            priority,
            model_id: model_id.to_string(),
        };

        let batch_size = request.documents.len();

        // Find suitable worker
        let worker = match self
            .registry
            .find_worker_for_batch(&request.documents)
            .await
        {
            Ok(worker) => worker,
            Err(e) => {
                error!("Failed to find worker for batch: {}", e);
                return Err(anyhow!("No suitable worker available: {}", e));
            }
        };

        info!(
            "Selected worker {} for batch {} with {} documents",
            worker.id, request.request_id, batch_size
        );

        // Process with retries
        self.send_batch_with_retry(&worker, request).await
    }

    // Send batch to worker with retry logic
    async fn send_batch_with_retry(
        &self,
        worker: &Worker,
        request: BatchProcessRequest,
    ) -> Result<BatchProcessResponse> {
        let mut last_error = None;
        let batch_size = request.documents.len();

        for retry in 0..=self.config.max_retries {
            // Check if circuit breaker is open
            let circuit_open = {
                let breakers = self.registry.circuit_breakers.lock().await;
                if let Some(breaker) = breakers.get(&worker.id) {
                    !breaker.can_request()
                } else {
                    false
                }
            };

            if circuit_open {
                warn!(
                    "Circuit breaker open for worker {}, skipping retry",
                    worker.id
                );
                return Err(anyhow!("Circuit breaker open for worker {}", worker.id));
            }

            // If not the first retry, wait with exponential backoff
            if retry > 0 {
                let delay = BASE_RETRY_DELAY_MS * 2u64.pow(retry as u32 - 1);
                info!(
                    "Retry {}/{} for batch {} to worker {}, waiting {}ms",
                    retry, self.config.max_retries, request.request_id, worker.id, delay
                );
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }

            // Send request to worker
            let start_time = Instant::now();
            let result = self.send_batch_to_worker(worker, request.clone()).await;
            let elapsed = start_time.elapsed().as_millis() as f64;

            match result {
                Ok(response) => {
                    // Record success metrics
                    self.registry.record_success(&worker.id).await;
                    self.registry
                        .record_response_time(&worker.id, elapsed)
                        .await;
                    debug!(
                        "Batch {} processed successfully by worker {} in {:.2}ms",
                        request.request_id, worker.id, elapsed
                    );

                    // Update worker metrics
                    counter!("batch.processed", 1);
                    counter!("documents.processed", batch_size as u64);
                    histogram!("batch.processing_time_ms", response.processing_time_ms);

                    return Ok(response);
                }
                Err(e) => {
                    // Record failure for circuit breaker
                    self.registry.record_failure(&worker.id).await;
                    last_error = Some(e.to_string());

                    // Update worker metrics
                    counter!("batch.failed", 1);

                    warn!(
                        "Batch {} to worker {} failed: {}",
                        request.request_id,
                        worker.id,
                        last_error.as_ref().unwrap()
                    );
                }
            }
        }

        // All retries failed
        error!(
            "All retries failed for batch {} to worker {}",
            request.request_id, worker.id
        );

        Err(anyhow!(
            "Failed to process batch after {} retries: {}",
            self.config.max_retries,
            last_error.unwrap_or_else(|| "Unknown error".to_string())
        ))
    }

    // Send batch to a specific worker
    async fn send_batch_to_worker(
        &self,
        worker: &Worker,
        request: BatchProcessRequest,
    ) -> Result<BatchProcessResponse> {
        let server_address = if let Some(ip) = &worker.ip_address {
            if ip.contains(':') {
                // ip_address already has a port, use it directly
                ip.clone()
            } else {
                // No port in ip_address, append default
                format!("{}:3000", ip)
            }
        } else {
            // Fall back to hostname with default port
            format!("{}:3000", worker.hostname)
        };

        let worker_url = format!("http://{}/api/batches", server_address);

        // Send request to worker
        let response = self
            .http_client
            .post(&worker_url)
            .header("X-API-Key", &self.config.api_key)
            .json(&request)
            .send()
            .await?;

        // Handle response
        if response.status().is_success() {
            let batch_response: BatchProcessResponse = response.json().await?;
            Ok(batch_response)
        } else {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            Err(anyhow!(
                "Worker returned error status {}: {}",
                status,
                error_text
            ))
        }
    }

    // Process multiple batches in parallel
    pub async fn process_batches_parallel(
        &self,
        batches: Vec<(Vec<TokenizedDocument>, String)>, // (documents, model_id)
        max_concurrency: Option<usize>,
    ) -> Vec<Result<BatchProcessResponse>> {
        let concurrency = max_concurrency.unwrap_or(self.config.max_concurrent_requests);

        info!(
            "Processing {} batches with concurrency {}",
            batches.len(),
            concurrency
        );

        // Process batches in parallel with bounded concurrency
        let results = stream::iter(batches)
            .map(|(documents, model_id)| {
                let client = self.clone();
                async move { client.process_batch(documents, &model_id, None).await }
            })
            .buffer_unordered(concurrency)
            .collect::<Vec<_>>()
            .await;

        results
    }

    // Check health of a worker
    pub async fn check_worker_health(&self, worker_id: &str) -> Result<bool> {
        let worker = match self.registry.get_worker(worker_id).await {
            Some(w) => w,
            None => return Err(anyhow!("Worker not found: {}", worker_id)),
        };

        let server_address = if let Some(ip) = &worker.ip_address {
            if ip.contains(':') {
                // ip_address already has a port, use it directly
                ip.clone()
            } else {
                // No port in ip_address, append default
                format!("{}:3000", ip)
            }
        } else {
            // Fall back to hostname with default port
            format!("{}:3000", worker.hostname)
        };

        let url = format!("http://{}/api/health", server_address);

        match self.http_client.get(&url).send().await {
            Ok(response) => {
                let healthy = response.status().is_success();

                // Update worker status based on health check
                if healthy {
                    self.registry.reset_circuit_breaker(worker_id).await;
                } else {
                    self.registry.record_failure(worker_id).await;
                }

                Ok(healthy)
            }
            Err(_) => {
                self.registry.record_failure(worker_id).await;
                Ok(false)
            }
        }
    }

    // Reset circuit breaker for a worker
    pub async fn reset_worker_circuit_breaker(&self, worker_id: &str) -> Result<()> {
        if let Some(_) = self.registry.get_worker(worker_id).await {
            self.registry.reset_circuit_breaker(worker_id).await;
            Ok(())
        } else {
            Err(anyhow!("Worker not found: {}", worker_id))
        }
    }

    // Start a worker discovery task
    pub async fn start_discovery(&self, discovery_service: Arc<dyn WorkerDiscovery>) {
        let registry = self.registry.clone();
        let interval_secs = self.config.worker_discovery_interval_secs;

        info!("Starting worker discovery with interval {}s", interval_secs);

        // Spawn discovery task
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));

            loop {
                interval.tick().await;

                // Discover workers
                match discovery_service.discover_workers().await {
                    Ok(workers) => {
                        for worker in workers {
                            registry.register_worker(worker).await;
                        }
                    }
                    Err(e) => {
                        error!("Worker discovery failed: {}", e);
                    }
                }

                // Clean up stale workers
                let removed = registry.clean_stale_workers().await;
                if !removed.is_empty() {
                    info!("Removed {} stale workers", removed.len());
                }
            }
        });
    }

    // Get all registered workers
    pub async fn get_all_workers(&self) -> Vec<Worker> {
        // In a synchronous context, we need to be careful about how we access async values
        // This method is a simplification that returns currently registered workers
        // from the registry by creating a blocking task

        let registry = self.registry.clone();

        // Create a runtime for the blocking operation (only in contexts where we don't have a runtime)
        let workers = tokio::task::block_in_place(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create runtime");

            rt.block_on(async move {
                let workers_map = registry.workers.read().await;
                workers_map.values().cloned().collect::<Vec<_>>()
            })
        });

        workers
    }

    // Get circuit breaker status for all workers
    pub async fn get_circuit_breaker_status(&self) -> HashMap<String, bool> {
        let breakers = self.registry.circuit_breakers.lock().await;
        breakers
            .iter()
            .map(|(id, state)| (id.clone(), state.is_open))
            .collect()
    }
}

// Worker discovery trait
#[async_trait]
pub trait WorkerDiscovery: Send + Sync {
    async fn discover_workers(&self) -> Result<Vec<Worker>>;
}

// Database worker discovery implementation
pub struct DatabaseWorkerDiscovery {
    pool: sqlx::PgPool,
}

impl DatabaseWorkerDiscovery {
    pub fn new(pool: sqlx::PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl WorkerDiscovery for DatabaseWorkerDiscovery {
    async fn discover_workers(&self) -> Result<Vec<Worker>> {
        let workers = crate::db::get_all_workers(&self.pool, Some(WorkerType::Inference)).await?;
        Ok(workers)
    }
}

// Registry worker discovery implementation (for testing)
pub struct RegistryWorkerDiscovery {
    workers: Vec<Worker>,
}

impl RegistryWorkerDiscovery {
    pub fn new(workers: Vec<Worker>) -> Self {
        Self { workers }
    }
}

#[async_trait]
impl WorkerDiscovery for RegistryWorkerDiscovery {
    async fn discover_workers(&self) -> Result<Vec<Worker>> {
        Ok(self.workers.clone())
    }
}

// Re-export ServiceWithTaxonomiesRow for compatibility
pub use crate::types::types::ServiceWithTaxonomiesRow;
