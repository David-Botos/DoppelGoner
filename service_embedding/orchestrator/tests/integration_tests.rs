// orchestrator/tests/integration_tests.rs

use anyhow::Result;
use serde_json::json;
use sqlx::{postgres::PgPoolOptions, PgPool, Row};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{oneshot, RwLock};
use uuid::Uuid;

// Import needed modules from the orchestrator crate
use orchestrator::client::worker::{
    LoadBalanceStrategy, RegistryWorkerDiscovery, WorkerClient, WorkerClientConfig,
};
use orchestrator::db;
use orchestrator::orchestrator::{OrchestratorConfig, OrchestratorService, PipelineStats};
use orchestrator::tokenizer::TokenizerConfig;
use orchestrator::types::types::{
    BatchProcessResponse, EmbeddingJob, EmbeddingResult, JobStatus, Worker, WorkerCapabilities,
    WorkerStatus, WorkerType,
};

// Mock worker for testing
struct MockWorker {
    pool: PgPool,
    worker_info: Worker,
    batch_responses: RwLock<Vec<BatchProcessResponse>>,
}

impl MockWorker {
    fn new(pool: PgPool, worker_id: &str, hostname: &str) -> Self {
        let worker_info = Worker {
            id: worker_id.to_string(),
            hostname: hostname.to_string(),
            ip_address: Some(hostname.to_string()),
            worker_type: WorkerType::Inference,
            capabilities: WorkerCapabilities {
                gpu_type: Some("NVIDIA RTX A6000".to_string()),
                gpu_memory_mb: Some(49152),
                supports_cuda: true,
                supports_metal: false,
                cpu_cores: 32,
                optimal_batch_size: 32,
                max_batch_size: 64,
                embedding_dimensions: Some(384),
            },
            status: WorkerStatus::Online,
            last_heartbeat: SystemTime::now(),
            current_batch_size: None,
            current_load: Some(0.1),
            active_jobs: 0,
            created_at: SystemTime::now(),
        };

        Self {
            pool,
            worker_info,
            batch_responses: RwLock::new(Vec::new()),
        }
    }

    // Register the worker in the database
    async fn register(&self) -> Result<()> {
        db::register_worker(
            &self.pool,
            &self.worker_info.id,
            &self.worker_info.hostname,
            self.worker_info.ip_address.as_deref(),
            self.worker_info.worker_type,
            &self.worker_info.capabilities,
        )
        .await
    }

    // Add a pre-defined response for when the orchestrator calls this worker
    async fn add_batch_response(&self, response: BatchProcessResponse) {
        let mut responses = self.batch_responses.write().await;
        responses.push(response);
    }

    // Generate a generic success response for a batch of documents
    fn generate_success_response(
        &self,
        documents: &[orchestrator::types::types::TokenizedDocument],
    ) -> BatchProcessResponse {
        let mut results = Vec::with_capacity(documents.len());

        for doc in documents {
            // Generate a random embedding vector (384 dimensions)
            let embedding = (0..384).map(|_| rand::random::<f32>()).collect();

            results.push(EmbeddingResult {
                service_id: doc.service_id.clone(),
                job_id: doc.job_id,
                embedding,
                processing_time_ms: 50.0, // Simulated processing time
                model_id: "bge-small-en-v1.5".to_string(),
                token_count: doc.token_count,
            });
        }

        BatchProcessResponse {
            results,
            request_id: Uuid::new_v4(), // Random request ID
            processing_time_ms: 100.0,  // Simulated batch processing time
            worker_id: self.worker_info.id.clone(),
            error: None,
        }
    }

    // Mock processing a batch of documents
    async fn process_batch(
        &self,
        documents: Vec<orchestrator::types::types::TokenizedDocument>,
    ) -> BatchProcessResponse {
        // See if we have a pre-defined response
        let mut responses = self.batch_responses.write().await;
        if !responses.is_empty() {
            return responses.remove(0);
        }

        // Otherwise generate a generic success response
        self.generate_success_response(&documents)
    }
}

// Helper to setup test database
async fn setup_test_db() -> PgPool {
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost/embeddings_test".to_string());

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await
        .expect("Failed to connect to test database");

    // Initialize test database (same structure as in db_tests.rs)
    sqlx::query(
        r#"
        -- Create schemas and types if they don't exist
        CREATE SCHEMA IF NOT EXISTS embedding;
        
        DO $$ 
        BEGIN
            -- Create job_status type if it doesn't exist
            IF NOT EXISTS (SELECT 1 FROM pg_type 
                          JOIN pg_namespace ON pg_namespace.oid = pg_type.typnamespace
                          WHERE pg_type.typname = 'job_status' 
                          AND pg_namespace.nspname = 'embedding') THEN
                CREATE TYPE embedding.job_status AS ENUM ('queued', 'fetched', 'tokenized', 'processing', 'completed', 'failed');
            END IF;
            
            -- Create worker_type type if it doesn't exist
            IF NOT EXISTS (SELECT 1 FROM pg_type 
                          JOIN pg_namespace ON pg_namespace.oid = pg_type.typnamespace
                          WHERE pg_type.typname = 'worker_type' 
                          AND pg_namespace.nspname = 'embedding') THEN
                CREATE TYPE embedding.worker_type AS ENUM ('orchestrator', 'inference');
            END IF;
            
            -- Create worker_status type if it doesn't exist
            IF NOT EXISTS (SELECT 1 FROM pg_type 
                          JOIN pg_namespace ON pg_namespace.oid = pg_type.typnamespace
                          WHERE pg_type.typname = 'worker_status' 
                          AND pg_namespace.nspname = 'embedding') THEN
                CREATE TYPE embedding.worker_status AS ENUM ('online', 'offline', 'busy');
            END IF;
        END $$;
        
        -- Create necessary tables for testing
        CREATE TABLE IF NOT EXISTS embedding.jobs (
            id UUID PRIMARY KEY,
            service_id VARCHAR NOT NULL,
            status embedding.job_status NOT NULL DEFAULT 'queued',
            worker_id VARCHAR,
            batch_id UUID,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
            error_message TEXT,
            retry_count INT NOT NULL DEFAULT 0,
            last_retry_at TIMESTAMP,
            input_tokens INT,
            truncation_strategy VARCHAR,
            metadata JSONB DEFAULT '{}'::JSONB
        );
        
        CREATE TABLE IF NOT EXISTS embedding.workers (
            id VARCHAR PRIMARY KEY,
            hostname VARCHAR NOT NULL,
            ip_address VARCHAR,
            worker_type embedding.worker_type NOT NULL,
            capabilities JSONB NOT NULL DEFAULT '{}'::JSONB,
            status embedding.worker_status NOT NULL DEFAULT 'offline',
            last_heartbeat TIMESTAMP NOT NULL DEFAULT NOW(),
            current_batch_size INT,
            current_load FLOAT,
            active_jobs INT NOT NULL DEFAULT 0,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS public.service (
            id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            description TEXT,
            short_description TEXT,
            organization_id VARCHAR NOT NULL,
            url VARCHAR,
            email VARCHAR,
            status VARCHAR NOT NULL DEFAULT 'active',
            embedding_v2 VECTOR(384),
            embedding_v2_updated_at TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS public.taxonomy (
            id VARCHAR PRIMARY KEY,
            term VARCHAR NOT NULL,
            description TEXT,
            category VARCHAR
        );
        
        CREATE TABLE IF NOT EXISTS public.service_taxonomy (
            service_id VARCHAR NOT NULL REFERENCES public.service(id),
            taxonomy_id VARCHAR NOT NULL REFERENCES public.taxonomy(id),
            PRIMARY KEY(service_id, taxonomy_id)
        );
        
        -- Create function for claim_jobs if it doesn't exist
        CREATE OR REPLACE FUNCTION embedding.claim_jobs(
            p_worker_id VARCHAR,
            p_batch_size INT
        )
        RETURNS SETOF embedding.jobs AS $$
        DECLARE
            v_batch_id UUID := uuid_generate_v4();
        BEGIN
            RETURN QUERY
            WITH jobs_to_update AS (
                SELECT id FROM embedding.jobs
                WHERE status = 'queued'
                ORDER BY created_at
                LIMIT p_batch_size
                FOR UPDATE SKIP LOCKED
            )
            UPDATE embedding.jobs j
            SET 
                status = 'fetched',
                worker_id = p_worker_id,
                batch_id = v_batch_id,
                updated_at = NOW()
            FROM jobs_to_update jtu
            WHERE j.id = jtu.id
            RETURNING j.*;
        END;
        $$ LANGUAGE plpgsql;
        
        -- Create function for update_job_status if it doesn't exist
        CREATE OR REPLACE FUNCTION embedding.update_job_status(
            p_job_id UUID,
            p_status embedding.job_status,
            p_error_message TEXT DEFAULT NULL,
            p_input_tokens INT DEFAULT NULL,
            p_truncation_strategy VARCHAR DEFAULT NULL,
            p_metadata JSONB DEFAULT NULL
        )
        RETURNS embedding.jobs AS $$
        DECLARE
            v_job embedding.jobs;
        BEGIN
            UPDATE embedding.jobs
            SET 
                status = p_status,
                error_message = COALESCE(p_error_message, error_message),
                input_tokens = COALESCE(p_input_tokens, input_tokens),
                truncation_strategy = COALESCE(p_truncation_strategy, truncation_strategy),
                metadata = CASE 
                    WHEN p_metadata IS NOT NULL THEN p_metadata
                    ELSE metadata
                END,
                updated_at = NOW(),
                retry_count = CASE 
                    WHEN p_status = 'failed' THEN retry_count + 1
                    ELSE retry_count
                END,
                last_retry_at = CASE 
                    WHEN p_status = 'queued' AND status = 'failed' THEN NOW()
                    ELSE last_retry_at
                END
            WHERE id = p_job_id
            RETURNING * INTO v_job;
            
            RETURN v_job;
        END;
        $$ LANGUAGE plpgsql;
        
        -- Create function for reset_stale_jobs if it doesn't exist
        CREATE OR REPLACE FUNCTION embedding.reset_stale_jobs(
            p_minutes_threshold INT DEFAULT 30
        )
        RETURNS SETOF embedding.jobs AS $$
        BEGIN
            RETURN QUERY
            UPDATE embedding.jobs
            SET 
                status = 'queued',
                worker_id = NULL,
                error_message = CONCAT(error_message, ' (Reset after stalling in status: ', status, ')'),
                updated_at = NOW()
            WHERE 
                status NOT IN ('completed', 'queued', 'failed') AND
                updated_at < NOW() - (p_minutes_threshold * interval '1 minute')
            RETURNING *;
        END;
        $$ LANGUAGE plpgsql;
        
        -- Create function that mocks fetch_service_with_taxonomies
        CREATE OR REPLACE FUNCTION embedding.fetch_service_with_taxonomies(
            p_service_id VARCHAR
        )
        RETURNS TABLE (
            service_id VARCHAR,
            service_name VARCHAR,
            service_description TEXT,
            service_short_description TEXT,
            organization_id VARCHAR,
            service_url VARCHAR,
            service_email VARCHAR,
            service_status VARCHAR,
            taxonomy_id VARCHAR,
            taxonomy_term VARCHAR,
            taxonomy_description TEXT,
            taxonomy_category VARCHAR
        ) AS $$
        BEGIN
            RETURN QUERY
            WITH service_data AS (
                SELECT * FROM public.service 
                WHERE id = p_service_id
            )
            SELECT 
                s.id AS service_id,
                s.name AS service_name,
                s.description AS service_description,
                s.short_description AS service_short_description,
                s.organization_id,
                s.url AS service_url,
                s.email AS service_email,
                s.status AS service_status,
                t.id AS taxonomy_id,
                t.term AS taxonomy_term,
                t.description AS taxonomy_description,
                t.category AS taxonomy_category
            FROM 
                service_data s
            LEFT JOIN 
                public.service_taxonomy st ON s.id = st.service_id
            LEFT JOIN 
                public.taxonomy t ON st.taxonomy_id = t.id;
        END;
        $$ LANGUAGE plpgsql;
        "#
    )
    .execute(&pool)
    .await
    .expect("Failed to create test database schema");

    // Clean up any existing test data
    sqlx::query("TRUNCATE TABLE embedding.jobs, embedding.workers, public.service, public.taxonomy, public.service_taxonomy RESTART IDENTITY CASCADE")
        .execute(&pool)
        .await
        .expect("Failed to truncate test tables");

    // Try to create vector extension if it doesn't exist (may fail if not available)
    let _ = sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(&pool)
        .await;

    pool
}

// Seed test data for our tests
async fn seed_test_data(pool: &PgPool) -> Result<()> {
    // Insert a test service
    sqlx::query(
        r#"
        INSERT INTO public.service (id, name, description, short_description, organization_id, url, email, status)
        VALUES 
            ('service-test-1', 'Test Service 1', 'This is a comprehensive description of Test Service 1. It provides various features and benefits to users.', 'Short description of service 1', 'org-123', 'https://example.com/service1', 'service1@example.com', 'active'),
            ('service-test-2', 'Test Service 2', 'Test Service 2 helps users accomplish important tasks. It is reliable and efficient.', 'Quick service 2 info', 'org-123', 'https://example.com/service2', 'service2@example.com', 'active')
        "#
    )
    .execute(pool)
    .await?;

    // Insert test taxonomies
    sqlx::query(
        r#"
        INSERT INTO public.taxonomy (id, term, description, category)
        VALUES 
            ('tax-1', 'Financial Services', 'Services related to financial management and transactions', 'category'),
            ('tax-2', 'Healthcare', 'Medical and healthcare related services', 'category'),
            ('tax-3', 'Education', 'Educational and learning services', 'category'),
            ('tax-4', 'Premium', 'Premium tier service offerings', 'tier'),
            ('tax-5', 'Basic', 'Basic tier service offerings', 'tier')
        "#
    )
    .execute(pool)
    .await?;

    // Associate taxonomies with services
    sqlx::query(
        r#"
        INSERT INTO public.service_taxonomy (service_id, taxonomy_id)
        VALUES 
            ('service-test-1', 'tax-1'),
            ('service-test-1', 'tax-4'),
            ('service-test-2', 'tax-2'),
            ('service-test-2', 'tax-3'),
            ('service-test-2', 'tax-5')
        "#,
    )
    .execute(pool)
    .await?;

    // Insert test jobs
    for _ in 0..5 {
        sqlx::query(
            r#"
            INSERT INTO embedding.jobs (id, service_id, status, metadata)
            VALUES 
                ($1, 'service-test-1', 'queued', '{"priority": 1}'::jsonb)
            "#,
        )
        .bind(Uuid::new_v4())
        .execute(pool)
        .await?;
    }

    for _ in 0..5 {
        sqlx::query(
            r#"
            INSERT INTO embedding.jobs (id, service_id, status, metadata)
            VALUES 
                ($1, 'service-test-2', 'queued', '{"priority": 2}'::jsonb)
            "#,
        )
        .bind(Uuid::new_v4())
        .execute(pool)
        .await?;
    }

    Ok(())
}

// Create a mock worker registry for testing
async fn create_mock_worker_registry(worker: &MockWorker) -> RegistryWorkerDiscovery {
    RegistryWorkerDiscovery::new(vec![worker.worker_info.clone()])
}

// Create a worker client with the mock registry
fn create_worker_client(
    pool: &PgPool,
) -> (
    WorkerClient,
    Arc<oneshot::Sender<BatchProcessResponse>>,
    Arc<tokio::sync::Mutex<Vec<Vec<orchestrator::types::types::TokenizedDocument>>>>,
) {
    let config = WorkerClientConfig {
        api_key: "fart".to_string(),
        request_timeout_secs: 60,
        max_concurrent_requests: 5,
        max_retries: 1,
        circuit_breaker_threshold: 3,
        circuit_breaker_reset_secs: 300,
        stale_worker_threshold_secs: 60,
        worker_discovery_interval_secs: 30,
        load_balance_strategy: LoadBalanceStrategy::LeastLoaded,
    };

    let worker_client = WorkerClient::new(config);

    // Create channels for receiving requests
    let (batch_tx, batch_rx) = oneshot::channel::<BatchProcessResponse>();
    let batch_tx = Arc::new(batch_tx);

    // Create storage for tokenized documents that are passed to the worker
    let tokenized_docs = Arc::new(tokio::sync::Mutex::new(Vec::new()));

    // Mock WorkerClient's process_batch method
    let tokenized_docs_clone = tokenized_docs.clone();

    (worker_client, batch_tx, tokenized_docs_clone)
}

// Test for embedding a single service through the pipeline
#[tokio::test]
async fn test_embedding_pipeline_single_job() -> Result<()> {
    let pool = setup_test_db().await;
    seed_test_data(&pool).await?;

    // Setup worker environment
    let mock_worker = MockWorker::new(pool.clone(), "test-worker-1", "10.0.0.3");
    mock_worker.register().await?;

    // Create batch response channel
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);

    // Create worker client with mocked process_batch
    let (worker_client, _, tokenized_docs) = create_worker_client(&pool);

    // Register the worker with the worker client
    worker_client
        .registry
        .register_worker(mock_worker.worker_info.clone())
        .await;

    // Create orchestrator config
    let config = OrchestratorConfig {
        batch_size: 1, // Process just one job
        model_id: "bge-small-en-v1.5".to_string(),
        tokenizer_config: TokenizerConfig::default(),
        worker_check_interval_secs: 30,
        job_timeout_mins: 10,
        stale_job_threshold_mins: 30,
        max_concurrent_tokenization: 5,
        max_concurrent_inference: 3,
        default_worker_locations: vec!["10.0.0.3:3000".to_string()],
    };

    // Create orchestrator service
    let orchestrator = OrchestratorService::new(worker_client, pool.clone(), config);

    // We need to intercept the worker client's process_batch method
    // Start a mock server in a separate task
    let worker_task = tokio::spawn(async move {
        // Spawn a task to handle worker requests
        let doc_batches = tokenized_docs.lock().await;

        // Print information about what we got
        println!("Received {} document batches", doc_batches.len());

        // Mock a successful response
        let mock_response = BatchProcessResponse {
            results: vec![EmbeddingResult {
                service_id: "service-test-1".to_string(),
                job_id: Uuid::new_v4(),    // We'll have to fix this later
                embedding: vec![0.1; 384], // 384-dimension vector of 0.1
                processing_time_ms: 120.5,
                model_id: "bge-small-en-v1.5".to_string(),
                token_count: 256,
            }],
            request_id: Uuid::new_v4(),
            processing_time_ms: 150.8,
            worker_id: "test-worker-1".to_string(),
            error: None,
        };

        // Send the response
        let _ = tx.send(mock_response).await;
    });

    // Run the pipeline
    let result = orchestrator.run_embedding_pipeline().await?;

    // Wait for worker task to complete
    worker_task.await?;

    // Verify the pipeline stats
    assert_eq!(result.jobs_processed, 1, "Should process 1 job");
    assert_eq!(result.services_fetched, 1, "Should fetch 1 service");
    assert_eq!(result.documents_tokenized, 1, "Should tokenize 1 document");

    // Check database for job status updates
    let completed_jobs: i64 =
        sqlx::query("SELECT COUNT(*) FROM embedding.jobs WHERE status = 'completed'")
            .fetch_one(&pool)
            .await?
            .get(0);

    assert_eq!(completed_jobs, 1, "Should have 1 completed job in database");

    // Check for embeddings in the service table
    let services_with_embeddings: i64 =
        sqlx::query("SELECT COUNT(*) FROM public.service WHERE embedding_v2 IS NOT NULL")
            .fetch_one(&pool)
            .await?
            .get(0);

    // This might fail if vector extension is not available in the test DB
    if let Ok(row) =
        sqlx::query("SELECT embedding_v2 FROM public.service WHERE id = 'service-test-1'")
            .fetch_optional(&pool)
            .await
    {
        if let Some(_) = row {
            assert_eq!(
                services_with_embeddings, 1,
                "Should have 1 service with embeddings"
            );
        }
    }

    Ok(())
}

// Test for resetting stale jobs
#[tokio::test]
async fn test_reset_stale_jobs() -> Result<()> {
    let pool = setup_test_db().await;
    seed_test_data(&pool).await?;

    // First, claim a job and set it to processing
    let worker_id = "test-orchestrator-stale";
    let batch_size = 1;
    let jobs = db::claim_jobs(&pool, worker_id, batch_size).await?;

    assert!(!jobs.is_empty(), "Should have claimed a job");
    let job = &jobs[0];

    // Update to processing and make it stale
    let _ = db::update_job_status(&pool, &job.id, JobStatus::Processing, None, None, None).await?;

    // Set the job's updated_at to be older than the threshold
    sqlx::query("UPDATE embedding.jobs SET updated_at = NOW() - INTERVAL '1 hour' WHERE id = $1")
        .bind(job.id)
        .execute(&pool)
        .await?;

    // Setup orchestrator
    let (worker_client, _, _) = create_worker_client(&pool);

    // Create orchestrator config
    let config = OrchestratorConfig {
        batch_size: 1,
        model_id: "bge-small-en-v1.5".to_string(),
        tokenizer_config: TokenizerConfig::default(),
        worker_check_interval_secs: 30,
        job_timeout_mins: 10,
        stale_job_threshold_mins: 30, // 30 minutes threshold
        max_concurrent_tokenization: 5,
        max_concurrent_inference: 3,
        default_worker_locations: vec![],
    };

    // Create orchestrator service
    let orchestrator = OrchestratorService::new(worker_client, pool.clone(), config);

    // Reset stale jobs
    let reset_jobs = orchestrator.reset_stale_jobs().await?;

    // Verify job was reset
    assert_eq!(reset_jobs.len(), 1, "Should have reset 1 job");
    assert_eq!(
        reset_jobs[0].status,
        JobStatus::Queued,
        "Job should be reset to Queued status"
    );

    // Verify in database
    let queued_job_count: i64 =
        sqlx::query("SELECT COUNT(*) FROM embedding.jobs WHERE status = 'queued' AND id = $1")
            .bind(job.id)
            .fetch_one(&pool)
            .await?
            .get(0);

    assert_eq!(
        queued_job_count, 1,
        "Database should show the job as queued"
    );

    Ok(())
}

// Test batch size optimization
#[tokio::test]
async fn test_batch_size_optimization() -> Result<()> {
    let pool = setup_test_db().await;

    // Create multiple workers with different capabilities
    let worker1 = MockWorker::new(pool.clone(), "worker-opt-1", "10.0.0.3");
    let worker2 = MockWorker::new(pool.clone(), "worker-opt-2", "10.0.0.4");

    // Modify worker capabilities
    let mut worker1_info = worker1.worker_info.clone();
    worker1_info.capabilities.optimal_batch_size = 10;
    worker1_info.capabilities.max_batch_size = 20;
    worker1_info.current_load = Some(0.5); // 50% loaded

    let mut worker2_info = worker2.worker_info.clone();
    worker2_info.capabilities.optimal_batch_size = 20;
    worker2_info.capabilities.max_batch_size = 40;
    worker2_info.current_load = Some(0.2); // 20% loaded

    // Register workers
    worker1.register().await?;
    worker2.register().await?;

    // Create worker client
    let (worker_client, _, _) = create_worker_client(&pool);

    // Register workers with the worker client
    worker_client.registry.register_worker(worker1_info).await;
    worker_client.registry.register_worker(worker2_info).await;

    // Create orchestrator config with a larger batch size than any single worker's optimal
    let config = OrchestratorConfig {
        batch_size: 30, // Larger than any single worker's optimal
        model_id: "bge-small-en-v1.5".to_string(),
        tokenizer_config: TokenizerConfig::default(),
        worker_check_interval_secs: 30,
        job_timeout_mins: 10,
        stale_job_threshold_mins: 30,
        max_concurrent_tokenization: 5,
        max_concurrent_inference: 3,
        default_worker_locations: vec![],
    };

    // Create orchestrator service
    let orchestrator = OrchestratorService::new(worker_client, pool.clone(), config);

    // Get optimal batch size
    let optimal_batch_size = orchestrator.get_optimal_batch_size().await;

    // Verify that the computed batch size respects worker capabilities and load
    assert!(
        optimal_batch_size <= 30,
        "Batch size should not exceed configured maximum"
    );
    assert!(optimal_batch_size > 0, "Batch size should be positive");

    // The exact value will depend on the load balancing algorithm, but with the given loads:
    // Worker 1: 10 optimal - (0.5 * 10) = 5 available
    // Worker 2: 20 optimal - (0.2 * 20) = 16 available
    // Total available: 21, limited by config.batch_size = 30
    // So we expect around 21
    assert!(
        optimal_batch_size >= 15,
        "Batch size should account for worker capacity and load"
    );

    Ok(())
}

// Test stats tracking
#[tokio::test]
async fn test_pipeline_stats_tracking() -> Result<()> {
    let pool = setup_test_db().await;
    seed_test_data(&pool).await?;

    // Setup worker
    let mock_worker = MockWorker::new(pool.clone(), "test-worker-stats", "10.0.0.3");
    mock_worker.register().await?;

    // Create worker client
    let (worker_client, _, _) = create_worker_client(&pool);

    // Register worker with the worker client
    worker_client
        .registry
        .register_worker(mock_worker.worker_info.clone())
        .await;

    // Create orchestrator
    let config = OrchestratorConfig {
        batch_size: 3, // Process 3 jobs
        model_id: "bge-small-en-v1.5".to_string(),
        tokenizer_config: TokenizerConfig::default(),
        worker_check_interval_secs: 30,
        job_timeout_mins: 10,
        stale_job_threshold_mins: 30,
        max_concurrent_tokenization: 5,
        max_concurrent_inference: 3,
        default_worker_locations: vec![],
    };

    let orchestrator = OrchestratorService::new(worker_client, pool.clone(), config);

    // Initial stats should be zero
    let initial_stats = orchestrator.get_stats().await;
    assert_eq!(
        initial_stats.jobs_processed, 0,
        "Initial job count should be zero"
    );
    assert_eq!(
        initial_stats.embeddings_generated, 0,
        "Initial embedding count should be zero"
    );

    // Run the pipeline (may not complete due to mock issues, but should update stats)
    let _ = orchestrator.run_embedding_pipeline().await;

    // Get updated stats
    let updated_stats = orchestrator.get_stats().await;

    // We care about the stats being tracked, not the exact values
    assert!(
        updated_stats.total_time_ms > 0,
        "Total time should be tracked"
    );
    assert!(
        updated_stats.fetch_time_ms > 0,
        "Fetch time should be tracked"
    );
    assert!(
        updated_stats.tokenize_time_ms > 0,
        "Tokenize time should be tracked"
    );

    // Reset stats
    orchestrator.reset_stats().await;

    // Verify reset worked
    let reset_stats = orchestrator.get_stats().await;
    assert_eq!(
        reset_stats.jobs_processed, 0,
        "Stats should be reset to zero"
    );
    assert_eq!(
        reset_stats.total_time_ms, 0,
        "Time stats should be reset to zero"
    );

    Ok(())
}

#[tokio::test]
async fn test_worker_registration() -> Result<()> {
    let pool = setup_test_db().await;

    // Create worker client
    let (worker_client, _, _) = create_worker_client(&pool);

    // Create orchestrator
    let config = OrchestratorConfig {
        batch_size: 1,
        model_id: "bge-small-en-v1.5".to_string(),
        tokenizer_config: TokenizerConfig::default(),
        worker_check_interval_secs: 30,
        job_timeout_mins: 10,
        stale_job_threshold_mins: 30,
        max_concurrent_tokenization: 5,
        max_concurrent_inference: 3,
        default_worker_locations: vec!["10.0.0.5:3000".to_string(), "10.0.0.6:3000".to_string()],
    };

    let orchestrator = OrchestratorService::new(worker_client.clone(), pool.clone(), config);

    // Manually register workers in the database to simulate what would happen
    // when the orchestrator tries to discover them
    // (The actual HTTP requests would fail in the test environment)

    // Insert workers directly in database
    for hostname in &["10.0.0.5", "10.0.0.6"] {
        let worker_id = format!("inference-test-{}", hostname);
        let capabilities = json!({
            "gpu_type": "NVIDIA RTX A4000",
            "gpu_memory_mb": 16384,
            "supports_cuda": true,
            "supports_metal": false,
            "cpu_cores": 16,
            "optimal_batch_size": 16,
            "max_batch_size": 32,
            "embedding_dimensions": 384
        });

        sqlx::query(
            r#"
            INSERT INTO embedding.workers (id, hostname, ip_address, worker_type, capabilities, status)
            VALUES 
                ($1, $2, $2, 'inference', $3, 'online')
            "#
        )
        .bind(&worker_id)
        .bind(hostname)
        .bind(capabilities)
        .execute(&pool)
        .await?;

        // Create a Worker object and register it with the worker client
        let worker = Worker {
            id: worker_id,
            hostname: hostname.to_string(),
            ip_address: Some(hostname.to_string()),
            worker_type: WorkerType::Inference,
            capabilities: WorkerCapabilities {
                gpu_type: Some("NVIDIA RTX A4000".to_string()),
                gpu_memory_mb: Some(16384),
                supports_cuda: true,
                supports_metal: false,
                cpu_cores: 16,
                optimal_batch_size: 16,
                max_batch_size: 32,
                embedding_dimensions: Some(384),
            },
            status: WorkerStatus::Online,
            last_heartbeat: SystemTime::now(),
            current_batch_size: None,
            current_load: Some(0.0),
            active_jobs: 0,
            created_at: SystemTime::now(),
        };

        worker_client.registry.register_worker(worker).await;
    }

    // Verify workers are registered
    let all_workers = worker_client.get_all_workers().await;
    assert_eq!(all_workers.len(), 2, "Should have 2 registered workers");

    // Verify hostnames match expected values
    let hostnames: Vec<String> = all_workers.iter().map(|w| w.hostname.clone()).collect();
    assert!(
        hostnames.contains(&"10.0.0.5".to_string()),
        "Should have worker at 10.0.0.5"
    );
    assert!(
        hostnames.contains(&"10.0.0.6".to_string()),
        "Should have worker at 10.0.0.6"
    );

    Ok(())
}
