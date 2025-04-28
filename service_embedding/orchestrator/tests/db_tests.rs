// orchestrator/tests/db_tests.rs

use anyhow::Result;
use serde_json::json;
use sqlx::types::time::PrimitiveDateTime;
use sqlx::{postgres::PgPoolOptions, PgPool, Row};
use std::time::SystemTime;
use uuid::Uuid;

// Import the orchestrator crate
use orchestrator::db;
use orchestrator::types::types::{
    EmbeddingJob, JobStatus, ServiceWithTaxonomiesRow, WorkerCapabilities, WorkerStatus, WorkerType,
};

// Helper to create a test database connection
async fn setup_test_db() -> PgPool {
    // Load environment variables from .env
    dotenv::dotenv().ok();
    
    // Match your production connection string construction
    let host = std::env::var("POSTGRES_HOST").unwrap_or_else(|_| "localhost".to_string());
    let port = std::env::var("POSTGRES_PORT").unwrap_or_else(|_| "5432".to_string());
    let user = std::env::var("POSTGRES_USER").unwrap_or_else(|_| "postgres".to_string());
    let pass = std::env::var("POSTGRES_PASSWORD").unwrap_or_default();
    let db = std::env::var("POSTGRES_DB").unwrap_or_else(|_| "dataplatform".to_string());

    let database_url = format!("postgres://{}:{}@{}:{}/{}", user, pass, host, port, db);

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await
        .expect("Failed to connect to database");

    // Only clean existing test data - don't recreate schema
    sqlx::query(
        r#"
        TRUNCATE TABLE 
            embedding.embedding_jobs,
            embedding.workers,
            public.service,
            public.taxonomy_term,
            public.service_taxonomy
        RESTART IDENTITY CASCADE
        "#
    )
    .execute(&pool)
    .await
    .expect("Failed to truncate test tables");

    pool
}

// Seed test data for our tests
async fn seed_test_data(pool: &PgPool) -> Result<()> {
    sqlx::query(
        r#"
        INSERT INTO public.organization (id, name)
        VALUES ('org-123', 'Test Organization')
        "#,
    )
    .execute(pool)
    .await?;
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
        INSERT INTO public.taxonomy_term (id, term, description, category)
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
    sqlx::query(
        r#"
        INSERT INTO embedding.embedding_jobs (id, service_id, status, metadata)
        VALUES 
            ($1, 'service-test-1', 'queued', '{"priority": 1}'::jsonb),
            ($2, 'service-test-2', 'queued', '{"priority": 2}'::jsonb)
        "#,
    )
    .bind(Uuid::new_v4())
    .bind(Uuid::new_v4())
    .execute(pool)
    .await?;

    // Insert test worker
    let capabilities = json!({
        "gpu_type": "NVIDIA RTX A6000",
        "gpu_memory_mb": 49152,
        "supports_cuda": true,
        "supports_metal": false,
        "cpu_cores": 32,
        "optimal_batch_size": 32,
        "max_batch_size": 64,
        "embedding_dimensions": 384
    });

    sqlx::query(
        r#"
        INSERT INTO embedding.workers (id, hostname, ip_address, worker_type, capabilities, status)
        VALUES 
            ('worker-test-1', '10.0.0.3', '10.0.0.3', 'inference', $1, 'online')
        "#,
    )
    .bind(capabilities)
    .execute(pool)
    .await?;

    Ok(())
}

#[tokio::test]
async fn test_claim_jobs() -> Result<()> {
    let pool = setup_test_db().await;
    seed_test_data(&pool).await?;

    // Claim the test jobs
    let worker_id = "test-orchestrator-1";
    let batch_size = 2;
    let jobs = db::claim_jobs(&pool, worker_id, batch_size).await?;

    // Verify the claimed jobs
    assert_eq!(jobs.len(), 2, "Should claim 2 test jobs");

    // Check that all jobs have the expected structure and values
    for job in &jobs {
        assert_eq!(
            job.worker_id,
            Some(worker_id.to_string()),
            "Worker ID should be set"
        );
        assert_eq!(job.status, JobStatus::Fetched, "Status should be Fetched");
        assert!(job.batch_id.is_some(), "Batch ID should be set");
        assert_eq!(job.retry_count, 0, "Retry count should be 0");
        assert!(
            job.service_id.starts_with("service-test-"),
            "Service ID should match pattern"
        );
    }

    // Verify jobs are marked as fetched in the database
    let count: i64 = sqlx::query("SELECT COUNT(*) FROM embedding.embedding_jobs WHERE status = 'fetched'")
        .fetch_one(&pool)
        .await?
        .get(0);

    assert_eq!(count, 2, "Database should show 2 jobs as fetched");

    Ok(())
}

#[tokio::test]
async fn test_fetch_service_with_taxonomies() -> Result<()> {
    let pool = setup_test_db().await;
    seed_test_data(&pool).await?;

    // Fetch service data
    let service_rows = db::fetch_service_with_taxonomies(&pool, "service-test-1").await?;

    // Verify the service data
    assert!(
        !service_rows.is_empty(),
        "Should have at least one row of service data"
    );

    // First row should have service data
    let first_row = &service_rows[0];
    assert_eq!(
        first_row.service_id, "service-test-1",
        "Service ID should match"
    );
    assert_eq!(
        first_row.service_name.as_deref().unwrap_or(""),
        "Test Service 1",
        "Service name should match"
    );
    assert!(
        first_row.service_description.is_some(),
        "Service description should exist"
    );
    assert_eq!(
        first_row.organization_id, "org-123",
        "Organization ID should match"
    );

    // Check if we have the right number of taxonomy associations
    // Service 1 has 2 taxonomies
    assert_eq!(
        service_rows.len(),
        2,
        "Service 1 should have 2 taxonomy associations"
    );

    // Verify taxonomy associations
    let tax_ids: Vec<&str> = service_rows
        .iter()
        .filter_map(|row| row.taxonomy_id.as_deref())
        .collect();

    assert!(
        tax_ids.contains(&"tax-1"),
        "Should have Financial Services taxonomy"
    );
    assert!(tax_ids.contains(&"tax-4"), "Should have Premium taxonomy");

    Ok(())
}

#[tokio::test]
async fn test_update_job_status() -> Result<()> {
    let pool = setup_test_db().await;
    seed_test_data(&pool).await?;

    // First, claim a job to work with
    let worker_id = "test-orchestrator-2";
    let batch_size = 1;
    let jobs = db::claim_jobs(&pool, worker_id, batch_size).await?;

    assert!(!jobs.is_empty(), "Should have claimed at least one job");
    let job = &jobs[0];

    // Test updating to tokenized status
    let token_count = 256;
    let tokenized_job = db::update_job_status(
        &pool,
        &job.id,
        JobStatus::Tokenized,
        None,
        Some(token_count),
        None,
    )
    .await?;

    assert_eq!(
        tokenized_job.status,
        JobStatus::Tokenized,
        "Status should be Tokenized"
    );
    assert_eq!(
        tokenized_job.input_tokens,
        Some(token_count),
        "Token count should be set"
    );

    // Test updating to processing status
    let processing_job =
        db::update_job_status(&pool, &job.id, JobStatus::Processing, None, None, None).await?;

    assert_eq!(
        processing_job.status,
        JobStatus::Processing,
        "Status should be Processing"
    );

    // Test updating to completed status with metadata
    let metadata = json!({
        "model_id": "bge-small-en-v1.5",
        "processing_time_ms": 120.5,
        "embedding_dimensions": 384
    });

    let completed_job = db::update_job_status(
        &pool,
        &job.id,
        JobStatus::Completed,
        None,
        None,
        Some(&metadata),
    )
    .await?;

    assert_eq!(
        completed_job.status,
        JobStatus::Completed,
        "Status should be Completed"
    );

    // Extract the model_id from the metadata to verify it's stored correctly
    let model_id: String =
        serde_json::from_value(completed_job.metadata.get("model_id").unwrap().clone())?;

    assert_eq!(
        model_id, "bge-small-en-v1.5",
        "Metadata should be saved correctly"
    );

    // Test updating to failed status with error message
    let job2 = &jobs[0]; // We'll use the same job for this example

    let failed_job = db::update_job_status(
        &pool,
        &job2.id,
        JobStatus::Failed,
        Some("Test error message"),
        None,
        None,
    )
    .await?;

    assert_eq!(
        failed_job.status,
        JobStatus::Failed,
        "Status should be Failed"
    );
    assert_eq!(
        failed_job.error_message.as_deref().unwrap_or(""),
        "Test error message",
        "Error message should be set"
    );
    assert_eq!(
        failed_job.retry_count, 1,
        "Retry count should be incremented"
    );

    Ok(())
}

#[tokio::test]
async fn test_reset_stale_jobs() -> Result<()> {
    let pool = setup_test_db().await;
    seed_test_data(&pool).await?;

    // First, claim a job and set its status to processing
    let worker_id = "test-orchestrator-3";
    let batch_size = 1;
    let jobs = db::claim_jobs(&pool, worker_id, batch_size).await?;

    assert!(!jobs.is_empty(), "Should have claimed at least one job");
    let job = &jobs[0];

    // Update to processing status
    let _ = db::update_job_status(&pool, &job.id, JobStatus::Processing, None, None, None).await?;

    // Set the job's updated_at to be older than the threshold
    sqlx::query("UPDATE embedding.embedding_jobs SET updated_at = NOW() - INTERVAL '1 hour' WHERE id = $1")
        .bind(job.id)
        .execute(&pool)
        .await?;

    // Now reset stale jobs
    let threshold_mins = 30; // 30 minutes
    let reset_jobs = db::reset_stale_jobs(&pool, threshold_mins).await?;

    assert_eq!(reset_jobs.len(), 1, "Should have reset 1 stale job");
    assert_eq!(
        reset_jobs[0].id, job.id,
        "Reset job should match our test job"
    );
    assert_eq!(
        reset_jobs[0].status,
        JobStatus::Queued,
        "Status should be reset to Queued"
    );
    assert!(
        reset_jobs[0].worker_id.is_none(),
        "Worker ID should be cleared"
    );
    assert!(
        reset_jobs[0]
            .error_message
            .as_deref()
            .unwrap_or("")
            .contains("Reset after stalling"),
        "Error message should mention the reset"
    );

    Ok(())
}

#[tokio::test]
async fn test_register_worker() -> Result<()> {
    let pool = setup_test_db().await;

    // Create a worker to register
    let worker_id = "test-worker-1";
    let hostname = "10.0.0.5";
    let ip_address = Some("10.0.0.5");
    let worker_type = WorkerType::Inference;

    let capabilities = WorkerCapabilities {
        gpu_type: Some("NVIDIA RTX A4000".to_string()),
        gpu_memory_mb: Some(16384),
        supports_cuda: true,
        supports_metal: false,
        cpu_cores: 16,
        optimal_batch_size: 16,
        max_batch_size: 32,
        embedding_dimensions: Some(384),
    };

    // Register the worker
    db::register_worker(
        &pool,
        worker_id,
        hostname,
        ip_address.as_deref(),
        worker_type,
        &capabilities,
    )
    .await?;

    // Verify the worker was registered correctly
    let worker_row = sqlx::query("SELECT * FROM embedding.workers WHERE id = $1")
        .bind(worker_id)
        .fetch_one(&pool)
        .await?;

    assert_eq!(
        worker_row.get::<String, _>("id"),
        worker_id,
        "Worker ID should match"
    );
    assert_eq!(
        worker_row.get::<String, _>("hostname"),
        hostname,
        "Hostname should match"
    );
    assert_eq!(
        worker_row.get::<String, _>("worker_type"),
        "inference",
        "Worker type should be inference"
    );
    assert_eq!(
        worker_row.get::<String, _>("status"),
        "online",
        "Status should be online by default"
    );

    // Verify capabilities were serialized correctly
    let capabilities_json: serde_json::Value = worker_row.get("capabilities");
    assert_eq!(
        capabilities_json["gpu_type"], "NVIDIA RTX A4000",
        "GPU type should match"
    );
    assert_eq!(
        capabilities_json["gpu_memory_mb"], 16384,
        "GPU memory should match"
    );
    assert_eq!(
        capabilities_json["optimal_batch_size"], 16,
        "Optimal batch size should match"
    );

    Ok(())
}

#[tokio::test]
async fn test_update_worker_status() -> Result<()> {
    let pool = setup_test_db().await;
    seed_test_data(&pool).await?;

    // Test updating an existing worker's status
    let worker_id = "worker-test-1"; // From seed data
    let new_status = WorkerStatus::Busy;

    // Update status
    db::update_worker_status(&pool, worker_id, new_status).await?;

    // Verify status was updated
    let status: String = sqlx::query("SELECT status FROM embedding.workers WHERE id = $1")
        .bind(worker_id)
        .fetch_one(&pool)
        .await?
        .get(0);

    assert_eq!(status, "busy", "Worker status should be updated to busy");

    // Test last_heartbeat was updated
    let last_heartbeat: PrimitiveDateTime =
        sqlx::query("SELECT last_heartbeat FROM embedding.workers WHERE id = $1")
            .bind(worker_id)
            .fetch_one(&pool)
            .await?
            .get(0);

    let last_heartbeat_system_time = db::timestamp_to_system_time(last_heartbeat);

    let now = SystemTime::now();
    let diff = now
        .duration_since(last_heartbeat_system_time)
        .unwrap()
        .as_secs();

    assert!(
        diff < 5,
        "Last heartbeat should be updated to recent timestamp"
    );

    Ok(())
}

#[tokio::test]
async fn test_get_worker_by_id() -> Result<()> {
    let pool = setup_test_db().await;
    seed_test_data(&pool).await?;

    // Get an existing worker
    let worker_id = "worker-test-1"; // From seed data
    let worker_opt = db::get_worker_by_id(&pool, worker_id).await?;

    assert!(worker_opt.is_some(), "Should find the test worker");

    let worker = worker_opt.unwrap();
    assert_eq!(worker.id, worker_id, "Worker ID should match");
    assert_eq!(worker.hostname, "10.0.0.3", "Hostname should match");
    assert_eq!(
        worker.worker_type,
        WorkerType::Inference,
        "Worker type should be inference"
    );
    assert_eq!(
        worker.status,
        WorkerStatus::Online,
        "Status should be online"
    );

    // Verify capabilities are properly converted
    assert_eq!(
        worker.capabilities.gpu_type.as_deref().unwrap(),
        "NVIDIA RTX A6000",
        "GPU type should match"
    );
    assert_eq!(
        worker.capabilities.embedding_dimensions.unwrap(),
        384,
        "Embedding dimensions should match"
    );

    // Test with a non-existent worker
    let nonexistent_id = "nonexistent-worker";
    let nonexistent_opt = db::get_worker_by_id(&pool, nonexistent_id).await?;

    assert!(
        nonexistent_opt.is_none(),
        "Should not find a non-existent worker"
    );

    Ok(())
}

#[tokio::test]
async fn test_get_all_workers() -> Result<()> {
    let pool = setup_test_db().await;
    seed_test_data(&pool).await?;

    // Add another worker with different type
    let orchestrator_capabilities = json!({
        "cpu_cores": 16,
        "optimal_batch_size": 10,
        "max_batch_size": 20
    });

    sqlx::query(
        r#"
        INSERT INTO embedding.workers (id, hostname, ip_address, worker_type, capabilities, status)
        VALUES 
            ('orchestrator-1', '10.0.0.2', '10.0.0.2', 'orchestrator', $1, 'online')
        "#,
    )
    .bind(orchestrator_capabilities)
    .execute(&pool)
    .await?;

    // Get all workers
    let all_workers = db::get_all_workers(&pool, None).await?;

    assert_eq!(all_workers.len(), 2, "Should have 2 workers in total");

    // Get only inference workers
    let inference_workers = db::get_all_workers(&pool, Some(WorkerType::Inference)).await?;

    assert_eq!(inference_workers.len(), 1, "Should have 1 inference worker");
    assert_eq!(
        inference_workers[0].worker_type,
        WorkerType::Inference,
        "Worker type should be inference"
    );

    // Get only orchestrator workers
    let orchestrator_workers = db::get_all_workers(&pool, Some(WorkerType::Orchestrator)).await?;

    assert_eq!(
        orchestrator_workers.len(),
        1,
        "Should have 1 orchestrator worker"
    );
    assert_eq!(
        orchestrator_workers[0].worker_type,
        WorkerType::Orchestrator,
        "Worker type should be orchestrator"
    );

    Ok(())
}

// Add this to the bottom of your tests to clean up resources
async fn teardown_test_db(pool: &PgPool) {
    // Not dropping tables to avoid conflicts with other tests running in parallel
    // Just cleaning the data is sufficient
    let _ = sqlx::query("TRUNCATE TABLE embedding.embedding_jobs, embedding.workers, public.service, public.taxonomy_term, public.service_taxonomy RESTART IDENTITY CASCADE")
        .execute(pool)
        .await;
}
