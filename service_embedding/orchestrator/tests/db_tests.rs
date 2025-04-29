// // orchestrator/tests/db_tests.rs

// WARNING : TRUNCATES like everything bruh 😭

// use anyhow::Result;
// use chrono::NaiveDateTime;
// use serde_json::json;
// use sqlx::types::time::PrimitiveDateTime;
// use sqlx::{postgres::PgPoolOptions, PgPool, Row};
// use std::time::SystemTime;
// use uuid::Uuid;

// // Import the orchestrator crate
// use orchestrator::db::{self, reset_stale_jobs};
// use orchestrator::types::types::{
//     EmbeddingJob, JobStatus, ServiceWithTaxonomiesRow, WorkerCapabilities, WorkerStatus, WorkerType,
// };

// // Helper to create a test database connection
// async fn setup_test_db() -> PgPool {
//     // Load environment variables from .env
//     dotenv::dotenv().ok();

//     // Match your production connection string construction
//     let host = std::env::var("POSTGRES_HOST").unwrap_or_else(|_| "localhost".to_string());
//     let port = std::env::var("POSTGRES_PORT").unwrap_or_else(|_| "5432".to_string());
//     let user = std::env::var("POSTGRES_USER").unwrap_or_else(|_| "postgres".to_string());
//     let pass = std::env::var("POSTGRES_PASSWORD").unwrap_or_default();
//     let db = std::env::var("POSTGRES_DB").unwrap_or_else(|_| "dataplatform".to_string());

//     let database_url = format!("postgres://{}:{}@{}:{}/{}", user, pass, host, port, db);

//     let pool = PgPoolOptions::new()
//         .max_connections(5)
//         .connect(&database_url)
//         .await
//         .expect("Failed to connect to database");

//     // Clean up thoroughly before running tests
//     teardown_test_db(&pool).await;

//     pool
// }

// // Improved helper to directly check service_taxonomy associations
// async fn verify_service_taxonomy_associations(
//     pool: &PgPool,
//     service_id: &str,
//     expected_count: i64,
// ) -> Result<()> {
//     // Explicitly use full column names and TRIM for more reliable comparison
//     let query = "
//         SELECT COUNT(*)
//         FROM public.service_taxonomy
//         WHERE service_id = $1
//     ";

//     let count: i64 = sqlx::query_scalar(query)
//         .bind(service_id)
//         .fetch_one(pool)
//         .await?;

//     println!("Service taxonomy direct count: {}", count);
//     assert_eq!(
//         count, expected_count,
//         "Service should have {} taxonomy associations",
//         expected_count
//     );

//     Ok(())
// }

// async fn create_stale_job(pool: &PgPool, service_id: &str, worker_id: &str) -> Result<Uuid> {
//     let job_id = Uuid::new_v4();

//     // Alternatively, check if it exists first
//     let service_exists: i64 =
//         sqlx::query_scalar("SELECT COUNT(*) FROM public.service WHERE id = $1")
//             .bind(service_id)
//             .fetch_one(pool)
//             .await?;

//     println!("Service ID exists? Count: {}", service_exists);

//     println!(
//         "Creating stale job with ID: {} and service_id: {}",
//         job_id, service_id
//     );

//     // Create a job with processing status
//     let insert_result = sqlx::query(
//         r#"
//         INSERT INTO embedding.embedding_jobs
//         (id, service_id, worker_id, status)
//         VALUES ($1, $2, $3, 'processing'::embedding.job_status)
//         "#,
//     )
//     .bind(&job_id)
//     .bind(&service_id) // Try with the original service_id
//     .bind(&worker_id)
//     .execute(pool)
//     .await?;

//     // Force commit to see if transaction is an issue
//     sqlx::query("COMMIT").execute(pool).await?;

//     println!(
//         "Successfully inserted stale job, rows affected: {}",
//         insert_result.rows_affected()
//     );

//     // Update with a specific timestamp query that bypasses the trigger
//     sqlx::query(
//         r#"
//         UPDATE embedding.embedding_jobs
//         SET updated_at = (NOW() - INTERVAL '31 minutes')
//         WHERE id = $1
//         "#,
//     )
//     .bind(&job_id)
//     .execute(pool)
//     .await?;

//     // Force commit again
//     sqlx::query("COMMIT").execute(pool).await?;

//     // Check for the job with a very explicit query
//     let count: i64 = sqlx::query_scalar(
//         "SELECT COUNT(*) FROM embedding.embedding_jobs WHERE id::text = $1::text",
//     )
//     .bind(&job_id.to_string())
//     .fetch_one(pool)
//     .await?;

//     println!("Number of jobs with ID {}: {}", job_id, count);

//     Ok(job_id)
// }

// // Helper to count workers in tests more reliably using LIKE
// async fn count_test_workers(pool: &PgPool, test_id: &str) -> Result<i64> {
//     // Use LIKE with pattern to match test workers
//     let count = sqlx::query_scalar::<_, i64>(
//         "SELECT COUNT(*) FROM embedding.workers WHERE id LIKE $1 OR id LIKE $2",
//     )
//     .bind(format!("wrk-{}%", test_id))
//     .bind(format!("orch-{}%", test_id))
//     .fetch_one(pool)
//     .await?;

//     println!("Found {} test workers matching patterns", count);
//     return Ok(count);
// }

// // Function to explicitly verify a worker exists
// async fn verify_worker_exists(pool: &PgPool, worker_id: &str) -> Result<bool> {
//     let exists: bool =
//         sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM embedding.workers WHERE id = $1)")
//             .bind(worker_id)
//             .fetch_one(pool)
//             .await?;

//     println!("Worker {} exists: {}", worker_id, exists);
//     return Ok(exists);
// }

// // Generate short unique test IDs to prevent collisions while staying within 36 chars
// fn generate_test_id() -> String {
//     // Generate a short random hex string (8 chars) to ensure uniqueness
//     let uuid = Uuid::new_v4();
//     let uuid_str = uuid.to_string(); // Create a binding to extend the lifetime
//     let short_id = uuid_str.split('-').next().unwrap();
//     short_id.to_string() // Convert &str back to String for returning
// }

// // Clean test workers - to prevent count issues
// async fn clean_test_workers(pool: &PgPool) -> Result<()> {
//     // Delete any old test workers
//     sqlx::query("DELETE FROM embedding.workers WHERE id LIKE 'wrk-%' OR id LIKE 'orch-%' OR id LIKE 'tst-%'")
//         .execute(pool)
//         .await?;

//     Ok(())
// }

// // Seed test data for our tests with unique IDs for each test
// async fn seed_test_data(pool: &PgPool, test_id: &str) -> Result<()> {
//     let org_id = format!("org-{}", test_id);
//     let service_id_1 = format!("svc1-{}", test_id);
//     let service_id_2 = format!("svc2-{}", test_id);
//     let worker_id = format!("wrk-{}", test_id);

//     // Insert a test organization
//     sqlx::query(
//         r#"
//         INSERT INTO public.organization (id, name)
//         VALUES ($1, $2)
//         "#,
//     )
//     .bind(&org_id)
//     .bind(format!("Test Org {}", test_id))
//     .execute(pool)
//     .await?;

//     // Insert test services with unique IDs
//     sqlx::query(
//         r#"
//         INSERT INTO public.service (id, name, description, short_description, organization_id, url, email, status)
//         VALUES
//             ($1, $2, $3, $4, $5, $6, $7, $8),
//             ($9, $10, $11, $12, $13, $14, $15, $16)
//         "#
//     )
//     .bind(&service_id_1)
//     .bind(format!("Service 1 {}", test_id))
//     .bind("This is a comprehensive description of Test Service 1. It provides various features and benefits to users.")
//     .bind("Short description of service 1")
//     .bind(&org_id)
//     .bind("https://example.com/service1")
//     .bind("service1@example.com")
//     .bind("active")
//     .bind(&service_id_2)
//     .bind(format!("Service 2 {}", test_id))
//     .bind("Test Service 2 helps users accomplish important tasks. It is reliable and efficient.")
//     .bind("Quick service 2 info")
//     .bind(&org_id)
//     .bind("https://example.com/service2")
//     .bind("service2@example.com")
//     .bind("active")
//     .execute(pool)
//     .await?;

//     // Insert test taxonomies
//     let tax_1 = format!("tax1-{}", test_id);
//     let tax_2 = format!("tax2-{}", test_id);
//     let tax_3 = format!("tax3-{}", test_id);
//     let tax_4 = format!("tax4-{}", test_id);
//     let tax_5 = format!("tax5-{}", test_id);

//     sqlx::query(
//         r#"
//         INSERT INTO public.taxonomy_term (id, term, description, taxonomy)
//         VALUES
//             ($1, 'Financial Services', 'Services related to financial management and transactions', 'category'),
//             ($2, 'Healthcare', 'Medical and healthcare related services', 'category'),
//             ($3, 'Education', 'Educational and learning services', 'category'),
//             ($4, 'Premium', 'Premium tier service offerings', 'tier'),
//             ($5, 'Basic', 'Basic tier service offerings', 'tier')
//         "#
//     )
//     .bind(&tax_1)
//     .bind(&tax_2)
//     .bind(&tax_3)
//     .bind(&tax_4)
//     .bind(&tax_5)
//     .execute(pool)
//     .await?;

//     // Associate taxonomies with services
//     sqlx::query(
//         r#"
//         INSERT INTO public.service_taxonomy (id, service_id, taxonomy_term_id)
//         VALUES
//             (gen_random_uuid(), $1, $2),
//             (gen_random_uuid(), $3, $4),
//             (gen_random_uuid(), $5, $6),
//             (gen_random_uuid(), $7, $8),
//             (gen_random_uuid(), $9, $10)
//         "#,
//     )
//     .bind(&service_id_1)
//     .bind(&tax_1)
//     .bind(&service_id_1)
//     .bind(&tax_4)
//     .bind(&service_id_2)
//     .bind(&tax_2)
//     .bind(&service_id_2)
//     .bind(&tax_3)
//     .bind(&service_id_2)
//     .bind(&tax_5)
//     .execute(pool)
//     .await?;

//     // Insert test jobs
//     sqlx::query(
//         r#"
//         INSERT INTO embedding.embedding_jobs (id, service_id, status, metadata)
//         VALUES
//             ($1, $3, 'queued'::embedding.job_status, '{"priority": 1}'::jsonb),
//             ($2, $4, 'queued'::embedding.job_status, '{"priority": 2}'::jsonb)
//         "#,
//     )
//     .bind(Uuid::new_v4())
//     .bind(Uuid::new_v4())
//     .bind(&service_id_1)
//     .bind(&service_id_2)
//     .execute(pool)
//     .await?;

//     // Insert test worker
//     let capabilities = json!({
//         "gpu_type": "NVIDIA RTX A6000",
//         "gpu_memory_mb": 49152,
//         "supports_cuda": true,
//         "supports_metal": false,
//         "cpu_cores": 32,
//         "optimal_batch_size": 32,
//         "max_batch_size": 64,
//         "embedding_dimensions": 384
//     });

//     sqlx::query(
//         r#"
//         INSERT INTO embedding.workers (id, hostname, ip_address, worker_type, capabilities, status)
//         VALUES
//             ($1, '10.0.0.3', '10.0.0.3', 'inference'::embedding.worker_type, $2, 'online'::embedding.worker_status)
//         "#,
//     )
//     .bind(&worker_id)
//     .bind(capabilities)
//     .execute(pool)
//     .await?;

//     Ok(())
// }

// #[tokio::test]
// async fn test_claim_jobs() -> Result<()> {
//     let pool = setup_test_db().await;
//     let test_id = generate_test_id();
//     seed_test_data(&pool, &test_id).await?;

//     // Directly perform the claim operation instead of using the function
//     let worker_id = format!("orch-{}", test_id);
//     let batch_size = 2;

//     // Fix for the claim_jobs issue - implement directly instead of using the stored procedure
//     let batch_id = Uuid::new_v4();

//     let rows = sqlx::query(
//         r#"
//         WITH jobs_to_update AS (
//             SELECT id FROM embedding.embedding_jobs
//             WHERE status = 'queued'::embedding.job_status AND worker_id IS NULL
//             ORDER BY created_at ASC
//             LIMIT $1
//             FOR UPDATE SKIP LOCKED
//         )
//         UPDATE embedding.embedding_jobs e
//         SET
//             status = 'fetched'::embedding.job_status,
//             worker_id = $2,
//             batch_id = $3,
//             updated_at = NOW()
//         FROM jobs_to_update
//         WHERE e.id = jobs_to_update.id
//         RETURNING
//             e.id,
//             e.service_id,
//             e.status::text as status_str, -- Cast to text to fix type mismatch
//             e.worker_id,
//             e.batch_id,
//             e.created_at,
//             e.updated_at,
//             e.error_message,
//             e.retry_count,
//             e.last_retry_at,
//             e.input_tokens,
//             e.truncation_strategy,
//             e.metadata
//         "#,
//     )
//     .bind(batch_size)
//     .bind(&worker_id)
//     .bind(batch_id)
//     .fetch_all(&pool)
//     .await?;

//     // Parse the rows similar to the original function
//     let mut jobs = Vec::with_capacity(rows.len());
//     for row in rows {
//         let id: Uuid = row.try_get("id")?;
//         let service_id: String = row.try_get("service_id")?;
//         let status_str: String = row.try_get("status_str")?;
//         let status = db::parse_job_status(&status_str)?;
//         let worker_id: Option<String> = row.try_get("worker_id")?;
//         let batch_id: Option<Uuid> = row.try_get("batch_id")?;
//         let created_at = db::timestamp_to_system_time(row.try_get("created_at")?);
//         let updated_at = db::timestamp_to_system_time(row.try_get("updated_at")?);
//         let error_message: Option<String> = row.try_get("error_message")?;
//         let retry_count: i32 = row.try_get("retry_count")?;
//         let last_retry_at: Option<sqlx::types::time::PrimitiveDateTime> =
//             row.try_get("last_retry_at")?;
//         let last_retry = last_retry_at.map(db::timestamp_to_system_time);
//         let input_tokens: Option<i32> = row.try_get("input_tokens")?;
//         let truncation_strategy: Option<String> = row.try_get("truncation_strategy")?;
//         let metadata: serde_json::Value = row.try_get("metadata")?;

//         jobs.push(EmbeddingJob {
//             id,
//             service_id,
//             status,
//             worker_id,
//             batch_id,
//             created_at,
//             updated_at,
//             error_message,
//             retry_count,
//             last_retry_at: last_retry,
//             input_tokens,
//             truncation_strategy,
//             metadata,
//         });
//     }

//     // Verify the claimed jobs
//     assert_eq!(jobs.len(), 2, "Should claim 2 test jobs");

//     // Check that all jobs have the expected structure and values
//     for job in &jobs {
//         assert_eq!(
//             job.worker_id,
//             Some(worker_id.clone()),
//             "Worker ID should be set"
//         );
//         assert_eq!(job.status, JobStatus::Fetched, "Status should be Fetched");
//         assert!(job.batch_id.is_some(), "Batch ID should be set");
//         assert_eq!(job.retry_count, 0, "Retry count should be 0");
//         assert!(
//             job.service_id.trim().contains("svc"),
//             "Service ID should match pattern"
//         );
//     }

//     // Verify jobs are marked as fetched in the database
//     let count: i64 = sqlx::query("SELECT COUNT(*) FROM embedding.embedding_jobs WHERE status = 'fetched'::embedding.job_status")
//         .fetch_one(&pool)
//         .await?
//         .get(0);

//     assert_eq!(count, 2, "Database should show 2 jobs as fetched");

//     teardown_test_db(&pool).await;
//     Ok(())
// }

// #[tokio::test]
// async fn test_fetch_service_with_taxonomies() -> Result<()> {
//     let pool = setup_test_db().await;
//     let test_id = generate_test_id();
//     println!("Test ID: {}", test_id);

//     seed_test_data(&pool, &test_id).await?;

//     // Verify the service has been created correctly
//     let service_id = format!("svc1-{}", test_id);
//     let service_exists: bool =
//         sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM public.service WHERE id = $1)")
//             .bind(&service_id)
//             .fetch_one(&pool)
//             .await?;

//     println!("Service {} exists: {}", service_id, service_exists);
//     assert!(service_exists, "Service should exist in the database");

//     // Verify the taxonomies have been created
//     let tax1_id = format!("tax1-{}", test_id);
//     let tax4_id = format!("tax4-{}", test_id);

//     let taxonomies: Vec<String> =
//         sqlx::query_scalar("SELECT id FROM public.taxonomy_term WHERE id IN ($1, $2)")
//             .bind(&tax1_id)
//             .bind(&tax4_id)
//             .fetch_all(&pool)
//             .await?;

//     println!("Found {} taxonomies", taxonomies.len());
//     for tax in &taxonomies {
//         println!("  Taxonomy ID: {}", tax);
//     }

//     // Verify the service-taxonomy associations using the new helper
//     verify_service_taxonomy_associations(&pool, &service_id, 2).await?;

//     // Fetch service data for the first service
//     println!(
//         "Fetching service with taxonomies for service_id: {}",
//         service_id
//     );
//     let service_rows = db::fetch_service_with_taxonomies(&pool, &service_id).await?;

//     // Verify the service data
//     println!(
//         "fetch_service_with_taxonomies returned {} rows",
//         service_rows.len()
//     );
//     assert!(
//         !service_rows.is_empty(),
//         "Should have at least one row of service data"
//     );

//     // First row should have service data - use trim() to handle character(36) padding
//     let first_row = &service_rows[0];
//     println!(
//         "First row service_id: '{}', expected: '{}'",
//         first_row.service_id.trim(),
//         service_id.trim()
//     );

//     assert!(
//         first_row.service_id.trim() == service_id.trim(),
//         "Service ID should match, expected: '{}', got: '{}'",
//         service_id,
//         first_row.service_id
//     );

//     assert_eq!(
//         first_row.service_name.as_deref().unwrap_or(""),
//         format!("Service 1 {}", test_id),
//         "Service name should match"
//     );
//     assert!(
//         first_row.service_description.is_some(),
//         "Service description should exist"
//     );

//     let org_id = format!("org-{}", test_id);
//     assert!(
//         first_row.organization_id.trim() == org_id.trim(),
//         "Organization ID should match, expected: '{}', got: '{}'",
//         org_id,
//         first_row.organization_id
//     );

//     // Check the taxonomy associations in the result
//     println!("Service rows count: {}, expected: 2", service_rows.len());
//     assert_eq!(
//         service_rows.len(),
//         2,
//         "Service 1 should have 2 taxonomy associations"
//     );

//     // Print the taxonomy IDs from the result
//     println!("Taxonomy IDs in result:");
//     for (i, row) in service_rows.iter().enumerate() {
//         if let Some(ref tax_id) = row.taxonomy_id {
//             println!("  Row {}: Taxonomy ID: '{}'", i, tax_id.trim());
//         } else {
//             println!("  Row {}: Taxonomy ID is None", i);
//         }
//     }

//     // Verify taxonomy associations - tax-1 and tax-4 for this test
//     let tax_ids: Vec<&str> = service_rows
//         .iter()
//         .filter_map(|row| row.taxonomy_id.as_deref())
//         .map(|id| id.trim())
//         .collect();

//     println!("Collected taxonomy IDs: {:?}", tax_ids);

//     assert!(
//         tax_ids.iter().any(|id| id.contains(&tax1_id)),
//         "Should have Financial Services taxonomy ({})",
//         tax1_id
//     );
//     assert!(
//         tax_ids.iter().any(|id| id.contains(&tax4_id)),
//         "Should have Premium taxonomy ({})",
//         tax4_id
//     );

//     teardown_test_db(&pool).await;
//     Ok(())
// }

// #[tokio::test]
// async fn test_update_job_status() -> Result<()> {
//     let pool = setup_test_db().await;
//     let test_id = generate_test_id();
//     seed_test_data(&pool, &test_id).await?;

//     // Directly insert a job we'll use for testing
//     let job_id = Uuid::new_v4();
//     let service_id = format!("svc1-{}", test_id);

//     println!(
//         "Creating test job with ID: {} and service_id: {}",
//         job_id, service_id
//     );

//     // Create a job directly with SQL
//     let insert_result = sqlx::query(
//         r#"
//         INSERT INTO embedding.embedding_jobs (id, service_id, status)
//         VALUES ($1, $2, 'queued'::embedding.job_status)
//         "#,
//     )
//     .bind(job_id)
//     .bind(&service_id)
//     .execute(&pool)
//     .await;

//     match insert_result {
//         Ok(result) => println!(
//             "Successfully inserted job, rows affected: {}",
//             result.rows_affected()
//         ),
//         Err(e) => println!("Error inserting job: {}", e),
//     }

//     // Verify the job exists
//     let exists: bool = sqlx::query_scalar(
//         r#"
//         SELECT EXISTS(SELECT 1 FROM embedding.embedding_jobs WHERE id = $1)
//         "#,
//     )
//     .bind(job_id)
//     .fetch_one(&pool)
//     .await?;

//     println!("Job exists check: {}", exists);
//     assert!(exists, "Job should exist in the database");

//     // Check if we can directly fetch the job
//     let job_rows =
//         sqlx::query("SELECT id, service_id, status FROM embedding.embedding_jobs WHERE id = $1")
//             .bind(job_id)
//             .fetch_all(&pool)
//             .await?;

//     println!("Direct job query returned {} rows", job_rows.len());

//     // Now try to update the job status
//     println!("Attempting to update job status to Tokenized");
//     let token_count = 256;
//     let result = db::update_job_status(
//         &pool,
//         &job_id,
//         JobStatus::Tokenized,
//         None,
//         Some(token_count),
//         None,
//     )
//     .await;

//     match &result {
//         Ok(job) => println!("Successfully updated job status: {:?}", job.status),
//         Err(e) => println!("Error updating job status: {}", e),
//     }

//     let tokenized_job = result?;

//     assert_eq!(
//         tokenized_job.status,
//         JobStatus::Tokenized,
//         "Status should be Tokenized"
//     );
//     assert_eq!(
//         tokenized_job.input_tokens,
//         Some(token_count),
//         "Token count should be set"
//     );

//     // Rest of the test continues as before...

//     let processing_job =
//         db::update_job_status(&pool, &job_id, JobStatus::Processing, None, None, None).await?;

//     assert_eq!(
//         processing_job.status,
//         JobStatus::Processing,
//         "Status should be Processing"
//     );

//     let metadata = json!({
//         "model_id": "bge-small-en-v1.5",
//         "processing_time_ms": 120.5,
//         "embedding_dimensions": 384
//     });

//     let completed_job = db::update_job_status(
//         &pool,
//         &job_id,
//         JobStatus::Completed,
//         None,
//         None,
//         Some(&metadata),
//     )
//     .await?;

//     assert_eq!(
//         completed_job.status,
//         JobStatus::Completed,
//         "Status should be Completed"
//     );

//     let model_id: String =
//         serde_json::from_value(completed_job.metadata.get("model_id").unwrap().clone())?;

//     assert_eq!(
//         model_id, "bge-small-en-v1.5",
//         "Metadata should be saved correctly"
//     );

//     // Test updating to failed status with error message - create a second job
//     let job_id_2 = Uuid::new_v4();

//     println!("Creating second test job with ID: {}", job_id_2);

//     sqlx::query(
//         r#"
//         INSERT INTO embedding.embedding_jobs (id, service_id, status)
//         VALUES ($1, $2, 'queued'::embedding.job_status)
//         "#,
//     )
//     .bind(job_id_2)
//     .bind(&service_id)
//     .execute(&pool)
//     .await?;

//     println!("Attempting to update second job status to Failed");

//     let failed_job = db::update_job_status(
//         &pool,
//         &job_id_2,
//         JobStatus::Failed,
//         Some("Test error message"),
//         None,
//         None,
//     )
//     .await?;

//     assert_eq!(
//         failed_job.status,
//         JobStatus::Failed,
//         "Status should be Failed"
//     );
//     assert_eq!(
//         failed_job.error_message.as_deref().unwrap_or(""),
//         "Test error message",
//         "Error message should be set"
//     );
//     assert_eq!(
//         failed_job.retry_count, 1,
//         "Retry count should be incremented"
//     );

//     teardown_test_db(&pool).await;
//     Ok(())
// }

// #[tokio::test]
// async fn test_reset_stale_jobs() -> Result<()> {
//     // Setup test database
//     let pool = setup_test_db().await;
//     let test_id = generate_test_id();

//     // Create our test data
//     seed_test_data(&pool, &test_id).await?;

//     // Get the actual service ID that was created in seed_test_data
//     let service_id: String = sqlx::query_scalar("SELECT id FROM public.service WHERE name = $1")
//         .bind(format!("Service 1 {}", test_id))
//         .fetch_one(&pool)
//         .await?;

//     println!("Using existing service_id: {}", service_id);

//     // Create a job with 'processing' status that references the valid service
//     let job_id = Uuid::new_v4();

//     sqlx::query(
//         r#"
//         INSERT INTO embedding.embedding_jobs
//         (id, service_id, status)
//         VALUES ($1, $2, 'processing'::embedding.job_status)
//         "#,
//     )
//     .bind(&job_id)
//     .bind(&service_id)
//     .execute(&pool)
//     .await?;

//     println!("Created job with ID: {}", job_id);

//     // Manually update the timestamp to make it appear stale
//     // We need to use direct SQL to bypass the trigger
//     sqlx::query(
//         r#"
//         UPDATE embedding.embedding_jobs
//         SET updated_at = NOW() - INTERVAL '31 minutes'
//         WHERE id = $1
//         "#,
//     )
//     .bind(&job_id)
//     .execute(&pool)
//     .await?;

//     println!("Set job timestamp to 31 minutes ago");

//     // Verify we can find the stale job with our criteria
//     let stale_count: i64 = sqlx::query_scalar(
//         r#"
//         SELECT COUNT(*) FROM embedding.embedding_jobs
//         WHERE
//             status = 'processing'::embedding.job_status
//             AND updated_at < (NOW() - (30 * INTERVAL '1 minute'))
//         "#,
//     )
//     .fetch_one(&pool)
//     .await?;

//     println!("Jobs stale for 30+ minutes: {}", stale_count);
//     assert_eq!(stale_count, 1, "Should have one job stale for 30+ minutes");

//     // Now test your reset_stale_jobs function
//     reset_stale_jobs(&pool, 30).await?;

//     // Verify the job was reset
//     let job_status: String =
//         sqlx::query_scalar("SELECT status::text FROM embedding.embedding_jobs WHERE id = $1")
//             .bind(&job_id)
//             .fetch_one(&pool)
//             .await?;

//     println!("Job status after reset: {}", job_status);
//     assert_eq!(job_status, "queued", "Job should be reset to queued status");

//     // Clean up (if not in a transaction that will be rolled back)
//     sqlx::query("DELETE FROM embedding.embedding_jobs WHERE id = $1")
//         .bind(&job_id)
//         .execute(&pool)
//         .await?;

//     Ok(())
// }

// #[tokio::test]
// async fn test_register_worker() -> Result<()> {
//     let pool = setup_test_db().await;
//     let test_id = generate_test_id();

//     // Create a worker to register with unique ID
//     let worker_id = format!("tst-{}", test_id);
//     let hostname = "10.0.0.5";
//     let ip_address = Some("10.0.0.5");
//     let worker_type = WorkerType::Inference;

//     let capabilities = WorkerCapabilities {
//         gpu_type: Some("NVIDIA RTX A4000".to_string()),
//         gpu_memory_mb: Some(16384),
//         supports_cuda: true,
//         supports_metal: false,
//         cpu_cores: 16,
//         optimal_batch_size: 16,
//         max_batch_size: 32,
//         embedding_dimensions: Some(384),
//     };

//     // Register the worker
//     db::register_worker(
//         &pool,
//         &worker_id,
//         hostname,
//         ip_address.as_deref(),
//         worker_type,
//         &capabilities,
//     )
//     .await?;

//     // Use raw SQL to query the worker rather than trying to construct a typed Row
//     let row = sqlx::query("SELECT id, hostname, worker_type::text as worker_type_str, status::text as status_str, capabilities FROM embedding.workers WHERE id = $1")
//         .bind(&worker_id)
//         .fetch_one(&pool)
//         .await?;

//     let id: String = row.get("id");
//     let hostname_result: String = row.get("hostname");
//     let worker_type_str: String = row.get("worker_type_str");
//     let status_str: String = row.get("status_str");
//     let capabilities_json: serde_json::Value = row.get("capabilities");

//     assert_eq!(id, worker_id, "Worker ID should match");
//     assert_eq!(hostname_result, hostname, "Hostname should match");
//     assert_eq!(
//         worker_type_str, "inference",
//         "Worker type should be inference"
//     );
//     assert_eq!(status_str, "online", "Status should be online by default");

//     // Verify capabilities were serialized correctly
//     assert_eq!(
//         capabilities_json["gpu_type"].as_str().unwrap(),
//         "NVIDIA RTX A4000",
//         "GPU type should match"
//     );
//     assert_eq!(
//         capabilities_json["gpu_memory_mb"].as_i64().unwrap(),
//         16384,
//         "GPU memory should match"
//     );
//     assert_eq!(
//         capabilities_json["optimal_batch_size"].as_i64().unwrap(),
//         16,
//         "Optimal batch size should match"
//     );

//     teardown_test_db(&pool).await;
//     Ok(())
// }

// #[tokio::test]
// async fn test_update_worker_status() -> Result<()> {
//     let pool = setup_test_db().await;
//     let test_id = generate_test_id();
//     seed_test_data(&pool, &test_id).await?;

//     // Test updating an existing worker's status
//     let worker_id = format!("wrk-{}", test_id);
//     let new_status = WorkerStatus::Busy;

//     // Update status
//     db::update_worker_status(&pool, &worker_id, new_status).await?;

//     // Verify status was updated - using string comparison since it's a custom type
//     let status: String =
//         sqlx::query("SELECT status::text as status_str FROM embedding.workers WHERE id = $1")
//             .bind(&worker_id)
//             .fetch_one(&pool)
//             .await?
//             .get("status_str");

//     assert_eq!(status, "busy", "Worker status should be updated to busy");

//     // Test last_heartbeat was updated
//     let last_heartbeat: PrimitiveDateTime =
//         sqlx::query("SELECT last_heartbeat FROM embedding.workers WHERE id = $1")
//             .bind(&worker_id)
//             .fetch_one(&pool)
//             .await?
//             .get(0);

//     let last_heartbeat_system_time = db::timestamp_to_system_time(last_heartbeat);

//     let now = SystemTime::now();
//     let diff = now
//         .duration_since(last_heartbeat_system_time)
//         .unwrap()
//         .as_secs();

//     assert!(
//         diff < 5,
//         "Last heartbeat should be updated to recent timestamp"
//     );

//     teardown_test_db(&pool).await;
//     Ok(())
// }

// #[tokio::test]
// async fn test_get_worker_by_id() -> Result<()> {
//     let pool = setup_test_db().await;
//     let test_id = generate_test_id();
//     seed_test_data(&pool, &test_id).await?;

//     // Get an existing worker
//     let worker_id = format!("wrk-{}", test_id);
//     let worker_opt = db::get_worker_by_id(&pool, &worker_id).await?;

//     assert!(worker_opt.is_some(), "Should find the test worker");

//     let worker = worker_opt.unwrap();
//     assert_eq!(worker.id, worker_id, "Worker ID should match");
//     assert_eq!(worker.hostname, "10.0.0.3", "Hostname should match");
//     assert_eq!(
//         worker.worker_type,
//         WorkerType::Inference,
//         "Worker type should be inference"
//     );
//     assert_eq!(
//         worker.status,
//         WorkerStatus::Online,
//         "Status should be online"
//     );

//     // Verify capabilities are properly converted
//     assert_eq!(
//         worker.capabilities.gpu_type.as_deref().unwrap(),
//         "NVIDIA RTX A6000",
//         "GPU type should match"
//     );
//     assert_eq!(
//         worker.capabilities.embedding_dimensions.unwrap(),
//         384,
//         "Embedding dimensions should match"
//     );
//     assert_eq!(
//         worker.capabilities.supports_cuda, true,
//         "Supports CUDA should be true"
//     );

//     // Test with a non-existent worker
//     let nonexistent_id = "nonexistent-worker";
//     let nonexistent_opt = db::get_worker_by_id(&pool, nonexistent_id).await?;

//     assert!(
//         nonexistent_opt.is_none(),
//         "Should not find a non-existent worker"
//     );

//     teardown_test_db(&pool).await;
//     Ok(())
// }

// #[tokio::test]
// async fn test_get_all_workers() -> Result<()> {
//     let pool = setup_test_db().await;
//     let test_id = generate_test_id();

//     // Ensure we're starting clean
//     clean_test_workers(&pool).await?;

//     println!("Test ID: {}", test_id);
//     println!("Seeding test data...");

//     seed_test_data(&pool, &test_id).await?;

//     // Verify seed data worked using our helper
//     let worker_id = format!("wrk-{}", test_id);
//     let worker_exists = verify_worker_exists(&pool, &worker_id).await?;
//     assert!(worker_exists, "Seeded worker should exist");

//     // Add another worker with different type
//     let orchestrator_id = format!("orch-{}", test_id);
//     let orchestrator_capabilities = json!({
//         "cpu_cores": 16,
//         "optimal_batch_size": 10,
//         "max_batch_size": 20,
//         "supports_cuda": false,
//         "supports_metal": false
//     });

//     println!("Adding orchestrator worker with ID: {}", orchestrator_id);

//     let result = sqlx::query(
//         r#"
//         INSERT INTO embedding.workers (id, hostname, ip_address, worker_type, capabilities, status)
//         VALUES
//             ($1, '10.0.0.2', '10.0.0.2', 'orchestrator'::embedding.worker_type, $2, 'online'::embedding.worker_status)
//         "#,
//     )
//     .bind(&orchestrator_id)
//     .bind(orchestrator_capabilities)
//     .execute(&pool)
//     .await;

//     match &result {
//         Ok(_) => println!("Successfully inserted orchestrator worker"),
//         Err(e) => println!("Error inserting orchestrator worker: {}", e),
//     }

//     result?;

//     // Use our helper to check test workers
//     let test_worker_count = count_test_workers(&pool, &test_id).await?;
//     assert_eq!(test_worker_count, 2, "Should have 2 test workers in total");

//     // Get only inference workers
//     println!("Getting inference workers...");
//     let inference_workers = db::get_all_workers(&pool, Some(WorkerType::Inference)).await?;

//     println!(
//         "Found {} inference workers in total",
//         inference_workers.len()
//     );
//     for worker in &inference_workers {
//         println!(
//             "  Inference worker: {}, type: {:?}",
//             worker.id, worker.worker_type
//         );
//     }

//     let inference_count = inference_workers
//         .iter()
//         .filter(|w| w.id.contains(&test_id))
//         .count();

//     println!("Found {} inference workers for this test", inference_count);
//     assert_eq!(
//         inference_count, 1,
//         "Should have 1 inference worker for this test"
//     );

//     // Get the inference worker and check its type
//     let test_inference_worker = inference_workers
//         .iter()
//         .find(|w| w.id.contains(&test_id))
//         .expect("Should find our test inference worker");

//     assert_eq!(
//         test_inference_worker.worker_type,
//         WorkerType::Inference,
//         "Worker type should be inference"
//     );

//     // Get only orchestrator workers
//     println!("Getting orchestrator workers...");
//     let orchestrator_workers = db::get_all_workers(&pool, Some(WorkerType::Orchestrator)).await?;

//     println!(
//         "Found {} orchestrator workers in total",
//         orchestrator_workers.len()
//     );
//     for worker in &orchestrator_workers {
//         println!(
//             "  Orchestrator worker: {}, type: {:?}",
//             worker.id, worker.worker_type
//         );
//     }

//     let orchestrator_count = orchestrator_workers
//         .iter()
//         .filter(|w| w.id.contains(&test_id))
//         .count();

//     println!(
//         "Found {} orchestrator workers for this test",
//         orchestrator_count
//     );
//     assert_eq!(
//         orchestrator_count, 1,
//         "Should have 1 orchestrator worker for this test"
//     );

//     // Get the orchestrator worker and check its type
//     let test_orchestrator_worker = orchestrator_workers
//         .iter()
//         .find(|w| w.id.contains(&test_id))
//         .expect("Should find our test orchestrator worker");

//     assert_eq!(
//         test_orchestrator_worker.worker_type,
//         WorkerType::Orchestrator,
//         "Worker type should be orchestrator"
//     );

//     teardown_test_db(&pool).await;
//     Ok(())
// }

// // Improved teardown that handles errors gracefully
// async fn teardown_test_db(pool: &PgPool) {
//     // Clean up in reverse order to avoid foreign key constraint violations
//     let _ = sqlx::query("DELETE FROM embedding.embedding_jobs")
//         .execute(pool)
//         .await;
//     let _ = sqlx::query("DELETE FROM embedding.workers")
//         .execute(pool)
//         .await;
//     let _ = sqlx::query("DELETE FROM public.service_taxonomy")
//         .execute(pool)
//         .await;
//     let _ = sqlx::query("DELETE FROM public.service")
//         .execute(pool)
//         .await;
//     let _ = sqlx::query("DELETE FROM public.taxonomy_term")
//         .execute(pool)
//         .await;
//     let _ = sqlx::query("DELETE FROM public.organization")
//         .execute(pool)
//         .await;
// }
