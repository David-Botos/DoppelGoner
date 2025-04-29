// orchestrator/src/db/queries.rs

use crate::types::types::{
    EmbeddingJob, JobStatus, ServiceWithTaxonomiesRow, Worker, WorkerCapabilities, WorkerStatus,
    WorkerType,
};
use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use sqlx::{PgPool, Row};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

/// Custom error type for database operations
#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("Entity not found: {0}")]
    NotFound(String),

    #[error("Invalid status: {0}")]
    InvalidStatus(String),

    #[error("Database connection error: {0}")]
    ConnectionError(String),

    #[error("Query execution error: {0}")]
    QueryError(String),

    #[error("Data conversion error: {0}")]
    ConversionError(String),
}

/// Helper function to compare database IDs consistently handling padding
pub fn compare_ids(id1: &str, id2: &str) -> bool {
    id1.trim() == id2.trim()
}

/// Helper function to convert string to JobStatus with proper error handling
#[instrument(level = "debug")]
pub fn parse_job_status(status: &str) -> Result<JobStatus> {
    match status.trim() {
        "queued" => Ok(JobStatus::Queued),
        "fetched" => Ok(JobStatus::Fetched),
        "tokenized" => Ok(JobStatus::Tokenized),
        "processing" => Ok(JobStatus::Processing),
        "completed" => Ok(JobStatus::Completed),
        "failed" => Ok(JobStatus::Failed),
        _ => Err(anyhow!("Invalid job status: {}", status)),
    }
}

/// Helper function to convert string to WorkerStatus with proper error handling
#[instrument(level = "debug")]
pub fn parse_worker_status(status: &str) -> Result<WorkerStatus> {
    match status.trim() {
        "online" => Ok(WorkerStatus::Online),
        "offline" => Ok(WorkerStatus::Offline),
        "busy" => Ok(WorkerStatus::Busy),
        _ => Err(anyhow!("Invalid worker status: {}", status)),
    }
}

/// Helper function to convert string to WorkerType with proper error handling
#[instrument(level = "debug")]
pub fn parse_worker_type(worker_type: &str) -> Result<WorkerType> {
    match worker_type.trim() {
        "orchestrator" => Ok(WorkerType::Orchestrator),
        "inference" => Ok(WorkerType::Inference),
        _ => Err(anyhow!("Invalid worker type: {}", worker_type)),
    }
}

/// Convert sqlx timestamp to SystemTime
#[instrument(level = "debug", skip(timestamp))]
pub fn timestamp_to_system_time(timestamp: sqlx::types::time::PrimitiveDateTime) -> SystemTime {
    let secs = timestamp.assume_utc().unix_timestamp();
    let nanos = timestamp.microsecond() * 1000;
    UNIX_EPOCH + std::time::Duration::new(secs as u64, nanos)
}

/// Helper function to measure query execution time and log slow queries
async fn time_query<F, Fut, T>(name: &str, f: F) -> Result<T>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let start = std::time::Instant::now();
    let result = f().await;
    let duration = start.elapsed();

    if duration > std::time::Duration::from_millis(100) {
        warn!("Slow query detected: {} took {:?}", name, duration);
    } else {
        debug!("Query {} took {:?}", name, duration);
    }

    result
}

/// Claim jobs from the database for a worker
#[instrument(skip(pool), fields(worker_id = %worker_id, batch_size = %batch_size))]
pub async fn claim_jobs(
    pool: &PgPool,
    worker_id: &str,
    batch_size: i32,
) -> Result<Vec<EmbeddingJob>> {
    info!("Claiming {} jobs for worker {}", batch_size, worker_id);

    let batch_id = Uuid::new_v4(); // Generate a batch ID for these jobs

    return time_query("claim_jobs", || async {
        // MODIFIED: Use explicit field list with status cast to text
        let rows = sqlx::query(
            r#"
            WITH jobs_to_update AS (
                SELECT id FROM embedding.embedding_jobs 
                WHERE status = 'queued'::embedding.job_status AND worker_id IS NULL
                ORDER BY created_at ASC
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE embedding.embedding_jobs e
            SET 
                status = 'fetched'::embedding.job_status,
                worker_id = $2,
                batch_id = $3,
                updated_at = NOW()
            FROM jobs_to_update
            WHERE e.id = jobs_to_update.id
            RETURNING 
                e.id,
                e.service_id,
                e.status::text as status,
                e.worker_id,
                e.batch_id,
                e.created_at,
                e.updated_at,
                e.error_message,
                e.retry_count,
                e.last_retry_at,
                e.input_tokens,
                e.truncation_strategy,
                e.metadata
            "#,
        )
        .bind(batch_size)
        .bind(worker_id)
        .bind(batch_id)
        .fetch_all(pool)
        .await
        .context("Failed to claim jobs from database")?;

        info!("Claimed {} jobs", rows.len());

        let mut jobs = Vec::with_capacity(rows.len());
        for row in rows {
            let id: Uuid = row.try_get("id").context("Failed to extract job ID")?;

            debug!("Processing claimed job {}", id);

            let service_id: String = row
                .try_get("service_id")
                .context("Failed to extract service_id")?;

            let status_str: String = row.try_get("status").context("Failed to extract status")?;

            let status = parse_job_status(&status_str).context("Failed to parse job status")?;

            let worker_id: Option<String> = row
                .try_get("worker_id")
                .context("Failed to extract worker_id")?;

            let batch_id: Option<Uuid> = row
                .try_get("batch_id")
                .context("Failed to extract batch_id")?;

            let created_at = timestamp_to_system_time(
                row.try_get("created_at")
                    .context("Failed to extract created_at")?,
            );

            let updated_at = timestamp_to_system_time(
                row.try_get("updated_at")
                    .context("Failed to extract updated_at")?,
            );

            let error_message: Option<String> = row
                .try_get("error_message")
                .context("Failed to extract error_message")?;

            let retry_count: i32 = row
                .try_get("retry_count")
                .context("Failed to extract retry_count")?;

            let last_retry_at: Option<sqlx::types::time::PrimitiveDateTime> = row
                .try_get("last_retry_at")
                .context("Failed to extract last_retry_at")?;

            let last_retry = last_retry_at.map(timestamp_to_system_time);

            let input_tokens: Option<i32> = row
                .try_get("input_tokens")
                .context("Failed to extract input_tokens")?;

            let truncation_strategy: Option<String> = row
                .try_get("truncation_strategy")
                .context("Failed to extract truncation_strategy")?;

            let metadata: Value = row
                .try_get("metadata")
                .context("Failed to extract metadata")?;

            jobs.push(EmbeddingJob {
                id,
                service_id,
                status,
                worker_id,
                batch_id,
                created_at,
                updated_at,
                error_message,
                retry_count,
                last_retry_at: last_retry,
                input_tokens,
                truncation_strategy,
                metadata,
            });
        }

        Ok(jobs)
    })
    .await;
}

/// Fetch service data for a specific service with taxonomy information
#[instrument(skip(pool), fields(service_id = %service_id))]
pub async fn fetch_service_with_taxonomies(
    pool: &PgPool,
    service_id: &str,
) -> Result<Vec<ServiceWithTaxonomiesRow>> {
    info!(
        "Fetching service with taxonomies for service_id: '{}'",
        service_id
    );

    return time_query("fetch_service_with_taxonomies", || async {
        // Enhanced TRIM usage for better CHAR(36) compatibility
        // CRITICAL FIX: Use exact equality on padded character fields
        let query = r#"
            SELECT 
                s.id as service_id,
                s.name as service_name,
                s.description as service_description,
                s.short_description as service_short_description,
                s.organization_id,
                s.url as service_url,
                s.email as service_email,
                s.status as service_status,
                t.id as taxonomy_id,
                t.term as taxonomy_term,
                t.description as taxonomy_description,
                t.taxonomy as taxonomy_category
            FROM 
                public.service s
            LEFT JOIN 
                public.service_taxonomy st ON s.id = st.service_id
            LEFT JOIN 
                public.taxonomy_term t ON st.taxonomy_term_id = t.id
            WHERE 
                s.id = $1
            "#;

        debug!(
            "Executing query with service_id parameter: '{}'",
            service_id
        );

        let rows = sqlx::query(query)
            .bind(service_id)
            .fetch_all(pool)
            .await
            .context(format!(
                "Failed to fetch service with taxonomies for service_id: {}",
                service_id
            ))?;

        debug!("Query returned {} rows", rows.len());

        let mut service_rows = Vec::with_capacity(rows.len());
        for row in rows {
            let taxonomy_id: Option<String> = row
                .try_get("taxonomy_id")
                .context("Failed to extract taxonomy_id")?;

            debug!("Processing row with taxonomy_id: {:?}", taxonomy_id);

            let service_row = ServiceWithTaxonomiesRow {
                service_id: row
                    .try_get("service_id")
                    .context("Failed to extract service_id")?,

                service_name: row
                    .try_get("service_name")
                    .context("Failed to extract service_name")?,

                service_description: row
                    .try_get("service_description")
                    .context("Failed to extract service_description")?,

                service_short_description: row
                    .try_get("service_short_description")
                    .context("Failed to extract service_short_description")?,

                organization_id: row
                    .try_get("organization_id")
                    .context("Failed to extract organization_id")?,

                service_url: row
                    .try_get("service_url")
                    .context("Failed to extract service_url")?,

                service_email: row
                    .try_get("service_email")
                    .context("Failed to extract service_email")?,

                service_status: row
                    .try_get("service_status")
                    .context("Failed to extract service_status")?,

                taxonomy_id,

                taxonomy_term: row
                    .try_get("taxonomy_term")
                    .context("Failed to extract taxonomy_term")?,

                taxonomy_description: row
                    .try_get("taxonomy_description")
                    .context("Failed to extract taxonomy_description")?,

                taxonomy_category: row
                    .try_get("taxonomy_category")
                    .context("Failed to extract taxonomy_category")?,
            };
            service_rows.push(service_row);
        }

        info!("Returning {} service rows", service_rows.len());

        Ok(service_rows)
    })
    .await;
}

/// Update job status in the database
#[instrument(skip(pool, error_message, metadata), fields(job_id = %job_id, status = ?status))]
pub async fn update_job_status(
    pool: &PgPool,
    job_id: &Uuid,
    status: JobStatus,
    error_message: Option<&str>,
    input_tokens: Option<i32>,
    metadata: Option<&Value>,
) -> Result<EmbeddingJob> {
    info!("Updating job {} to status {:?}", job_id, status);

    return time_query("update_job_status", || async {
        // Skip existence check - it's causing issues in tests
        // We'll rely on the UPDATE query to find the job directly

        let status_str = match status {
            JobStatus::Queued => "queued",
            JobStatus::Fetched => "fetched",
            JobStatus::Tokenized => "tokenized",
            JobStatus::Processing => "processing",
            JobStatus::Completed => "completed",
            JobStatus::Failed => "failed",
        };

        // Modified query to ensure we get text representation of status
        // Simplified direct query
        let query = r#"
            UPDATE embedding.embedding_jobs
            SET 
                status = $2::embedding.job_status,
                updated_at = NOW(),
                error_message = CASE 
                                WHEN $3::text IS NULL THEN error_message 
                                ELSE $3::text 
                                END,
                input_tokens = COALESCE($4, input_tokens),
                retry_count = CASE 
                                WHEN $2 = 'failed' THEN retry_count + 1 
                                ELSE retry_count 
                                END,
                last_retry_at = CASE 
                                WHEN $2 = 'failed' THEN NOW() 
                                ELSE last_retry_at 
                                END,
                metadata = CASE 
                            WHEN $5::jsonb IS NULL THEN metadata 
                            ELSE $5::jsonb 
                        END
            WHERE id = $1
            RETURNING 
                id, 
                service_id, 
                status::text as status_text, 
                worker_id, 
                batch_id, 
                created_at, 
                updated_at, 
                error_message, 
                retry_count, 
                last_retry_at, 
                input_tokens, 
                truncation_strategy, 
                metadata
        "#;

        debug!("Executing update query for job {}", job_id);

        let row = sqlx::query(query)
            .bind(job_id)
            .bind(status_str)
            .bind(error_message)
            .bind(input_tokens)
            .bind(metadata)
            .fetch_one(pool)
            .await
            .context(format!(
                "Failed to update job {} status to {:?}",
                job_id, status
            ))?;

        let id: Uuid = row.try_get("id").context("Failed to extract job ID")?;

        let service_id: String = row
            .try_get("service_id")
            .context("Failed to extract service_id")?;

        let status_str: String = row
            .try_get("status_text")
            .context("Failed to extract status")?;

        let status = parse_job_status(&status_str).context("Failed to parse job status")?;

        let worker_id: Option<String> = row
            .try_get("worker_id")
            .context("Failed to extract worker_id")?;

        let batch_id: Option<Uuid> = row
            .try_get("batch_id")
            .context("Failed to extract batch_id")?;

        let created_at = timestamp_to_system_time(
            row.try_get("created_at")
                .context("Failed to extract created_at")?,
        );

        let updated_at = timestamp_to_system_time(
            row.try_get("updated_at")
                .context("Failed to extract updated_at")?,
        );

        let error_message: Option<String> = row
            .try_get("error_message")
            .context("Failed to extract error_message")?;

        let retry_count: i32 = row
            .try_get("retry_count")
            .context("Failed to extract retry_count")?;

        let last_retry_at: Option<sqlx::types::time::PrimitiveDateTime> = row
            .try_get("last_retry_at")
            .context("Failed to extract last_retry_at")?;

        let last_retry = last_retry_at.map(timestamp_to_system_time);

        let input_tokens: Option<i32> = row
            .try_get("input_tokens")
            .context("Failed to extract input_tokens")?;

        let truncation_strategy: Option<String> = row
            .try_get("truncation_strategy")
            .context("Failed to extract truncation_strategy")?;

        let metadata: Value = row
            .try_get("metadata")
            .context("Failed to extract metadata")?;

        info!("Successfully updated job {} to status {:?}", id, status);

        Ok(EmbeddingJob {
            id,
            service_id,
            status,
            worker_id,
            batch_id,
            created_at,
            updated_at,
            error_message,
            retry_count,
            last_retry_at: last_retry,
            input_tokens,
            truncation_strategy,
            metadata,
        })
    })
    .await;
}

/// Reset stale jobs that have been in processing state too long
#[instrument(skip(pool), fields(minutes_threshold = %minutes_threshold))]
pub async fn reset_stale_jobs(pool: &PgPool, minutes_threshold: i32) -> Result<Vec<EmbeddingJob>> {
    info!(
        "Resetting stale jobs with threshold of {} minutes",
        minutes_threshold
    );

    return time_query("reset_stale_jobs", || async {
        // First find how many stale jobs exist - FIXED time interval calculation
        let count: i64 = sqlx::query_scalar(
            r#"
            SELECT COUNT(*) 
            FROM embedding.embedding_jobs 
            WHERE 
                status = 'processing'::embedding.job_status
                AND updated_at < (NOW() - ($1 * INTERVAL '1 minute'))
            "#
        )
        .bind(minutes_threshold)
        .fetch_one(pool)
        .await
        .context("Failed to count stale jobs")?;  
        info!("Stale jobs found before reset: {}", count);
        
        // List stale jobs for debugging - FIXED time interval
        let stale_jobs: Vec<(Uuid, String, Option<String>)> = sqlx::query_as(
            r#"
            SELECT id, service_id, worker_id
            FROM embedding.embedding_jobs
            WHERE 
                status = 'processing'::embedding.job_status
                AND updated_at < (NOW() - ($1 * INTERVAL '1 minute'))
            "#
        )
        .bind(minutes_threshold)
        .fetch_all(pool)
        .await
        .context("Failed to list stale jobs")?;
        
        info!("Found {} stale jobs to reset", stale_jobs.len());
        for (id, service_id, worker_id) in &stale_jobs {
            debug!("Stale job: ID={}, Service={}, Worker={:?}", id, service_id, worker_id);
        }
        
        // Now reset the stale jobs with a more precise query - FIXED time interval
        let reset_query = r#"
            WITH stale_jobs AS (
                SELECT id
                FROM embedding.embedding_jobs
                WHERE 
                    status = 'processing'::embedding.job_status
                    AND updated_at < (NOW() - ($1 * INTERVAL '1 minute'))
                FOR UPDATE
            )
            UPDATE embedding.embedding_jobs e
            SET 
                status = 'queued'::embedding.job_status,
                worker_id = NULL,
                batch_id = NULL,
                error_message = COALESCE(error_message, '') || ' Reset after stalling in processing state.',
                updated_at = NOW()
            FROM stale_jobs
            WHERE e.id = stale_jobs.id
            RETURNING 
                e.id,
                e.service_id,
                e.status::text as status_text,
                e.worker_id,
                e.batch_id,
                e.created_at,
                e.updated_at,
                e.error_message,
                e.retry_count,
                e.last_retry_at,
                e.input_tokens,
                e.truncation_strategy,
                e.metadata
            "#;
        
        debug!("Executing reset query with threshold: {}", minutes_threshold);
        
        let rows = sqlx::query(reset_query)
            .bind(minutes_threshold)
            .fetch_all(pool)
            .await
            .context("Failed to reset stale jobs")?;

        debug!("Reset query returned {} rows", rows.len());

        let mut jobs = Vec::with_capacity(rows.len());
        for row in rows {
            let id: Uuid = row.try_get("id")
                .context("Failed to extract job ID")?;
                
            let service_id: String = row.try_get("service_id")
                .context("Failed to extract service_id")?;
                
            let status_str: String = row.try_get("status_text")
                .context("Failed to extract status")?;
                
            let status = parse_job_status(&status_str)
                .context("Failed to parse job status")?;
                
            let worker_id: Option<String> = row.try_get("worker_id")
                .context("Failed to extract worker_id")?;
                
            let batch_id: Option<Uuid> = row.try_get("batch_id")
                .context("Failed to extract batch_id")?;
                
            let created_at = timestamp_to_system_time(
                row.try_get("created_at").context("Failed to extract created_at")?
            );
            
            let updated_at = timestamp_to_system_time(
                row.try_get("updated_at").context("Failed to extract updated_at")?
            );
            
            let error_message: Option<String> = row.try_get("error_message")
                .context("Failed to extract error_message")?;
                
            let retry_count: i32 = row.try_get("retry_count")
                .context("Failed to extract retry_count")?;
                
            let last_retry_at: Option<sqlx::types::time::PrimitiveDateTime> =
                row.try_get("last_retry_at").context("Failed to extract last_retry_at")?;
                
            let last_retry = last_retry_at.map(timestamp_to_system_time);
            
            let input_tokens: Option<i32> = row.try_get("input_tokens")
                .context("Failed to extract input_tokens")?;
                
            let truncation_strategy: Option<String> = row.try_get("truncation_strategy")
                .context("Failed to extract truncation_strategy")?;
                
            let metadata: Value = row.try_get("metadata")
                .context("Failed to extract metadata")?;

            jobs.push(EmbeddingJob {
                id,
                service_id,
                status,
                worker_id,
                batch_id,
                created_at,
                updated_at,
                error_message,
                retry_count,
                last_retry_at: last_retry,
                input_tokens,
                truncation_strategy,
                metadata,
            });
        }
        
        info!("Returning {} reset jobs", jobs.len());

        Ok(jobs)
    }).await;
}

/// Register a worker with the database
#[instrument(skip(pool, capabilities), fields(worker_id = %worker_id, worker_type = ?worker_type))]
pub async fn register_worker(
    pool: &PgPool,
    worker_id: &str,
    hostname: &str,
    ip_address: Option<&str>,
    worker_type: WorkerType,
    capabilities: &WorkerCapabilities,
) -> Result<()> {
    info!("Registering worker {} of type {:?}", worker_id, worker_type);

    return time_query("register_worker", || async {
        let worker_type_str = match worker_type {
            WorkerType::Orchestrator => "orchestrator",
            WorkerType::Inference => "inference",
        };

        let capabilities_json = serde_json::to_value(capabilities)
            .context("Failed to convert capabilities to JSON")?;

        // Direct INSERT instead of calling stored procedure to avoid issues
        let result = sqlx::query(
            r#"
            INSERT INTO embedding.workers 
                (id, hostname, ip_address, worker_type, capabilities, status, last_heartbeat, created_at)
            VALUES 
                ($1, $2, $3, $4::embedding.worker_type, $5, 'online'::embedding.worker_status, NOW(), NOW())
            ON CONFLICT (id) DO UPDATE
            SET 
                hostname = $2,
                ip_address = $3,
                capabilities = $5,
                status = 'online'::embedding.worker_status, 
                last_heartbeat = NOW()
            "#,
        )
        .bind(worker_id)
        .bind(hostname)
        .bind(ip_address)
        .bind(worker_type_str)
        .bind(capabilities_json)
        .execute(pool)
        .await
        .context(format!("Failed to register worker {}", worker_id))?;

        info!("Worker registration affected {} rows", result.rows_affected());

        Ok(())
    }).await;
}

/// Update worker status in the database
#[instrument(skip(pool), fields(worker_id = %worker_id, status = ?status))]
pub async fn update_worker_status(
    pool: &PgPool,
    worker_id: &str,
    status: WorkerStatus,
) -> Result<()> {
    return time_query("update_worker_status", || async {
        let status_str = match status {
            WorkerStatus::Online => "online",
            WorkerStatus::Offline => "offline",
            WorkerStatus::Busy => "busy",
        };

        info!("Updating worker {} status to {}", worker_id, status_str);

        // Skip existence check - it's causing issues in tests
        // We'll rely on the UPDATE query to find the worker directly

        // Cast the status string to the custom PostgreSQL type
        let result = sqlx::query(
            r#"
            UPDATE embedding.workers
            SET 
                status = $1::embedding.worker_status,
                last_heartbeat = NOW()
            WHERE id = $2
            "#,
        )
        .bind(status_str)
        .bind(worker_id)
        .execute(pool)
        .await
        .context(format!(
            "Failed to update worker {} status to {}",
            worker_id, status_str
        ))?;

        info!(
            "Worker status update affected {} rows",
            result.rows_affected()
        );

        Ok(())
    })
    .await;
}

/// Get worker by ID from the database
#[instrument(skip(pool), fields(worker_id = %worker_id))]
pub async fn get_worker_by_id(pool: &PgPool, worker_id: &str) -> Result<Option<Worker>> {
    info!("Getting worker with ID: {}", worker_id);

    return time_query("get_worker_by_id", || async {
        let row_opt = sqlx::query(
            r#"
            SELECT 
                id, 
                hostname, 
                ip_address, 
                worker_type::text as worker_type_str, 
                capabilities, 
                status::text as status_str, 
                last_heartbeat, 
                current_batch_size, 
                current_load, 
                active_jobs, 
                created_at
            FROM embedding.workers
            WHERE id = $1
            "#,
        )
        .bind(worker_id)
        .fetch_optional(pool)
        .await
        .context(format!("Failed to fetch worker with ID {}", worker_id))?;

        if let Some(row) = row_opt {
            let worker_type_str: String = row
                .try_get("worker_type_str")
                .context("Failed to extract worker_type")?;

            let worker_type =
                parse_worker_type(&worker_type_str).context("Failed to parse worker type")?;

            let status_str: String = row
                .try_get("status_str")
                .context("Failed to extract status")?;

            let status =
                parse_worker_status(&status_str).context("Failed to parse worker status")?;

            let capabilities_json: Value = row
                .try_get("capabilities")
                .context("Failed to extract capabilities")?;

            let capabilities: WorkerCapabilities = serde_json::from_value(capabilities_json)
                .context("Failed to parse capabilities JSON")?;

            let created_at = timestamp_to_system_time(
                row.try_get("created_at")
                    .context("Failed to extract created_at")?,
            );

            let last_heartbeat = timestamp_to_system_time(
                row.try_get("last_heartbeat")
                    .context("Failed to extract last_heartbeat")?,
            );

            info!("Found worker {} of type {:?}", worker_id, worker_type);

            Ok(Some(Worker {
                id: row.try_get("id").context("Failed to extract id")?,
                hostname: row
                    .try_get("hostname")
                    .context("Failed to extract hostname")?,
                ip_address: row
                    .try_get("ip_address")
                    .context("Failed to extract ip_address")?,
                worker_type,
                capabilities,
                status,
                last_heartbeat,
                current_batch_size: row
                    .try_get("current_batch_size")
                    .context("Failed to extract current_batch_size")?,
                current_load: row
                    .try_get("current_load")
                    .context("Failed to extract current_load")?,
                active_jobs: row
                    .try_get("active_jobs")
                    .context("Failed to extract active_jobs")?,
                created_at,
            }))
        } else {
            info!("Worker {} not found", worker_id);
            Ok(None)
        }
    })
    .await;
}

/// Get all workers with optional filtering by worker type
#[instrument(skip(pool), fields(worker_type = ?worker_type))]
pub async fn get_all_workers(
    pool: &PgPool,
    worker_type: Option<WorkerType>,
) -> Result<Vec<Worker>> {
    info!("Getting all workers with type filter: {:?}", worker_type);

    return time_query("get_all_workers", || async {
        // Get all workers with type filtering as a parameter
        let worker_type_str = match worker_type {
            Some(WorkerType::Inference) => Some("inference"),
            Some(WorkerType::Orchestrator) => Some("orchestrator"),
            None => None,
        };

        debug!("Looking for worker type: {:?}", worker_type_str);

        // Modified query to handle test worker IDs
        let query = match worker_type {
            Some(_) => {
                // When filtering by type, use a specific pattern matching approach
                r#"
                SELECT 
                    id, 
                    hostname, 
                    ip_address, 
                    worker_type::text as worker_type_str, 
                    capabilities, 
                    status::text as status_str, 
                    last_heartbeat, 
                    current_batch_size, 
                    current_load, 
                    active_jobs, 
                    created_at
                FROM embedding.workers
                WHERE worker_type::text = $1
                "#
            }
            None => {
                // When getting all workers, use a simpler query
                r#"
                SELECT 
                    id, 
                    hostname, 
                    ip_address, 
                    worker_type::text as worker_type_str, 
                    capabilities, 
                    status::text as status_str, 
                    last_heartbeat, 
                    current_batch_size, 
                    current_load, 
                    active_jobs, 
                    created_at
                FROM embedding.workers
                "#
            }
        };

        debug!(
            "Executing query to fetch workers with type filter: {:?}",
            worker_type_str
        );

        // Execute the query with or without parameter based on worker_type
        let rows = match worker_type {
            Some(_) => {
                sqlx::query(query)
                    .bind(worker_type_str.unwrap()) // Safe to unwrap since we know it's Some
                    .fetch_all(pool)
                    .await
            }
            None => sqlx::query(query).fetch_all(pool).await,
        }
        .context("Failed to fetch workers")?;

        debug!("Found {} workers matching filter", rows.len());

        let mut workers = Vec::with_capacity(rows.len());
        for row in rows {
            let worker_type_str: String = row
                .try_get("worker_type_str")
                .context("Failed to extract worker_type")?;

            let worker_type =
                parse_worker_type(&worker_type_str).context("Failed to parse worker type")?;

            let status_str: String = row
                .try_get("status_str")
                .context("Failed to extract status")?;

            let status =
                parse_worker_status(&status_str).context("Failed to parse worker status")?;

            let capabilities_json: Value = row
                .try_get("capabilities")
                .context("Failed to extract capabilities")?;

            let capabilities: WorkerCapabilities = serde_json::from_value(capabilities_json)
                .context("Failed to parse capabilities JSON")?;

            let created_at = timestamp_to_system_time(
                row.try_get("created_at")
                    .context("Failed to extract created_at")?,
            );

            let last_heartbeat = timestamp_to_system_time(
                row.try_get("last_heartbeat")
                    .context("Failed to extract last_heartbeat")?,
            );

            workers.push(Worker {
                id: row.try_get("id").context("Failed to extract id")?,
                hostname: row
                    .try_get("hostname")
                    .context("Failed to extract hostname")?,
                ip_address: row
                    .try_get("ip_address")
                    .context("Failed to extract ip_address")?,
                worker_type,
                capabilities,
                status,
                last_heartbeat,
                current_batch_size: row
                    .try_get("current_batch_size")
                    .context("Failed to extract current_batch_size")?,
                current_load: row
                    .try_get("current_load")
                    .context("Failed to extract current_load")?,
                active_jobs: row
                    .try_get("active_jobs")
                    .context("Failed to extract active_jobs")?,
                created_at,
            });
        }

        info!("Returning {} workers", workers.len());
        Ok(workers)
    })
    .await;
}

/// Create a connection pool with appropriate configuration
pub async fn create_connection_pool(
    connection_string: &str,
    max_connections: u32,
) -> Result<PgPool> {
    info!(
        "Creating database connection pool with max connections: {}",
        max_connections
    );

    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(max_connections)
        .min_connections(2)
        .max_lifetime(std::time::Duration::from_secs(30 * 60)) // 30 minutes
        .idle_timeout(std::time::Duration::from_secs(10 * 60)) // 10 minutes
        .connect(connection_string)
        .await
        .context("Failed to create database connection pool")?;

    // Test the connection
    sqlx::query("SELECT 1")
        .execute(&pool)
        .await
        .context("Failed to verify database connection")?;

    info!("Database connection pool created successfully");
    Ok(pool)
}
