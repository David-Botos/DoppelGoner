// orchestrator/src/db/queries.rs

use crate::types::types::{
    EmbeddingJob, JobStatus, ServiceWithTaxonomiesRow, Worker, WorkerCapabilities, WorkerStatus,
    WorkerType,
};
use anyhow::{anyhow, Result};
use serde_json::Value;
use sqlx::{PgPool, Row};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

// Helper function to convert string to JobStatus
pub fn parse_job_status(status: &str) -> Result<JobStatus> {
    match status {
        "queued" => Ok(JobStatus::Queued),
        "fetched" => Ok(JobStatus::Fetched),
        "tokenized" => Ok(JobStatus::Tokenized),
        "processing" => Ok(JobStatus::Processing),
        "completed" => Ok(JobStatus::Completed),
        "failed" => Ok(JobStatus::Failed),
        _ => Err(anyhow!("Invalid job status: {}", status)),
    }
}

// Helper function to convert string to WorkerStatus
pub fn parse_worker_status(status: &str) -> Result<WorkerStatus> {
    match status {
        "online" => Ok(WorkerStatus::Online),
        "offline" => Ok(WorkerStatus::Offline),
        "busy" => Ok(WorkerStatus::Busy),
        _ => Err(anyhow!("Invalid worker status: {}", status)),
    }
}

// Helper function to convert string to WorkerType
pub fn parse_worker_type(worker_type: &str) -> Result<WorkerType> {
    match worker_type {
        "orchestrator" => Ok(WorkerType::Orchestrator),
        "inference" => Ok(WorkerType::Inference),
        _ => Err(anyhow!("Invalid worker type: {}", worker_type)),
    }
}

// Convert sqlx timestamp to SystemTime
pub fn timestamp_to_system_time(timestamp: sqlx::types::time::PrimitiveDateTime) -> SystemTime {
    let secs = timestamp.assume_utc().unix_timestamp();
    let nanos = timestamp.microsecond() * 1000;
    UNIX_EPOCH + std::time::Duration::new(secs as u64, nanos)
}

// Claim jobs from the database
pub async fn claim_jobs(
    pool: &PgPool,
    worker_id: &str,
    batch_size: i32,
) -> Result<Vec<EmbeddingJob>> {
    let rows = sqlx::query(
        r#"
        SELECT * FROM embedding.claim_jobs($1, $2)
        "#,
    )
    .bind(worker_id)
    .bind(batch_size)
    .fetch_all(pool)
    .await?;

    let mut jobs = Vec::with_capacity(rows.len());
    for row in rows {
        let id: Uuid = row.try_get("id")?;
        let service_id: String = row.try_get("service_id")?;
        let status_str: String = row.try_get("status")?;
        let status = parse_job_status(&status_str)?;
        let worker_id: Option<String> = row.try_get("worker_id")?;
        let batch_id: Option<Uuid> = row.try_get("batch_id")?;
        let created_at = timestamp_to_system_time(row.try_get("created_at")?);
        let updated_at = timestamp_to_system_time(row.try_get("updated_at")?);
        let error_message: Option<String> = row.try_get("error_message")?;
        let retry_count: i32 = row.try_get("retry_count")?;
        let last_retry_at: Option<sqlx::types::time::PrimitiveDateTime> =
            row.try_get("last_retry_at")?;
        let last_retry = last_retry_at.map(timestamp_to_system_time);
        let input_tokens: Option<i32> = row.try_get("input_tokens")?;
        let truncation_strategy: Option<String> = row.try_get("truncation_strategy")?;
        let metadata: Value = row.try_get("metadata")?;

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
}

// Fetch service data for a specific service
pub async fn fetch_service_with_taxonomies(
    pool: &PgPool,
    service_id: &str,
) -> Result<Vec<ServiceWithTaxonomiesRow>> {
    let rows = sqlx::query(
        r#"
        SELECT * FROM embedding.fetch_service_with_taxonomies($1)
        "#,
    )
    .bind(service_id)
    .fetch_all(pool)
    .await?;

    let mut service_rows = Vec::with_capacity(rows.len());
    for row in rows {
        let service_row = ServiceWithTaxonomiesRow {
            service_id: row.try_get("service_id")?,
            service_name: row.try_get("service_name")?,
            service_description: row.try_get("service_description")?,
            service_short_description: row.try_get("service_short_description")?,
            organization_id: row.try_get("organization_id")?,
            service_url: row.try_get("service_url")?,
            service_email: row.try_get("service_email")?,
            service_status: row.try_get("service_status")?,
            taxonomy_id: row.try_get("taxonomy_id")?,
            taxonomy_term: row.try_get("taxonomy_term")?,
            taxonomy_description: row.try_get("taxonomy_description")?,
            taxonomy_category: row.try_get("taxonomy_category")?,
        };
        service_rows.push(service_row);
    }

    Ok(service_rows)
}

// Update job status
pub async fn update_job_status(
    pool: &PgPool,
    job_id: &Uuid,
    status: JobStatus,
    error_message: Option<&str>,
    input_tokens: Option<i32>,
    metadata: Option<&Value>,
) -> Result<EmbeddingJob> {
    let status_str = match status {
        JobStatus::Queued => "queued",
        JobStatus::Fetched => "fetched",
        JobStatus::Tokenized => "tokenized",
        JobStatus::Processing => "processing",
        JobStatus::Completed => "completed",
        JobStatus::Failed => "failed",
    };

    let row = sqlx::query(
        r#"
        SELECT * FROM embedding.update_job_status($1, $2::embedding.job_status, $3, $4, NULL, $5)
        "#,
    )
    .bind(job_id)
    .bind(status_str)
    .bind(error_message)
    .bind(input_tokens)
    .bind(metadata)
    .fetch_one(pool)
    .await?;

    let id: Uuid = row.try_get("id")?;
    let service_id: String = row.try_get("service_id")?;
    let status_str: String = row.try_get("status")?;
    let status = parse_job_status(&status_str)?;
    let worker_id: Option<String> = row.try_get("worker_id")?;
    let batch_id: Option<Uuid> = row.try_get("batch_id")?;
    let created_at = timestamp_to_system_time(row.try_get("created_at")?);
    let updated_at = timestamp_to_system_time(row.try_get("updated_at")?);
    let error_message: Option<String> = row.try_get("error_message")?;
    let retry_count: i32 = row.try_get("retry_count")?;
    let last_retry_at: Option<sqlx::types::time::PrimitiveDateTime> =
        row.try_get("last_retry_at")?;
    let last_retry = last_retry_at.map(timestamp_to_system_time);
    let input_tokens: Option<i32> = row.try_get("input_tokens")?;
    let truncation_strategy: Option<String> = row.try_get("truncation_strategy")?;
    let metadata: Value = row.try_get("metadata")?;

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
}

// Reset stale jobs
pub async fn reset_stale_jobs(pool: &PgPool, minutes_threshold: i32) -> Result<Vec<EmbeddingJob>> {
    let rows = sqlx::query(
        r#"
        SELECT * FROM embedding.reset_stale_jobs($1)
        "#,
    )
    .bind(minutes_threshold)
    .fetch_all(pool)
    .await?;

    let mut jobs = Vec::with_capacity(rows.len());
    for row in rows {
        let id: Uuid = row.try_get("id")?;
        let service_id: String = row.try_get("service_id")?;
        let status_str: String = row.try_get("status")?;
        let status = parse_job_status(&status_str)?;
        let worker_id: Option<String> = row.try_get("worker_id")?;
        let batch_id: Option<Uuid> = row.try_get("batch_id")?;
        let created_at = timestamp_to_system_time(row.try_get("created_at")?);
        let updated_at = timestamp_to_system_time(row.try_get("updated_at")?);
        let error_message: Option<String> = row.try_get("error_message")?;
        let retry_count: i32 = row.try_get("retry_count")?;
        let last_retry_at: Option<sqlx::types::time::PrimitiveDateTime> =
            row.try_get("last_retry_at")?;
        let last_retry = last_retry_at.map(timestamp_to_system_time);
        let input_tokens: Option<i32> = row.try_get("input_tokens")?;
        let truncation_strategy: Option<String> = row.try_get("truncation_strategy")?;
        let metadata: Value = row.try_get("metadata")?;

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
}

// Register a worker with the database
pub async fn register_worker(
    pool: &PgPool,
    worker_id: &str,
    hostname: &str,
    ip_address: Option<&str>,
    worker_type: WorkerType,
    capabilities: &WorkerCapabilities,
) -> Result<()> {
    let worker_type_str = match worker_type {
        WorkerType::Orchestrator => "orchestrator",
        WorkerType::Inference => "inference",
    };

    let capabilities_json = serde_json::to_value(capabilities)?;

    sqlx::query(
        r#"
        SELECT * FROM embedding.register_worker($1, $2, $3, $4::embedding.worker_type, $5)
        "#,
    )
    .bind(worker_id)
    .bind(hostname)
    .bind(ip_address)
    .bind(worker_type_str)
    .bind(capabilities_json)
    .execute(pool)
    .await?;

    Ok(())
}

// Update worker status
pub async fn update_worker_status(
    pool: &PgPool,
    worker_id: &str,
    status: WorkerStatus,
) -> Result<()> {
    let status_str = match status {
        WorkerStatus::Online => "online",
        WorkerStatus::Offline => "offline",
        WorkerStatus::Busy => "busy",
    };

    sqlx::query(
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
    .await?;

    Ok(())
}

// Get worker by ID
pub async fn get_worker_by_id(pool: &PgPool, worker_id: &str) -> Result<Option<Worker>> {
    let row_opt = sqlx::query(
        r#"
        SELECT 
            id, 
            hostname, 
            ip_address, 
            worker_type, 
            capabilities, 
            status, 
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
    .await?;

    if let Some(row) = row_opt {
        let worker_type_str: String = row.try_get("worker_type")?;
        let worker_type = parse_worker_type(&worker_type_str)?;

        let status_str: String = row.try_get("status")?;
        let status = parse_worker_status(&status_str)?;

        let capabilities_json: Value = row.try_get("capabilities")?;
        let capabilities: WorkerCapabilities = serde_json::from_value(capabilities_json)?;

        let created_at = timestamp_to_system_time(row.try_get("created_at")?);
        let last_heartbeat = timestamp_to_system_time(row.try_get("last_heartbeat")?);

        Ok(Some(Worker {
            id: row.try_get("id")?,
            hostname: row.try_get("hostname")?,
            ip_address: row.try_get("ip_address")?,
            worker_type,
            capabilities,
            status,
            last_heartbeat,
            current_batch_size: row.try_get("current_batch_size")?,
            current_load: row.try_get("current_load")?,
            active_jobs: row.try_get("active_jobs")?,
            created_at,
        }))
    } else {
        Ok(None)
    }
}

// Get all workers
pub async fn get_all_workers(
    pool: &PgPool,
    worker_type: Option<WorkerType>,
) -> Result<Vec<Worker>> {
    let worker_type_filter = match worker_type {
        Some(WorkerType::Inference) => "WHERE worker_type = 'inference'",
        Some(WorkerType::Orchestrator) => "WHERE worker_type = 'orchestrator'",
        None => "",
    };

    let query = format!(
        r#"
        SELECT 
            id, 
            hostname, 
            ip_address, 
            worker_type, 
            capabilities, 
            status, 
            last_heartbeat, 
            current_batch_size, 
            current_load, 
            active_jobs, 
            created_at
        FROM embedding.workers
        {}
        "#,
        worker_type_filter
    );

    let rows = sqlx::query(&query).fetch_all(pool).await?;

    let mut workers = Vec::with_capacity(rows.len());
    for row in rows {
        let worker_type_str: String = row.try_get("worker_type")?;
        let worker_type = parse_worker_type(&worker_type_str)?;

        let status_str: String = row.try_get("status")?;
        let status = parse_worker_status(&status_str)?;

        let capabilities_json: Value = row.try_get("capabilities")?;
        let capabilities: WorkerCapabilities = serde_json::from_value(capabilities_json)?;

        let created_at = timestamp_to_system_time(row.try_get("created_at")?);
        let last_heartbeat = timestamp_to_system_time(row.try_get("last_heartbeat")?);

        workers.push(Worker {
            id: row.try_get("id")?,
            hostname: row.try_get("hostname")?,
            ip_address: row.try_get("ip_address")?,
            worker_type,
            capabilities,
            status,
            last_heartbeat,
            current_batch_size: row.try_get("current_batch_size")?,
            current_load: row.try_get("current_load")?,
            active_jobs: row.try_get("active_jobs")?,
            created_at,
        });
    }

    Ok(workers)
}
