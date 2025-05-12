// src/db.rs

use anyhow::{Context, Result};
use bb8::Pool;
use bb8_postgres::PostgresConnectionManager;
use log::{debug, info, warn};
use std::time::Duration;
use tokio_postgres::{Config, GenericClient, NoTls}; // Added GenericClient
use uuid::Uuid;

use crate::{
    models::{NewClusterFormationEdge, NewSuggestedAction},
    reinforcement::FeedbackItem,
};

pub type PgPool = Pool<PostgresConnectionManager<NoTls>>;

/// Reads environment variables and constructs a PostgreSQL config.
/// Matches behavior and settings of the TS `hetznerPostgresConfig`.
fn build_pg_config() -> Config {
    let mut config = Config::new();

    let host = std::env::var("POSTGRES_HOST").unwrap_or_else(|e| {
        warn!("POSTGRES_HOST not found in environment: {}", e);
        "10.0.0.1".to_string()
    });

    let port_str = std::env::var("POSTGRES_PORT").unwrap_or_else(|e| {
        warn!("POSTGRES_PORT not found in environment: {}", e);
        "5432".to_string()
    });
    let port = port_str.parse::<u16>().unwrap_or_else(|e| {
        warn!("Invalid POSTGRES_PORT format: {}", e);
        5432
    });

    let dbname = std::env::var("POSTGRES_DB").unwrap_or_else(|e| {
        warn!("POSTGRES_DB not found in environment: {}", e);
        "dataplatform".to_string()
    });

    let user = std::env::var("POSTGRES_USER").unwrap_or_else(|e| {
        warn!("POSTGRES_USER not found in environment: {}", e);
        "postgres".to_string()
    });

    let password = std::env::var("POSTGRES_PASSWORD").unwrap_or_else(|e| {
        warn!("POSTGRES_PASSWORD not found in environment: {}", e);
        "".to_string()
    });

    info!("Database connection parameters:");
    info!("  Host: {}", host);
    info!("  Port: {}", port);
    info!("  Database: {}", dbname);
    info!("  User: {}", user);
    info!(
        "  Password: {}",
        if password.is_empty() {
            "[empty]"
        } else {
            "[set]"
        }
    );

    config
        .host(&host)
        .port(port)
        .dbname(&dbname)
        .user(&user)
        .password(&password);

    config.application_name("deduplication");
    config.connect_timeout(Duration::from_secs(10));

    config
}

/// Initializes the database connection pool.
pub async fn connect() -> Result<PgPool> {
    debug!("Environment variables:");
    for (key, value) in std::env::vars() {
        if key.starts_with("POSTGRES_") {
            debug!(
                "  {}: {}",
                key,
                if key == "POSTGRES_PASSWORD" {
                    "[hidden]"
                } else {
                    &value
                }
            );
        }
    }

    let config = build_pg_config();
    info!("Connecting to PostgreSQL database...");
    let manager = PostgresConnectionManager::new(config, NoTls);

    info!("Building connection pool...");
    let pool = Pool::builder()
        .max_size(20)
        .min_idle(Some(5))
        .idle_timeout(Some(Duration::from_secs(90)))
        .connection_timeout(Duration::from_secs(15))
        .build(manager)
        .await
        .context("Failed to build database connection pool")?;

    info!("Testing database connection...");
    {
        let conn = pool
            .get()
            .await
            .context("Failed to get initial test connection from pool")?;

        let row = conn
            .query_one("SELECT 1", &[])
            .await
            .context("Test query 'SELECT 1' failed")?;

        let result: i32 = row.get(0);
        if result == 1 {
            info!("Database connection test successful");
        } else {
            warn!(
                "Database connection test returned unexpected value: {}",
                result
            );
            return Err(anyhow::anyhow!(
                "Database connection test failed: unexpected result from SELECT 1"
            ));
        }
    }

    info!("Database connection pool initialized successfully");
    Ok(pool)
}

pub fn load_env_from_file(file_path: &str) -> Result<()> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    info!("Loading environment variables from file: {}", file_path);
    let file = File::open(file_path).context(format!("Failed to open env file: {}", file_path))?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line.context("Failed to read line from env file")?;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        if let Some(idx) = line.find('=') {
            let key = line[..idx].trim();
            let value = line[idx + 1..].trim();
            if std::env::var(key).is_err() {
                debug!("Setting env var from file: {}", key);
                // SAFETY: Setting environment variables like this is common,
                // but be aware of potential thread-safety issues if other threads
                // read environment variables concurrently without synchronization.
                // std::env::set_var is not guaranteed to be thread-safe by POSIX.
                // However, in many contexts (like initial setup), this is acceptable.
                std::env::set_var(key, value);
            } else {
                debug!("Env var already set, skipping: {}", key);
            }
        }
    }
    Ok(())
}

pub async fn insert_suggestion(
    conn: &impl GenericClient, // Use GenericClient for flexibility (Transaction or Connection)
    suggestion: &NewSuggestedAction,
) -> Result<Uuid> {
    const INSERT_SUGGESTION_SQL: &str = "
        INSERT INTO clustering_metadata.suggested_actions (
            pipeline_run_id, action_type, entity_id, group_id_1, group_id_2, cluster_id,
            triggering_confidence, details, reason_code, reason_message, priority, status,
            reviewer_id, reviewed_at, review_notes
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        RETURNING id";

    let row = conn
        .query_one(
            INSERT_SUGGESTION_SQL,
            &[
                &suggestion.pipeline_run_id,
                &suggestion.action_type,
                &suggestion.entity_id,
                &suggestion.group_id_1,
                &suggestion.group_id_2,
                &suggestion.cluster_id,
                &suggestion.triggering_confidence,
                &suggestion.details,
                &suggestion.reason_code,
                &suggestion.reason_message,
                &suggestion.priority,
                &suggestion.status,
                &suggestion.reviewer_id,
                &suggestion.reviewed_at,
                &suggestion.review_notes,
            ],
        )
        .await
        .context("Failed to insert suggested_action")?;

    let suggestion_id: Uuid = row.get(0);
    Ok(suggestion_id)
}

pub async fn update_suggestion_review(
    pool: &PgPool,
    suggestion_id: Uuid,
    reviewer_id: String,
    new_status: String,
    review_notes: Option<String>,
) -> Result<u64> {
    // This function was not using transactions or prepare originally for its main statement,
    // so its core logic remains similar, just ensuring good context for errors.
    // It gets a connection from the pool.
    let conn = pool
        .get()
        .await
        .context("Failed to get DB connection for update_suggestion_review")?;

    const UPDATE_SUGGESTION_SQL: &str = "
        UPDATE clustering_metadata.suggested_actions
         SET status = $1,
             reviewer_id = $2,
             reviewed_at = CURRENT_TIMESTAMP,
             review_notes = $3
         WHERE id = $4";

    conn.execute(
        UPDATE_SUGGESTION_SQL,
        &[&new_status, &reviewer_id, &review_notes, &suggestion_id],
    )
    .await
    .context("Failed to update suggested_action review")
}

pub async fn insert_cluster_formation_edge(
    conn: &impl GenericClient, // Use GenericClient
    edge: &NewClusterFormationEdge,
) -> Result<Uuid> {
    const INSERT_EDGE_SQL: &str = "
        INSERT INTO clustering_metadata.cluster_formation_edges (
            pipeline_run_id, source_group_id, target_group_id,
            calculated_edge_weight, contributing_shared_entities
        ) VALUES ($1, $2, $3, $4, $5)
        RETURNING id";

    let row = conn
        .query_one(
            INSERT_EDGE_SQL,
            &[
                &edge.pipeline_run_id,
                &edge.source_group_id,
                &edge.target_group_id,
                &edge.calculated_edge_weight,
                &edge.contributing_shared_entities,
            ],
        )
        .await
        .context("Failed to insert cluster_formation_edge")?;

    let edge_id: Uuid = row.get(0);
    Ok(edge_id)
}

pub async fn get_confidence_for_entity_in_group(
    conn: &impl GenericClient,
    entity_id_to_check: &str,
    pair_group_id: &str,
) -> Result<Option<f64>> {
    // This function was already using prepare on the GenericClient,
    // which is fine for a single, well-defined query.
    // For consistency with the request to reduce prepare, we can change it,
    // but prepare here is not part of a "massive query building" problem.
    // Let's change it for consistency.
    const SELECT_CONFIDENCE_SQL: &str = "
        SELECT entity_id_1, entity_id_2, confidence_score
         FROM public.entity_group
         WHERE id = $1";

    match conn
        .query_opt(SELECT_CONFIDENCE_SQL, &[&pair_group_id])
        .await
    {
        Ok(Some(row)) => {
            let entity_id_1_from_db: String = row.get("entity_id_1");
            let entity_id_2_from_db: String = row.get("entity_id_2");
            let confidence_score: Option<f64> = row.get("confidence_score");

            if entity_id_1_from_db == entity_id_to_check
                || entity_id_2_from_db == entity_id_to_check
            {
                Ok(confidence_score)
            } else {
                warn!(
                    "Entity {} not found in pair {} (contains {} and {}). This might indicate a logic issue.",
                    entity_id_to_check, pair_group_id, entity_id_1_from_db, entity_id_2_from_db
                );
                Ok(None)
            }
        }
        Ok(None) => {
            warn!("No entity_group found with id: {}", pair_group_id);
            Ok(None)
        }
        Err(e) => Err(anyhow::anyhow!(e).context(format!(
            "Failed to query entity_group for pair_id: {}",
            pair_group_id
        ))),
    }
}

// Helper to fetch feedback (to be implemented, e.g., in feedback_processor.rs)
pub async fn fetch_recent_feedback_items(pool: &PgPool) -> Result<Vec<FeedbackItem>> {
    // Query clustering_metadata.human_review_decisions and human_review_method_feedback
    // Transform into Vec<FeedbackItem>
    // This should probably fetch items that haven't been processed by a training run yet.
    // For now, a placeholder:
    let client = pool.get().await?;
    let rows = client
        .query(
            "SELECT entity_id1, entity_id2, method_type, confidence, was_correct
         FROM view_rl_feedback_items -- Assuming a view or query exists for this
         WHERE processed_for_training = FALSE -- Example flag
         LIMIT 1000", // Limit batch size
            &[],
        )
        .await
        .context("Failed to fetch feedback items from DB")?;

    let mut items = Vec::new();
    for row in rows {
        items.push(FeedbackItem {
            entity_id1: row.get(0),
            entity_id2: row.get(1),
            method_type: row.get(2),
            confidence: row.get(3),
            was_correct: row.get(4),
        });
    }
    // After fetching, mark them as processed_for_training = TRUE in the DB.
    // This part is crucial and needs careful implementation.
    Ok(items)
}
