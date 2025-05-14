// src/bin/identify_entities.rs
use anyhow::{Context, Result};
use chrono::Utc;
use log::{info, warn};
use std::{path::Path, time::Instant};
use uuid::Uuid;

use dedupe_lib::{db, entity_organizations};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    info!("Starting identify_entities pipeline component");
    let start_time = Instant::now();

    // Try to load .env file if it exists
    let env_paths = [".env", ".env.local", "../.env"];
    let mut loaded_env = false;

    for path in env_paths.iter() {
        if Path::new(path).exists() {
            if let Err(e) = db::load_env_from_file(path) {
                warn!("Failed to load environment from {}: {}", path, e);
            } else {
                info!("Loaded environment variables from {}", path);
                loaded_env = true;
                break;
            }
        }
    }

    if !loaded_env {
        info!("No .env file found, using environment variables from system");
    }

    // Connect to the database (now async)
    let pool = db::connect()
        .await
        .context("Failed to connect to database")?;
    info!("Successfully connected to the database");

    // Generate a run ID for traceability
    let run_id = Uuid::new_v4().to_string();
    let run_timestamp = Utc::now().naive_utc();
    info!("Run ID: {}", run_id);

    // Run the entity identification component
    info!("Running identify_entities...");
    let entity_count = entity_organizations::extract_entities(&pool)
        .await
        .context("Failed to identify entities")?;

    info!("Entity identification step complete");
    info!(
        "Discovered or created {} mapping(s) between organization and entity tables.",
        entity_count.len()
    );

    // Link entities to their features
    info!("Linking entity features...");
    entity_organizations::link_entity_features(&pool, &entity_count)
        .await
        .context("Failed to link entity features")?;
    info!("Entity features linked successfully");

    // Get the total count of entities
    let conn = pool
        .get()
        .await
        .context("Failed to get DB connection for counting entities")?;
    let total_entities_row = conn
        .query_one("SELECT COUNT(*) FROM public.entity", &[])
        .await
        .context("Failed to count total entities")?;
    let total_entities_count: i64 = total_entities_row.get(0);

    let elapsed = start_time.elapsed();
    info!(
        "identify_entities completed in {:.2?}. Total entities in system: {}",
        elapsed, total_entities_count
    );

    Ok(())
}
