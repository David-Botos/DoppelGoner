// src/bin/extract_context_features.rs
use anyhow::{Context, Result};
use log::{info, warn};
use std::{path::Path, time::Instant};

use dedupe_lib::{db, entity_organizations};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    info!("Starting context feature extraction pipeline component");
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

    // Count entities to check if there are any to process
    let conn = pool
        .get()
        .await
        .context("Failed to get DB connection for counting entities")?;
    let total_entities_row = conn
        .query_one("SELECT COUNT(*) FROM public.entity", &[])
        .await
        .context("Failed to count total entities")?;
    let total_entities_count: i64 = total_entities_row.get(0);

    if total_entities_count == 0 {
        info!("No entities found. Please run identify_entities first.");
        return Ok(());
    }

    // Run the context feature extraction component
    info!("Running extract_and_store_all_entity_context_features...");
    match entity_organizations::extract_and_store_all_entity_context_features(&pool).await {
        Ok(features_count) => info!("Successfully extracted and stored context features for {} entities.", features_count),
        Err(e) => warn!("Context feature extraction failed: {}. Some operations may not have ML-guided features.", e),
    }

    let elapsed = start_time.elapsed();
    info!("Context feature extraction completed in {:.2?}", elapsed);

    Ok(())
}
