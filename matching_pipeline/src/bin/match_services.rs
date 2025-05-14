// src/bin/match_services.rs
use anyhow::{Context, Result};
use log::{info, warn};
use std::{path::Path, time::Instant};

use dedupe_lib::{db, service_matching};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    info!("Starting service matching pipeline component");
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

    // Connect to the database
    let pool = db::connect()
        .await
        .context("Failed to connect to database")?;
    info!("Successfully connected to the database");

    // Check if there are clusters to process
    let conn = pool.get().await.context("Failed to get DB connection")?;
    let clusters_query = "SELECT COUNT(*) FROM public.group_cluster";
    let clusters_row = conn
        .query_one(clusters_query, &[])
        .await
        .context("Failed to count clusters")?;
    let clusters_count: i64 = clusters_row.get(0);

    if clusters_count == 0 {
        info!("No clusters found. Please run cluster consolidation first.");
        return Ok(());
    }

    info!(
        "Found {} clusters to process for service matching",
        clusters_count
    );

    // Run the service matching component
    info!("Running service matching...");
    let result = service_matching::semantic_geospatial::match_services(&pool)
        .await
        .context("Failed to match services")?;

    info!(
        "Service matching completed successfully. Created {} service matches.",
        result.groups_created
    );

    info!(
        "Method statistics: {} matches created, avg similarity: {:.2}",
        result.stats.groups_created, result.stats.avg_confidence
    );

    // Get additional service match stats
    let conn = pool
        .get()
        .await
        .context("Failed to get DB connection for stats")?;
    let stats_query = "
        SELECT 
            COUNT(*) as total_matches,
            COALESCE(AVG(similarity_score), 0.0) as avg_similarity,
            COUNT(DISTINCT group_cluster_id) as clusters_with_matches
        FROM 
            public.service_match
    ";

    let row = conn
        .query_one(stats_query, &[])
        .await
        .context("Failed to query service match statistics")?;

    let total_matches: i64 = row.get(0);
    let avg_similarity: f64 = row.get(1);
    let clusters_with_matches: i64 = row.get(2);

    info!("Service match statistics:");
    info!("  Total service matches: {}", total_matches);
    info!("  Average similarity score: {:.2}", avg_similarity);
    info!("  Clusters with service matches: {}", clusters_with_matches);

    let elapsed = start_time.elapsed();
    info!("Service matching completed in {:.2?}", elapsed);

    Ok(())
}
