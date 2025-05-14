// src/bin/consolidate_clusters.rs
use anyhow::{Context, Result};
use log::{info, warn};
use std::{path::Path, sync::Arc, time::Instant};
use tokio::sync::Mutex;
use uuid::Uuid;

use dedupe_lib::{consolidate_clusters, db, reinforcement};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    info!("Starting cluster consolidation pipeline component");
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

    // Generate a run ID for traceability
    let run_id = Uuid::new_v4().to_string();
    info!("Run ID: {}", run_id);

    // Check if there are unassigned groups
    let conn = pool.get().await.context("Failed to get DB connection")?;
    let unassigned_query =
        "SELECT COUNT(*) FROM public.entity_group WHERE group_cluster_id IS NULL";
    let groups_row = conn
        .query_one(unassigned_query, &[])
        .await
        .context("Failed to count unassigned groups")?;
    let unassigned_groups: i64 = groups_row.get(0);

    info!(
        "Found {} groups that need cluster assignment",
        unassigned_groups
    );

    if unassigned_groups == 0 {
        info!("No groups require clustering. Skipping consolidation.");
        return Ok(());
    }

    // Initialize the ML reinforcement orchestrator
    info!("Initializing ML-guided matching reinforcement orchestrator");
    let reinforcement_orchestrator_instance = reinforcement::MatchingOrchestrator::new(&pool)
        .await
        .context("Failed to initialize ML reinforcement orchestrator")?;

    let reinforcement_orchestrator = Arc::new(Mutex::new(reinforcement_orchestrator_instance));

    // Run the cluster consolidation component
    info!("Running cluster consolidation...");
    let clusters_created =
        consolidate_clusters::process_clusters(&pool, Some(&reinforcement_orchestrator), &run_id)
            .await
            .context("Failed to process clusters")?;

    info!(
        "Cluster consolidation completed successfully. Created {} clusters.",
        clusters_created
    );

    // Get additional cluster stats
    let conn = pool
        .get()
        .await
        .context("Failed to get DB connection for stats")?;
    let stats_query = "
        SELECT 
            COUNT(id) as total_clusters,
            COALESCE(AVG(entity_count), 0.0) as avg_entities_per_cluster,
            COALESCE(AVG(group_count), 0.0) as avg_groups_per_cluster,
            COALESCE(MAX(entity_count), 0) as largest_cluster_size
        FROM 
            public.group_cluster
    ";

    let row = conn
        .query_one(stats_query, &[])
        .await
        .context("Failed to query cluster statistics")?;

    let total_clusters: i64 = row.get(0);
    let avg_entities_per_cluster: f64 = row.get(1);
    let avg_groups_per_cluster: f64 = row.get(2);
    let largest_cluster_size: i64 = row.get(3);

    info!("Cluster statistics:");
    info!("  Total clusters: {}", total_clusters);
    info!(
        "  Average entities per cluster: {:.2}",
        avg_entities_per_cluster
    );
    info!(
        "  Average groups per cluster: {:.2}",
        avg_groups_per_cluster
    );
    info!("  Largest cluster size: {}", largest_cluster_size);

    let elapsed = start_time.elapsed();
    info!("Cluster consolidation completed in {:.2?}", elapsed);

    Ok(())
}
