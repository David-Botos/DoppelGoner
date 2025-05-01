// src/main.rs
use anyhow::{Context, Result};
use chrono::Utc;
use log::{info, warn};
use results::MatchMethodStats;
use uuid::Uuid;
// use services::match_services::find_service_matches;
use std::{
    collections::HashMap,
    path::Path,
    time::{Duration, Instant},
};

mod consolidate_clusters;
mod db;
mod entity_organizations;
mod matching;
mod models;
mod results;
mod service_matching;

use db::PgPool;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    info!("Starting HSDS entity grouping and clustering pipeline");
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

    // Capture timing information for each phase
    let mut phase_times = HashMap::new();

    // Run the pipeline in sequence
    let stats: results::PipelineStats = run_pipeline(&pool, &mut phase_times).await?;

    let elapsed = start_time.elapsed();
    info!(
        "Pipeline completed in {:.2?}. Processed: {} entities, {} groups, {} clusters, {} service matches",
        elapsed,
        stats.total_entities,
        stats.total_groups,
        stats.total_clusters,
        stats.total_service_matches
    );

    // Generate and store report
    let description = Some("Full pipeline run with standard matching configuration".to_string());
    results::generate_report(&pool, stats, &phase_times, description).await?;

    Ok(())
}

async fn run_pipeline(
    pool: &PgPool,
    phase_times: &mut HashMap<String, Duration>,
) -> Result<results::PipelineStats> {
    let run_id = Uuid::new_v4().to_string();
    let run_timestamp = Utc::now().naive_utc();

    // Initialize more complete stats structure
    let mut stats = results::PipelineStats {
        run_id,
        run_timestamp,
        description: Some("Regular pipeline run".to_string()),

        total_entities: 0,
        total_groups: 0,
        total_clusters: 0,
        total_service_matches: 0,

        // Initialize timing fields - will be updated later
        entity_processing_time: 0.0,
        matching_time: 0.0,
        clustering_time: 0.0,
        service_matching_time: 0.0,
        total_processing_time: 0.0,

        // These will be populated during the pipeline
        method_stats: Vec::new(),
        cluster_stats: None,
        service_stats: None,
    };

    info!("Pipeline started. Progress: [0/4] phases (0%)");

    // Phase 1: Entity identification
    info!("Phase 1: Entity identification");
    let phase1_start = Instant::now();
    stats.total_entities = identify_entities(pool).await?;
    let phase1_duration = phase1_start.elapsed();
    phase_times.insert("entity_identification".to_string(), phase1_duration);
    stats.entity_processing_time = phase1_duration.as_secs_f64();
    info!(
        "Identified {} entities in {:.2?}",
        stats.total_entities,
        phase1_start.elapsed()
    );
    info!("Pipeline progress: [1/4] phases (25%)");

    // Phase 2: Entity matching - now returns method stats too
    info!("Phase 2: Entity matching");
    let phase2_start = Instant::now();
    let (total_groups, method_stats) = run_matching_pipeline(pool).await?;
    stats.total_groups = total_groups;
    stats.method_stats = method_stats;
    let phase2_duration = phase2_start.elapsed();
    phase_times.insert("entity_matching".to_string(), phase2_duration);
    stats.matching_time = phase2_duration.as_secs_f64();
    info!(
        "Created {} entity groups in {:.2?}",
        stats.total_groups,
        phase2_start.elapsed()
    );
    info!("Pipeline progress: [2/4] phases (50%)");

    // Phase 3: Cluster consolidation
    info!("Phase 3: Cluster consolidation");
    let phase3_start = Instant::now();
    stats.total_clusters = consolidate_clusters(pool).await?;
    let phase3_duration = phase3_start.elapsed();
    phase_times.insert("cluster_consolidation".to_string(), phase3_duration);
    stats.clustering_time = phase3_duration.as_secs_f64();
    info!(
        "Formed {} clusters in {:.2?}",
        stats.total_clusters,
        phase3_start.elapsed()
    );
    info!("Pipeline progress: [3/4] phases (75%)");

    // Phase 4: Service matching
    info!("Phase 4: Service matching");
    let phase4_start = Instant::now();
    let service_match_stats = service_matching::semantic_geospatial::match_services(pool)
        .await
        .context("failed to match services")?;
    stats.total_service_matches = service_match_stats.groups_created;
    stats.method_stats.push(service_match_stats.stats);

    let phase4_duration = phase4_start.elapsed();
    phase_times.insert("service_matching".to_string(), phase4_duration);
    stats.service_matching_time = phase4_duration.as_secs_f64();

    // Calculate total processing time
    stats.total_processing_time = stats.entity_processing_time
        + stats.matching_time
        + stats.clustering_time
        + stats.service_matching_time;

    // Collect additional statistics
    let (cluster_stats, service_stats) = tokio::join!(
        results::collect_cluster_stats(pool),
        results::collect_service_stats(pool)
    );

    stats.cluster_stats = cluster_stats?;
    stats.service_stats = service_stats?;

    Ok(stats)
}

async fn identify_entities(pool: &PgPool) -> Result<usize> {
    // First, extract entities from organizations
    let org_entities = entity_organizations::extract_entities(pool)
        .await
        .context("Failed to extract entities from organizations")?;

    // Then, link entities to their features (services, phones, etc.)
    let linked_entities = entity_organizations::link_entity_features(pool, &org_entities)
        .await
        .context("Failed to link entity features")?;

    Ok(linked_entities)
}

async fn run_matching_pipeline(pool: &PgPool) -> Result<(usize, Vec<MatchMethodStats>)> {
    let mut total_groups = 0;
    let mut method_stats = Vec::new();
    let matching_methods = 5; // Total number of matching methods
    let mut completed_methods = 0;

    // Email matching (highest precision)
    info!(
        "Running email matching [{}/{}]",
        completed_methods + 1,
        matching_methods
    );
    let start = Instant::now();
    let email_result = matching::email::find_matches(pool)
        .await
        .context("Failed during email matching")?;

    total_groups += email_result.groups_created;
    method_stats.push(email_result.stats);
    completed_methods += 1;
    info!(
        "Email matching created {} groups in {:.2?} [{}/{}] ({:.0}%)",
        email_result.groups_created,
        start.elapsed(),
        completed_methods,
        matching_methods,
        (completed_methods as f32 / matching_methods as f32) * 100.0
    );

    // Phone matching
    info!(
        "Running phone matching [{}/{}]",
        completed_methods + 1,
        matching_methods
    );
    let start = Instant::now();
    let phone_result = matching::phone::find_matches(pool)
        .await
        .context("Failed during phone matching")?;

    total_groups += phone_result.groups_created;
    method_stats.push(phone_result.stats);
    completed_methods += 1;
    info!(
        "Phone matching created {} groups in {:.2?} [{}/{}] ({:.0}%)",
        phone_result.groups_created,
        start.elapsed(),
        completed_methods,
        matching_methods,
        (completed_methods as f32 / matching_methods as f32) * 100.0
    );

    // URL matching
    info!(
        "Running URL matching [{}/{}]",
        completed_methods + 1,
        matching_methods
    );
    let start = Instant::now();
    let url_result = matching::url::find_matches(pool)
        .await
        .context("Failed during URL matching")?;

    total_groups += url_result.groups_created;
    method_stats.push(url_result.stats);
    completed_methods += 1;
    info!(
        "URL matching created {} groups in {:.2?} [{}/{}] ({:.0}%)",
        url_result.groups_created,
        start.elapsed(),
        completed_methods,
        matching_methods,
        (completed_methods as f32 / matching_methods as f32) * 100.0
    );

    // Address matching
    info!(
        "Running address matching [{}/{}]",
        completed_methods + 1,
        matching_methods
    );
    let start = Instant::now();
    let address_result = matching::address::find_matches(pool)
        .await
        .context("Failed during address matching")?;

    total_groups += address_result.groups_created;
    method_stats.push(address_result.stats);
    completed_methods += 1;
    info!(
        "Address matching created {} groups in {:.2?} [{}/{}] ({:.0}%)",
        address_result.groups_created,
        start.elapsed(),
        completed_methods,
        matching_methods,
        (completed_methods as f32 / matching_methods as f32) * 100.0
    );

    // Geospatial matching with service similarity checks
    info!(
        "Running enhanced geospatial matching with service similarity [{}/{}]",
        completed_methods + 1,
        matching_methods
    );
    let start = Instant::now();
    let geospatial_match_result = matching::geospatial::find_matches(pool)
        .await
        .context("Failed during geospatial matching")?;
    total_groups += geospatial_match_result.groups_created;
    completed_methods += 1;

    // Add the geospatial stats
    method_stats.push(geospatial_match_result.stats);

    info!(
        "Enhanced geospatial matching created {} groups in {:.2?} [{}/{}] ({:.0}%)",
        geospatial_match_result.groups_created,
        start.elapsed(),
        completed_methods,
        matching_methods,
        (completed_methods as f32 / matching_methods as f32) * 100.0
    );

    Ok((total_groups, method_stats))
}

async fn consolidate_clusters(pool: &PgPool) -> Result<usize> {
    // Get the unassigned group count before we start
    let conn = pool.get().await.context("Failed to get DB connection")?;

    let unassigned_query = "SELECT COUNT(*) FROM entity_group WHERE group_cluster_id IS NULL";
    let groups_row = conn
        .query_one(unassigned_query, &[])
        .await
        .context("Failed to count groups")?;
    let unassigned_groups: i64 = groups_row.get(0);

    info!(
        "Found {} groups that need cluster assignment",
        unassigned_groups
    );

    // Call the actual implementation
    let clusters = consolidate_clusters::process_clusters(pool)
        .await
        .context("Failed to process clusters")?;

    Ok(clusters)
}

// async fn match_services(pool: &PgPool) -> Result<usize> {
//     // Get the cluster count before we start
//     let conn = pool.get().await.context("Failed to get DB connection")?;

//     let clusters_query = "SELECT COUNT(*) FROM group_cluster";
//     let clusters_row = conn
//         .query_one(clusters_query, &[])
//         .await
//         .context("Failed to count clusters")?;
//     let cluster_count: i64 = clusters_row.get(0);

//     info!("Processing {} clusters for service matching", cluster_count);

//     // Check if there are any existing matches
//     let existing_matches_query = "SELECT COUNT(*) FROM service_match";
//     let existing_row = conn
//         .query_one(existing_matches_query, &[])
//         .await
//         .context("Failed to count existing matches")?;
//     let existing_matches: i64 = existing_row.get(0);

//     if existing_matches > 0 {
//         info!("Found {} existing service matches", existing_matches);
//     }

//     // Call the actual implementation
//     let matches = find_service_matches(pool)
//         .await
//         .context("Failed to find service matches")?;

//     info!("Created {} new service matches", matches);

//     Ok(matches)
// }
