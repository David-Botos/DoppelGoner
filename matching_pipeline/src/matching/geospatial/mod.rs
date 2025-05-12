// src/matching/geospatial/mod.rs

mod config;
mod core;
mod database;
mod postgis;
mod service_utils;
mod utils;

use anyhow::{Context, Result};
use chrono::Utc;
use log::{error, info, warn};
use service_utils::SERVICE_SIMILARITY_THRESHOLD; // Added error for explicit error logging
use std::collections::HashSet;
use std::time::Instant;
use tokio::sync::Mutex;

use crate::db::PgPool;
use crate::models::{EntityId, MatchMethodType, MatchResult}; // Added EntityId for HashSet initialization
use crate::reinforcement;
use crate::results::{AnyMatchResult, EnhancedGeospatialMatchResult, MatchMethodStats};
// Assuming SERVICE_SIMILARITY_THRESHOLD is defined in app_config or geospatial::config
// For clarity, let's assume it's in geospatial::config or app_config.
// If it's from service_utils, ensure that's clear.
// Using app_config as per original core.rs implies it's a general config.
use crate::config as app_config;

const GEO_PIPELINE_BATCH_SIZE: usize = 500;
const MAX_ERRORS_BEFORE_ABORT: usize = 3;

pub async fn find_matches(
    pool: &PgPool,
    reinforcement_orchestrator: Option<&Mutex<reinforcement::MatchingOrchestrator>>,
    pipeline_run_id: &str,
) -> Result<AnyMatchResult> {
    info!(
        "Starting pairwise geospatial matching with RL (run ID: {})...",
        pipeline_run_id
    );
    let overall_start_time = Instant::now();

    let main_conn = pool
        .get()
        .await
        .context("Geospatial: DB conn for initial data fetch")?;

    let existing_geospatial_pairs = database::get_existing_geospatial_pairs(&main_conn).await?;
    info!(
        "Fetched {} existing geospatial pairs.",
        existing_geospatial_pairs.len()
    );

    let all_unpaired_locations_result =
        database::get_locations_for_unpaired_entities(&main_conn).await?;
    let all_unpaired_locations = all_unpaired_locations_result.locations;
    let has_postgis = all_unpaired_locations_result.has_postgis;
    drop(main_conn); // Release connection early

    if all_unpaired_locations.is_empty() {
        info!("No new (unpaired) locations to process for geospatial matching. Skipping.");
        let geospatial_result = EnhancedGeospatialMatchResult {
            groups_created: 0,
            stats: MatchMethodStats {
                method_type: MatchMethodType::Geospatial,
                groups_created: 0,
                entities_matched: 0,
                avg_confidence: 0.0,
                avg_group_size: 0.0,
            },
            // These specific stats might need more intricate collection if they are critical.
            rejected_by_service_similarity: 0,
            avg_service_similarity: 0.0,
            service_similarity_threshold: SERVICE_SIMILARITY_THRESHOLD, // Assuming it's in app_config
        };
        return Ok(AnyMatchResult::Geospatial(geospatial_result));
    }
    info!(
        "Found {} locations for unpaired entities. PostGIS available: {}.",
        all_unpaired_locations.len(),
        has_postgis
    );

    // Initialize MatchResult manually
    let mut cumulative_match_result = MatchResult {
        groups_created: 0,
        entities_matched: 0,
        entities_added: 0,
        entities_skipped: 0,
        processed_entities: HashSet::<EntityId>::new(), // Initialize with an empty HashSet
    };

    let mut error_count = 0;

    // Get DbStatements (SQL strings) once.
    // This no longer requires a transaction or await.
    let db_statements = database::get_statements();

    for (batch_idx, location_chunk) in all_unpaired_locations
        .chunks(GEO_PIPELINE_BATCH_SIZE)
        .enumerate()
    {
        let batch_start_time = Instant::now();
        info!(
            "Geospatial: Processing batch {}/{} ({} locations)",
            batch_idx + 1,
            (all_unpaired_locations.len() + GEO_PIPELINE_BATCH_SIZE - 1) / GEO_PIPELINE_BATCH_SIZE,
            location_chunk.len()
        );

        let mut batch_db_conn = pool
            .get()
            .await
            .context("Geospatial: DB conn for batch transaction")?;
        let tx = batch_db_conn
            .transaction()
            .await
            .context("Geospatial: Start batch transaction")?;

        let now_utc = Utc::now().naive_utc();

        match core::perform_geospatial_pairwise_matching_in_transaction(
            &tx,            // Pass the transaction for this batch
            &db_statements, // Pass the struct with SQL strings
            location_chunk,
            &existing_geospatial_pairs,
            has_postgis,
            &now_utc,
            pool, // Pass the pool for RL context fetching (outside tx)
            reinforcement_orchestrator,
            pipeline_run_id,
        )
        .await
        {
            Ok(batch_run_result) => {
                tx.commit()
                    .await
                    .context("Geospatial: Commit batch transaction")?;

                cumulative_match_result.groups_created += batch_run_result.groups_created;
                for entity_id in batch_run_result.processed_entities.iter() {
                    // processed_entities from batch_run_result are those involved in NEW pairs in THIS batch.
                    cumulative_match_result
                        .processed_entities
                        .insert(entity_id.clone());
                }
                cumulative_match_result.entities_skipped += batch_run_result.entities_skipped;

                info!("Geospatial batch {} completed in {:.2?}. New pairs this batch: {}. Unique entities added to pairs this batch: {}.", batch_idx + 1, batch_start_time.elapsed(), batch_run_result.groups_created, batch_run_result.entities_matched);
            }
            Err(e) => {
                // Transaction will be rolled back automatically on drop if not committed.
                warn!(
                    "Geospatial batch {} failed: {}. Rolling back transaction.",
                    batch_idx + 1,
                    e
                );
                // Log the detailed error context
                error!("Detailed error for batch {}: {:?}", batch_idx + 1, e);
                error_count += 1;
                if error_count >= MAX_ERRORS_BEFORE_ABORT {
                    error!("Geospatial matching aborted after {} errors.", error_count);
                    return Err(anyhow::anyhow!(
                        "Geospatial matching aborted after {} critical batch errors.",
                        error_count
                    )
                    .context(e));
                }
            }
        }
    }

    // Final count of unique entities matched across all batches for this geospatial run.
    cumulative_match_result.entities_matched = cumulative_match_result.processed_entities.len();
    cumulative_match_result.entities_added = cumulative_match_result.entities_matched;

    // For avg_confidence, it's tricky to get an accurate real-time average of newly created pairs
    // without querying them back or accumulating scores. This remains a placeholder.
    // A more accurate approach would be to query entity_group for method='geospatial' and pipeline_run_id=current
    // after all batches are done, then calculate the average.
    let avg_confidence_for_report = if cumulative_match_result.groups_created > 0 {
        // Placeholder: In a real scenario, you might fetch and average actual confidences.
        // For now, let's assume a default if groups were made.
        // Ensure MIN_GEO_PAIR_CONFIDENCE_THRESHOLD is accessible, e.g., from app_config
        app_config::MIN_GEO_PAIR_CONFIDENCE_THRESHOLD + 0.1 // Example
    } else {
        0.0
    };

    let final_method_stats = MatchMethodStats {
        method_type: MatchMethodType::Geospatial,
        groups_created: cumulative_match_result.groups_created,
        entities_matched: cumulative_match_result.entities_matched,
        avg_confidence: avg_confidence_for_report,
        avg_group_size: if cumulative_match_result.groups_created > 0 {
            2.0
        } else {
            0.0
        }, // Assuming pairs
    };

    info!(
        "Geospatial pairwise matching finished in {:.2?}. Total new pairs created: {}. Total unique entities involved in new pairs: {}.",
        overall_start_time.elapsed(),
        final_method_stats.groups_created,
        final_method_stats.entities_matched
    );

    let geospatial_result = EnhancedGeospatialMatchResult {
        groups_created: final_method_stats.groups_created,
        stats: final_method_stats,
        // These detailed stats would need to be explicitly collected from core.rs if required for the final report.
        // For example, core.rs could return a struct that includes these counts.
        rejected_by_service_similarity: 0,
        avg_service_similarity: 0.0,
        service_similarity_threshold: SERVICE_SIMILARITY_THRESHOLD,
    };

    Ok(AnyMatchResult::Geospatial(geospatial_result))
}
