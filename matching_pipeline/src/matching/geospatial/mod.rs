// src/matching/geospatial/mod.rs

// Main module file that exports public functionality and orchestrates the geospatial matching
mod config;
mod core;
mod database;
mod postgis;
mod service_utils;
mod utils;

use anyhow::{Context, Result};
use chrono::Utc;
use log::{info, warn};
use service_utils::SERVICE_SIMILARITY_THRESHOLD;
use std::collections::HashSet;
use std::time::Instant;
use tokio::sync::Mutex;

use crate::db::PgPool;
use crate::models::{MatchMethodType, MatchResult};
use crate::reinforcement;
use crate::results::{EnhancedGeospatialMatchResult, MatchMethodStats};

// Constants for batch processing
const BATCH_SIZE: usize = 50; // Process 50 locations per batch
const MAX_ERRORS_BEFORE_ABORT: usize = 3; // Maximum number of errors before aborting

/// Main entry point for geospatial matching
pub async fn find_matches(
    pool: &PgPool,
    reinforcement_orchestrator: Option<&Mutex<reinforcement::MatchingOrchestrator>>,
) -> Result<EnhancedGeospatialMatchResult> {
    info!(
        "Starting geospatial matching with semantic service similarity check{}...",
        if reinforcement_orchestrator.is_some() {
            " and ML guidance"
        } else {
            ""
        }
    );
    let start_time = Instant::now();

    let conn = pool
        .get()
        .await
        .context("Failed to get DB connection for geospatial matching")?;

    // Get all entities that are already part of a geospatial match group
    info!("Finding entities already processed by geospatial matching");
    let processed_entities = database::get_processed_entities(&conn).await?;

    info!(
        "Found {} entities already processed by geospatial matching",
        processed_entities.len()
    );

    // Query for unprocessed locations
    let location_results = database::get_unprocessed_locations(&conn, &processed_entities).await?;

    if location_results.locations.is_empty() {
        info!("No new locations to process, finishing early");
        return Ok(EnhancedGeospatialMatchResult {
            groups_created: 0,
            stats: MatchMethodStats {
                method_type: MatchMethodType::Custom(
                    "geospatial_with_service_similarity".to_string(),
                ),
                groups_created: 0,
                entities_matched: 0,
                avg_confidence: 0.0,
                avg_group_size: 0.0,
            },
            rejected_by_service_similarity: 0,
            avg_service_similarity: 0.0,
            service_similarity_threshold: SERVICE_SIMILARITY_THRESHOLD,
        });
    }

    info!(
        "Found {} location records for unprocessed entities",
        location_results.locations.len()
    );
    info!(
        "{}",
        if location_results.has_postgis {
            "Using PostGIS for geospatial operations"
        } else {
            "PostGIS not available, using fallback calculations"
        }
    );

    // Load existing groups with their centroids
    info!("Loading existing geospatial groups with their centroids");
    let group_results = database::get_existing_groups(&conn, location_results.has_postgis).await?;

    info!(
        "Found {} existing geospatial groups",
        group_results.groups.len()
    );

    // Process in batches to avoid transaction bloat and memory issues
    let total_locations = location_results.locations.len();
    let mut overall_result = MatchResult {
        entities_added: 0,
        entities_skipped: 0,
        groups_created: 0,
        processed_entities: HashSet::new(),
    };
    let mut error_count = 0;

    // Track statistics for reporting
    let mut confidence_scores = Vec::new();
    let mut group_sizes = Vec::new();
    let mut total_entities_matched = 0;

    // Track service similarity statistics
    let mut service_similarity_scores = Vec::new();
    let mut rejected_by_service_similarity = 0;

    // Process matching to existing groups first (all batches)
    if !group_results.groups.is_empty() {
        info!(
            "Processing matches to existing groups in batches of {}",
            BATCH_SIZE
        );

        for (batch_index, location_batch) in
            location_results.locations.chunks(BATCH_SIZE).enumerate()
        {
            let batch_start = Instant::now();
            info!(
                "Processing batch {}/{} (locations {}-{} of {})",
                batch_index + 1,
                (total_locations + BATCH_SIZE - 1) / BATCH_SIZE,
                batch_index * BATCH_SIZE + 1,
                (batch_index * BATCH_SIZE + location_batch.len()).min(total_locations),
                total_locations
            );

            // Begin transaction for this batch
            let mut batch_conn = pool
                .get()
                .await
                .context("Failed to get connection for batch")?;
            let tx = batch_conn
                .transaction()
                .await
                .context("Failed to start transaction for batch")?;

            // Prepare database statements
            let db_statements = database::prepare_statements(&tx).await?;

            let now = Utc::now().naive_utc();

            // Skip entities already processed in previous batches
            let filtered_batch: Vec<_> = location_batch
                .iter()
                .filter(|(entity_id, _, _)| !overall_result.processed_entities.contains(entity_id))
                .cloned()
                .collect();

            if filtered_batch.is_empty() {
                info!(
                    "Batch {} has no unprocessed entities, skipping",
                    batch_index + 1
                );
                continue;
            }

            info!(
                "Batch {} has {} entities to process",
                batch_index + 1,
                filtered_batch.len()
            );

            // Process this batch - pass ML orchestrator to core function
            match core::match_to_existing_groups(
                &tx,
                &db_statements,
                &filtered_batch,
                &group_results,
                location_results.has_postgis,
                &now,
                pool,                       // Pass pool for ML calls
                reinforcement_orchestrator, // Pass ML orchestrator
            )
            .await
            {
                Ok(batch_result) => {
                    // Collect statistics for this batch
                    if let Some((avg_confidence, avg_group_size, entities_in_batch)) =
                        collect_batch_stats(&tx, &batch_result).await?
                    {
                        confidence_scores.push(avg_confidence);
                        group_sizes.push(avg_group_size);
                        total_entities_matched += entities_in_batch;
                    }

                    // Collect service similarity statistics
                    let batch_service_stats =
                        collect_service_similarity_stats(&tx, &batch_result).await?;
                    rejected_by_service_similarity += batch_service_stats.0;
                    service_similarity_scores.extend(batch_service_stats.1);

                    // Commit this batch transaction
                    tx.commit()
                        .await
                        .context("Failed to commit batch transaction")?;

                    // Update overall results
                    overall_result.entities_added += batch_result.entities_added;
                    overall_result.entities_skipped += batch_result.entities_skipped;
                    overall_result
                        .processed_entities
                        .extend(batch_result.processed_entities);

                    info!(
                        "Batch {} complete: added {} entities to existing groups, skipped {} in {:.2?}",
                        batch_index + 1,
                        batch_result.entities_added,
                        batch_result.entities_skipped,
                        batch_start.elapsed()
                    );
                }
                Err(err) => {
                    // Log error and roll back transaction
                    warn!("Error processing batch {}: {}", batch_index + 1, err);
                    error_count += 1;

                    if error_count >= MAX_ERRORS_BEFORE_ABORT {
                        return Err(anyhow::anyhow!(
                            "Aborting after {} consecutive errors in batch processing",
                            error_count
                        ));
                    }
                }
            }
        }
    }

    // Now process remaining entities that need new groups
    // Filter the locations to only include those not yet processed
    let remaining_locations: Vec<_> = location_results
        .locations
        .into_iter()
        .filter(|(entity_id, _, _)| !overall_result.processed_entities.contains(entity_id))
        .collect();

    info!(
        "{} entities added to existing groups, {} remaining to process",
        overall_result.entities_added,
        remaining_locations.len()
    );

    // Process remaining entities to form new groups (also in batches)
    if !remaining_locations.is_empty() {
        info!("Processing new group creation in batches of {}", BATCH_SIZE);

        for (batch_index, location_batch) in remaining_locations.chunks(BATCH_SIZE).enumerate() {
            let batch_start = Instant::now();
            info!(
                "Processing new groups batch {}/{} ({} locations)",
                batch_index + 1,
                (remaining_locations.len() + BATCH_SIZE - 1) / BATCH_SIZE,
                location_batch.len()
            );

            // Begin transaction for this batch
            let mut batch_conn = pool
                .get()
                .await
                .context("Failed to get connection for batch")?;
            let tx = batch_conn
                .transaction()
                .await
                .context("Failed to start transaction for batch")?;

            // Prepare database statements
            let db_statements = database::prepare_statements(&tx).await?;

            let now = Utc::now().naive_utc();

            // Process this batch - pass ML orchestrator to core function
            match core::create_new_groups(
                &tx,
                &db_statements,
                location_batch,
                location_results.has_postgis,
                &now,
                pool,                       // Pass pool for ML calls
                reinforcement_orchestrator, // Pass ML orchestrator
            )
            .await
            {
                Ok(batch_result) => {
                    // Collect statistics for this batch of new groups
                    if let Some((avg_confidence, avg_group_size, entities_in_batch)) =
                        collect_new_groups_stats(&tx, &batch_result).await?
                    {
                        confidence_scores.push(avg_confidence);
                        group_sizes.push(avg_group_size);
                        total_entities_matched += entities_in_batch;
                    }

                    // Collect service similarity statistics
                    let batch_service_stats =
                        collect_service_similarity_stats(&tx, &batch_result).await?;
                    rejected_by_service_similarity += batch_service_stats.0;
                    service_similarity_scores.extend(batch_service_stats.1);

                    // Commit this batch transaction
                    tx.commit()
                        .await
                        .context("Failed to commit batch transaction")?;

                    // Update overall results
                    overall_result.entities_added += batch_result.entities_added;
                    overall_result.groups_created += batch_result.groups_created;
                    overall_result
                        .processed_entities
                        .extend(batch_result.processed_entities);

                    info!(
                        "New groups batch {} complete: created {} groups with {} entities in {:.2?}",
                        batch_index + 1,
                        batch_result.groups_created,
                        batch_result.entities_added,
                        batch_start.elapsed()
                    );
                }
                Err(err) => {
                    // Log error and roll back transaction
                    warn!(
                        "Error processing new groups batch {}: {}",
                        batch_index + 1,
                        err
                    );
                    error_count += 1;

                    if error_count >= MAX_ERRORS_BEFORE_ABORT {
                        return Err(anyhow::anyhow!(
                            "Aborting after {} consecutive errors in batch processing",
                            error_count
                        ));
                    }
                }
            }
        }
    }

    // Calculate average confidence score and group size
    let avg_confidence: f64 = if !confidence_scores.is_empty() {
        (confidence_scores.iter().sum::<f64>() / confidence_scores.len() as f64).into()
    } else {
        0.8 // Default confidence for geospatial matching
    };

    let avg_group_size: f64 = if !group_sizes.is_empty() {
        (group_sizes.iter().sum::<f64>() / group_sizes.len() as f64).into()
    } else {
        0.0
    };

    // Calculate average service similarity
    let avg_service_similarity: f64 = if !service_similarity_scores.is_empty() {
        service_similarity_scores.iter().sum::<f64>() / service_similarity_scores.len() as f64
    } else {
        0.0
    };

    // Create method stats
    let method_stats = MatchMethodStats {
        method_type: MatchMethodType::Custom("geospatial_with_service_similarity".to_string()),
        groups_created: overall_result.groups_created,
        entities_matched: total_entities_matched,
        avg_confidence,
        avg_group_size,
    };

    let elapsed = start_time.elapsed();
    info!(
        "Geospatial matching with service similarity complete: created {} new entity groups and added {} entities to existing groups in {:.2?}",
        overall_result.groups_created, overall_result.entities_added, elapsed
    );

    info!(
        "Service similarity statistics: {} potential matches rejected, average similarity score: {:.4}",
        rejected_by_service_similarity, avg_service_similarity
    );

    Ok(EnhancedGeospatialMatchResult {
        groups_created: overall_result.groups_created,
        stats: method_stats,
        rejected_by_service_similarity,
        avg_service_similarity,
        service_similarity_threshold: SERVICE_SIMILARITY_THRESHOLD,
    })
}

/// Collect statistics for a batch of entities added to existing groups
/// Returns (avg_confidence, avg_group_size, total_entities) if any entities were added
async fn collect_batch_stats(
    tx: &tokio_postgres::Transaction<'_>,
    batch_result: &MatchResult,
) -> Result<Option<(f64, f64, usize)>> {
    if batch_result.entities_added == 0 {
        return Ok(None);
    }

    // Default confidence for geospatial matches is relatively high
    let avg_confidence = 0.8;

    // Query to get the sizes of the groups these entities were added to
    let query = "
        SELECT 
            AVG(eg.entity_count) as avg_size
        FROM 
            entity_group eg
        JOIN 
            group_method gm ON eg.id = gm.entity_group_id
        WHERE 
            gm.method_type = 'geospatial'
            AND eg.updated_at >= NOW() - INTERVAL '5 minutes'
    ";

    let row = tx
        .query_one(query, &[])
        .await
        .context("Failed to query group statistics")?;
    let avg_group_size: f64 = row.get("avg_size");

    Ok(Some((
        avg_confidence,
        avg_group_size,
        batch_result.entities_added,
    )))
}

/// Collect statistics for a batch of newly created groups
/// Returns (avg_confidence, avg_group_size, total_entities) if any groups were created
async fn collect_new_groups_stats(
    tx: &tokio_postgres::Transaction<'_>,
    batch_result: &MatchResult,
) -> Result<Option<(f64, f64, usize)>> {
    if batch_result.groups_created == 0 {
        return Ok(None);
    }

    // Default confidence for geospatial matches is relatively high
    let avg_confidence: f64 = 0.8;

    // For new groups, we know the group size is just entities_added / groups_created
    let avg_group_size: f64 =
        (batch_result.entities_added as f32 / batch_result.groups_created as f32).into();

    Ok(Some((
        avg_confidence,
        avg_group_size,
        batch_result.entities_added,
    )))
}

/// Collect service similarity statistics for a batch
/// Returns (rejected_count, similarity_scores)
async fn collect_service_similarity_stats(
    tx: &tokio_postgres::Transaction<'_>,
    batch_result: &MatchResult,
) -> Result<(usize, Vec<f64>)> {
    // Query for recorded service similarity scores
    // This assumes we've logged them in the database during matching
    let query = "
        SELECT 
            value
        FROM 
            matching_metadata
        WHERE 
            type = 'service_similarity'
            AND created_at >= NOW() - INTERVAL '5 minutes'
    ";

    let rows = match tx.query(query, &[]).await {
        Ok(rows) => rows,
        Err(e) => {
            warn!("Failed to query service similarity scores: {}", e);
            return Ok((0, Vec::new()));
        }
    };

    let similarity_scores: Vec<f64> = rows
        .iter()
        .filter_map(|row| row.get::<_, Option<f64>>("value"))
        .collect();

    // Query for rejected matches
    let reject_query = "
        SELECT 
            COUNT(*)
        FROM 
            matching_metadata
        WHERE 
            type = 'service_similarity_reject'
            AND created_at >= NOW() - INTERVAL '5 minutes'
    ";

    let rejected_count = match tx.query_one(reject_query, &[]).await {
        Ok(row) => row.get::<_, i64>(0) as usize,
        Err(e) => {
            warn!("Failed to query rejected matches: {}", e);
            0
        }
    };

    Ok((rejected_count, similarity_scores))
}
