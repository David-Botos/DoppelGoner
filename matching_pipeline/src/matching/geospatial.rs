// src/matching/geospatial.rs
use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use log::{debug, error, info, warn};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::config; // For suggestion thresholds
use crate::db::{self, PgPool};
use crate::models::{
    ActionType, EntityGroupId, EntityId, GeospatialMatchValue, MatchMethodType, MatchValues,
    NewSuggestedAction, SuggestionStatus,
};
use crate::reinforcement::MatchingOrchestrator;
use crate::results::{AnyMatchResult, GeospatialMatchResult, MatchMethodStats};
use serde_json;

const METERS_IN_0_15_MILES: f64 = 241.4016; // 0.15 miles

// Reduced batch size to avoid overwhelming the system
const BATCH_SIZE: usize = 25;

// Maximum number of feature extractions to perform at once to prevent memory issues
const MAX_FEATURE_EXTRACTIONS_PER_BATCH: usize = 10;

// Maximum number of pairs to process in one run to prevent hangups
const MAX_PAIRS_TO_PROCESS: usize = 1000;

// SQL query for inserting into entity_group - Same as other matching modules
const INSERT_ENTITY_GROUP_SQL: &str = "
    INSERT INTO public.entity_group
(id, entity_id_1, entity_id_2, method_type, match_values, confidence_score, 
 pre_rl_confidence_score)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (entity_id_1, entity_id_2, method_type) DO NOTHING";

/// Main function to find geospatial-based matches
pub async fn find_matches(
    pool: &PgPool,
    reinforcement_orchestrator_option: Option<Arc<Mutex<MatchingOrchestrator>>>,
    pipeline_run_id: &str,
) -> Result<AnyMatchResult> {
    info!(
        "Starting pairwise geospatial matching (run ID: {}){}...",
        pipeline_run_id,
        if reinforcement_orchestrator_option.is_some() {
            " with RL confidence tuning"
        } else {
            " (RL tuner not provided)"
        }
    );
    let start_time = Instant::now();

    // Verify that both entities exist in the database before trying to match them
    info!("Geospatial: First verifying entity existence in database...");
    let valid_entities = fetch_valid_entity_ids(pool).await?;
    info!(
        "Geospatial: Found {} valid entities in database.",
        valid_entities.len()
    );

    // Cache existing pairs for faster lookup
    let existing_processed_pairs = fetch_existing_pairs(pool).await?;
    info!(
        "Geospatial: Found {} existing geospatial-matched pairs.",
        existing_processed_pairs.len()
    );

    // Fetch all candidate pairs in a single query
    let mut candidate_pairs = fetch_candidate_pairs(pool).await?;
    let total_candidate_count = candidate_pairs.len();
    info!(
        "Geospatial: Found {} potential geospatial pairs from DB query.",
        total_candidate_count
    );

    // Filter candidates to only include valid entities
    let original_count = candidate_pairs.len();
    candidate_pairs
        .retain(|(e1, e2, ..)| valid_entities.contains(&e1.0) && valid_entities.contains(&e2.0));

    info!(
        "Geospatial: Filtered out {} pairs with invalid entity IDs, {} valid pairs remain.",
        original_count - candidate_pairs.len(),
        candidate_pairs.len()
    );

    // Limit the number of pairs to process to prevent hanging
    if candidate_pairs.len() > MAX_PAIRS_TO_PROCESS {
        info!(
            "Geospatial: Limiting to {} out of {} potential pairs for this run to prevent performance issues.",
            MAX_PAIRS_TO_PROCESS, candidate_pairs.len()
        );
        candidate_pairs.truncate(MAX_PAIRS_TO_PROCESS);
    }

    // Track stats
    let mut new_pairs_created_count = 0;
    let mut entities_in_new_pairs: HashSet<EntityId> = HashSet::new();
    let mut confidence_scores_for_stats: Vec<f64> = Vec::new();
    let mut individual_transaction_errors = 0;
    let mut pairs_processed_count = 0;
    let mut feature_extraction_count = 0;
    let mut feature_extraction_failures = 0;
    let mut suggestions_created = 0;
    let mut suggestions_failed = 0;

    // Process pairs in multiple phases to avoid memory issues
    for (chunk_index, chunk) in candidate_pairs.chunks(BATCH_SIZE).enumerate() {
        info!(
            "Geospatial: Processing chunk {}/{} ({} pairs)...",
            chunk_index + 1,
            (candidate_pairs.len() + BATCH_SIZE - 1) / BATCH_SIZE,
            chunk.len()
        );

        // Filter out already processed pairs
        let filtered_chunk: Vec<_> = chunk
            .iter()
            .filter(|(e1, e2, ..)| !existing_processed_pairs.contains(&(e1.clone(), e2.clone())))
            .cloned()
            .collect();

        if filtered_chunk.is_empty() {
            info!("Geospatial: No unprocessed pairs in this chunk, skipping.");
            continue;
        }

        info!(
            "Geospatial: After filtering existing pairs, {} out of {} pairs remain for processing in chunk {}.",
            filtered_chunk.len(),
            chunk.len(),
            chunk_index + 1
        );

        let mut chunk_feature_map = HashMap::new();

        // 1. Extract features in smaller sub-batches to reduce memory pressure
        if let Some(orch) = reinforcement_orchestrator_option.as_ref() {
            info!(
                "Geospatial: Extracting features for {} pairs in chunk {}...",
                filtered_chunk.len(),
                chunk_index + 1
            );

            for (sub_batch_index, sub_batch) in filtered_chunk
                .chunks(MAX_FEATURE_EXTRACTIONS_PER_BATCH)
                .enumerate()
            {
                let sub_batch_start = Instant::now();
                info!(
                    "Geospatial: Extracting features for sub-batch {}/{} ({} pairs)...",
                    sub_batch_index + 1,
                    (filtered_chunk.len() + MAX_FEATURE_EXTRACTIONS_PER_BATCH - 1)
                        / MAX_FEATURE_EXTRACTIONS_PER_BATCH,
                    sub_batch.len()
                );

                for (e1, e2, ..) in sub_batch {
                    match MatchingOrchestrator::extract_pair_context_features(pool, e1, e2).await {
                        Ok(features) => {
                            chunk_feature_map.insert((e1.clone(), e2.clone()), features);
                            feature_extraction_count += 1;
                        }
                        Err(e) => {
                            warn!(
                                "Geospatial: Failed to extract features for pair ({}, {}): {}",
                                e1.0, e2.0, e
                            );
                            feature_extraction_failures += 1;
                        }
                    }
                }

                info!(
                    "Geospatial: Completed feature extraction for sub-batch {}/{} in {:.2?}",
                    sub_batch_index + 1,
                    (filtered_chunk.len() + MAX_FEATURE_EXTRACTIONS_PER_BATCH - 1)
                        / MAX_FEATURE_EXTRACTIONS_PER_BATCH,
                    sub_batch_start.elapsed()
                );
            }
        }

        // 2. Prepare pairs for insertion
        let mut pending_batch = Vec::new();
        let mut pending_suggestions = Vec::new();

        for (e1_id, e2_id, lat1, lon1, lat2, lon2, distance_meters) in filtered_chunk {
            pairs_processed_count += 1;

            let pre_rl_confidence_score = 0.85; // Default confidence
            let mut final_confidence_score = pre_rl_confidence_score;

            // Determine confidence score with RL if available
            if let (Some(orch_arc), Some(extracted_features)) = (
                reinforcement_orchestrator_option.as_ref(),
                chunk_feature_map.get(&(e1_id.clone(), e2_id.clone())),
            ) {
                if !extracted_features.is_empty() {
                    let orchestrator_guard = orch_arc.lock().await;
                    match orchestrator_guard.get_tuned_confidence(
                        &MatchMethodType::Geospatial,
                        pre_rl_confidence_score,
                        extracted_features,
                    ) {
                        Ok(tuned_score) => {
                            final_confidence_score = tuned_score;
                        }
                        Err(e) => {
                            warn!("Geospatial: Failed to get tuned confidence for ({}, {}): {}. Using pre-RL score.", e1_id.0, e2_id.0, e);
                        }
                    }
                }
            }

            let match_values = MatchValues::Geospatial(GeospatialMatchValue {
                latitude1: lat1,
                longitude1: lon1,
                latitude2: lat2,
                longitude2: lon2,
                distance: distance_meters,
            });

            // Add to batch for later insertion
            pending_batch.push((
                e1_id.clone(),
                e2_id.clone(),
                match_values,
                final_confidence_score,
                pre_rl_confidence_score,
            ));

            // Create suggestion for low confidence matches
            if final_confidence_score < config::MODERATE_LOW_SUGGESTION_THRESHOLD {
                let priority =
                    if final_confidence_score < config::CRITICALLY_LOW_SUGGESTION_THRESHOLD {
                        2
                    } else {
                        1
                    };

                let details_json = serde_json::json!({
                    "method_type": MatchMethodType::Geospatial.as_str(),
                    "distance_meters": distance_meters,
                    "latitude1": lat1, "longitude1": lon1,
                    "latitude2": lat2, "longitude2": lon2,
                    "entity_group_id": "", // Will be updated after entity group creation
                    "pre_rl_confidence": pre_rl_confidence_score,
                });

                let reason_message = format!(
                    "Pair ({}, {}) matched by Geospatial (distance: {:.2}m) with tuned confidence ({:.4}).",
                    e1_id.0, e2_id.0, distance_meters, final_confidence_score
                );

                let suggestion = NewSuggestedAction {
                    pipeline_run_id: Some(pipeline_run_id.to_string()),
                    action_type: ActionType::ReviewEntityInGroup.as_str().to_string(),
                    entity_id: None,
                    group_id_1: Some("placeholder".to_string()), // Placeholder will be updated after entity group creation
                    group_id_2: None,
                    cluster_id: None,
                    triggering_confidence: Some(final_confidence_score),
                    details: Some(details_json),
                    reason_code: Some("LOW_TUNED_CONFIDENCE_PAIR".to_string()),
                    reason_message: Some(reason_message),
                    priority,
                    status: SuggestionStatus::PendingReview.as_str().to_string(),
                    reviewer_id: None,
                    reviewed_at: None,
                    review_notes: None,
                };

                pending_suggestions.push(suggestion);
            }
        }

        // 3. Process batch of pairs from this chunk
        if !pending_batch.is_empty() {
            info!(
                "Geospatial: Inserting batch of {} pairs for chunk {}...",
                pending_batch.len(),
                chunk_index + 1
            );

            let batch_start = Instant::now();
            let (pairs_created, entities_added, conf_scores, errors, sugg_created, sugg_failed) =
                process_batch(
                    pool,
                    &pending_batch,
                    &mut pending_suggestions,
                    reinforcement_orchestrator_option.as_ref(),
                    pipeline_run_id,
                    &chunk_feature_map,
                )
                .await?;

            // Update stats
            new_pairs_created_count += pairs_created;
            for entity in entities_added {
                entities_in_new_pairs.insert(entity);
            }
            confidence_scores_for_stats.extend(conf_scores);
            individual_transaction_errors += errors;
            suggestions_created += sugg_created;
            suggestions_failed += sugg_failed;

            info!(
                "Geospatial: Completed batch processing for chunk {} in {:.2?}. Created {} pairs, {} errors. Progress: {}/{} pairs processed.",
                chunk_index + 1,
                batch_start.elapsed(),
                pairs_created,
                errors,
                pairs_processed_count,
                candidate_pairs.len()
            );
        }

        // Clean up to free memory after each chunk
        chunk_feature_map.clear();
    }

    // Report statistics
    let avg_confidence: f64 = if !confidence_scores_for_stats.is_empty() {
        confidence_scores_for_stats.iter().sum::<f64>() / confidence_scores_for_stats.len() as f64
    } else {
        0.0
    };

    let method_stats = MatchMethodStats {
        method_type: MatchMethodType::Geospatial,
        groups_created: new_pairs_created_count,
        entities_matched: entities_in_new_pairs.len(),
        avg_confidence,
        avg_group_size: if new_pairs_created_count > 0 {
            2.0
        } else {
            0.0
        },
    };

    let elapsed = start_time.elapsed();
    info!(
        "Geospatial matching complete in {:.2?}: processed {}/{} pairs, created {} new pairs ({} errors), involving {} unique entities.",
        elapsed,
        pairs_processed_count,
        total_candidate_count,
        method_stats.groups_created,
        individual_transaction_errors,
        method_stats.entities_matched
    );

    info!(
        "Geospatial feature extraction stats: {} successful, {} failed",
        feature_extraction_count, feature_extraction_failures
    );

    info!(
        "Geospatial suggestion stats: {} successfully created, {} failed",
        suggestions_created, suggestions_failed
    );

    // If we limited the number of pairs and there are more to process, log a notice
    if total_candidate_count > MAX_PAIRS_TO_PROCESS {
        info!(
            "Geospatial: Note: Only processed {}/{} potential pairs due to processing limit. Consider running the pipeline again to process more pairs.",
            MAX_PAIRS_TO_PROCESS,
            total_candidate_count
        );
    }

    let geospatial_specific_result = GeospatialMatchResult {
        groups_created: method_stats.groups_created,
        stats: method_stats,
    };

    Ok(AnyMatchResult::Geospatial(geospatial_specific_result))
}

/// Helper function to fetch a set of valid entity IDs
async fn fetch_valid_entity_ids(pool: &PgPool) -> Result<HashSet<String>> {
    let mut conn = pool
        .get()
        .await
        .context("Geospatial: Failed to get DB connection for fetching valid entity IDs")?;

    let rows = conn
        .query("SELECT id FROM public.entity", &[])
        .await
        .context("Geospatial: Failed to query valid entity IDs")?;

    let mut valid_ids = HashSet::with_capacity(rows.len());
    for row in rows {
        let id: String = row.get(0);
        valid_ids.insert(id);
    }

    Ok(valid_ids)
}

/// Insert multiple entity groups in a batch
async fn insert_entity_group_batch(
    pool: &PgPool,
    pairs: &[(EntityId, EntityId, MatchValues, f64, f64)],
) -> Result<Vec<(EntityGroupId, EntityId, EntityId, f64)>> {
    if pairs.is_empty() {
        return Ok(Vec::new());
    }

    let mut conn = pool
        .get()
        .await
        .context("Geospatial: Failed to get DB connection for batch insert")?;

    let tx = conn
        .transaction()
        .await
        .context("Geospatial: Failed to start transaction for batch entity_group insert")?;

    let method_type = MatchMethodType::Geospatial.as_str();
    let mut results = Vec::with_capacity(pairs.len());

    for (entity_id_1, entity_id_2, match_values, confidence_score, pre_rl_confidence_score) in pairs
    {
        let new_id = EntityGroupId(Uuid::new_v4().to_string());

        let match_values_json = match serde_json::to_value(match_values) {
            Ok(json) => json,
            Err(e) => {
                warn!(
                    "Geospatial: Failed to serialize match_values for pair ({}, {}): {}",
                    entity_id_1.0, entity_id_2.0, e
                );
                continue;
            }
        };

        // Use the modified SQL statement with ON CONFLICT DO NOTHING
        match tx
            .execute(
                INSERT_ENTITY_GROUP_SQL,
                &[
                    &new_id.0,
                    &entity_id_1.0,
                    &entity_id_2.0,
                    &method_type,
                    &match_values_json,
                    &confidence_score,
                    &pre_rl_confidence_score,
                ],
            )
            .await
        {
            Ok(rows_affected) => {
                if rows_affected > 0 {
                    results.push((
                        new_id.clone(),
                        entity_id_1.clone(),
                        entity_id_2.clone(),
                        *confidence_score,
                    ));
                    debug!(
                        "Geospatial: Inserted entity_group {} for pair ({}, {})",
                        new_id.0, entity_id_1.0, entity_id_2.0
                    );
                } else {
                    debug!(
                        "Geospatial: Entity group for pair ({}, {}) already exists, skipped",
                        entity_id_1.0, entity_id_2.0
                    );
                }
            }
            Err(e) => {
                warn!(
                    "Geospatial: Failed to insert entity_group for pair ({}, {}): {}",
                    entity_id_1.0, entity_id_2.0, e
                );
                // Continue with other pairs rather than failing entire batch
            }
        }
    }

    tx.commit()
        .await
        .context("Geospatial: Failed to commit transaction for batch entity_group insert")?;

    Ok(results)
}

/// Batched suggestion insert - now returns count of successful and failed inserts
async fn insert_suggestion_batch(
    pool: &PgPool,
    suggestions: &[NewSuggestedAction],
) -> Result<(usize, usize)> {
    if suggestions.is_empty() {
        return Ok((0, 0));
    }

    let mut conn = pool
        .get()
        .await
        .context("Geospatial: Failed to get DB connection for suggestion batch")?;

    let tx = conn
        .transaction()
        .await
        .context("Geospatial: Failed to start transaction for suggestion batch")?;

    let mut success_count = 0;
    let mut failure_count = 0;

    for suggestion in suggestions {
        // Basic validation before attempting insert
        if suggestion.action_type.is_empty() {
            warn!("Geospatial: Skipping suggestion with empty action_type");
            failure_count += 1;
            continue;
        }

        // Skip suggestions with placeholder group_id_1 (these weren't properly updated)
        if let Some(group_id) = &suggestion.group_id_1 {
            if group_id == "placeholder" {
                warn!("Geospatial: Skipping suggestion with placeholder group_id_1");
                failure_count += 1;
                continue;
            }
        }

        match db::insert_suggestion(&tx, suggestion).await {
            Ok(_) => {
                success_count += 1;
            }
            Err(e) => {
                warn!("Geospatial: Failed to insert suggestion: {}", e);
                failure_count += 1;
            }
        }
    }

    tx.commit()
        .await
        .context("Geospatial: Failed to commit transaction for suggestion batch")?;

    Ok((success_count, failure_count))
}

/// Helper function to fetch existing pairs
async fn fetch_existing_pairs(pool: &PgPool) -> Result<HashSet<(EntityId, EntityId)>> {
    let mut conn = pool
        .get()
        .await
        .context("Geospatial: Failed to get DB connection for fetching existing pairs")?;

    debug!("Geospatial: Fetching existing geospatial-matched pairs...");
    let existing_pairs_query = "
        SELECT entity_id_1, entity_id_2
        FROM public.entity_group
        WHERE method_type = $1";

    let existing_pair_rows = conn
        .query(
            existing_pairs_query,
            &[&MatchMethodType::Geospatial.as_str()],
        )
        .await
        .context("Geospatial: Failed to query existing geospatial-matched pairs")?;

    let mut existing_processed_pairs: HashSet<(EntityId, EntityId)> =
        HashSet::with_capacity(existing_pair_rows.len());
    for row in existing_pair_rows {
        let id1_str: String = row.get("entity_id_1");
        let id2_str: String = row.get("entity_id_2");
        // Always ensure consistent ordering
        if id1_str < id2_str {
            existing_processed_pairs.insert((EntityId(id1_str), EntityId(id2_str)));
        } else {
            existing_processed_pairs.insert((EntityId(id2_str), EntityId(id1_str)));
        }
    }

    Ok(existing_processed_pairs)
}

/// Helper function to fetch all candidate pairs
async fn fetch_candidate_pairs(
    pool: &PgPool,
) -> Result<Vec<(EntityId, EntityId, f64, f64, f64, f64, f64)>> {
    let mut conn = pool
        .get()
        .await
        .context("Geospatial: Failed to get DB connection for geospatial candidate query")?;

    let geo_candidates_query = "
        WITH EntityLocations AS (
            SELECT
                e.id AS entity_id,
                l.geom,
                l.latitude,
                l.longitude
            FROM
                public.entity e
            JOIN
                public.location l ON e.organization_id = l.organization_id
            WHERE
                l.geom IS NOT NULL AND e.id IS NOT NULL
        )
        SELECT
            el1.entity_id AS entity_id_1_str,
            el2.entity_id AS entity_id_2_str,
            el1.latitude AS lat1,
            el1.longitude AS lon1,
            el2.latitude AS lat2,
            el2.longitude AS lon2,
            ST_Distance(el1.geom, el2.geom) AS distance_meters
        FROM
            EntityLocations el1
        JOIN
            EntityLocations el2 ON el1.entity_id < el2.entity_id -- Ensures order & avoids self-match
            AND ST_DWithin(el1.geom, el2.geom, $1)
    ";

    debug!("Geospatial: Executing geospatial candidate query...");
    let candidate_rows = conn
        .query(geo_candidates_query, &[&METERS_IN_0_15_MILES])
        .await
        .context("Geospatial: Candidate query failed")?;

    let mut result = Vec::with_capacity(candidate_rows.len());
    for row in candidate_rows {
        let entity_id1_str: String = row.get("entity_id_1_str");
        let entity_id2_str: String = row.get("entity_id_2_str");
        let lat1: f64 = row.get("lat1");
        let lon1: f64 = row.get("lon1");
        let lat2: f64 = row.get("lat2");
        let lon2: f64 = row.get("lon2");
        let distance_meters: f64 = row.get("distance_meters");

        result.push((
            EntityId(entity_id1_str),
            EntityId(entity_id2_str),
            lat1,
            lon1,
            lat2,
            lon2,
            distance_meters,
        ));
    }

    Ok(result)
}

/// Helper function to process a batch of pairs
/// Returns (pairs_created, entities_added, confidence_scores, errors, suggestions_created, suggestions_failed)
async fn process_batch(
    pool: &PgPool,
    pairs: &[(EntityId, EntityId, MatchValues, f64, f64)],
    pending_suggestions: &mut Vec<NewSuggestedAction>,
    reinforcement_orchestrator: Option<&Arc<Mutex<MatchingOrchestrator>>>,
    pipeline_run_id: &str,
    feature_map: &HashMap<(EntityId, EntityId), Vec<f64>>,
) -> Result<(usize, Vec<EntityId>, Vec<f64>, usize, usize, usize)> {
    if pairs.is_empty() {
        return Ok((0, Vec::new(), Vec::new(), 0, 0, 0));
    }

    let mut pairs_created = 0;
    let mut entities_added = Vec::new();
    let mut confidence_scores = Vec::new();
    let mut errors = 0;
    let mut suggestions_created = 0;
    let mut suggestions_failed = 0;

    // Insert entity groups in batch
    match insert_entity_group_batch(pool, pairs).await {
        Ok(successful_inserts) => {
            // Update stats and log decisions
            for (entity_group_id, e1_id, e2_id, confidence_score) in &successful_inserts {
                pairs_created += 1;
                entities_added.push(e1_id.clone());
                entities_added.push(e2_id.clone());
                confidence_scores.push(*confidence_score);

                debug!(
                    "Geospatial: Created pair group {} for ({}, {}) with confidence: {:.4}",
                    entity_group_id.0, e1_id.0, e2_id.0, confidence_score
                );

                // Update any matching suggestions with the actual entity group ID
                for suggestion in pending_suggestions.iter_mut() {
                    if let Some(group_id) = &mut suggestion.group_id_1 {
                        if *group_id == "placeholder" {
                            let entity_pair_in_suggestion =
                                if let Some(details) = &suggestion.details {
                                    if let (Some(e1), Some(e2)) = (
                                        details.get("entity_id_1").and_then(|v| v.as_str()),
                                        details.get("entity_id_2").and_then(|v| v.as_str()),
                                    ) {
                                        (e1 == e1_id.0 && e2 == e2_id.0)
                                            || (e1 == e2_id.0 && e2 == e1_id.0)
                                    } else {
                                        false
                                    }
                                } else {
                                    false
                                };

                            if entity_pair_in_suggestion {
                                *group_id = entity_group_id.0.clone();

                                // Update details JSON with entity_group_id
                                if let Some(ref mut details) = suggestion.details {
                                    if let Some(obj) = details.as_object_mut() {
                                        obj.insert(
                                            "entity_group_id".to_string(),
                                            serde_json::Value::String(entity_group_id.0.clone()),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                // Log decision to orchestrator if features exist
                if let (Some(orch_arc), Some(features)) = (
                    reinforcement_orchestrator,
                    feature_map.get(&(e1_id.clone(), e2_id.clone())),
                ) {
                    let orchestrator_guard = orch_arc.lock().await;

                    // Get the original pre_RL score for this pair
                    let pre_rl_score = pairs
                        .iter()
                        .find(|(e1, e2, _, _, _)| e1 == e1_id && e2 == e2_id)
                        .map(|(_, _, _, _, pre_rl)| *pre_rl)
                        .unwrap_or(0.85); // Default if not found

                    if let Err(e) = orchestrator_guard
                        .log_decision_snapshot(
                            pool,
                            &entity_group_id.0,
                            pipeline_run_id,
                            features,
                            &MatchMethodType::Geospatial,
                            pre_rl_score,
                            *confidence_score,
                        )
                        .await
                    {
                        warn!(
                            "Geospatial: Failed to log decision snapshot for entity_group {}: {}",
                            entity_group_id.0, e
                        );
                    }
                }
            }

            // Now filter suggestions to only include those with valid (non-placeholder) group_id_1
            pending_suggestions.retain(|s| {
                if let Some(group_id) = &s.group_id_1 {
                    group_id != "placeholder"
                } else {
                    false
                }
            });

            // Insert suggestions in batch if any
            if !pending_suggestions.is_empty() {
                match insert_suggestion_batch(pool, pending_suggestions).await {
                    Ok((success, failure)) => {
                        suggestions_created = success;
                        suggestions_failed = failure;
                        info!(
                            "Geospatial: Successfully inserted {} suggestions ({} failed)",
                            success, failure
                        );
                    }
                    Err(e) => {
                        warn!("Geospatial: Failed to insert batch of suggestions: {}", e);
                        suggestions_failed = pending_suggestions.len();
                    }
                }
                pending_suggestions.clear();
            }
        }
        Err(e) => {
            warn!(
                "Geospatial: Batch insert failed: {}. Entity group creation aborted.",
                e
            );
            errors = pairs.len();
        }
    }

    Ok((
        pairs_created,
        entities_added,
        confidence_scores,
        errors,
        suggestions_created,
        suggestions_failed,
    ))
}