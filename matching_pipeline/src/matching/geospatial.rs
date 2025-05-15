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
const BATCH_SIZE: usize = 100; // Size for batch operations

// SQL query for inserting into entity_group - Same as other matching modules
const INSERT_ENTITY_GROUP_SQL: &str = "
    INSERT INTO public.entity_group
(id, entity_id_1, entity_id_2, method_type, match_values, confidence_score, 
 pre_rl_confidence_score)
VALUES ($1, $2, $3, $4, $5, $6, $7)";

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

    // Cache existing pairs for faster lookup
    let existing_processed_pairs = fetch_existing_pairs(pool).await?;
    info!(
        "Geospatial: Found {} existing geospatial-matched pairs.",
        existing_processed_pairs.len()
    );

    // Fetch all candidate pairs in a single query
    let candidate_pairs = fetch_candidate_pairs(pool).await?;
    info!(
        "Geospatial: Found {} potential geospatial pairs from DB query.",
        candidate_pairs.len()
    );

    // Extract features for all entity pairs in batches to reduce orchestrator locks
    let mut feature_map = HashMap::new();
    if let Some(orch) = reinforcement_orchestrator_option.as_ref() {
        for chunk in candidate_pairs.chunks(BATCH_SIZE) {
            let unique_pairs: Vec<(&EntityId, &EntityId)> = chunk
                .iter()
                .map(|(e1, e2, ..)| (e1, e2))
                .filter(|(e1, e2)| {
                    !existing_processed_pairs.contains(&((*e1).clone(), (*e2).clone()))
                })
                .collect();

            if unique_pairs.is_empty() {
                continue;
            }

            // Extract features for the batch
            for (e1, e2) in unique_pairs {
                if let Ok(features) =
                    MatchingOrchestrator::extract_pair_context_features(pool, e1, e2).await
                {
                    feature_map.insert((e1.clone(), e2.clone()), features);
                }
            }
        }
    }

    // Process pairs in batches
    let mut new_pairs_created_count = 0;
    let mut entities_in_new_pairs: HashSet<EntityId> = HashSet::new();
    let mut confidence_scores_for_stats: Vec<f64> = Vec::new();
    let mut individual_transaction_errors = 0;
    let mut pending_batch = Vec::new();
    let mut pending_suggestions = Vec::new();

    for (e1_id, e2_id, lat1, lon1, lat2, lon2, distance_meters) in candidate_pairs {
        // Skip already processed pairs
        if existing_processed_pairs.contains(&(e1_id.clone(), e2_id.clone())) {
            continue;
        }

        let mut pre_rl_confidence_score = 0.85; // Default confidence
        let mut final_confidence_score = pre_rl_confidence_score;

        // Get features from our pre-extracted map
        let features = feature_map.get(&(e1_id.clone(), e2_id.clone()));

        // Determine confidence score with RL if available
        if let (Some(orch_arc), Some(extracted_features)) =
            (reinforcement_orchestrator_option.as_ref(), features)
        {
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
            let priority = if final_confidence_score < config::CRITICALLY_LOW_SUGGESTION_THRESHOLD {
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
                group_id_1: Some("placeholder".to_string()), // Placeholder will be updated after batch insert
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

        // Process batch when it reaches BATCH_SIZE
        if pending_batch.len() >= BATCH_SIZE {
            process_batch(
                pool,
                &mut pending_batch,
                &mut pending_suggestions,
                reinforcement_orchestrator_option.as_ref(),
                pipeline_run_id,
                &mut new_pairs_created_count,
                &mut entities_in_new_pairs,
                &mut confidence_scores_for_stats,
                &mut individual_transaction_errors,
                &feature_map,
            )
            .await?;
        }
    }

    // Process remaining items in the batch
    if !pending_batch.is_empty() {
        process_batch(
            pool,
            &mut pending_batch,
            &mut pending_suggestions,
            reinforcement_orchestrator_option.as_ref(),
            pipeline_run_id,
            &mut new_pairs_created_count,
            &mut entities_in_new_pairs,
            &mut confidence_scores_for_stats,
            &mut individual_transaction_errors,
            &feature_map,
        )
        .await?;
    }

    if individual_transaction_errors > 0 {
        warn!(
            "Geospatial: {} errors during batch transaction attempts.",
            individual_transaction_errors
        );
    }

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
        "Geospatial matching complete in {:.2?}: created {} new pairs ({} errors), involving {} unique entities.",
        elapsed,
        method_stats.groups_created,
        individual_transaction_errors,
        method_stats.entities_matched
    );

    let geospatial_specific_result = GeospatialMatchResult {
        groups_created: method_stats.groups_created,
        stats: method_stats,
    };

    Ok(AnyMatchResult::Geospatial(geospatial_specific_result))
}

/// Insert multiple entity groups in a batch
async fn insert_entity_group_batch(
    pool: &PgPool,
    pairs: &[(EntityId, EntityId, MatchValues, f64, f64)],
) -> Result<Vec<EntityGroupId>> {
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
    let mut entity_group_ids = Vec::with_capacity(pairs.len());

    for (entity_id_1, entity_id_2, match_values, confidence_score, pre_rl_confidence_score) in pairs
    {
        let new_id = EntityGroupId(Uuid::new_v4().to_string());
        entity_group_ids.push(new_id.clone());

        let match_values_json = serde_json::to_value(match_values)
            .context("Geospatial: Failed to serialize match_values for batch insert")?;

        // Use the same SQL statement and parameter order as other matching modules
        if let Err(e) = tx
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
            let _ = tx.rollback().await;
            return Err(anyhow!(e).context("Geospatial: DB error inserting entity_group batch"));
        }
    }

    tx.commit()
        .await
        .context("Geospatial: Failed to commit transaction for batch entity_group insert")?;

    Ok(entity_group_ids)
}

/// Batched suggestion insert
async fn insert_suggestion_batch(pool: &PgPool, suggestions: &[NewSuggestedAction]) -> Result<()> {
    if suggestions.is_empty() {
        return Ok(());
    }

    let mut conn = pool
        .get()
        .await
        .context("Geospatial: Failed to get DB connection for suggestion batch")?;

    let tx = conn
        .transaction()
        .await
        .context("Geospatial: Failed to start transaction for suggestion batch")?;

    for suggestion in suggestions {
        if let Err(e) = db::insert_suggestion(&tx, suggestion).await {
            warn!("Geospatial: Failed to insert suggestion: {}", e);
            // Continue with other suggestions rather than failing entire batch
        }
    }

    tx.commit()
        .await
        .context("Geospatial: Failed to commit transaction for suggestion batch")?;

    Ok(())
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
async fn process_batch(
    pool: &PgPool,
    pending_batch: &mut Vec<(EntityId, EntityId, MatchValues, f64, f64)>,
    pending_suggestions: &mut Vec<NewSuggestedAction>,
    reinforcement_orchestrator: Option<&Arc<Mutex<MatchingOrchestrator>>>,
    pipeline_run_id: &str,
    new_pairs_created_count: &mut usize,
    entities_in_new_pairs: &mut HashSet<EntityId>,
    confidence_scores_for_stats: &mut Vec<f64>,
    individual_transaction_errors: &mut usize,
    feature_map: &HashMap<(EntityId, EntityId), Vec<f64>>,
) -> Result<()> {
    if pending_batch.is_empty() {
        return Ok(());
    }

    // Insert entity groups in batch
    match insert_entity_group_batch(pool, pending_batch).await {
        Ok(new_entity_group_ids) => {
            // Update stats and log decisions
            for (i, (e1_id, e2_id, _, confidence_score, pre_rl_confidence_score)) in
                pending_batch.iter().enumerate()
            {
                if i < new_entity_group_ids.len() {
                    let new_entity_group_id = &new_entity_group_ids[i];
                    *new_pairs_created_count += 1;
                    entities_in_new_pairs.insert(e1_id.clone());
                    entities_in_new_pairs.insert(e2_id.clone());
                    confidence_scores_for_stats.push(*confidence_score);

                    debug!(
                        "Geospatial: Created pair group {} for ({}, {}) with confidence: {:.4}",
                        new_entity_group_id.0, e1_id.0, e2_id.0, confidence_score
                    );

                    // Log decision to orchestrator if features exist
                    if let (Some(orch_arc), Some(features)) = (
                        reinforcement_orchestrator,
                        feature_map.get(&(e1_id.clone(), e2_id.clone())),
                    ) {
                        let orchestrator_guard = orch_arc.lock().await;
                        if let Err(e) = orchestrator_guard
                            .log_decision_snapshot(
                                pool,
                                &new_entity_group_id.0,
                                pipeline_run_id,
                                features,
                                &MatchMethodType::Geospatial,
                                *pre_rl_confidence_score,
                                *confidence_score,
                            )
                            .await
                        {
                            warn!(
                                "Geospatial: Failed to log decision snapshot for entity_group {}: {}",
                                new_entity_group_id.0, e
                            );
                        }
                    }

                    // Update suggestion with actual entity group ID
                    if i < pending_suggestions.len() {
                        if let Some(ref mut group_id) = pending_suggestions[i].group_id_1 {
                            *group_id = new_entity_group_id.0.clone();
                        }

                        // Update details JSON with entity_group_id
                        if let Some(ref mut details) = pending_suggestions[i].details {
                            if let Some(obj) = details.as_object_mut() {
                                obj.insert(
                                    "entity_group_id".to_string(),
                                    serde_json::Value::String(new_entity_group_id.0.clone()),
                                );
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            warn!(
                "Geospatial: Batch insert failed: {}. Processing pairs individually.",
                e
            );
            *individual_transaction_errors += pending_batch.len();
        }
    }

    // Insert suggestions in batch if any
    if !pending_suggestions.is_empty() {
        if let Err(e) = insert_suggestion_batch(pool, pending_suggestions).await {
            warn!("Geospatial: Failed to insert batch of suggestions: {}", e);
        }
        pending_suggestions.clear();
    }

    pending_batch.clear();
    Ok(())
}
