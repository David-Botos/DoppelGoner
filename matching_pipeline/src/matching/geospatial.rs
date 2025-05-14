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

// Helper function (can be shared or made local like in email.rs)
// This should ideally be in a shared db utility module if used by multiple matchers.
async fn insert_single_entity_group_pair(
    pool: &PgPool,
    entity_id_1: &EntityId,
    entity_id_2: &EntityId,
    method_type: &MatchMethodType,
    match_values: &MatchValues,
    confidence_score: f64,
    pre_rl_confidence_score: f64,
) -> Result<EntityGroupId> {
    let mut conn = pool
        .get()
        .await
        .context("Failed to get DB connection from pool for single insert")?;
    let tx = conn
        .transaction()
        .await
        .context("Failed to start transaction for single entity_group insert")?;

    let new_entity_group_id = EntityGroupId(Uuid::new_v4().to_string());
    let now = Utc::now().naive_utc();
    let match_values_json = serde_json::to_value(match_values)
        .context("Failed to serialize match_values for single insert")?;

    // Ensure entity_id_1 < entity_id_2 for the uq_entity_pair_method constraint
    let (id1, id2, mv_json) = if entity_id_1.0 < entity_id_2.0 {
        (entity_id_1, entity_id_2, match_values_json)
    } else {
        warn!("Geospatial: Pair ({},{}) was not pre-ordered. This should ideally be handled by the query.", entity_id_1.0, entity_id_2.0);
        (entity_id_2, entity_id_1, match_values_json)
    };

    let insert_query = "
        INSERT INTO public.entity_group
        (id, entity_id_1, entity_id_2, method_type, match_values, confidence_score, pre_rl_confidence_score, created_at, updated_at, version)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 1)
    ";

    match tx
        .execute(
            insert_query,
            &[
                &new_entity_group_id.0,
                &id1.0,
                &id2.0,
                &method_type.as_str(),
                &mv_json,
                &confidence_score,
                &pre_rl_confidence_score,
                &now,
                &now,
            ],
        )
        .await
    {
        Ok(_) => {
            tx.commit()
                .await
                .context("Failed to commit transaction for single entity_group insert")?;
            Ok(new_entity_group_id)
        }
        Err(e) => {
            let r_err = tx.rollback().await;
            if let Err(rollback_err) = r_err {
                error!("Failed to rollback transaction after insert error for pair ({}, {}), method {}: {}. Original error: {}",
                       id1.0, id2.0, method_type.as_str(), rollback_err, e);
            } else {
                error!(
                    "Rolled back transaction for pair ({}, {}), method {}. Insert error: {}",
                    id1.0,
                    id2.0,
                    method_type.as_str(),
                    e
                );
            }
            Err(anyhow!(e).context(format!(
                "DB error inserting entity_group for pair ({}, {}), method {}",
                id1.0,
                id2.0,
                method_type.as_str()
            )))
        }
    }
}

pub async fn find_matches(
    pool: &PgPool,
    reinforcement_orchestrator: Option<&Arc<Mutex<MatchingOrchestrator>>>, // Changed to Arc
    pipeline_run_id: &str,
) -> Result<AnyMatchResult> {
    info!(
        "Starting pairwise geospatial matching (run ID: {}){}...",
        pipeline_run_id,
        if reinforcement_orchestrator.is_some() {
            " with ML guidance"
        } else {
            ""
        }
    );
    let start_time = Instant::now();

    let mut initial_read_conn = pool
        .get()
        .await
        .context("Failed to get DB connection for initial geospatial matching reads")?;

    debug!("Fetching existing geospatial-matched pairs...");
    let existing_pairs_query = "
        SELECT entity_id_1, entity_id_2
        FROM public.entity_group
        WHERE method_type = $1";
    let existing_pair_rows = initial_read_conn
        .query(
            existing_pairs_query,
            &[&MatchMethodType::Geospatial.as_str()],
        )
        .await
        .context("Failed to query existing geospatial-matched pairs")?;

    let mut existing_processed_pairs: HashSet<(EntityId, EntityId)> = HashSet::new();
    for row in existing_pair_rows {
        let id1_str: String = row.get("entity_id_1");
        let id2_str: String = row.get("entity_id_2");
        // The insert helper should handle order, but good to be consistent here too
        if id1_str < id2_str {
            existing_processed_pairs.insert((EntityId(id1_str), EntityId(id2_str)));
        } else {
            existing_processed_pairs.insert((EntityId(id2_str), EntityId(id1_str)));
        }
    }
    info!(
        "Found {} existing geospatial-matched pairs.",
        existing_processed_pairs.len()
    );

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
    debug!("Executing geospatial candidate query...");
    let candidate_rows = initial_read_conn
        .query(geo_candidates_query, &[&METERS_IN_0_15_MILES])
        .await
        .context("Geospatial candidate query failed")?;
    info!(
        "Found {} potential geospatial pairs from DB query.",
        candidate_rows.len()
    );

    drop(initial_read_conn); // Release connection to pool

    let mut new_pairs_created_count = 0;
    let mut entities_in_new_pairs: HashSet<EntityId> = HashSet::new();
    let mut confidence_scores_for_stats: Vec<f64> = Vec::new();
    let mut individual_transaction_errors = 0;

    for row in candidate_rows {
        let entity_id1_str: String = row.get("entity_id_1_str");
        let entity_id2_str: String = row.get("entity_id_2_str");

        let e1_id = EntityId(entity_id1_str);
        let e2_id = EntityId(entity_id2_str);

        // Query ensures e1_id.0 < e2_id.0
        if existing_processed_pairs.contains(&(e1_id.clone(), e2_id.clone())) {
            debug!(
                "Pair ({}, {}) already processed by geospatial method. Skipping.",
                e1_id.0, e2_id.0
            );
            continue;
        }

        let lat1: f64 = row.get("lat1");
        let lon1: f64 = row.get("lon1");
        let lat2: f64 = row.get("lat2");
        let lon2: f64 = row.get("lon2");
        let distance_meters: f64 = row.get("distance_meters");

        let mut final_confidence_score = 0.85; // Default if no RL
        let mut predicted_method_type_from_ml = MatchMethodType::Geospatial; // Default to current method
        let mut features_for_logging: Option<Vec<f64>> = None;
        let mut ml_prediction_confidence_for_logging: Option<f64> = None;

        let pre_rl_confidence_score = final_confidence_score;

        if let Some(orchestrator_arc_mutex) = reinforcement_orchestrator {
            // Note: `extract_pair_context` is a static method on MatchingOrchestrator
            match MatchingOrchestrator::extract_pair_context(pool, &e1_id, &e2_id).await {
                Ok(features) => {
                    features_for_logging = Some(features.clone());
                    let orchestrator_guard = orchestrator_arc_mutex.lock().await;
                    match orchestrator_guard.predict_method_with_context(&features) {
                        Ok((predicted_method, rl_conf)) => {
                            predicted_method_type_from_ml = predicted_method;
                            final_confidence_score = rl_conf;
                            ml_prediction_confidence_for_logging = Some(rl_conf);
                            info!("ML guidance for geo pair ({}, {}): Predicted Method: {:?}, Confidence: {:.4}", e1_id.0, e2_id.0, predicted_method_type_from_ml, final_confidence_score);
                        }
                        Err(e) => {
                            warn!("ML prediction failed for geo pair ({}, {}): {}. Using default confidence for this method.", e1_id.0, e2_id.0, e);
                            // Use a default confidence specific to geospatial if ML fails,
                            // or let the predefined 'final_confidence_score' (0.85) stand.
                        }
                    }
                }
                Err(e) => {
                    warn!("Context extraction failed for geo pair ({}, {}): {}. Using default confidence.", e1_id.0, e2_id.0, e);
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

        match insert_single_entity_group_pair(
            pool,
            &e1_id,
            &e2_id,
            &MatchMethodType::Geospatial,
            &match_values,
            final_confidence_score,
            pre_rl_confidence_score,
        )
        .await
        {
            Ok(new_entity_group_id) => {
                new_pairs_created_count += 1;
                entities_in_new_pairs.insert(e1_id.clone());
                entities_in_new_pairs.insert(e2_id.clone());
                confidence_scores_for_stats.push(final_confidence_score);
                existing_processed_pairs.insert((e1_id.clone(), e2_id.clone())); // Add to processed set

                info!(
                    "SUCCESS: Created new geospatial pair group {} for ({}, {}) with distance {:.2}m, final confidence: {:.4}",
                    new_entity_group_id.0, e1_id.0, e2_id.0, distance_meters, final_confidence_score
                );

                if let Some(orchestrator_arc_mutex) = reinforcement_orchestrator {
                    let mut orchestrator_guard = orchestrator_arc_mutex.lock().await;
                    if let Err(e) = orchestrator_guard
                        .log_match_result(
                            pool,
                            &e1_id,
                            &e2_id,
                            &MatchMethodType::Geospatial, // Log that Geospatial method led to this
                            final_confidence_score,
                            true, // Assuming match made by system is 'correct' for logging
                            features_for_logging.as_ref(),
                            Some(&predicted_method_type_from_ml), // What ML thought
                            ml_prediction_confidence_for_logging, // ML's confidence in its own prediction
                        )
                        .await
                    {
                        warn!(
                            "Failed to log geospatial match result to orchestrator for ({},{}): {}",
                            e1_id.0, e2_id.0, e
                        );
                    }
                }

                // Suggest review for low-confidence matches
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
                        "entity_group_id": &new_entity_group_id.0,
                        "rl_predicted_method": predicted_method_type_from_ml.as_str(),
                        "rl_confidence_in_prediction": ml_prediction_confidence_for_logging,
                    });
                    let reason_message = format!(
                        "Pair ({}, {}) matched by Geospatial (distance: {:.2}m) with RL-derived confidence ({:.4}). RL predicted method: {:?}.",
                        e1_id.0, e2_id.0, distance_meters, final_confidence_score, predicted_method_type_from_ml
                    );

                    let suggestion = NewSuggestedAction {
                        pipeline_run_id: Some(pipeline_run_id.to_string()),
                        action_type: ActionType::ReviewEntityInGroup.as_str().to_string(),
                        entity_id: None,
                        group_id_1: Some(new_entity_group_id.0.clone()),
                        group_id_2: None,
                        cluster_id: None,
                        triggering_confidence: Some(final_confidence_score),
                        details: Some(details_json),
                        reason_code: Some("LOW_RL_CONFIDENCE_PAIR".to_string()),
                        reason_message: Some(reason_message),
                        priority,
                        status: SuggestionStatus::PendingReview.as_str().to_string(),
                        reviewer_id: None,
                        reviewed_at: None,
                        review_notes: None,
                    };
                    // Insert suggestion (consider batching these for performance)
                    let mut temp_conn_sugg = pool
                        .get()
                        .await
                        .context("Failed to get temp conn for geo suggestion")?;
                    let sugg_tx = temp_conn_sugg
                        .transaction()
                        .await
                        .context("Failed to start geo suggestion tx")?;
                    if let Err(e) = db::insert_suggestion(&sugg_tx, &suggestion).await {
                        warn!("Failed to log suggestion for low confidence geo pair ({}, {}): {}. Suggestion: {:?}", e1_id.0, e2_id.0, e, suggestion);
                        if let Err(rb_err) = sugg_tx.rollback().await {
                            error!("Failed to rollback sugg_tx: {}", rb_err);
                        }
                    } else {
                        if let Err(c_err) = sugg_tx.commit().await {
                            error!("Failed to commit sugg_tx: {}", c_err);
                        }
                    }
                }
            }
            Err(e) => {
                individual_transaction_errors += 1;
                // Error is logged by insert_single_entity_group_pair
            }
        }
    }

    if individual_transaction_errors > 0 {
        warn!(
            "Encountered {} errors during individual geospatial pair transaction attempts.",
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
        "Pairwise geospatial matching complete in {:.2?}: created {} new pairs ({} errors), involving {} unique entities.",
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
