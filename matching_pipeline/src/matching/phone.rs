// src/matching/phone.rs
use anyhow::{Context, Result};
use chrono::Utc;
use log::{debug, info, warn};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::config;
use crate::db::{self, PgPool};
use crate::models::{
    ActionType, EntityGroupId, EntityId, MatchMethodType, MatchValues, NewSuggestedAction,
    PhoneMatchValue, SuggestionStatus,
};
use crate::reinforcement::MatchingOrchestrator;
use crate::results::{AnyMatchResult, MatchMethodStats, PhoneMatchResult};

// SQL query for inserting into entity_group
const INSERT_ENTITY_GROUP_SQL: &str = "
    INSERT INTO public.entity_group
    (id, entity_id_1, entity_id_2, method_type, match_values, confidence_score, created_at, updated_at, version)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 1)";

pub async fn find_matches(
    pool: &PgPool,
    reinforcement_orchestrator: Option<&Mutex<MatchingOrchestrator>>,
    pipeline_run_id: &str,
) -> Result<AnyMatchResult> {
    info!(
        "Starting pairwise phone matching (run ID: {}){}...",
        pipeline_run_id,
        if reinforcement_orchestrator.is_some() {
            " with ML guidance"
        } else {
            ""
        }
    );
    let start_time = Instant::now();

    let conn = pool
        .get()
        .await
        .context("Failed to get DB connection for phone matching")?;

    // 1. Fetch existing phone-matched pairs
    debug!("Fetching existing phone-matched pairs...");
    let existing_pairs_query = "
        SELECT entity_id_1, entity_id_2
        FROM public.entity_group
        WHERE method_type = $1";
    let existing_pair_rows = conn
        .query(existing_pairs_query, &[&MatchMethodType::Phone.as_str()])
        .await
        .context("Failed to query existing phone-matched pairs")?;

    let mut existing_processed_pairs: HashSet<(EntityId, EntityId)> = HashSet::new();
    for row in existing_pair_rows {
        let id1: String = row.get("entity_id_1");
        let id2: String = row.get("entity_id_2");
        if id1 < id2 {
            existing_processed_pairs.insert((EntityId(id1), EntityId(id2)));
        } else {
            existing_processed_pairs.insert((EntityId(id2), EntityId(id1)));
        }
    }
    info!(
        "Found {} existing phone-matched pairs.",
        existing_processed_pairs.len()
    );

    // 2. Fetch Phone Data for all entities
    let phone_query = "
        SELECT e.id as entity_id, p.number, p.extension
        FROM entity e
        JOIN entity_feature ef ON e.id = ef.entity_id
        JOIN phone p ON ef.table_id = p.id AND ef.table_name = 'phone'
        WHERE p.number IS NOT NULL AND p.number != ''
    ";
    debug!("Executing phone query for all entities...");
    let phone_rows = conn
        .query(phone_query, &[])
        .await
        .context("Failed to query entities with phone numbers")?;
    info!(
        "Found {} phone records across all entities.",
        phone_rows.len()
    );

    let mut phone_map: HashMap<String, HashMap<EntityId, (String, Option<String>)>> =
        HashMap::new();
    for row in &phone_rows {
        let entity_id = EntityId(row.get("entity_id"));
        let number: String = row.get("number");
        let extension: Option<String> = row
            .try_get("extension")
            .ok()
            .flatten()
            .filter(|s: &String| !s.is_empty());
        let normalized_phone = normalize_phone(&number);

        if !normalized_phone.is_empty() {
            phone_map
                .entry(normalized_phone)
                .or_default()
                .insert(entity_id, (number, extension));
        }
    }
    info!(
        "Processed {} unique normalized phone numbers.",
        phone_map.len()
    );

    // 3. Database Operations (No transaction spanning all inserts)
    debug!("Starting pairwise phone matching inserts (non-transactional for the whole batch).");

    let now = Utc::now().naive_utc();
    let mut new_pairs_created_count = 0;
    let mut entities_in_new_pairs: HashSet<EntityId> = HashSet::new();
    let mut confidence_scores_for_stats: Vec<f64> = Vec::new();

    // 4. Process each normalized phone and form pairs
    for (normalized_shared_phone, current_entity_map) in phone_map {
        if current_entity_map.len() < 2 {
            continue;
        }

        let entities_sharing_phone: Vec<_> = current_entity_map.iter().collect();

        for i in 0..entities_sharing_phone.len() {
            for j in (i + 1)..entities_sharing_phone.len() {
                let (entity_id1_obj, (original_phone1, original_ext1)) = entities_sharing_phone[i];
                let (entity_id2_obj, (original_phone2, original_ext2)) = entities_sharing_phone[j];

                let (e1_id, e1_orig_phone, e1_orig_ext, e2_id, e2_orig_phone, e2_orig_ext) =
                    if entity_id1_obj.0 < entity_id2_obj.0 {
                        (
                            entity_id1_obj,
                            original_phone1,
                            original_ext1,
                            entity_id2_obj,
                            original_phone2,
                            original_ext2,
                        )
                    } else {
                        (
                            entity_id2_obj,
                            original_phone2,
                            original_ext2,
                            entity_id1_obj,
                            original_phone1,
                            original_ext1,
                        )
                    };

                if existing_processed_pairs.contains(&(e1_id.clone(), e2_id.clone())) {
                    debug!(
                        "Pair ({}, {}) already processed by phone method. Skipping.",
                        e1_id.0, e2_id.0
                    );
                    continue;
                }

                let base_confidence = if e1_orig_ext == e2_orig_ext {
                    0.95
                } else {
                    0.85
                };
                let mut final_confidence_score = base_confidence;
                let mut predicted_method_type_from_ml = MatchMethodType::Phone;
                let mut features_for_logging: Option<Vec<f64>> = None;

                if let Some(orchestrator_mutex) = reinforcement_orchestrator {
                    match MatchingOrchestrator::extract_pair_context(pool, e1_id, e2_id).await {
                        // Pass pool
                        Ok(features) => {
                            features_for_logging = Some(features.clone());
                            let orchestrator_guard = orchestrator_mutex.lock().await;
                            match orchestrator_guard.predict_method_with_context(&features) {
                                Ok((predicted_method, rl_conf)) => {
                                    predicted_method_type_from_ml = predicted_method;
                                    final_confidence_score = rl_conf;
                                    info!("ML guidance for phone pair ({}, {}): Predicted Method: {:?}, Confidence: {:.4}", e1_id.0, e2_id.0, predicted_method_type_from_ml, final_confidence_score);
                                }
                                Err(e) => {
                                    warn!("ML prediction failed for phone pair ({}, {}): {}. Using base confidence {:.2}.", e1_id.0, e2_id.0, e, base_confidence);
                                    // final_confidence_score remains base_confidence
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Context extraction failed for phone pair ({}, {}): {}. Using base confidence {:.2}.", e1_id.0, e2_id.0, e, base_confidence);
                            // final_confidence_score remains base_confidence
                        }
                    }
                }

                let match_values = MatchValues::Phone(PhoneMatchValue {
                    original_phone1: e1_orig_phone.clone(),
                    original_phone2: e2_orig_phone.clone(),
                    normalized_shared_phone: normalized_shared_phone.clone(),
                    extension1: e1_orig_ext.clone(),
                    extension2: e2_orig_ext.clone(),
                });
                let match_values_json = serde_json::to_value(&match_values).with_context(|| {
                    format!(
                        "Failed to serialize phone MatchValues for pair ({}, {})",
                        e1_id.0, e2_id.0
                    )
                })?;

                let new_entity_group_id = EntityGroupId(Uuid::new_v4().to_string());

                let insert_result = conn
                    .execute(
                        INSERT_ENTITY_GROUP_SQL,
                        &[
                            &new_entity_group_id.0,
                            &e1_id.0,
                            &e2_id.0,
                            &MatchMethodType::Phone.as_str(),
                            &match_values_json,
                            &final_confidence_score,
                            &now,
                            &now,
                        ],
                    )
                    .await;

                match insert_result {
                    Ok(_) => {
                        new_pairs_created_count += 1;
                        entities_in_new_pairs.insert(e1_id.clone());
                        entities_in_new_pairs.insert(e2_id.clone());
                        confidence_scores_for_stats.push(final_confidence_score);
                        existing_processed_pairs.insert((e1_id.clone(), e2_id.clone()));

                        info!(
                            "Created new phone pair group {} for ({}, {}) with shared phone '{}', confidence: {:.4}",
                            new_entity_group_id.0, e1_id.0, e2_id.0, normalized_shared_phone, final_confidence_score
                        );

                        if let Some(orchestrator_mutex) = reinforcement_orchestrator {
                            let mut orchestrator_guard = orchestrator_mutex.lock().await;
                            if let Err(e) = orchestrator_guard
                                .log_match_result(
                                    pool,
                                    e1_id, // Clone or ensure EntityId is Copy
                                    e2_id,
                                    &predicted_method_type_from_ml,
                                    final_confidence_score,
                                    true,
                                    features_for_logging.as_ref(),
                                    Some(&predicted_method_type_from_ml),
                                    Some(final_confidence_score),
                                )
                                .await
                            {
                                warn!("Failed to log phone match result to entity_match_pairs for ({},{}): {}", e1_id.0, e2_id.0, e);
                            }
                        }

                        if final_confidence_score < config::MODERATE_LOW_SUGGESTION_THRESHOLD {
                            let priority = if final_confidence_score
                                < config::CRITICALLY_LOW_SUGGESTION_THRESHOLD
                            {
                                2
                            } else {
                                1
                            };
                            let details_json = serde_json::json!({
                                "method_type": MatchMethodType::Phone.as_str(),
                                "matched_value": &normalized_shared_phone,
                                "original_phone1": e1_orig_phone, "extension1": e1_orig_ext,
                                "original_phone2": e2_orig_phone, "extension2": e2_orig_ext,
                                "entity_group_id": &new_entity_group_id.0,
                                "rl_predicted_method": predicted_method_type_from_ml.as_str(),
                            });
                            let reason_message = format!(
                                "Pair ({}, {}) matched by Phone with low RL confidence ({:.4}). RL predicted: {:?}.",
                                e1_id.0, e2_id.0, final_confidence_score, predicted_method_type_from_ml
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
                            if let Err(e) = db::insert_suggestion(&*conn, &suggestion).await {
                                // Pass &conn
                                warn!("Failed to log suggestion for low confidence phone pair ({}, {}): {}. This operation was attempted on the main connection.", e1_id.0, e2_id.0, e);
                            }
                        }
                    }
                    Err(e) => {
                        let error_message = format!(
                            "Failed to insert phone pair group for ({}, {}) with shared phone '{}'",
                            e1_id.0, e2_id.0, normalized_shared_phone
                        );
                        if let Some(db_err) = e.as_db_error() {
                            if db_err.constraint() == Some("uq_entity_pair_method") {
                                debug!("{}: Pair already exists (DB constraint uq_entity_pair_method). Skipping.", error_message);
                                existing_processed_pairs.insert((e1_id.clone(), e2_id.clone()));
                            } else {
                                warn!("{}: Database error: {:?}. SQLState: {:?}, Detail: {:?}, Hint: {:?}",
                                    error_message, db_err, db_err.code(), db_err.detail(), db_err.hint());
                            }
                        } else {
                            warn!("{}: Non-database error: {}", error_message, e);
                        }
                    }
                }
            }
        }
    }
    // No transaction to commit.
    debug!("Finished processing phone pairs.");

    let avg_confidence: f64 = if !confidence_scores_for_stats.is_empty() {
        confidence_scores_for_stats.iter().sum::<f64>() / confidence_scores_for_stats.len() as f64
    } else {
        0.0
    };

    let method_stats = MatchMethodStats {
        method_type: MatchMethodType::Phone,
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
        "Pairwise phone matching complete in {:.2?}: created {} new pairs, involving {} unique entities. Errors for individual pairs (if any) were logged above.",
        elapsed,
        method_stats.groups_created,
        method_stats.entities_matched
    );

    let phone_result = PhoneMatchResult {
        groups_created: method_stats.groups_created,
        stats: method_stats,
    };

    Ok(AnyMatchResult::Phone(phone_result))
}

/// Normalize a phone number by:
/// - Removing all non-numeric characters
/// - Basic handling for US country code (stripping leading '1' if 11 digits and length becomes 10)
/// - Returns empty string if normalization results in a number outside typical lengths.
fn normalize_phone(phone: &str) -> String {
    let digits_only: String = phone.chars().filter(|c| c.is_ascii_digit()).collect();

    if digits_only.len() == 11 && digits_only.starts_with('1') {
        // Standard 10-digit US number after stripping country code
        return digits_only[1..].to_string();
    }

    // Basic validation for common phone number lengths (e.g., 7 to 15 digits)
    // This is a very simple heuristic and might need adjustment based on expected phone number formats.
    if digits_only.len() >= 7 && digits_only.len() <= 15 {
        return digits_only;
    }

    // If it doesn't match common patterns or is too short/long after stripping non-digits,
    // consider it invalid for matching purposes.
    debug!(
        "Phone number '{}' normalized to '{}', which is considered invalid for matching.",
        phone, digits_only
    );
    String::new()
}
