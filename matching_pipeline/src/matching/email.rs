// src/matching/email.rs
use anyhow::{anyhow, Context, Result}; // Added anyhow for error creation
use chrono::Utc;
use log::{debug, error, info, warn}; // Added error log level
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::config;
use crate::db::{self, PgPool}; // db::connect is available if needed, but PgPool is passed
use crate::models::{
    ActionType,
    EmailMatchValue, // Specific MatchValue for Email
    // EntityGroup, // We'll construct the values directly for insertion
    EntityGroupId,
    EntityId,
    MatchMethodType,
    MatchValues, // Enum holding different MatchValue types
    NewSuggestedAction,
    SuggestionStatus,
};
use crate::reinforcement::MatchingOrchestrator;
use crate::results::{AnyMatchResult, EmailMatchResult, MatchMethodStats}; // Removed PairMlResult as it's part of orchestrator logic now
use serde_json;

// Helper function to insert a single entity group pair in its own transaction
async fn insert_single_entity_group_pair(
    pool: &PgPool,
    entity_id_1: &EntityId,
    entity_id_2: &EntityId,
    method_type: &MatchMethodType,
    match_values: &MatchValues,
    confidence_score: f64,
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

    let insert_query = "
        INSERT INTO public.entity_group
        (id, entity_id_1, entity_id_2, method_type, match_values, confidence_score, created_at, updated_at, version)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 1)
    ";

    match tx
        .execute(
            insert_query,
            &[
                &new_entity_group_id.0,
                &entity_id_1.0,
                &entity_id_2.0,
                &method_type.as_str(),
                &match_values_json,
                &confidence_score,
                &now, // created_at
                &now, // updated_at
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
            // Attempt to rollback, though it might also fail if connection is broken
            let r_err = tx.rollback().await;
            if let Err(rollback_err) = r_err {
                error!("Failed to rollback transaction after insert error for pair ({}, {}), method {}: {}. Original error: {}",
                       entity_id_1.0, entity_id_2.0, method_type.as_str(), rollback_err, e);
            } else {
                error!(
                    "Rolled back transaction for pair ({}, {}), method {}. Insert error: {}",
                    entity_id_1.0,
                    entity_id_2.0,
                    method_type.as_str(),
                    e
                );
            }
            Err(anyhow!(e).context(format!(
                "DB error inserting entity_group for pair ({}, {}), method {}",
                entity_id_1.0,
                entity_id_2.0,
                method_type.as_str()
            )))
        }
    }
}

pub async fn find_matches(
    pool: &PgPool, // Use the passed-in pool
    reinforcement_orchestrator: Option<&Mutex<MatchingOrchestrator>>,
    pipeline_run_id: &str,
) -> Result<AnyMatchResult> {
    info!(
        "Starting pairwise email matching (run ID: {}){}...",
        pipeline_run_id,
        if reinforcement_orchestrator.is_some() {
            " with ML guidance"
        } else {
            ""
        }
    );
    let start_time = Instant::now();

    // Get a connection for initial reads (fetching existing pairs, all emails)
    // This connection is not used for the iterative inserts.
    let mut initial_read_conn = pool
        .get()
        .await
        .context("Failed to get DB connection for initial email matching reads")?;

    debug!("Fetching existing email-matched pairs...");
    let existing_pairs_query = "
        SELECT entity_id_1, entity_id_2
        FROM public.entity_group
        WHERE method_type = $1";
    let existing_pair_rows =
        initial_read_conn // Use the dedicated connection for reads
            .query(existing_pairs_query, &[&MatchMethodType::Email.as_str()])
            .await
            .context("Failed to query existing email-matched pairs")?;

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
        "Found {} existing email-matched pairs.",
        existing_processed_pairs.len()
    );

    let email_query = "
        SELECT 'organization' as source, e.id as entity_id, o.email, o.name as entity_name
        FROM entity e
        JOIN organization o ON e.organization_id = o.id
        WHERE o.email IS NOT NULL AND o.email != ''
        UNION ALL
        SELECT 'service' as source, e.id as entity_id, s.email, s.name as entity_name
        FROM entity e
        JOIN entity_feature ef ON e.id = ef.entity_id
        JOIN service s ON ef.table_id = s.id AND ef.table_name = 'service'
        WHERE s.email IS NOT NULL AND s.email != ''
    ";
    debug!("Executing email query for all entities...");
    let email_rows = initial_read_conn // Use the dedicated connection for reads
        .query(email_query, &[])
        .await
        .context("Failed to query entities with emails")?;
    info!(
        "Found {} email records across all entities.",
        email_rows.len()
    );

    // Drop the initial_read_conn to return it to the pool before starting loops
    drop(initial_read_conn);

    let mut email_map: HashMap<String, HashMap<EntityId, String>> = HashMap::new();
    for row in &email_rows {
        let entity_id = EntityId(row.get("entity_id"));
        let email: String = row.get("email");
        let normalized = normalize_email(&email);
        if !normalized.is_empty() {
            email_map
                .entry(normalized)
                .or_default()
                .insert(entity_id, email);
        }
    }
    info!("Processed {} unique normalized emails.", email_map.len());

    let mut new_pairs_created_count = 0;
    let mut entities_in_new_pairs: HashSet<EntityId> = HashSet::new();
    let mut confidence_scores_for_stats: Vec<f64> = Vec::new();
    let mut individual_transaction_errors = 0;

    for (normalized_shared_email, current_entity_map) in email_map {
        if current_entity_map.len() < 2 {
            continue;
        }

        let entities_sharing_email: Vec<_> = current_entity_map.iter().collect();

        for i in 0..entities_sharing_email.len() {
            for j in (i + 1)..entities_sharing_email.len() {
                let (entity_id1_obj, original_email1) = entities_sharing_email[i];
                let (entity_id2_obj, original_email2) = entities_sharing_email[j];

                let (e1_id, e1_orig_email, e2_id, e2_orig_email) =
                    if entity_id1_obj.0 < entity_id2_obj.0 {
                        (
                            entity_id1_obj,
                            original_email1,
                            entity_id2_obj,
                            original_email2,
                        )
                    } else {
                        (
                            entity_id2_obj,
                            original_email2,
                            entity_id1_obj,
                            original_email1,
                        )
                    };

                if existing_processed_pairs.contains(&(e1_id.clone(), e2_id.clone())) {
                    debug!(
                        "Pair ({}, {}) already processed by email method. Skipping.",
                        e1_id.0, e2_id.0
                    );
                    continue;
                }

                let mut final_confidence_score = 1.0;
                let mut predicted_method_type_from_ml = MatchMethodType::Email;
                let mut features_for_logging: Option<Vec<f64>> = None;

                if let Some(orchestrator_mutex) = reinforcement_orchestrator {
                    match MatchingOrchestrator::extract_pair_context(pool, e1_id, e2_id).await {
                        Ok(features) => {
                            features_for_logging = Some(features.clone());
                            let orchestrator_guard = orchestrator_mutex.lock().await;
                            match orchestrator_guard.predict_method_with_context(&features) {
                                Ok((predicted_method, rl_conf)) => {
                                    predicted_method_type_from_ml = predicted_method;
                                    final_confidence_score = rl_conf;
                                    info!("ML guidance for pair ({}, {}): Predicted Method: {:?}, Confidence: {:.4}", e1_id.0, e2_id.0, predicted_method_type_from_ml, final_confidence_score);
                                }
                                Err(e) => {
                                    warn!("ML prediction failed for email pair ({}, {}): {}. Using default confidence.", e1_id.0, e2_id.0, e);
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Context extraction failed for email pair ({}, {}): {}. Using default confidence.", e1_id.0, e2_id.0, e);
                        }
                    }
                }

                let match_values = MatchValues::Email(EmailMatchValue {
                    original_email1: e1_orig_email.clone(),
                    original_email2: e2_orig_email.clone(),
                    normalized_shared_email: normalized_shared_email.clone(),
                });

                // Perform the insert in its own transaction
                match insert_single_entity_group_pair(
                    pool,
                    e1_id,
                    e2_id,
                    &MatchMethodType::Email,
                    &match_values,
                    final_confidence_score,
                )
                .await
                {
                    Ok(new_entity_group_id) => {
                        new_pairs_created_count += 1;
                        entities_in_new_pairs.insert(e1_id.clone());
                        entities_in_new_pairs.insert(e2_id.clone());
                        confidence_scores_for_stats.push(final_confidence_score);
                        existing_processed_pairs.insert((e1_id.clone(), e2_id.clone()));

                        info!(
                            "SUCCESS: Created new email pair group {} for ({}, {}) with shared email '{}', confidence: {:.4}",
                            new_entity_group_id.0, e1_id.0, e2_id.0, normalized_shared_email, final_confidence_score
                        );

                        if let Some(orchestrator_mutex) = reinforcement_orchestrator {
                            let mut orchestrator_guard = orchestrator_mutex.lock().await;
                            if let Err(e) = orchestrator_guard
                                .log_match_result(
                                    pool,
                                    e1_id,
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
                                warn!("Failed to log email match result to entity_match_pairs for ({},{}): {}", e1_id.0, e2_id.0, e);
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
                                "method_type": MatchMethodType::Email.as_str(),
                                "matched_value": &normalized_shared_email,
                                "original_email1": e1_orig_email,
                                "original_email2": e2_orig_email,
                                "entity_group_id": &new_entity_group_id.0,
                                "rl_predicted_method": predicted_method_type_from_ml.as_str(),
                            });
                            let reason_message = format!(
                                "Pair ({}, {}) matched by Email with low RL confidence ({:.4}). RL predicted: {:?}.",
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
                            // For suggestions, we might want to collect them and insert them in a batch
                            // or handle them outside the critical path of pair insertion if they also fail.
                            // For now, attempting to insert it directly. This also needs a transaction or separate handling.
                            // This example will use a temporary connection for simplicity, but batching is better.
                            let mut temp_conn_for_suggestion = pool
                                .get()
                                .await
                                .context("Failed to get temp conn for suggestion")?;
                            let suggestion_tx = temp_conn_for_suggestion
                                .transaction()
                                .await
                                .context("Failed to start suggestion tx")?;
                            if let Err(e) = db::insert_suggestion(&suggestion_tx, &suggestion).await
                            {
                                warn!("Failed to log suggestion for low confidence email pair ({}, {}): {}. Suggestion details: {:?}", e1_id.0, e2_id.0, e, suggestion);
                                // Decide if suggestion failure should stop the pair processing. Probably not.
                                // Attempt to rollback suggestion_tx but continue.
                                if let Err(s_rb_err) = suggestion_tx.rollback().await {
                                    error!("Failed to rollback suggestion_tx: {}", s_rb_err);
                                }
                            } else {
                                if let Err(s_commit_err) = suggestion_tx.commit().await {
                                    error!("Failed to commit suggestion_tx: {}", s_commit_err);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        individual_transaction_errors += 1;
                        // Error is already logged by insert_single_entity_group_pair
                        // We continue to the next pair
                    }
                }
            }
        }
    }

    if individual_transaction_errors > 0 {
        warn!(
            "Encountered {} errors during individual email pair transaction attempts.",
            individual_transaction_errors
        );
    }

    let avg_confidence: f64 = if !confidence_scores_for_stats.is_empty() {
        confidence_scores_for_stats.iter().sum::<f64>() / confidence_scores_for_stats.len() as f64
    } else {
        0.0
    };

    let method_stats = MatchMethodStats {
        method_type: MatchMethodType::Email,
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
        "Pairwise email matching complete in {:.2?}: created {} new pairs ({} individual transaction errors), involving {} unique entities.",
        elapsed,
        method_stats.groups_created,
        individual_transaction_errors,
        method_stats.entities_matched
    );

    let email_specific_result = EmailMatchResult {
        groups_created: method_stats.groups_created,
        stats: method_stats,
    };

    Ok(AnyMatchResult::Email(email_specific_result))
}

// normalize_email function remains the same
fn normalize_email(email: &str) -> String {
    let email_trimmed = email.trim().to_lowercase();
    if !email_trimmed.contains('@') {
        return email_trimmed;
    }

    let parts: Vec<&str> = email_trimmed.splitn(2, '@').collect();
    if parts.len() != 2 {
        return email_trimmed;
    }

    let (local_part_full, domain_part) = (parts[0], parts[1]);
    let local_part_no_plus = local_part_full.split('+').next().unwrap_or("").to_string();

    let final_local_part = if domain_part == "gmail.com" || domain_part == "googlemail.com" {
        local_part_no_plus.replace('.', "")
    } else {
        local_part_no_plus
    };

    let final_domain_part = match domain_part {
        "googlemail.com" => "gmail.com",
        _ => domain_part,
    };

    if final_local_part.is_empty() {
        return String::new();
    }
    format!("{}@{}", final_local_part, final_domain_part)
}
