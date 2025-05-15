// src/matching/email.rs
use anyhow::{anyhow, Context, Result}; // Added anyhow for error creation
use chrono::Utc;
use log::{debug, error, info, warn}; // Added error log level
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
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

// SQL for inserting entity_group, now includes new confirmed_status and nullable review fields
const INSERT_ENTITY_GROUP_SQL: &str = "
    INSERT INTO public.entity_group
(id, entity_id_1, entity_id_2, method_type, match_values, confidence_score, 
 pre_rl_confidence_score)
VALUES ($1, $2, $3, $4, $5, $6, $7)";

// Helper function to insert a single entity group pair.
async fn insert_single_entity_group_pair(
    pool: &PgPool,
    entity_id_1: &EntityId,
    entity_id_2: &EntityId,
    method_type: &MatchMethodType,
    match_values: &MatchValues,
    final_confidence_score: f64,  // This is the tuned score
    pre_rl_confidence_score: f64, // The original heuristic score
) -> Result<EntityGroupId> {
    let mut conn = pool
        .get()
        .await
        .context("Email: Failed to get DB connection for single insert")?;
    let tx = conn
        .transaction()
        .await
        .context("Email: Failed to start transaction for single entity_group insert")?;

    let new_entity_group_id = EntityGroupId(Uuid::new_v4().to_string());
    let match_values_json = serde_json::to_value(match_values)
        .context("Email: Failed to serialize match_values for single insert")?;

    match tx
        .execute(
            INSERT_ENTITY_GROUP_SQL,
            &[
                &new_entity_group_id.0,
                &entity_id_1.0,
                &entity_id_2.0,
                &method_type.as_str(),
                &match_values_json,
                &final_confidence_score,  // Stored in confidence_score column
                &pre_rl_confidence_score, // Stored in pre_rl_confidence_score column
            ],
        )
        .await
    {
        Ok(_) => {
            tx.commit()
                .await
                .context("Email: Failed to commit transaction for single entity_group insert")?;
            Ok(new_entity_group_id)
        }
        Err(e) => {
            let r_err = tx.rollback().await;
            if let Err(rollback_err) = r_err {
                error!("Email: Failed to rollback transaction after insert error for pair ({}, {}), method {}: {}. Original error: {}",
                       entity_id_1.0, entity_id_2.0, method_type.as_str(), rollback_err, e);
            } else {
                error!(
                    "Email: Rolled back transaction for pair ({}, {}), method {}. Insert error: {}",
                    entity_id_1.0,
                    entity_id_2.0,
                    method_type.as_str(),
                    e
                );
            }
            Err(anyhow!(e).context(format!(
                "Email: DB error inserting entity_group for pair ({}, {}), method {}",
                entity_id_1.0,
                entity_id_2.0,
                method_type.as_str()
            )))
        }
    }
}

pub async fn find_matches(
    pool: &PgPool,
    reinforcement_orchestrator_option: Option<Arc<Mutex<MatchingOrchestrator>>>,
    pipeline_run_id: &str,
) -> Result<AnyMatchResult> {
    info!(
        "Starting V1 pairwise email matching (run ID: {}){}...",
        pipeline_run_id,
        if reinforcement_orchestrator_option.is_some() {
            " with RL confidence tuning"
        } else {
            " (RL tuner not provided)"
        }
    );
    let start_time = Instant::now();

    let mut initial_read_conn = pool
        .get()
        .await
        .context("Email: Failed to get DB connection for initial reads")?;

    let existing_processed_pairs: HashSet<(EntityId, EntityId)> =
        fetch_existing_pairs(&*initial_read_conn, MatchMethodType::Email).await?;
    info!(
        "Email: Found {} existing email-matched pairs.",
        existing_processed_pairs.len()
    );

    // Fetch email data (same as before)
    let email_query = "
        SELECT 'organization' as source, e.id as entity_id, o.email, o.name as entity_name
        FROM entity e
        JOIN organization o ON e.organization_id = o.id
        WHERE o.email IS NOT NULL AND o.email != ''
        UNION ALL
        SELECT 'service' as source, e.id as entity_id, s.email, s.name as entity_name
        FROM public.entity e
        JOIN public.entity_feature ef ON e.id = ef.entity_id
        JOIN public.service s ON ef.table_id = s.id AND ef.table_name = 'service'
        WHERE s.email IS NOT NULL AND s.email != ''
    ";
    let email_rows = initial_read_conn
        .query(email_query, &[])
        .await
        .context("Email: Failed to query entities with emails")?;
    info!(
        "Email: Found {} email records across all entities.",
        email_rows.len()
    );
    drop(initial_read_conn); // Release connection

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
    let email_frequency = calculate_email_frequency(&email_map);

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
                    continue;
                }

                let mut pre_rl_confidence_score = 1.0; // Base for email
                if is_generic_organizational_email(&normalized_shared_email) {
                    pre_rl_confidence_score *= 0.9;
                }
                let email_count = email_frequency
                    .get(&normalized_shared_email)
                    .cloned()
                    .unwrap_or(1);
                if email_count > 10 {
                    pre_rl_confidence_score *= 0.85;
                } else if email_count > 5 {
                    pre_rl_confidence_score *= 0.92;
                }

                let mut final_confidence_score = pre_rl_confidence_score; // Default to pre-RL
                let mut features_for_snapshot: Option<Vec<f64>> = None;

                if let Some(ro_arc) = reinforcement_orchestrator_option.as_ref() {
                    match MatchingOrchestrator::extract_pair_context_features(pool, e1_id, e2_id).await {
                        Ok(features_vec) => {
                            if !features_vec.is_empty() {
                                features_for_snapshot = Some(features_vec.clone());
                                let orchestrator_guard = ro_arc.lock().await;
                                match orchestrator_guard.get_tuned_confidence(
                                    &MatchMethodType::Email,
                                    pre_rl_confidence_score,
                                    &features_vec,
                                ) {
                                    Ok(tuned_score) => final_confidence_score = tuned_score,
                                    Err(e) => warn!("Email: Failed to get tuned confidence for ({}, {}): {}. Using pre-RL score.", e1_id.0, e2_id.0, e),
                                }
                            } else {
                                warn!("Email: Extracted features vector is empty for pair ({}, {}). Using pre-RL score.", e1_id.0, e2_id.0);
                            }
                        }
                        Err(e) => warn!("Email: Failed to extract features for ({}, {}): {}. Using pre-RL score.", e1_id.0, e2_id.0, e),
                    }
                }

                let match_values = MatchValues::Email(EmailMatchValue {
                    original_email1: e1_orig_email.clone(),
                    original_email2: e2_orig_email.clone(),
                    normalized_shared_email: normalized_shared_email.clone(),
                });

                match insert_single_entity_group_pair(
                    pool,
                    e1_id,
                    e2_id,
                    &MatchMethodType::Email,
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
                        // existing_processed_pairs.insert((e1_id.clone(), e2_id.clone())); // Already handled by initial fetch

                        if let (Some(ro_arc), Some(ref features)) = (
                            reinforcement_orchestrator_option.as_ref(),
                            features_for_snapshot.as_ref(),
                        ) {
                            let orchestrator_guard = ro_arc.lock().await;
                            if let Err(e) = orchestrator_guard
                                .log_decision_snapshot(
                                    pool,
                                    &new_entity_group_id.0,
                                    pipeline_run_id,
                                    features,
                                    &MatchMethodType::Email,
                                    pre_rl_confidence_score,
                                    final_confidence_score,
                                )
                                .await
                            {
                                warn!(
                                    "Email: Failed to log decision snapshot for {}: {}",
                                    new_entity_group_id.0, e
                                );
                            }
                        }

                        if final_confidence_score < config::MODERATE_LOW_SUGGESTION_THRESHOLD {
                            // ... (suggestion logging logic, ensure it uses a connection from pool for db::insert_suggestion)
                            let mut temp_conn_for_suggestion = pool
                                .get()
                                .await
                                .context("Email: Failed to get temp conn for suggestion")?;
                            let suggestion_tx = temp_conn_for_suggestion
                                .transaction()
                                .await
                                .context("Email: Failed to start suggestion tx")?;
                            // ... (construct suggestion, details_json should not include rl_predicted_method)
                            let details_json = serde_json::json!({
                                "method_type": MatchMethodType::Email.as_str(),
                                "matched_value": &normalized_shared_email,
                                "original_email1": e1_orig_email,
                                "original_email2": e2_orig_email,
                                "entity_group_id": &new_entity_group_id.0,
                                "pre_rl_confidence": pre_rl_confidence_score,
                            });
                            let reason_message = format!(
                                "Pair ({}, {}) matched by Email with low tuned confidence ({:.4}). Pre-RL: {:.2}.",
                                e1_id.0, e2_id.0, final_confidence_score, pre_rl_confidence_score
                            );
                            let suggestion = NewSuggestedAction {
                                // ...
                                reason_message: Some(reason_message),
                                details: Some(details_json),
                                // ...
                                pipeline_run_id: Some(pipeline_run_id.to_string()),
                                action_type: ActionType::ReviewEntityInGroup.as_str().to_string(),
                                entity_id: None,
                                group_id_1: Some(new_entity_group_id.0.clone()),
                                group_id_2: None,
                                cluster_id: None,
                                triggering_confidence: Some(final_confidence_score),
                                reason_code: Some("LOW_TUNED_CONFIDENCE_PAIR".to_string()),
                                priority: if final_confidence_score
                                    < config::CRITICALLY_LOW_SUGGESTION_THRESHOLD
                                {
                                    2
                                } else {
                                    1
                                },
                                status: SuggestionStatus::PendingReview.as_str().to_string(),
                                reviewer_id: None,
                                reviewed_at: None,
                                review_notes: None,
                            };
                            if let Err(e) = db::insert_suggestion(&suggestion_tx, &suggestion).await
                            {
                                warn!("Email: Failed to log suggestion: {}", e);
                                let _ = suggestion_tx.rollback().await; // Attempt rollback
                            } else {
                                suggestion_tx
                                    .commit()
                                    .await
                                    .context("Email: Failed to commit suggestion tx")?;
                            }
                        }
                    }
                    Err(e) => {
                        individual_transaction_errors += 1;
                        // Error already logged by insert_single_entity_group_pair
                    }
                }
            }
        }
    }

    if individual_transaction_errors > 0 {
        warn!(
            "Email: {} errors during individual pair transaction attempts.",
            individual_transaction_errors
        );
    }

    let avg_confidence = if !confidence_scores_for_stats.is_empty() {
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

    info!(
        "Email matching complete in {:.2?}: {} new pairs, {} unique entities.",
        start_time.elapsed(),
        method_stats.groups_created,
        method_stats.entities_matched
    );
    Ok(AnyMatchResult::Email(EmailMatchResult {
        groups_created: method_stats.groups_created,
        stats: method_stats,
    }))
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
        String::new()
    } else {
        format!("{}@{}", final_local_part, final_domain_part)
    }
}

fn is_generic_organizational_email(email: &str) -> bool {
    ["info@", "contact@", "office@", "admin@"]
        .iter()
        .any(|prefix| email.starts_with(prefix))
}

// Helper to fetch existing pairs for a given method type (can be moved to a shared db utility)
async fn fetch_existing_pairs(
    conn: &impl tokio_postgres::GenericClient,
    method_type: MatchMethodType,
) -> Result<HashSet<(EntityId, EntityId)>> {
    let query = "SELECT entity_id_1, entity_id_2 FROM public.entity_group WHERE method_type = $1";
    let rows = conn
        .query(query, &[&method_type.as_str()])
        .await
        .with_context(|| format!("Failed to query existing {:?}-matched pairs", method_type))?;

    let mut existing_pairs = HashSet::new();
    for row in rows {
        let id1_str: String = row.get("entity_id_1");
        let id2_str: String = row.get("entity_id_2");
        if id1_str < id2_str {
            existing_pairs.insert((EntityId(id1_str), EntityId(id2_str)));
        } else {
            existing_pairs.insert((EntityId(id2_str), EntityId(id1_str)));
        }
    }
    Ok(existing_pairs)
}
fn calculate_email_frequency(
    email_map: &HashMap<String, HashMap<EntityId, String>>,
) -> HashMap<String, usize> {
    let mut freq = HashMap::new();
    for (normalized_email, entities) in email_map {
        freq.insert(normalized_email.clone(), entities.len());
    }
    freq
}
