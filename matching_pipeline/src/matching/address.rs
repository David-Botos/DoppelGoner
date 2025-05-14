// src/matching/address.rs
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
    ActionType,
    AddressMatchValue, // Specific MatchValue for Address
    EntityGroupId,
    EntityId,
    MatchMethodType,
    MatchValues, // Enum holding different MatchValue types
    NewSuggestedAction,
    SuggestionStatus,
};
use crate::reinforcement::{self, MatchingOrchestrator};
use crate::results::{AddressMatchResult, AnyMatchResult, MatchMethodStats}; // PairMlResult might not be needed if not used
use serde_json;

// SQL query for inserting into entity_group
const INSERT_ENTITY_GROUP_SQL: &str = "
    INSERT INTO public.entity_group
    (id, entity_id_1, entity_id_2, method_type, match_values, confidence_score, pre_rl_confidence_score, created_at, updated_at, version)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 1)";

pub async fn find_matches(
    pool: &PgPool,
    reinforcement_orchestrator: Option<&Mutex<reinforcement::MatchingOrchestrator>>,
    pipeline_run_id: &str,
) -> Result<AnyMatchResult> {
    info!(
        "Starting pairwise address matching (run ID: {}){}...",
        pipeline_run_id,
        if reinforcement_orchestrator.is_some() {
            " with ML guidance"
        } else {
            ""
        }
    );
    let start_time = Instant::now();

    // Get a connection from the pool. This connection will be used for all operations.
    // No single overarching transaction will be used.
    let mut conn = pool
        .get()
        .await
        .context("Failed to get DB connection for address matching")?;

    // 1. Fetch existing address-matched pairs
    debug!("Fetching existing address-matched pairs...");
    let existing_pairs_query = "
        SELECT entity_id_1, entity_id_2
        FROM public.entity_group
        WHERE method_type = $1";
    let existing_pair_rows = conn
        .query(existing_pairs_query, &[&MatchMethodType::Address.as_str()])
        .await
        .context("Failed to query existing address-matched pairs")?;

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
        "Found {} existing address-matched pairs.",
        existing_processed_pairs.len()
    );

    // 2. Fetch Address Data for all entities
    let address_query = "
        SELECT
            e.id AS entity_id,
            a.id AS address_id,
            a.address_1,
            a.address_2,
            a.city,
            a.state_province,
            a.postal_code,
            a.country
        FROM
            public.entity e
            JOIN public.entity_feature ef ON e.id = ef.entity_id
            JOIN public.location l ON ef.table_id = l.id AND ef.table_name = 'location'
            JOIN public.address a ON a.location_id = l.id
        WHERE
            a.address_1 IS NOT NULL AND a.address_1 != ''
            AND a.city IS NOT NULL AND a.city != ''
    ";
    debug!("Executing address query for all entities...");
    let address_rows = conn
        .query(address_query, &[])
        .await
        .context("Failed to query addresses")?;
    info!(
        "Found {} address records across all entities.",
        address_rows.len()
    );

    let mut address_map: HashMap<String, HashMap<EntityId, String>> = HashMap::new();
    for row in &address_rows {
        let entity_id = EntityId(row.get("entity_id"));
        let address_1: String = row.get("address_1");
        let address_2: Option<String> = row.try_get("address_2").ok().flatten();
        let city: String = row.get("city");
        let state_province: String = row.get("state_province");
        let postal_code: String = row.get("postal_code");
        let country: String = row.get("country");

        let full_address = format!(
            "{}{}, {}, {} {}, {}",
            address_1,
            address_2
                .as_deref()
                .map_or("".to_string(), |a| format!(", {}", a.trim())),
            city.trim(),
            state_province.trim(),
            postal_code.trim(),
            country.trim()
        );
        let normalized_address = normalize_address(&full_address);

        if !normalized_address.is_empty() {
            address_map
                .entry(normalized_address)
                .or_default()
                .insert(entity_id, full_address);
        }
    }
    info!(
        "Processed {} unique normalized addresses.",
        address_map.len()
    );

    // 3. Database Operations (No transaction spanning all inserts)
    // Each insert will be an individual operation.
    debug!("Starting pairwise address matching inserts (non-transactional for the whole batch).");

    let now = Utc::now().naive_utc();
    let mut new_pairs_created_count = 0;
    let mut entities_in_new_pairs: HashSet<EntityId> = HashSet::new();
    let mut confidence_scores_for_stats: Vec<f64> = Vec::new();

    // 4. Process addresses and form pairs
    for (normalized_shared_address, current_entity_map) in address_map {
        if current_entity_map.len() < 2 {
            continue;
        }

        let entities_sharing_address: Vec<_> = current_entity_map.iter().collect();

        for i in 0..entities_sharing_address.len() {
            for j in (i + 1)..entities_sharing_address.len() {
                let (entity_id1_obj, original_address1) = entities_sharing_address[i];
                let (entity_id2_obj, original_address2) = entities_sharing_address[j];

                let (e1_id, e1_orig_addr, e2_id, e2_orig_addr) =
                    if entity_id1_obj.0 < entity_id2_obj.0 {
                        (
                            entity_id1_obj,
                            original_address1,
                            entity_id2_obj,
                            original_address2,
                        )
                    } else {
                        (
                            entity_id2_obj,
                            original_address2,
                            entity_id1_obj,
                            original_address1,
                        )
                    };

                if existing_processed_pairs.contains(&(e1_id.clone(), e2_id.clone())) {
                    debug!(
                        "Pair ({}, {}) already processed by address method. Skipping.",
                        e1_id.0, e2_id.0
                    );
                    continue;
                }

                let pre_rl_score = Some(1.0_f32);
                let mut final_confidence_score = 0.95;
                let mut predicted_method_type_from_ml = MatchMethodType::Address;
                let mut features_for_logging: Option<Vec<f64>> = None;

                // Check if units are different
                let unit1 = extract_unit(e1_orig_addr);
                let unit2 = extract_unit(e2_orig_addr);
                let has_different_units = !unit1.is_empty() && !unit2.is_empty() && unit1 != unit2;

                if has_different_units {
                    // Apply confidence penalty for different suites/units in the same building
                    final_confidence_score *= 0.85;
                    debug!(
                        "Applied unit difference penalty for pair ({}, {}) with units '{}' and '{}'", 
                        e1_id.0, e2_id.0, unit1, unit2
                    );
                }

                if let Some(orchestrator_mutex) = reinforcement_orchestrator {
                    // Assuming extract_pair_context can use a non-transactional pool or connection reference
                    match MatchingOrchestrator::extract_pair_context(pool, e1_id, e2_id).await {
                        Ok(features) => {
                            features_for_logging = Some(features.clone());
                            let orchestrator_guard = orchestrator_mutex.lock().await;
                            match orchestrator_guard.predict_method_with_context(&features) {
                                Ok((predicted_method, rl_conf)) => {
                                    predicted_method_type_from_ml = predicted_method;
                                    final_confidence_score = rl_conf;
                                    info!("ML guidance for address pair ({}, {}), normalized '{}': Predicted Method: {:?}, Confidence: {:.4}", e1_id.0, e2_id.0, normalized_shared_address, predicted_method_type_from_ml, final_confidence_score);
                                }
                                Err(e) => {
                                    warn!("ML prediction failed for address pair ({}, {}): {}. Using default confidence {:.2}.", e1_id.0, e2_id.0, e, final_confidence_score);
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Context extraction failed for address pair ({}, {}): {}. Using default confidence {:.2}.", e1_id.0, e2_id.0, e, final_confidence_score);
                        }
                    }
                }

                let match_values = MatchValues::Address(AddressMatchValue {
                    original_address1: e1_orig_addr.clone(),
                    original_address2: e2_orig_addr.clone(),
                    normalized_shared_address: normalized_shared_address.clone(),
                    pairwise_match_score: pre_rl_score,
                });
                let match_values_json = serde_json::to_value(&match_values).with_context(|| {
                    format!(
                        "Failed to serialize address MatchValues for pair ({}, {})",
                        e1_id.0, e2_id.0
                    )
                })?;

                let new_entity_group_id = EntityGroupId(Uuid::new_v4().to_string());

                // Execute insert directly on the connection
                let insert_result = conn
                    .execute(
                        INSERT_ENTITY_GROUP_SQL,
                        &[
                            &new_entity_group_id.0,
                            &e1_id.0,
                            &e2_id.0,
                            &MatchMethodType::Address.as_str(),
                            &match_values_json,
                            &final_confidence_score,
                            &(pre_rl_score.unwrap_or(1.0) as f64), // Convert the Option<f32> to f64
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
                            "Created new address pair group {} for ({}, {}) with shared address '{}', confidence: {:.4}",
                            new_entity_group_id.0, e1_id.0, e2_id.0, normalized_shared_address, final_confidence_score
                        );

                        if let Some(orchestrator_mutex) = reinforcement_orchestrator {
                            let mut orchestrator_guard = orchestrator_mutex.lock().await;
                            // log_match_result might need to be adapted if it expects a transaction
                            // or ensure it handles its own DB interaction carefully.
                            // Assuming it can use the pool or a direct connection if needed.
                            if let Err(e) = orchestrator_guard
                                .log_match_result(
                                    pool,
                                    e1_id, // Assuming EntityId is Clone or Copy
                                    e2_id, // Assuming EntityId is Clone or Copy
                                    &predicted_method_type_from_ml,
                                    final_confidence_score,
                                    true,
                                    features_for_logging.as_ref(),
                                    Some(&predicted_method_type_from_ml), // actual_method_type
                                    Some(final_confidence_score),         // actual_confidence
                                )
                                .await
                            {
                                warn!("Failed to log address match result to entity_match_pairs for ({},{}): {}", e1_id.0, e2_id.0, e);
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
                                "method_type": MatchMethodType::Address.as_str(),
                                "matched_value": &normalized_shared_address,
                                "original_address1": e1_orig_addr,
                                "original_address2": e2_orig_addr,
                                "entity_group_id": &new_entity_group_id.0,
                                "pre_rl_score": pre_rl_score,
                                "rl_predicted_method": predicted_method_type_from_ml.as_str(),
                            });
                            let reason_message = format!(
                                "Pair ({}, {}) matched by Address with low RL confidence ({:.4}). RL predicted: {:?}.",
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
                            // Pass the connection `conn` to insert_suggestion
                            if let Err(e) = db::insert_suggestion(&*conn, &suggestion).await {
                                warn!("Failed to log suggestion for low confidence address pair ({}, {}): {}. This operation was attempted on the main connection.", e1_id.0, e2_id.0, e);
                            }
                        }
                    }
                    Err(e) => {
                        // Detailed error logging for the specific insert failure
                        let error_message = format!("Failed to insert address pair group for ({}, {}) with shared address '{}'", e1_id.0, e2_id.0, normalized_shared_address);
                        if let Some(db_err) = e.as_db_error() {
                            if db_err.constraint() == Some("uq_entity_pair_method") {
                                debug!("{}: Pair already exists (DB constraint uq_entity_pair_method). Skipping.", error_message);
                                existing_processed_pairs.insert((e1_id.clone(), e2_id.clone()));
                            // Mark as processed to avoid retries if logic allows
                            } else {
                                warn!("{}: Database error: {:?}. SQLState: {:?}, Detail: {:?}, Hint: {:?}",
                                    error_message, db_err, db_err.code(), db_err.detail(), db_err.hint());
                            }
                        } else {
                            warn!("{}: Non-database error: {}", error_message, e);
                        }
                        // Consider if this error should halt the entire process or just this pair.
                        // Current logic: logs and continues.
                    }
                }
            }
        }
    }
    // No transaction to commit.

    debug!("Finished processing address pairs.");

    // 5. Calculate Statistics and Return
    let avg_confidence: f64 = if !confidence_scores_for_stats.is_empty() {
        confidence_scores_for_stats.iter().sum::<f64>() / confidence_scores_for_stats.len() as f64
    } else {
        0.0
    };

    let method_stats = MatchMethodStats {
        method_type: MatchMethodType::Address,
        groups_created: new_pairs_created_count,
        entities_matched: entities_in_new_pairs.len(),
        avg_confidence,
        avg_group_size: if new_pairs_created_count > 0 {
            2.0
        } else {
            0.0
        }, // Assuming pairs
    };

    let elapsed = start_time.elapsed();
    info!(
        "Pairwise address matching complete in {:.2?}: created {} new pairs, involving {} unique entities. Errors for individual pairs (if any) were logged above.",
        elapsed,
        method_stats.groups_created,
        method_stats.entities_matched
    );

    let address_result = AddressMatchResult {
        groups_created: method_stats.groups_created,
        stats: method_stats,
    };

    Ok(AnyMatchResult::Address(address_result))
}

/// Normalize an address by:
/// - Converting to lowercase
/// - Removing most punctuation (keeps alphanumeric and whitespace)
/// - Standardizing common abbreviations (street, road, etc.)
/// - Removing common unit designators (apt, suite, unit)
/// - Trimming extra whitespace
fn normalize_address(address: &str) -> String {
    let lower = address.to_lowercase();

    let mut normalized = lower
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '#')
        .collect::<String>();

    normalized = normalized
        .replace(" st ", " street ")
        .replace(" str ", " street ")
        .replace(" rd ", " road ")
        .replace(" ave ", " avenue ")
        .replace(" av ", " avenue ")
        .replace(" blvd ", " boulevard ")
        .replace(" blv ", " boulevard ")
        .replace(" dr ", " drive ")
        .replace(" ln ", " lane ")
        .replace(" ct ", " court ")
        .replace(" pl ", " place ")
        .replace(" sq ", " square ")
        .replace(" pkwy ", " parkway ")
        .replace(" cir ", " circle ");

    // Simplified removal of unit designators; consider regex for robustness
    let patterns_to_remove = [
        "apt ",
        "apartment ",
        "suite ",
        "ste ",
        "unit ",
        "#", // Handle '#' carefully
        "bldg ",
        "building ",
        "fl ",
        "floor ",
        "dept ",
        "department ",
        "room ",
        "rm ",
        "po box ",
        "p o box ",
        "p.o. box ",
    ];

    // Special handling for '#' which might be part of the number or a separator
    // If # is followed by a space then a digit, it's likely a unit. Otherwise, it might be part of the address number.
    // This is a heuristic.
    if let Some(idx) = normalized.find('#') {
        if normalized
            .chars()
            .nth(idx + 1)
            .map_or(false, |c| c.is_whitespace())
            && normalized
                .chars()
                .nth(idx + 2)
                .map_or(false, |c| c.is_alphanumeric())
        {
            // Likely pattern like "# 123" or "# A"
            let (before, after_pattern) = normalized.split_at(idx);
            let mut rest = after_pattern
                .trim_start_matches('#')
                .trim_start()
                .to_string();
            if let Some(space_idx) = rest.find(|c: char| c.is_whitespace() || c == ',') {
                rest = rest[space_idx..].to_string();
            } else {
                rest.clear();
            }
            normalized = format!("{}{}", before.trim_end(), rest.trim_start());
        }
        // Else, keep the '#' as it might be like '123# Main St' (uncommon but possible)
        // Or it was already filtered if not alphanumeric/whitespace.
    }
    // Remove other patterns
    for pattern_base in patterns_to_remove {
        // Iterate to catch multiple occurrences if not handled by split logic
        while let Some(idx) = normalized.find(pattern_base) {
            let (before, after_pattern_full) = normalized.split_at(idx);
            let mut rest_of_string = after_pattern_full
                .trim_start_matches(pattern_base)
                .to_string();

            // Remove the number/identifier after the pattern until a space or comma
            if let Some(end_of_unit_idx) =
                rest_of_string.find(|c: char| c.is_whitespace() || c == ',')
            {
                rest_of_string = rest_of_string[end_of_unit_idx..].to_string();
            } else {
                rest_of_string.clear(); // Pattern was at the end, or no clear separator
            }
            normalized = format!("{}{}", before.trim_end(), rest_of_string.trim_start());
            normalized = normalized.trim().to_string(); // Trim intermediate results
        }
    }

    // Final cleanup: multiple spaces to single, trim
    normalized
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

/// Extract unit or suite information from an address
/// Returns the extracted unit/suite or an empty string if none is found
fn extract_unit(address: &str) -> String {
    let lower = address.to_lowercase();

    // Common unit designators
    let unit_patterns = [
        "apt",
        "apartment",
        "suite",
        "ste",
        "unit",
        "#",
        "bldg",
        "building",
        "fl",
        "floor",
        "room",
        "rm",
    ];

    for pattern in unit_patterns {
        if let Some(idx) = lower.find(pattern) {
            // Get text after the pattern
            let after_pattern = &lower[idx..];

            // Find the end of the unit designator (next comma or end of string)
            if let Some(end_idx) = after_pattern.find(|c: char| c == ',' || c == ';') {
                return after_pattern[0..end_idx].trim().to_string();
            } else {
                // Look for next whitespace after a non-whitespace
                let mut found_non_space = false;
                let mut end_idx = 0;

                for (i, c) in after_pattern.char_indices() {
                    if !c.is_whitespace() {
                        found_non_space = true;
                    } else if found_non_space {
                        end_idx = i;
                        break;
                    }
                }

                if end_idx > 0 {
                    return after_pattern[0..end_idx].trim().to_string();
                } else {
                    return after_pattern.trim().to_string();
                }
            }
        }
    }

    String::new() // No unit found
}
