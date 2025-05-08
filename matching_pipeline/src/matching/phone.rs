// src/matching/phone.rs
use anyhow::{Context, Result};
use chrono::Utc;
use log::{debug, info, warn};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::db::PgPool;
use crate::models::{
    EntityGroup, EntityGroupId, EntityId, MatchMethodType, MatchValues, PhoneMatchValue,
};
use crate::reinforcement;
use crate::results::{MatchMethodStats, PhoneMatchResult};

pub async fn find_matches(
    pool: &PgPool,
    reinforcement_orchestrator: Option<&Mutex<reinforcement::MatchingOrchestrator>>,
) -> Result<PhoneMatchResult> {
    info!(
        "Starting phone matching{}...",
        if reinforcement_orchestrator.is_some() {
            " with ML guidance"
        } else {
            ""
        }
    );
    let start_time = Instant::now();

    let mut conn = pool
        .get()
        .await
        .context("Failed to get DB connection for phone matching")?;

    // First, get all entities that are already part of a phone match group
    debug!("Finding entities already processed by phone matching");
    let processed_query = "
        SELECT DISTINCT ge.entity_id
        FROM group_entity ge
        JOIN group_method gm ON ge.entity_group_id = gm.entity_group_id
        WHERE gm.method_type = 'phone'
    ";

    let processed_rows = conn
        .query(processed_query, &[])
        .await
        .context("Failed to query processed entities")?;

    let mut processed_entities = HashSet::new();
    for row in &processed_rows {
        let entity_id: String = row.get("entity_id");
        processed_entities.insert(EntityId(entity_id));
    }

    info!(
        "Found {} entities already processed by phone matching",
        processed_entities.len()
    );

    // Get existing phone groups and their normalized phone numbers
    debug!("Finding existing phone groups");
    let existing_groups_query = "
        SELECT 
            eg.id AS group_id,
            gm.id AS method_id,
            gm.match_values
        FROM 
            entity_group eg
            JOIN group_method gm ON eg.id = gm.entity_group_id
        WHERE 
            gm.method_type = 'phone'
    ";

    let existing_groups_rows = conn
        .query(existing_groups_query, &[])
        .await
        .context("Failed to query existing groups")?;

    // Map of normalized phone -> (group_id, method_id)
    let mut existing_phone_groups: HashMap<String, (String, String)> = HashMap::new();

    for row in &existing_groups_rows {
        let group_id: String = row.get("group_id");
        let method_id: String = row.get("method_id");
        let match_values_json: serde_json::Value = row.get("match_values");

        // Extract phones from the match_values JSON
        if let Ok(match_values) = serde_json::from_value::<MatchValues>(match_values_json.clone()) {
            if let MatchValues::Phone(phone_values) = match_values {
                for phone_value in &phone_values {
                    // Store mapping from normalized phone to group/method ids
                    existing_phone_groups.insert(
                        phone_value.normalized.clone(),
                        (group_id.clone(), method_id.clone()),
                    );
                }
            }
        }
    }

    info!(
        "Found {} unique normalized phone numbers in existing groups",
        existing_phone_groups.len()
    );

    // Query phones linked to entities
    let phone_query = "
        SELECT e.id as entity_id, p.number, p.extension 
        FROM entity e 
        JOIN entity_feature ef ON e.id = ef.entity_id 
        JOIN phone p ON ef.table_id = p.id 
        WHERE ef.table_name = 'phone' 
        AND p.number IS NOT NULL AND p.number != ''
        AND e.id NOT IN (
            SELECT ge.entity_id 
            FROM group_entity ge
            JOIN group_method gm ON ge.entity_group_id = gm.entity_group_id
            WHERE gm.method_type = 'phone'
        )
    ";

    debug!("Executing phone query");
    let phone_rows = conn
        .query(phone_query, &[])
        .await
        .context("Failed to query phones")?;

    info!("Found {} phone records to process", phone_rows.len());

    // Map to store normalized phone numbers -> map of entities with matching values
    // Using HashMap<normalized_phone, HashMap<entity_id, (original, extension)>> to avoid duplicates
    let mut phone_map: HashMap<String, HashMap<EntityId, (String, Option<String>)>> =
        HashMap::new();
    let mut processed_count = 0;

    // Process phone numbers
    for row in &phone_rows {
        let entity_id: String = row.get("entity_id");
        let entity_id = EntityId(entity_id);

        // Skip if this entity has already been processed
        if processed_entities.contains(&entity_id) {
            continue;
        }

        let number: String = row.get("number");
        let extension: Option<String> = row.try_get("extension").ok();

        let normalized = normalize_phone(&number);

        if !normalized.is_empty() {
            // Get or create the inner entity map for this normalized phone
            let entity_map = phone_map.entry(normalized).or_default();

            // Only add if this entity isn't already in the map for this phone
            entity_map.entry(entity_id).or_insert((number, extension));
        }

        processed_count += 1;
        if processed_count % 1000 == 0 {
            debug!("Processed {} phone records so far", processed_count);
        }
    }

    let unique_phones = phone_map.len();
    info!(
        "Found {} unique normalized phone numbers from unprocessed records",
        unique_phones
    );

    // Begin transaction for all database operations
    let tx = conn
        .transaction()
        .await
        .context("Failed to start transaction")?;

    // Prepare statements for reuse
    let group_stmt = tx
        .prepare(
            "
    INSERT INTO entity_group 
    (id, name, created_at, updated_at, confidence_score, entity_count, version) 
    VALUES ($1, $2, $3, $4, $5, $6, 1)
",
        )
        .await
        .context("Failed to prepare entity_group statement")?;

    let entity_stmt = tx
        .prepare(
            "
        INSERT INTO group_entity 
        (id, entity_group_id, entity_id, created_at) 
        VALUES ($1, $2, $3, $4)
    ",
        )
        .await
        .context("Failed to prepare group_entity statement")?;

    let method_stmt = tx
        .prepare(
            "
        INSERT INTO group_method 
        (id, entity_group_id, method_type, description, match_values, confidence_score, created_at) 
        VALUES ($1, $2, $3, $4, $5, $6, $7)
    ",
        )
        .await
        .context("Failed to prepare group_method statement")?;

    let update_group_count_stmt = tx
        .prepare(
            "
    UPDATE entity_group
    SET entity_count = entity_count + 1,
        updated_at = $1,
        version = version + 1
    WHERE id = $2
",
        )
        .await
        .context("Failed to prepare update_group_count statement")?;

    let update_method_stmt = tx
        .prepare(
            "
        UPDATE group_method
        SET match_values = $1
        WHERE id = $2
    ",
        )
        .await
        .context("Failed to prepare update_method statement")?;

    let now = Utc::now().naive_utc();
    let mut total_groups_created = 0;
    let mut total_entities_added = 0;
    let mut total_entities_matched = 0;
    let mut confidence_scores = Vec::new();
    let mut group_sizes = Vec::new();

    // Process phone numbers
    for (normalized_phone, entity_map) in phone_map {
        // Store value for logging
        let entity_map_length = entity_map.len();

        // Check if this phone already has a group
        if let Some((group_id, method_id)) = existing_phone_groups.get(&normalized_phone) {
            // Add these entities to the existing group

            // First get the current match values for this group
            let match_values_row = tx
                .query_one(
                    "SELECT match_values FROM group_method WHERE id = $1",
                    &[&method_id],
                )
                .await
                .context("Failed to get current match values")?;

            let match_values_json: serde_json::Value = match_values_row.get("match_values");

            // Parse the current match values
            let current_match_values = serde_json::from_value::<MatchValues>(match_values_json)
                .context("Failed to parse current match values")?;

            // Extract the phone values
            let mut phone_values = if let MatchValues::Phone(values) = current_match_values {
                values
            } else {
                Vec::new()
            };

            // Calculate confidence score based on extensions
            // Collect all extensions to check if they're consistent
            let all_extensions: Vec<Option<String>> = phone_values
                .iter()
                .map(|v| v.extension.clone())
                .chain(entity_map.iter().map(|(_, (_, ext))| ext.clone()))
                .collect();

            // If all entities have the same extension (or all have none), confidence is high
            // Otherwise, slightly lower confidence
            let all_same_extension = all_extensions.windows(2).all(|w| w[0] == w[1]);

            let confidence_score = if all_same_extension { 0.95 } else { 0.85 };
            confidence_scores.push(confidence_score);

            // Description for this match
            let description = if all_same_extension {
                format!("Matched on phone number: {}", normalized_phone)
            } else {
                format!(
                    "Matched on phone number: {} (different extensions)",
                    normalized_phone
                )
            };

            // Track which entities we've actually added
            let mut entities_added = 0;
            let mut entity_pairs = Vec::new();

            // Add the new entities to the group
            for (entity_id, (original, extension)) in entity_map {
                // Add entity to group
                tx.execute(
                    &entity_stmt,
                    &[&Uuid::new_v4().to_string(), &group_id, &entity_id.0, &now],
                )
                .await
                .context("Failed to insert group entity")?;

                // Update entity count for the group
                tx.execute(&update_group_count_stmt, &[&now, &group_id])
                    .await
                    .context("Failed to update group count")?;

                // Add to match values
                phone_values.push(PhoneMatchValue {
                    original,
                    normalized: normalized_phone.clone(),
                    extension,
                    entity_id: entity_id.clone(),
                });

                // Track entity pairs for ML feedback
                if reinforcement_orchestrator.is_some() {
                    // Store this entity with each existing entity in the group
                    // for ML feedback
                    for existing_value in &phone_values {
                        if existing_value.entity_id != entity_id {
                            entity_pairs
                                .push((entity_id.clone(), existing_value.entity_id.clone()));
                        }
                    }
                }

                entities_added += 1;
            }

            // Update the match values
            let updated_match_values = MatchValues::Phone(phone_values);
            let updated_json = serde_json::to_value(updated_match_values)
                .context("Failed to serialize updated match values")?;

            tx.execute(&update_method_stmt, &[&updated_json, &method_id])
                .await
                .context("Failed to update match values")?;

            // Log ML feedback for newly formed entity pairs
            if reinforcement_orchestrator.is_some() {
                // Get the group confidence score
                let query = "SELECT confidence_score FROM entity_group WHERE id = $1";
                let group_confidence: f64 = match tx.query_one(query, &[&group_id]).await {
                    Ok(row) => row.get("confidence_score"),
                    Err(e) => {
                        warn!(
                            "Failed to get confidence score for group {}: {}",
                            group_id, e
                        );
                        confidence_score // Use calculated confidence if query fails
                    }
                };

                // Lock the orchestrator for each call
                if let Some(orchestrator_ref) = reinforcement_orchestrator {
                    let mut orchestrator = orchestrator_ref.lock().await;

                    for (entity1, entity2) in entity_pairs {
                        match orchestrator
                            .log_match_result(
                                &MatchMethodType::Phone,
                                group_confidence,
                                true,
                                &entity1,
                                &entity2,
                            )
                            .await
                        {
                            Ok(_) => (),
                            Err(e) => warn!("Failed to log match result to ML system: {}", e),
                        }
                    }
                }
            }

            total_entities_added += entities_added;
            total_entities_matched += entities_added;

            // Query for updated group size
            let group_size_row = tx
                .query_one(
                    "SELECT entity_count FROM entity_group WHERE id = $1",
                    &[&group_id],
                )
                .await
                .context("Failed to get updated group size")?;

            let updated_group_size: i32 = group_size_row.get("entity_count");
            group_sizes.push(updated_group_size as f32);

            info!(
                "Added {} entities to existing phone group for {}",
                entities_added, normalized_phone
            );
        } else if entity_map.len() >= 2 {
            // Create a new group for this phone (only if multiple entities)

            // Convert the entity map to a vector of (EntityId, String, Option<String>) tuples
            let entities: Vec<(EntityId, String, Option<String>)> = entity_map
                .into_iter()
                .map(|(entity_id, (number, extension))| (entity_id, number, extension))
                .collect();

            let entity_count = entities.len() as i32;

            // Calculate confidence score based on extensions
            // If all entities have the same extension (or all have none), confidence is high
            // Otherwise, slightly lower confidence
            let all_same_extension = entities
                .iter()
                .map(|(_, _, ext)| ext.clone())
                .collect::<Vec<_>>()
                .windows(2)
                .all(|w| w[0] == w[1]);

            // Initialize default confidence score
            let mut confidence_score = if all_same_extension { 0.95 } else { 0.85 };

            // If the orchestrator is available, use it to get a ML-guided confidence
            if reinforcement_orchestrator.is_some() {
                let mut ml_confidence_scores = Vec::new();

                // Sample entity pairs to get confidence scores
                let max_pairs = 5;
                let pair_count = std::cmp::min(max_pairs, (entity_count * (entity_count - 1)) / 2);
                let mut pairs_checked = 0;

                // Check a sample of entity pairs
                'outer: for i in 0..entities.len() {
                    for j in (i + 1)..entities.len() {
                        if pairs_checked >= pair_count {
                            break 'outer;
                        }

                        // Get ML recommendation for this pair
                        if let Some(orchestrator_ref) = reinforcement_orchestrator {
                            // Lock for each method call
                            let mut orchestrator = orchestrator_ref.lock().await;
                            match orchestrator
                                .select_matching_method(pool, &entities[i].0, &entities[j].0)
                                .await
                            {
                                Ok((method, conf)) => {
                                    // If phone is recommended method, use its confidence
                                    if matches!(method, MatchMethodType::Phone) {
                                        ml_confidence_scores.push(conf);
                                    }
                                }
                                Err(e) => {
                                    // Log error but continue
                                    warn!("Error getting ML confidence: {}", e);
                                }
                            }
                        }

                        pairs_checked += 1;
                    }
                }

                // If we got any ML confidence scores, use their average
                if !ml_confidence_scores.is_empty() {
                    confidence_score = ml_confidence_scores.iter().sum::<f64>()
                        / ml_confidence_scores.len() as f64;
                    info!(
                        "Using ML-guided confidence {:.4} for phone group",
                        confidence_score
                    );
                } else {
                    info!("Using default confidence for phone group (no ML guidance)");
                }
            }

            confidence_scores.push(confidence_score);

            // Create a new entity group
            let group_id = EntityGroupId(Uuid::new_v4().to_string());

            let group = EntityGroup {
                id: group_id.clone(),
                name: Some(format!("Phone match on {}", normalized_phone)),
                group_cluster_id: None,
                created_at: now,
                updated_at: now,
                confidence_score,
                entity_count,
            };

            // Insert group
            tx.execute(
                &group_stmt,
                &[
                    &group.id.0,
                    &group.name,
                    &group.created_at,
                    &group.updated_at,
                    &group.confidence_score,
                    &group.entity_count,
                ],
            )
            .await
            .context("Failed to insert entity group")?;

            // Description for this match
            let description = if all_same_extension {
                format!("Matched on phone number: {}", normalized_phone)
            } else {
                format!(
                    "Matched on phone number: {} (different extensions)",
                    normalized_phone
                )
            };

            // Create match values for the group method
            let mut match_values = Vec::new();

            // Add all entities to the batch inserts
            for (entity_id, original, extension) in &entities {
                // Add to group_entity
                tx.execute(
                    &entity_stmt,
                    &[&Uuid::new_v4().to_string(), &group_id.0, &entity_id.0, &now],
                )
                .await
                .context("Failed to insert group entity")?;

                // Create match value for this entity
                match_values.push(PhoneMatchValue {
                    original: original.clone(),
                    normalized: normalized_phone.clone(),
                    extension: extension.clone(),
                    entity_id: entity_id.clone(),
                });
            }

            // Serialize match values to JSON
            let method_values = MatchValues::Phone(match_values);
            let match_values_json =
                serde_json::to_value(&method_values).context("Failed to serialize match values")?;

            // Insert group method
            tx.execute(
                &method_stmt,
                &[
                    &Uuid::new_v4().to_string(),
                    &group_id.0,
                    &MatchMethodType::Phone.as_str(),
                    &description,
                    &match_values_json,
                    &confidence_score,
                    &now,
                ],
            )
            .await
            .context("Failed to insert group method")?;

            // Log ML feedback for the newly created group
            if reinforcement_orchestrator.is_some() {
                // For a new group, log feedback for a sample of entity pairs
                let max_feedback_pairs = 10;
                let mut pairs_logged = 0;

                if let Some(orchestrator_ref) = reinforcement_orchestrator {
                    for i in 0..entities.len() {
                        for j in (i + 1)..entities.len() {
                            if pairs_logged >= max_feedback_pairs {
                                break;
                            }

                            let mut orchestrator = orchestrator_ref.lock().await;
                            // Phone matches are considered correct (true)
                            match orchestrator
                                .log_match_result(
                                    &MatchMethodType::Phone,
                                    confidence_score,
                                    true,
                                    &entities[i].0,
                                    &entities[j].0,
                                )
                                .await
                            {
                                Ok(_) => pairs_logged += 1,
                                Err(e) => warn!("Failed to log match result to ML system: {}", e),
                            }
                        }
                    }
                }
            }

            total_groups_created += 1;
            total_entities_matched += entity_count as usize;
            total_entities_added += entity_count as usize;

            group_sizes.push(entity_count as f32);

            info!(
                "Created new phone group for {} with {} entities (confidence: {:.4})",
                normalized_phone, entity_count, confidence_score
            );
        }
    }

    // Commit the transaction
    tx.commit().await.context("Failed to commit transaction")?;

    // Calculate average confidence score and group size
    let avg_confidence: f64 = if !confidence_scores.is_empty() {
        confidence_scores.iter().sum::<f64>() / confidence_scores.len() as f64
    } else {
        0.0
    };

    let avg_group_size: f64 = if !group_sizes.is_empty() {
        (group_sizes.iter().sum::<f32>() / group_sizes.len() as f32).into()
    } else {
        0.0
    };

    // Create method stats
    let method_stats = MatchMethodStats {
        method_type: MatchMethodType::Phone,
        groups_created: total_groups_created,
        entities_matched: total_entities_matched,
        avg_confidence,
        avg_group_size,
    };

    let elapsed = start_time.elapsed();
    info!(
        "Phone matching complete: created {} new entity groups and added {} entities to existing groups in {:.2?}",
        total_groups_created, total_entities_added, elapsed
    );

    Ok(PhoneMatchResult {
        groups_created: total_groups_created,
        stats: method_stats,
    })
}

/// Normalize a phone number by:
/// - Removing all non-numeric characters
/// - Handling country codes
/// - Standardizing format
fn normalize_phone(phone: &str) -> String {
    // 1. Remove all non-numeric characters (spaces, dashes, parentheses, etc.)
    let digits_only: String = phone.chars().filter(|c| c.is_ascii_digit()).collect();

    // Skip if we don't have enough digits
    if digits_only.len() < 7 {
        return String::new();
    }

    // 2. Handle different phone number formats
    // For US numbers:
    // - If 10 digits, assume US number
    // - If 11 digits and starts with 1, assume US number with country code

    if digits_only.len() == 10 {
        // Standard US number, keep as is
        return digits_only;
    } else if digits_only.len() == 11 && digits_only.starts_with('1') {
        // US number with country code, strip the leading 1
        return digits_only[1..].to_string();
    }

    // Return whatever we have for other cases
    // In a production system, you'd use a library to properly normalize international numbers
    digits_only
}
