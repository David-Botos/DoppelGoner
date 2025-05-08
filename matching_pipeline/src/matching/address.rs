// src/matching/address.rs
use anyhow::{Context, Result};
use chrono::Utc;
use log::{debug, info, warn};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::db::PgPool;
use crate::models::{
    AddressMatchValue, EntityGroup, EntityGroupId, EntityId, MatchMethodType, MatchValues,
};
use crate::reinforcement;
use crate::results::{AddressMatchResult, MatchMethodStats};

pub async fn find_matches(
    pool: &PgPool,
    reinforcement_orchestrator: Option<&Mutex<reinforcement::MatchingOrchestrator>>,
) -> Result<AddressMatchResult> {
    info!(
        "Starting address matching{}...",
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
        .context("Failed to get DB connection for address matching")?;

    // First, get all entities that are already part of an address match group
    debug!("Finding entities already processed by address matching");
    let processed_query = "
        SELECT DISTINCT ge.entity_id
        FROM group_entity ge
        JOIN group_method gm ON ge.entity_group_id = gm.entity_group_id
        WHERE gm.method_type = 'address'
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
        "Found {} entities already processed by address matching",
        processed_entities.len()
    );

    // Get existing address groups and their normalized addresses
    debug!("Finding existing address groups");
    let existing_groups_query = "
        SELECT 
            eg.id AS group_id,
            gm.id AS method_id,
            gm.match_values
        FROM 
            entity_group eg
            JOIN group_method gm ON eg.id = gm.entity_group_id
        WHERE 
            gm.method_type = 'address'
    ";

    let existing_groups_rows = conn
        .query(existing_groups_query, &[])
        .await
        .context("Failed to query existing groups")?;

    // Map of normalized address -> (group_id, method_id)
    let mut existing_address_groups: HashMap<String, (String, String)> = HashMap::new();

    for row in &existing_groups_rows {
        let group_id: String = row.get("group_id");
        let method_id: String = row.get("method_id");
        let match_values_json: serde_json::Value = row.get("match_values");

        // Parse the match values
        if let Ok(match_values) = serde_json::from_value::<MatchValues>(match_values_json.clone()) {
            if let MatchValues::Address(values) = match_values {
                if !values.is_empty() {
                    // Use the normalized address from the first value
                    let normalized = values[0].normalized.clone();
                    existing_address_groups.insert(normalized, (group_id, method_id));
                }
            }
        }
    }

    info!(
        "Found {} existing address groups",
        existing_address_groups.len()
    );

    // Query addresses linked to locations and entities
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
            entity e
            JOIN entity_feature ef ON e.id = ef.entity_id
            JOIN location l ON ef.table_id = l.id AND ef.table_name = 'location'
            JOIN address a ON a.location_id = l.id
        WHERE 
            a.address_1 IS NOT NULL 
            AND a.address_1 != ''
            AND a.city IS NOT NULL 
            AND a.city != ''
    ";

    debug!("Executing address query");
    let address_rows = conn
        .query(address_query, &[])
        .await
        .context("Failed to query addresses")?;

    info!("Found {} address records to process", address_rows.len());

    // Process addresses into a structured format
    let mut address_map: HashMap<String, HashMap<EntityId, String>> = HashMap::new();
    let mut processed_count = 0;

    for row in &address_rows {
        let entity_id: String = row.get("entity_id");
        let entity_id = EntityId(entity_id);

        // Skip if this entity has already been processed
        if processed_entities.contains(&entity_id) {
            continue;
        }

        let address_1: String = row.get("address_1");
        let address_2: Option<String> = row.try_get("address_2").unwrap_or(None);
        let city: String = row.get("city");
        let state_province: String = row.get("state_province");
        let postal_code: String = row.get("postal_code");
        let country: String = row.get("country");

        // Assemble full address
        let full_address = format!(
            "{}{}, {}, {} {}, {}",
            address_1,
            address_2.map_or("".to_string(), |a| format!(", {}", a)),
            city,
            state_province,
            postal_code,
            country
        );

        // Normalize the address
        let normalized_address = normalize_address(&full_address);

        if !normalized_address.is_empty() {
            // Get or create the inner entity map for this normalized address
            let entity_map = address_map.entry(normalized_address).or_default();

            // Only add if this entity isn't already in the map for this address
            entity_map.entry(entity_id).or_insert(full_address);
        }

        processed_count += 1;
        if processed_count % 1000 == 0 {
            debug!("Processed {} address records so far", processed_count);
        }
    }

    let unique_addresses = address_map.len();
    info!(
        "Found {} unique normalized addresses from unprocessed records",
        unique_addresses
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

    // Process addresses
    for (normalized_address, entity_map) in address_map {
        // Store value for logging before entity_map is consumed
        let entity_map_length = entity_map.len();
        // Check if this address already has a group
        if let Some((group_id, method_id)) = existing_address_groups.get(&normalized_address) {
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

            // Extract the address values
            let mut address_values = if let MatchValues::Address(values) = current_match_values {
                values
            } else {
                Vec::new()
            };

            // Track which entities we've actually added and entity pairs for ML logging
            let mut entities_added = 0;
            let mut entity_pairs = Vec::new();

            // Add the new entities to the group
            for (entity_id, original) in entity_map {
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
                address_values.push(AddressMatchValue {
                    original,
                    normalized: normalized_address.clone(),
                    match_score: Some(1.0),
                    entity_id: entity_id.clone(),
                });

                // Track entity pairs for ML feedback
                if reinforcement_orchestrator.is_some() {
                    // Store this entity with each existing entity in the group
                    // for ML feedback
                    for existing_value in &address_values {
                        if existing_value.entity_id != entity_id {
                            entity_pairs
                                .push((entity_id.clone(), existing_value.entity_id.clone()));
                        }
                    }
                }

                entities_added += 1;
            }

            // Update the match values
            let updated_match_values = MatchValues::Address(address_values);
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
                        0.95 // Default to 0.95 for address groups if query fails
                    }
                };

                // Lock the orchestrator for each call
                if let Some(orchestrator_ref) = reinforcement_orchestrator {
                    let mut orchestrator = orchestrator_ref.lock().await;

                    for (entity1, entity2) in entity_pairs {
                        match orchestrator
                            .log_match_result(
                                &MatchMethodType::Address,
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

            // Query for updated group size to track statistics
            let group_size_row = tx
                .query_one(
                    "SELECT entity_count FROM entity_group WHERE id = $1",
                    &[&group_id],
                )
                .await
                .context("Failed to get updated group size")?;

            let updated_group_size: i32 = group_size_row.get("entity_count");
            group_sizes.push(updated_group_size as f32);

            // Address matches have high confidence
            confidence_scores.push(0.95);

            info!(
                "Added {} entities to existing address group for normalized address",
                entity_map_length
            );
        } else if entity_map.len() >= 2 {
            // Create a new group for this address (only if multiple entities)

            // Convert the entity map to a vector of (EntityId, String) pairs
            let entities: Vec<(EntityId, String)> = entity_map.into_iter().collect();
            let entity_count = entities.len() as i32;

            // Initialize default confidence score - high for address matches
            let mut confidence_score = 0.95;

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
                                    // If address is recommended method, use its confidence
                                    if matches!(method, MatchMethodType::Address) {
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
                        "Using ML-guided confidence {:.4} for address group",
                        confidence_score
                    );
                } else {
                    info!("Using default confidence 0.95 for address group (no ML guidance)");
                }
            }

            // Create a new entity group
            let group_id = EntityGroupId(Uuid::new_v4().to_string());

            let group = EntityGroup {
                id: group_id.clone(),
                name: Some(format!("Address match")),
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

            // Create match values for the group method
            let mut match_values = Vec::new();

            // Add all entities to the inserts
            for (entity_id, original) in &entities {
                // Add to group_entity
                tx.execute(
                    &entity_stmt,
                    &[&Uuid::new_v4().to_string(), &group_id.0, &entity_id.0, &now],
                )
                .await
                .context("Failed to insert group entity")?;

                // Create match value for this entity
                match_values.push(AddressMatchValue {
                    original: original.clone(),
                    normalized: normalized_address.clone(),
                    match_score: Some(1.0),
                    entity_id: entity_id.clone(),
                });
            }

            // Serialize match values to JSON
            let method_values = MatchValues::Address(match_values);
            let match_values_json =
                serde_json::to_value(&method_values).context("Failed to serialize match values")?;

            // Insert group method
            tx.execute(
                &method_stmt,
                &[
                    &Uuid::new_v4().to_string(),
                    &group_id.0,
                    &MatchMethodType::Address.as_str(),
                    &format!("Matched on normalized address"),
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
                            // Address matches are considered correct (true)
                            match orchestrator
                                .log_match_result(
                                    &MatchMethodType::Address,
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

            // Address matches have high confidence
            confidence_scores.push(confidence_score);
            group_sizes.push(entity_count as f32);

            info!(
                "Created new address group with {} entities (confidence: {:.4})",
                entity_count, confidence_score
            );
        }
    }

    // Commit the transaction
    tx.commit().await.context("Failed to commit transaction")?;

    // Calculate average group size and confidence
    let avg_group_size: f64 = if !group_sizes.is_empty() {
        (group_sizes.iter().sum::<f32>() / group_sizes.len() as f32).into()
    } else {
        0.0
    };

    let avg_confidence: f64 = if !confidence_scores.is_empty() {
        confidence_scores.iter().sum::<f64>() / confidence_scores.len() as f64
    } else {
        0.0
    };

    // Create method stats
    let method_stats = MatchMethodStats {
        method_type: MatchMethodType::Address,
        groups_created: total_groups_created,
        entities_matched: total_entities_matched,
        avg_confidence,
        avg_group_size,
    };

    let elapsed = start_time.elapsed();
    info!(
        "Address matching complete: created {} new entity groups and added {} entities to existing groups in {:.2?}",
        total_groups_created, total_entities_added, elapsed
    );

    Ok(AddressMatchResult {
        groups_created: total_groups_created,
        stats: method_stats,
    })
}

/// Normalize an address by:
/// - Converting to lowercase
/// - Removing punctuation
/// - Standardizing common abbreviations
/// - Removing apartment/suite numbers
fn normalize_address(address: &str) -> String {
    // Convert to lowercase
    let lower = address.to_lowercase();

    // Remove punctuation and extra whitespace
    let mut normalized = lower
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>();

    normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");

    // Replace common abbreviations
    normalized = normalized
        .replace(" st ", " street ")
        .replace(" rd ", " road ")
        .replace(" ave ", " avenue ")
        .replace(" blvd ", " boulevard ")
        .replace(" dr ", " drive ")
        .replace(" ln ", " lane ")
        .replace(" apt ", " ")
        .replace(" suite ", " ")
        .replace(" unit ", " ")
        .replace(" #", " ");

    // Remove trailing commas and normalize state codes
    normalized = normalized.trim_end_matches(',').to_string();

    normalized
}
