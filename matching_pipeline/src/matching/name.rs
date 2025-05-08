// src/matching/name.rs

use anyhow::{Context, Result};
use chrono::Utc;
use log::{info, warn};
use regex::Regex;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use strsim::jaro_winkler;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::db::PgPool;
use crate::models::{
    Entity, EntityGroup, EntityGroupId, EntityId, GroupEntity, GroupMethod, MatchMethodType,
    MatchValues, OrganizationId,
};
use crate::reinforcement;
use crate::results::{MatchMethodStats, NameMatchResult};

// Configuration for name matching
const MIN_FUZZY_SIMILARITY: f32 = 0.85; // Threshold for fuzzy string matching
const MIN_SEMANTIC_SIMILARITY: f32 = 0.88; // Threshold for semantic similarity
const COMBINED_THRESHOLD: f32 = 0.86; // Threshold for combined approach
const FUZZY_WEIGHT: f32 = 0.4; // Weight for fuzzy matching in combined score
const SEMANTIC_WEIGHT: f32 = 0.6; // Weight for semantic matching in combined score

// Confidence scores based on match type
const CONFIDENCE_FUZZY: f32 = 0.80;
const CONFIDENCE_SEMANTIC: f32 = 0.90;
const CONFIDENCE_COMBINED: f32 = 0.85;

/// Represents a match between organization names
#[derive(Debug, Clone)]
struct NameMatch {
    entity_id: EntityId,
    original_name: String,
    normalized_name: String,
    match_score: f32,
    match_type: String, // "fuzzy", "semantic", or "combined"
}

/// Main function to find matches based on organization names
pub async fn find_matches(
    pool: &PgPool,
    reinforcement_orchestrator: Option<&Mutex<reinforcement::MatchingOrchestrator>>,
) -> Result<NameMatchResult> {
    info!(
        "Starting name-based entity matching{}...",
        if reinforcement_orchestrator.is_some() {
            " with ML guidance"
        } else {
            ""
        }
    );
    let start_time = std::time::Instant::now();

    // Get database connection
    let conn = pool.get().await.context("Failed to get DB connection")?;

    // 1. Get all entities that haven't been processed by name matching yet
    info!("Finding unprocessed entities for name matching");
    let unprocessed_entities = get_unprocessed_entities(&conn).await?;
    info!(
        "Found {} unprocessed entities for name matching",
        unprocessed_entities.len()
    );

    if unprocessed_entities.is_empty() {
        info!("No new entities to process for name matching");
        return Ok(NameMatchResult {
            groups_created: 0,
            stats: MatchMethodStats {
                method_type: MatchMethodType::Name,
                groups_created: 0,
                entities_matched: 0,
                avg_confidence: 0.0,
                avg_group_size: 0.0,
            },
        });
    }

    // 2. Get existing name-based groups
    info!("Retrieving existing name-based groups");
    let existing_groups = get_existing_name_groups(&conn).await?;
    info!("Found {} existing name-based groups", existing_groups.len());

    // 3. Process entities against existing groups first
    info!("Matching entities against existing groups");
    let mut processed_entities = HashSet::new();
    let mut entities_added_to_existing = 0;

    // Skip entities without names
    let valid_entities: Vec<&Entity> = unprocessed_entities
        .iter()
        .filter(|e| e.name.is_some() && !e.name.as_ref().unwrap().trim().is_empty())
        .collect();

    // Normalize names and get embeddings
    let mut entity_names = HashMap::new();
    for entity in &valid_entities {
        if let Some(name) = &entity.name {
            let normalized = normalize_name(name);
            entity_names.insert(entity.id.clone(), normalized);
        }
    }

    // Get organization embeddings for all relevant entities
    let org_embeddings = get_organization_embeddings(&conn, &valid_entities).await?;

    // Try to match against existing groups
    for entity in &valid_entities {
        // Skip if already processed in this run
        if processed_entities.contains(&entity.id) {
            continue;
        }

        let normalized_name = match entity_names.get(&entity.id) {
            Some(name) => name,
            None => continue, // Skip if no normalized name
        };

        let entity_embedding = org_embeddings
            .get(&entity.id)
            .and_then(|opt_vec| opt_vec.as_ref());

        if let Some(group_id) = match_to_existing_group(
            &conn,
            entity,
            normalized_name,
            entity_embedding,
            &existing_groups,
            &entity_names,
            &org_embeddings,
            reinforcement_orchestrator,
            pool,
        )
        .await?
        {
            processed_entities.insert(entity.id.clone());
            entities_added_to_existing += 1;
        }
    }

    info!(
        "Added {} entities to existing name groups",
        entities_added_to_existing
    );

    // 4. Process remaining entities to form new groups
    let remaining_entities: Vec<&Entity> = valid_entities
        .iter()
        .filter(|e| !processed_entities.contains(&e.id))
        .copied()
        .collect();

    info!(
        "Creating new groups for {} remaining entities",
        remaining_entities.len()
    );
    let mut new_groups = 0;
    let mut new_group_entity_count = 0;

    // Only proceed if we have entities to process
    if !remaining_entities.is_empty() {
        let (created_groups, matched_entities) = create_name_groups(
            &conn,
            &remaining_entities,
            &entity_names,
            &org_embeddings,
            reinforcement_orchestrator,
            pool,
        )
        .await?;

        new_groups = created_groups.len();
        new_group_entity_count = matched_entities.len();

        // Add processed entities to our tracking set
        for entity_id in matched_entities {
            processed_entities.insert(entity_id);
        }
    }

    // 5. Calculate final statistics
    let total_entities_matched = processed_entities.len();
    let total_groups = new_groups + existing_groups.len();

    let avg_confidence = if total_groups > 0 {
        calculate_name_match_confidence(&conn, existing_groups.keys().cloned().collect()).await?
    } else {
        0.0
    };

    let avg_group_size = if total_groups > 0 {
        calculate_name_group_size(&conn, total_groups).await?
    } else {
        0.0
    };

    info!(
        "Name matching completed in {:.2?}: {} total groups, {} entities matched",
        start_time.elapsed(),
        total_groups,
        total_entities_matched
    );

    Ok(NameMatchResult {
        groups_created: new_groups,
        stats: MatchMethodStats {
            method_type: MatchMethodType::Name,
            groups_created: total_groups,
            entities_matched: total_entities_matched,
            avg_confidence,
            avg_group_size,
        },
    })
}

/// Retrieve entities that haven't been processed by name matching
async fn get_unprocessed_entities(conn: &tokio_postgres::Client) -> Result<Vec<Entity>> {
    // Query for entities not yet processed by name matching
    let query = "
        SELECT e.* 
        FROM entity e
        WHERE NOT EXISTS (
            SELECT 1 
            FROM group_entity ge
            JOIN group_method gm ON ge.entity_group_id = gm.entity_group_id
            WHERE ge.entity_id = e.id 
            AND gm.method_type = 'name'
        )
        AND e.name IS NOT NULL
    ";

    let rows = conn.query(query, &[]).await?;
    let mut entities = Vec::with_capacity(rows.len());

    for row in &rows {
        let entity_id: String = row.get("id");
        let org_id: String = row.get("organization_id");
        let name: Option<String> = row.try_get("name").ok();
        let created_at = row.get("created_at");
        let updated_at = row.get("updated_at");
        let source_system: Option<String> = row.try_get("source_system").ok();
        let source_id: Option<String> = row.try_get("source_id").ok();

        entities.push(Entity {
            id: EntityId(entity_id),
            organization_id: OrganizationId(org_id),
            name,
            created_at,
            updated_at,
            source_system,
            source_id,
        });
    }

    Ok(entities)
}

/// Get existing name-based groups with their entities
async fn get_existing_name_groups(
    conn: &tokio_postgres::Client,
) -> Result<HashMap<String, Vec<EntityId>>> {
    // Query for groups created by name matching
    let query = "
        SELECT DISTINCT eg.id as group_id
        FROM entity_group eg
        JOIN group_method gm ON eg.id = gm.entity_group_id
        WHERE gm.method_type = 'name'
    ";

    let rows = conn.query(query, &[]).await?;
    let mut groups = HashMap::new();

    for row in &rows {
        let group_id: String = row.get("group_id");

        // For each group, get its entities
        let entity_query = "
            SELECT entity_id 
            FROM group_entity 
            WHERE entity_group_id = $1
        ";

        let entity_rows = conn.query(entity_query, &[&group_id]).await?;
        let mut entities = Vec::new();

        for entity_row in &entity_rows {
            let entity_id: String = entity_row.get("entity_id");
            entities.push(EntityId(entity_id));
        }

        groups.insert(group_id, entities);
    }

    Ok(groups)
}

/// Match an entity to existing groups
async fn match_to_existing_group(
    conn: &tokio_postgres::Client,
    entity: &Entity,
    normalized_name: &str,
    entity_embedding: Option<&Vec<f32>>,
    existing_groups: &HashMap<String, Vec<EntityId>>,
    entity_names: &HashMap<EntityId, String>,
    org_embeddings: &HashMap<EntityId, Option<Vec<f32>>>,
    reinforcement_orchestrator: Option<&Mutex<reinforcement::MatchingOrchestrator>>,
    pool: &PgPool,
) -> Result<Option<String>> {
    let mut best_group_id = None;
    let mut best_match_score = 0.0;
    let mut best_match_type = String::new();
    let mut best_match_entity = None;

    // For each group, try to match the entity
    for (group_id, group_entities) in existing_groups {
        // Check matches against each entity in this group
        for other_entity_id in group_entities {
            // Skip self-comparison
            if &entity.id == other_entity_id {
                continue;
            }

            // Get the other entity's normalized name
            let other_name = match entity_names.get(other_entity_id) {
                Some(name) => name,
                None => {
                    // If we don't have the normalized name, try to get the entity details
                    let query = "SELECT name FROM entity WHERE id = $1";
                    let row = match conn.query_opt(query, &[&other_entity_id.0]).await? {
                        Some(row) => row,
                        None => continue,
                    };

                    let other_entity_name: Option<String> = row.try_get("name").ok();
                    match other_entity_name {
                        Some(name) => &normalize_name(&name),
                        None => continue,
                    }
                }
            };

            // Calculate fuzzy string similarity
            let fuzzy_score = jaro_winkler(normalized_name, other_name) as f32;

            // Calculate semantic similarity if embeddings available
            let semantic_score = match (entity_embedding, org_embeddings.get(other_entity_id)) {
                (Some(emb1), Some(Some(emb2))) => cosine_similarity(emb1, emb2),
                _ => 0.0,
            };

            // Determine the combined score and match type
            let (combined_score, match_type) = if semantic_score >= MIN_SEMANTIC_SIMILARITY {
                // If semantic score is high enough, weight both scores
                let score = (fuzzy_score * FUZZY_WEIGHT) + (semantic_score * SEMANTIC_WEIGHT);
                (score, "combined".to_string())
            } else if fuzzy_score >= MIN_FUZZY_SIMILARITY {
                // If only fuzzy score is high enough
                (fuzzy_score, "fuzzy".to_string())
            } else {
                (0.0, "none".to_string())
            };

            // Check if this is a match and if it's better than previous matches
            if combined_score >= COMBINED_THRESHOLD && combined_score > best_match_score {
                best_match_score = combined_score;
                best_match_type = match_type;
                best_group_id = Some(group_id.clone());
                best_match_entity = Some(other_entity_id.clone());
            }
        }
    }

    // If we found a match, add the entity to the group
    if let Some(group_id) = &best_group_id {
        if let Some(original_name) = &entity.name {
            // Add to this group
            add_entity_to_name_group(
                conn,
                &entity.id,
                group_id,
                original_name,
                normalized_name,
                best_match_score,
                &best_match_type,
                best_match_entity.as_ref().unwrap(),
                reinforcement_orchestrator,
                pool,
            )
            .await?;
        }
    }

    Ok(best_group_id)
}

/// Create new groups from remaining entities
async fn create_name_groups(
    conn: &tokio_postgres::Client,
    entities: &[&Entity],
    entity_names: &HashMap<EntityId, String>,
    org_embeddings: &HashMap<EntityId, Option<Vec<f32>>>,
    reinforcement_orchestrator: Option<&Mutex<reinforcement::MatchingOrchestrator>>,
    pool: &PgPool,
) -> Result<(Vec<String>, HashSet<EntityId>)> {
    let entity_count = entities.len();
    let mut processed_entities = HashSet::new();
    let mut created_groups = Vec::new();

    if entity_count < 2 {
        // Need at least 2 entities to form a group
        return Ok((created_groups, processed_entities));
    }

    // Build similarity matrix between all pairs of entities
    let mut similarity_matrix = vec![vec![(0.0, String::new()); entity_count]; entity_count];

    for i in 0..entity_count {
        for j in (i + 1)..entity_count {
            let entity_i = entities[i];
            let entity_j = entities[j];

            // Get normalized names
            let name_i = match entity_names.get(&entity_i.id) {
                Some(name) => name,
                None => continue,
            };

            let name_j = match entity_names.get(&entity_j.id) {
                Some(name) => name,
                None => continue,
            };

            // Calculate fuzzy similarity
            let fuzzy_score = jaro_winkler(name_i, name_j) as f32;

            // Calculate semantic similarity if embeddings available
            let semantic_score = match (
                org_embeddings.get(&entity_i.id),
                org_embeddings.get(&entity_j.id),
            ) {
                (Some(Some(emb_i)), Some(Some(emb_j))) => cosine_similarity(emb_i, emb_j),
                _ => 0.0,
            };

            // If reinforcement_orchestrator is available, use it to get ML-guided confidence
            let mut ml_score = None;
            if let Some(orchestrator_ref) = reinforcement_orchestrator {
                let mut orchestrator = orchestrator_ref.lock().await;
                match orchestrator
                    .select_matching_method(pool, &entity_i.id, &entity_j.id)
                    .await
                {
                    Ok((method, conf)) => {
                        // If name is recommended method, use its confidence
                        if matches!(method, MatchMethodType::Name) {
                            ml_score = Some(conf);
                        }
                    }
                    Err(e) => {
                        // Log error but continue
                        warn!("Error getting ML confidence for name matching: {}", e);
                    }
                }
            }

            // Determine the combined score and match type
            let (combined_score, match_type) = if let Some(score) = ml_score {
                // If ML provided a score, use it preferentially but blend with semantic/fuzzy
                let standard_score = if semantic_score >= MIN_SEMANTIC_SIMILARITY {
                    // If semantic score is high enough, weight both scores
                    (fuzzy_score * FUZZY_WEIGHT) + (semantic_score * SEMANTIC_WEIGHT)
                } else if fuzzy_score >= MIN_FUZZY_SIMILARITY {
                    // If only fuzzy score is high enough
                    fuzzy_score
                } else {
                    0.0
                };

                // Blend ML score with standard score - 70% ML, 30% standard
                let blended = (score as f32 * 0.7) + (standard_score * 0.3);
                (blended, "ml_guided".to_string())
            } else if semantic_score >= MIN_SEMANTIC_SIMILARITY {
                // If semantic score is high enough, weight both scores
                let score = (fuzzy_score * FUZZY_WEIGHT) + (semantic_score * SEMANTIC_WEIGHT);
                (score, "combined".to_string())
            } else if fuzzy_score >= MIN_FUZZY_SIMILARITY {
                // If only fuzzy score is high enough
                (fuzzy_score, "fuzzy".to_string())
            } else {
                (0.0, "none".to_string())
            };

            // Store the score if it meets threshold
            if combined_score >= COMBINED_THRESHOLD {
                similarity_matrix[i][j] = (combined_score, match_type.clone());
                similarity_matrix[j][i] = (combined_score, match_type);
            }
        }
    }

    // Group formation using a greedy approach
    for i in 0..entity_count {
        let entity_i = entities[i];

        // Skip if already in a group
        if processed_entities.contains(&entity_i.id) {
            continue;
        }

        // Find all potential group members based on similarity
        let mut group_members = vec![i];
        let mut match_details = HashMap::new();

        for j in 0..entity_count {
            if i == j {
                continue;
            }

            let (score, match_type) = &similarity_matrix[i][j];

            if *score >= COMBINED_THRESHOLD {
                group_members.push(j);
                match_details.insert(j, (score.clone(), match_type.clone()));
            }
        }

        // Only create a group if we have at least 2 entities
        if group_members.len() >= 2 {
            // Create a new group
            let group_id = Uuid::new_v4().to_string();
            create_entity_group(conn, &group_id).await?;
            created_groups.push(group_id.clone());

            // Add all members to the group
            for &member_idx in &group_members {
                let member = entities[member_idx];

                // Skip if already processed
                if processed_entities.contains(&member.id) {
                    continue;
                }

                if let Some(original_name) = &member.name {
                    let normalized_name = match entity_names.get(&member.id) {
                        Some(name) => name.clone(),
                        None => normalize_name(original_name),
                    };

                    // Get match details for this member
                    let (score, match_type) = if member_idx == i {
                        // This is the first member (seed) of the group
                        // Find its best match in the group
                        let mut best_score = 0.0;
                        let mut best_type = "none".to_string();
                        let mut best_idx = 0;

                        for &other_idx in &group_members {
                            if other_idx != i {
                                let (s, t) = &similarity_matrix[i][other_idx];
                                if *s > best_score {
                                    best_score = *s;
                                    best_type = t.clone();
                                    best_idx = other_idx;
                                }
                            }
                        }

                        (best_score, best_type)
                    } else {
                        // Use the match details against the seed member
                        match match_details.get(&member_idx) {
                            Some((s, t)) => (*s, t.clone()),
                            None => (COMBINED_THRESHOLD, "fuzzy".to_string()), // Fallback
                        }
                    };

                    // Determine the matched entity for evidence
                    let matched_entity = if member_idx == i {
                        // The first entity's match is its strongest similar entity
                        let mut best_match_idx = 0;
                        let mut best_match_score = 0.0;

                        for &other_idx in &group_members {
                            if other_idx != i {
                                let (s, _) = similarity_matrix[i][other_idx];
                                if s > best_match_score {
                                    best_match_score = s;
                                    best_match_idx = other_idx;
                                }
                            }
                        }

                        entities[best_match_idx].id.clone()
                    } else {
                        // Other entities' matches are against the seed entity
                        entities[i].id.clone()
                    };

                    // Add to group with evidence
                    add_entity_to_name_group(
                        conn,
                        &member.id,
                        &group_id,
                        original_name,
                        &normalized_name,
                        score,
                        &match_type,
                        &matched_entity,
                        reinforcement_orchestrator,
                        pool,
                    )
                    .await?;

                    // Log ML feedback for this match pair
                    if let Some(orchestrator_ref) = reinforcement_orchestrator {
                        // Only log if this is a pair being formed (not self)
                        if &member.id != &matched_entity {
                            let mut orchestrator = orchestrator_ref.lock().await;
                            // Determine confidence based on match type
                            let confidence = match match_type.as_str() {
                                "semantic" => CONFIDENCE_SEMANTIC as f64,
                                "fuzzy" => CONFIDENCE_FUZZY as f64,
                                "ml_guided" => score as f64,
                                _ => CONFIDENCE_COMBINED as f64,
                            };

                            match orchestrator
                                .log_match_result(
                                    &MatchMethodType::Name,
                                    confidence,
                                    true, // Assume match is correct
                                    &member.id,
                                    &matched_entity,
                                )
                                .await
                            {
                                Ok(_) => {}
                                Err(e) => {
                                    warn!("Failed to log name match result to ML system: {}", e)
                                }
                            }
                        }
                    }

                    // Mark as processed
                    processed_entities.insert(member.id.clone());
                }
            }
        }
    }

    Ok((created_groups, processed_entities))
}

/// Create a new entity group
async fn create_entity_group(conn: &tokio_postgres::Client, group_id: &str) -> Result<()> {
    let now = Utc::now().naive_utc();

    let query = "
        INSERT INTO entity_group (
            id, name, created_at, updated_at, confidence_score, entity_count, version
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
    ";

    conn.execute(
        query,
        &[
            &group_id,
            &Option::<String>::None, // name
            &now,
            &now,
            &(0.0 as f64), // Initial confidence
            &(0 as i32),   // Initial entity count
            &(1 as i32),   // Initial version
        ],
    )
    .await?;

    Ok(())
}

/// Add an entity to a name-based group with evidence
async fn add_entity_to_name_group(
    conn: &tokio_postgres::Client,
    entity_id: &EntityId,
    group_id: &str,
    original_name: &str,
    normalized_name: &str,
    similarity_score: f32,
    match_type: &str,
    matched_entity_id: &EntityId,
    reinforcement_orchestrator: Option<&Mutex<reinforcement::MatchingOrchestrator>>,
    pool: &PgPool,
) -> Result<()> {
    let now = Utc::now().naive_utc();

    // Start a transaction manually
    conn.execute("BEGIN", &[]).await?;

    // Use a try block to handle errors
    let result = async {
        // 1. Add the entity to the group
        let mapping_id = Uuid::new_v4().to_string();
        let insert_mapping = "
            INSERT INTO group_entity (id, entity_group_id, entity_id, created_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT DO NOTHING
        ";

        conn.execute(
            insert_mapping,
            &[&mapping_id, &group_id, &entity_id.0, &now],
        )
        .await?;

        // 2. Determine confidence based on match type
        // Determine base confidence based on match type
        let mut confidence = match match_type {
            "semantic" => CONFIDENCE_SEMANTIC,
            "fuzzy" => CONFIDENCE_FUZZY,
            "ml_guided" => similarity_score, // Use the score directly for ML-guided matches
            _ => CONFIDENCE_COMBINED,
        };

        // If ML orchestrator is available, get ML-guided confidence
        if let Some(orchestrator_ref) = reinforcement_orchestrator {
            let mut orchestrator = orchestrator_ref.lock().await;
            match orchestrator
                .select_matching_method(pool, entity_id, matched_entity_id)
                .await
            {
                Ok((method, conf)) => {
                    // If name is the recommended method, use its confidence
                    if matches!(method, MatchMethodType::Name) {
                        confidence = conf as f32;
                        info!(
                            "Using ML-guided confidence {:.4} for name match",
                            confidence
                        );
                    }
                }
                Err(e) => {
                    // Log error but continue with standard confidence
                    warn!("Error getting ML confidence for name match: {}", e);
                }
            }
        }

        // 3. Create match evidence
        let match_value = json!({
            "entity_id": entity_id.0,
            "original": original_name,
            "normalized": normalized_name,
            "similarity_score": similarity_score,
            "match_type": match_type,
            "matched_entity_id": matched_entity_id.0
        });

        // 4. Check if a name method already exists for this group
        let method_query = "
            SELECT id, match_values 
            FROM group_method 
            WHERE entity_group_id = $1 AND method_type = 'name'
        ";

        let method_row = conn.query_opt(method_query, &[&group_id]).await?;

        if let Some(row) = method_row {
            // Update existing method
            let method_id: String = row.get("id");
            let match_values: serde_json::Value = row.get("match_values");

            // Append new match to existing array if it's an array, otherwise create new array
            let updated_values = if match_values.is_array() {
                let mut values = match_values.as_array().unwrap().clone();
                values.push(match_value);
                json!(values)
            } else {
                json!([match_value])
            };

            let update_method = "
                UPDATE group_method
                SET match_values = $1
                WHERE id = $2
            ";

            conn.execute(update_method, &[&updated_values, &method_id])
                .await?;
        } else {
            // Create new method record
            let method_id = Uuid::new_v4().to_string();
            let description = "Matched based on organization name similarity";

            let insert_method = "
                INSERT INTO group_method (
                    id, entity_group_id, method_type, description, 
                    match_values, confidence_score, created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            ";

            conn.execute(
                insert_method,
                &[
                    &method_id,
                    &group_id,
                    &"name", // method_type
                    &Some(description),
                    &json!([match_value]),
                    &Some(confidence as f64),
                    &now,
                ],
            )
            .await?;
        }

        // 5. Update entity count in the group
        let update_count = "
            UPDATE entity_group
            SET entity_count = (
                SELECT COUNT(*) FROM group_entity WHERE entity_group_id = $1
            )
            WHERE id = $1
        ";

        conn.execute(update_count, &[&group_id]).await?;

        // 6. Log ML feedback for this match
        if let Some(orchestrator_ref) = reinforcement_orchestrator {
            let mut orchestrator = orchestrator_ref.lock().await;
            match orchestrator
                .log_match_result(
                    &MatchMethodType::Name,
                    confidence as f64,
                    true, // Assume match is correct
                    entity_id,
                    matched_entity_id,
                )
                .await
            {
                Ok(_) => {
                    info!("Logged name match feedback to ML system");
                }
                Err(e) => {
                    warn!("Failed to log name match feedback to ML system: {}", e);
                }
            }
        }

        Ok::<_, anyhow::Error>(())
    }
    .await;

    // Commit or rollback based on result
    if result.is_ok() {
        conn.execute("COMMIT", &[]).await?;
    } else {
        let _ = conn.execute("ROLLBACK", &[]).await;
        return Err(result.unwrap_err());
    }

    Ok(())
}
/// Get organization embeddings for a set of entities
async fn get_organization_embeddings(
    conn: &tokio_postgres::Client,
    entities: &[&Entity],
) -> Result<HashMap<EntityId, Option<Vec<f32>>>> {
    let mut embeddings = HashMap::new();
    let mut org_ids = Vec::new();
    let mut entity_to_org = HashMap::new();

    // Collect organization IDs
    for entity in entities {
        org_ids.push(entity.organization_id.0.clone());
        entity_to_org.insert(entity.id.clone(), entity.organization_id.0.clone());
    }

    // Query embeddings in batches
    // Postgres can't handle too many parameters in one query
    let batch_size = 100;
    for batch in org_ids.chunks(batch_size) {
        let placeholders: Vec<String> = (1..=batch.len()).map(|i| format!("${}", i)).collect();

        let query = format!(
            "SELECT id, embedding FROM organization WHERE id IN ({}) AND embedding IS NOT NULL",
            placeholders.join(", ")
        );

        let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = batch
            .iter()
            .map(|id| id as &(dyn tokio_postgres::types::ToSql + Sync))
            .collect();

        let rows = conn.query(&query, &params[..]).await?;

        for row in &rows {
            let org_id: String = row.get("id");
            let embedding: Vec<f32> = row.get("embedding");

            // Match the embedding back to the entity
            for entity in entities {
                if entity.organization_id.0 == org_id {
                    embeddings.insert(entity.id.clone(), Some(embedding.clone()));
                }
            }
        }
    }

    // Fill in missing embeddings with None
    for entity in entities {
        if !embeddings.contains_key(&entity.id) {
            embeddings.insert(entity.id.clone(), None);
        }
    }

    Ok(embeddings)
}

/// Calculate average confidence for name-based groups
async fn calculate_name_match_confidence(
    conn: &tokio_postgres::Client,
    group_ids: Vec<String>,
) -> Result<f64> {
    if group_ids.is_empty() {
        return Ok(0.0);
    }

    // Build a query with all group IDs
    let placeholders: Vec<String> = (1..=group_ids.len()).map(|i| format!("${}", i)).collect();

    let query = format!(
        "SELECT AVG(confidence_score) as avg_confidence FROM group_method 
         WHERE entity_group_id IN ({}) AND method_type = 'name'",
        placeholders.join(", ")
    );

    let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = group_ids
        .iter()
        .map(|id| id as &(dyn tokio_postgres::types::ToSql + Sync))
        .collect();

    let row = conn.query_one(&query, &params[..]).await?;
    let avg_confidence: Option<f64> = row.get("avg_confidence");

    Ok(avg_confidence.unwrap_or(0.0))
}

/// Calculate average size of name-based groups
async fn calculate_name_group_size(
    conn: &tokio_postgres::Client,
    total_groups: usize,
) -> Result<f64> {
    let query = "
        SELECT AVG(entity_count)::float8 as avg_size
        FROM entity_group eg
        JOIN group_method gm ON eg.id = gm.entity_group_id
        WHERE gm.method_type = 'name'
    ";

    let row = conn.query_one(query, &[]).await?;
    let avg_size: Option<f64> = row.get("avg_size");

    Ok(avg_size.unwrap_or(0.0))
}

/// Normalize an organization name to improve matching
fn normalize_name(name: &str) -> String {
    let mut normalized = name.to_lowercase();

    // Remove common legal entity types
    let entity_types = [
        " inc",
        " incorporated",
        " corp",
        " corporation",
        " llc",
        " ltd",
        " limited",
        " lp",
        " llp",
        " foundation",
        " trust",
        " charitable trust",
        " co",
        " company",
        " non-profit",
        " nonprofit",
        " nfp",
        " association",
        " assn",
        " coop",
        " co-op",
        " cooperative",
        " npo",
        " organisation",
        " org",
        " organization",
        " coalition",
        " fund",
        " partnership",
        " academy",
        " consortium",
        " institute",
        " services",
        " group",
        " society",
        " network",
        " federation",
        " international",
        " global",
        " national",
        " alliance",
    ];

    for entity_type in &entity_types {
        // Remove from end of string
        if normalized.ends_with(entity_type) {
            let end_pos = normalized.len() - entity_type.len();
            normalized = normalized[..end_pos].to_string();
        }

        // Also check with a period (e.g., "inc.")
        let with_period = format!("{}.", entity_type);
        if normalized.ends_with(&with_period) {
            let end_pos = normalized.len() - with_period.len();
            normalized = normalized[..end_pos].to_string();
        }

        // Check within commas or parentheses
        let with_comma = format!("{},", entity_type);
        let in_parentheses = format!("({})", entity_type.trim());

        normalized = normalized.replace(&with_comma, "");
        normalized = normalized.replace(&in_parentheses, "");
    }

    // Standardize common abbreviations
    let abbr_regex = Regex::new(r"\b(ctr|cntr|cent|cen)\b").unwrap();
    normalized = abbr_regex.replace_all(&normalized, "center").to_string();

    let abbr_regex = Regex::new(r"\b(assoc|assn)\b").unwrap();
    normalized = abbr_regex
        .replace_all(&normalized, "association")
        .to_string();

    let abbr_regex = Regex::new(r"\b(dept|dpt)\b").unwrap();
    normalized = abbr_regex
        .replace_all(&normalized, "department")
        .to_string();

    let abbr_regex = Regex::new(r"\b(intl|int'l)\b").unwrap();
    normalized = abbr_regex
        .replace_all(&normalized, "international")
        .to_string();

    let abbr_regex = Regex::new(r"\b(nat'l|natl)\b").unwrap();
    normalized = abbr_regex.replace_all(&normalized, "national").to_string();

    let abbr_regex = Regex::new(r"\b(comm|cmty)\b").unwrap();
    normalized = abbr_regex.replace_all(&normalized, "community").to_string();

    let abbr_regex = Regex::new(r"\b(srv|svcs|serv|svc)\b").unwrap();
    normalized = abbr_regex.replace_all(&normalized, "service").to_string();

    // Remove common punctuation
    normalized = normalized.replace(
        [
            '&', '-', '.', ',', ':', ';', '\'', '"', '(', ')', '[', ']', '{', '}', '/', '\\',
        ],
        " ",
    );

    // Normalize whitespace
    let whitespace_regex = Regex::new(r"\s+").unwrap();
    normalized = whitespace_regex.replace_all(&normalized, " ").to_string();

    // Trim
    normalized.trim().to_string()
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    if vec1.len() != vec2.len() || vec1.is_empty() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    for i in 0..vec1.len() {
        dot_product += vec1[i] as f64 * vec2[i] as f64;
        norm1 += vec1[i] as f64 * vec1[i] as f64;
        norm2 += vec2[i] as f64 * vec2[i] as f64;
    }

    let magnitude = (norm1.sqrt() * norm2.sqrt()) as f64;
    if magnitude == 0.0 {
        return 0.0;
    }

    (dot_product / magnitude) as f32
}
