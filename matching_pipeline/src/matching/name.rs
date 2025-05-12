// src/matching/name.rs

use anyhow::{Context, Result};
use chrono::Utc;
use futures::stream::{self, StreamExt}; // Added for stream processing
use log::{debug, info, trace, warn};
use regex::Regex;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc, // Added for Arc
    time::Instant,
};
use strsim::jaro_winkler;
use tokio::sync::Mutex; // Mutex from tokio
use tokio::task::JoinError; // Import JoinError
use uuid::Uuid;

use crate::config;
use crate::db::{self, insert_suggestion, PgPool};
use crate::models::{
    ActionType, Entity, EntityGroupId, EntityId, MatchMethodType, MatchValues, NameMatchValue,
    NewSuggestedAction, OrganizationId, SuggestionStatus,
};
use crate::reinforcement::{self, MatchingOrchestrator}; // Ensure reinforcement module is correctly referenced
use crate::results::{AnyMatchResult, MatchMethodStats, NameMatchResult}; // For suggestion thresholds

// Configuration for name matching
const MIN_FUZZY_SIMILARITY_THRESHOLD: f32 = 0.85;
const MIN_SEMANTIC_SIMILARITY_THRESHOLD: f32 = 0.88;
const COMBINED_SIMILARITY_THRESHOLD: f32 = 0.86;
const FUZZY_WEIGHT: f32 = 0.4;
const SEMANTIC_WEIGHT: f32 = 0.6;
const INTERNAL_WORKERS_NAME_STRATEGY: usize = 2; // Number of concurrent tasks for pair processing

// SQL query for inserting into entity_group
const INSERT_ENTITY_GROUP_SQL: &str = "
    INSERT INTO public.entity_group
    (id, entity_id_1, entity_id_2, method_type, match_values, confidence_score, created_at, updated_at, version)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 1)";

// Helper struct to collect results from each parallel task if a new pair is made
#[derive(Debug)]
struct NamePairInsertData {
    entity_group_id: EntityGroupId,
    entity_id_1: EntityId,
    entity_id_2: EntityId,
    final_confidence_score: f64,
    // Fields needed for logging suggestions or other post-insert actions if any
    // For suggestions, we need:
    original_name1: String,
    original_name2: String,
    normalized_name1: String,
    normalized_name2: String,
    pre_rl_score: f32,
    pre_rl_match_type: String,
    predicted_method_type_from_ml: MatchMethodType,
    // For ML feedback logging
    features_for_logging: Option<Vec<f64>>,
}

/// Main function to find name-based matches.
pub async fn find_matches(
    pool: &PgPool,
    // reinforcement_orchestrator is passed as Option<&Mutex<...>> from main.rs,
    // but main.rs actually holds Arc<Mutex<...>>. We clone the Arc for tasks.
    reinforcement_orchestrator_arc_mutex: Option<Arc<Mutex<reinforcement::MatchingOrchestrator>>>,
    pipeline_run_id: &str,
) -> Result<AnyMatchResult> {
    info!(
        "Starting pairwise name-based entity matching (run ID: {}){}...",
        pipeline_run_id,
        if reinforcement_orchestrator_arc_mutex.is_some() {
            " with ML guidance"
        } else {
            ""
        }
    );
    let start_time = Instant::now();

    // Get a single connection for initial data fetching.
    let mut initial_conn = pool
        .get()
        .await
        .context("Failed to get DB connection for name matching initial reads")?;

    info!("Fetching all entities with names...");
    let all_entities_with_names_vec = get_all_entities_with_names(&*initial_conn).await?; // Pass &*conn
    info!(
        "Found {} entities with non-empty names.",
        all_entities_with_names_vec.len()
    );

    if all_entities_with_names_vec.len() < 2 {
        info!("Not enough entities with names to perform pairwise name matching.");
        let name_result = NameMatchResult {
            groups_created: 0,
            stats: MatchMethodStats {
                method_type: MatchMethodType::Name,
                groups_created: 0,
                entities_matched: 0,
                avg_confidence: 0.0,
                avg_group_size: 0.0,
            },
        };
        return Ok(AnyMatchResult::Name(name_result));
    }

    debug!("Fetching existing name-matched pairs...");
    let existing_pairs_query = "
        SELECT entity_id_1, entity_id_2
        FROM public.entity_group
        WHERE method_type = $1";
    let existing_pair_rows = initial_conn
        .query(existing_pairs_query, &[&MatchMethodType::Name.as_str()])
        .await
        .context("Failed to query existing name-matched pairs")?;

    let mut existing_processed_pairs_set: HashSet<(EntityId, EntityId)> = HashSet::new();
    for row in existing_pair_rows {
        let id1_str: String = row.get("entity_id_1");
        let id2_str: String = row.get("entity_id_2");
        // Ensure canonical order for the set
        if id1_str < id2_str {
            existing_processed_pairs_set.insert((EntityId(id1_str), EntityId(id2_str)));
        } else {
            existing_processed_pairs_set.insert((EntityId(id2_str), EntityId(id1_str)));
        }
    }
    info!(
        "Found {} existing name-matched pairs to potentially skip.",
        existing_processed_pairs_set.len()
    );

    info!("Normalizing names and fetching embeddings for all relevant entities...");
    let name_fetch_start = Instant::now();
    let mut entity_names_normalized_map: HashMap<EntityId, String> = HashMap::new();
    for entity in &all_entities_with_names_vec {
        if let Some(name) = &entity.name {
            entity_names_normalized_map.insert(entity.id.clone(), normalize_name(name));
        }
    }

    let org_embeddings_map =
        get_organization_embeddings(&*initial_conn, &all_entities_with_names_vec).await?;
    info!(
        "Names normalized and embeddings fetched in {:.2?}.",
        name_fetch_start.elapsed()
    );

    drop(initial_conn); // Release the initial connection

    // Prepare data for sharing with concurrent tasks
    let all_entities_arc = Arc::new(all_entities_with_names_vec);
    let entity_names_normalized_arc = Arc::new(entity_names_normalized_map);
    let org_embeddings_arc = Arc::new(org_embeddings_map);
    let existing_processed_pairs_arc = Arc::new(existing_processed_pairs_set);
    let pipeline_run_id_arc = Arc::new(pipeline_run_id.to_string());

    let num_entities = all_entities_arc.len();
    // Generate all unique pairs of indices
    let pair_indices: Vec<(usize, usize)> = (0..num_entities)
        .flat_map(|i| (i + 1..num_entities).map(move |j| (i, j)))
        .collect();

    info!(
        "Processing {} potential entity pairs for name similarity using {} concurrent workers...",
        pair_indices.len(),
        INTERNAL_WORKERS_NAME_STRATEGY
    );

    // Stream processing for concurrent pair evaluation
    // CORRECTED TYPE ANNOTATION FOR task_results:
    let task_results: Vec<Result<Result<Option<NamePairInsertData>, anyhow::Error>, JoinError>> = stream::iter(
        pair_indices,
    )
    .map(|(i, j)| {
        // Clone Arcs and other necessary data for each spawned task
        let entities_task_arc = all_entities_arc.clone();
        let names_normalized_task_arc = entity_names_normalized_arc.clone();
        let embeddings_task_arc = org_embeddings_arc.clone();
        let existing_pairs_task_arc = existing_processed_pairs_arc.clone();
        let pool_task_clone = pool.clone(); // PgPool is Arc-based, cheap to clone
        let ro_task_arc_mutex_clone = reinforcement_orchestrator_arc_mutex.clone(); // Clone Option<Arc<Mutex<...>>>
        let run_id_task_arc = pipeline_run_id_arc.clone();

        // Spawn a Tokio task for each pair
        // The `async move` block is the body of the concurrent task
        tokio::spawn(async move { // This async block returns Result<Option<NamePairInsertData>, anyhow::Error>
            let entity1 = &entities_task_arc[i];
            let entity2 = &entities_task_arc[j];

            // Ensure canonical order for entity IDs in the pair
            let (e1_id, e2_id) = if entity1.id.0 < entity2.id.0 {
                (entity1.id.clone(), entity2.id.clone())
            } else {
                (entity2.id.clone(), entity1.id.clone())
            };

            // Skip if pair already processed (based on initial fetch)
            if existing_pairs_task_arc.contains(&(e1_id.clone(), e2_id.clone())) {
                trace!(
                    "Pair ({}, {}) already processed by name method (in-memory check). Skipping.",
                    e1_id.0,
                    e2_id.0
                );
                return Ok(None); // Indicates skipped, not an error
            }

            let original_name1 = entity1.name.as_ref().cloned().unwrap_or_default();
            let original_name2 = entity2.name.as_ref().cloned().unwrap_or_default();
            let normalized_name1 = names_normalized_task_arc
                .get(&entity1.id)
                .cloned()
                .unwrap_or_default();
            let normalized_name2 = names_normalized_task_arc
                .get(&entity2.id)
                .cloned()
                .unwrap_or_default();

            if normalized_name1.is_empty() || normalized_name2.is_empty() {
                trace!(
                    "Skipping pair ({}, {}) due to empty normalized name(s).",
                    e1_id.0,
                    e2_id.0
                );
                return Ok(None);
            }

            // CPU-bound calculations: Jaro-Winkler and Cosine Similarity
            let fuzzy_score = jaro_winkler(&normalized_name1, &normalized_name2) as f32;

            let embedding1_opt = embeddings_task_arc.get(&entity1.id).and_then(|opt_emb| opt_emb.as_ref());
            let embedding2_opt = embeddings_task_arc.get(&entity2.id).and_then(|opt_emb| opt_emb.as_ref());

            let semantic_score = match (embedding1_opt, embedding2_opt) {
                (Some(emb1), Some(emb2)) => cosine_similarity(emb1, emb2),
                _ => 0.0,
            };

            let pre_rl_score;
            let pre_rl_match_type;

            if semantic_score >= MIN_SEMANTIC_SIMILARITY_THRESHOLD {
                pre_rl_score = (fuzzy_score * FUZZY_WEIGHT) + (semantic_score * SEMANTIC_WEIGHT);
                pre_rl_match_type = "combined".to_string();
            } else if fuzzy_score >= MIN_FUZZY_SIMILARITY_THRESHOLD {
                pre_rl_score = fuzzy_score;
                pre_rl_match_type = "fuzzy".to_string();
            } else {
                trace!("Pair ({}, {}) did not meet pre-RL thresholds (fuzzy: {:.2}, semantic: {:.2}). Skipping.", e1_id.0, e2_id.0, fuzzy_score, semantic_score);
                return Ok(None);
            }

            if pre_rl_score < COMBINED_SIMILARITY_THRESHOLD {
                trace!(
                    "Pair ({}, {}) pre-RL score {:.2} below combined threshold {}. Skipping RL.",
                    e1_id.0, e2_id.0, pre_rl_score, COMBINED_SIMILARITY_THRESHOLD
                );
                return Ok(None);
            }

            // Default confidence is pre-RL score
            let mut final_confidence_score = pre_rl_score as f64;
            let mut predicted_method_type_from_ml = MatchMethodType::Name; // Default if RL not used or fails
            let mut features_for_logging: Option<Vec<f64>> = None;

// Reinforcement Learning Orchestrator Interaction (Async)
if let Some(orchestrator_arc_mutex_ref) = ro_task_arc_mutex_clone.as_ref() { // <--- Use .as_ref()
    match MatchingOrchestrator::extract_pair_context(&pool_task_clone, &e1_id, &e2_id).await {
        Ok(features) => {
            features_for_logging = Some(features.clone());
            let orchestrator_guard = orchestrator_arc_mutex_ref.lock().await; // .lock() works on &Arc<Mutex<T>>
            match orchestrator_guard.predict_method_with_context(&features) {
                Ok((predicted_method, rl_conf)) => {
                    predicted_method_type_from_ml = predicted_method;
                    final_confidence_score = rl_conf;
                    info!("ML guidance for name pair ({}, {}): Predicted Method: {:?}, RL Confidence: {:.4}. Pre-RL score: {:.2} ({})", e1_id.0, e2_id.0, predicted_method_type_from_ml, final_confidence_score, pre_rl_score, pre_rl_match_type);
                }
                Err(e) => {
                    warn!("ML prediction failed for name pair ({}, {}): {}. Using pre-RL score {:.2} as confidence.", e1_id.0, e2_id.0, e, pre_rl_score);
                }
            }
        }
        Err(e) => {
            warn!("Context extraction failed for name pair ({}, {}): {}. Using pre-RL score {:.2} as confidence.", e1_id.0, e2_id.0, e, pre_rl_score);
        }
    }
} else {
    info!("No ML orchestrator. Using pre-RL score {:.2} ({}) as confidence for name pair ({}, {}).", pre_rl_score, pre_rl_match_type, e1_id.0, e2_id.0);
}

            // Construct MatchValues
            let match_values = MatchValues::Name(NameMatchValue {
                original_name1: original_name1.clone(),
                original_name2: original_name2.clone(),
                normalized_name1: normalized_name1.clone(),
                normalized_name2: normalized_name2.clone(),
                pre_rl_similarity_score: Some(pre_rl_score),
                pre_rl_match_type: Some(pre_rl_match_type.clone()),
            });
            let match_values_json = serde_json::to_value(&match_values)
                .with_context(|| format!("Failed to serialize NameMatchValue for pair ({}, {})", e1_id.0, e2_id.0))?;

            let new_entity_group_id = EntityGroupId(Uuid::new_v4().to_string());
            let now_utc = Utc::now().naive_utc();

            // Database Insert (Async, requires its own connection)
            // Get a connection from the pool for this task's insert operation
            let mut conn = pool_task_clone.get().await.context("Failed to get DB connection for name pair insert task")?;

            let insert_result = conn.execute(
                INSERT_ENTITY_GROUP_SQL,
                &[
                    &new_entity_group_id.0,
                    &e1_id.0,
                    &e2_id.0,
                    &MatchMethodType::Name.as_str(),
                    &match_values_json,
                    &final_confidence_score,
                    &now_utc, // created_at
                    &now_utc, // updated_at
                ],
            ).await;

            match insert_result {
                Ok(_) => {
                    info!(
                        "DB INSERT: Created new name pair group {} for ({}, {}) with pre-RL score {:.2} ({}), RL confidence: {:.4}",
                        new_entity_group_id.0, e1_id.0, e2_id.0, pre_rl_score, pre_rl_match_type, final_confidence_score
                    );

                    // Log ML feedback if orchestrator was used
                    if let Some(orchestrator_arc_mutex_feedback) = ro_task_arc_mutex_clone {
                         let mut orchestrator_guard_feedback = orchestrator_arc_mutex_feedback.lock().await;
                         if let Err(e) = orchestrator_guard_feedback.log_match_result(
                             &pool_task_clone, // Pool for DB ops within log_match_result
                             &e1_id,
                             &e2_id,
                             &predicted_method_type_from_ml, // Method predicted by ML
                             final_confidence_score,         // Confidence from ML (or pre-RL if ML failed)
                             true,                           // is_match (since we inserted)
                             features_for_logging.as_ref(),
                             Some(&MatchMethodType::Name),   // Actual method type that made the decision
                             Some(final_confidence_score),   // Actual confidence of this name match
                         ).await {
                             warn!("Failed to log name match result to entity_match_pairs for ({},{}): {}", e1_id.0, e2_id.0, e);
                         }
                    }

                    // Log suggestion if confidence is low
                    if final_confidence_score < config::MODERATE_LOW_SUGGESTION_THRESHOLD {
                        let priority = if final_confidence_score < config::CRITICALLY_LOW_SUGGESTION_THRESHOLD { 2 } else { 1 };
                        let details_json = serde_json::json!({
                            "method_type": MatchMethodType::Name.as_str(),
                            "original_name1": original_name1, "normalized_name1": normalized_name1, // Cloned original_name1
                            "original_name2": original_name2, "normalized_name2": normalized_name2, // Cloned original_name2
                            "pre_rl_score": pre_rl_score, "pre_rl_match_type": pre_rl_match_type.clone(),
                            "entity_group_id": &new_entity_group_id.0,
                            "rl_predicted_method": predicted_method_type_from_ml.as_str(),
                        });
                        let reason_message = format!(
                            "Pair ({}, {}) matched by Name with low RL confidence ({:.4}). Pre-RL: {:.2} ({}). RL Predicted: {:?}.",
                            e1_id.0, e2_id.0, final_confidence_score, pre_rl_score, pre_rl_match_type, predicted_method_type_from_ml
                        );
                        let suggestion = NewSuggestedAction {
                            pipeline_run_id: Some(run_id_task_arc.to_string()),
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
                        // conn is of type bb8::PooledConnection<...>
                        if let Err(e) = insert_suggestion(&*conn, &suggestion).await { // Note the &*conn
                            warn!("Failed to log suggestion for low confidence name pair ({}, {}): {}", e1_id.0, e2_id.0, e);
                        }
                    }

                    Ok(Some(NamePairInsertData {
                        entity_group_id: new_entity_group_id,
                        entity_id_1: e1_id.clone(),
                        entity_id_2: e2_id.clone(),
                        final_confidence_score,
                        original_name1,
                        original_name2,
                        normalized_name1,
                        normalized_name2,
                        pre_rl_score,
                        pre_rl_match_type,
                        predicted_method_type_from_ml,
                        features_for_logging,
                    }))
                }
                Err(e) => {
                    // Check for unique constraint violation (uq_entity_pair_method)
                    if let Some(db_err) = e.as_db_error() {
                        if db_err.constraint() == Some("uq_entity_pair_method") {
                            debug!(
                                "DB constraint prevented duplicate name pair for ({}, {}), method {}. Skipping.",
                                e1_id.0, e2_id.0, MatchMethodType::Name.as_str()
                            );
                            return Ok(None); // Successfully identified as duplicate by DB, not an error for the task
                        }
                    }
                    // For other DB errors or operational errors, propagate them
                    warn!("Failed to insert name pair group for ({}, {}): {}", e1_id.0, e2_id.0, e);
                    Err(e.into())
                }
            }
        })
    })
    .buffer_unordered(INTERNAL_WORKERS_NAME_STRATEGY) // Concurrency level
    .collect::<Vec<_>>() // Collect results of spawned Tokio tasks
    .await;

    // Aggregate results from all tasks
    let mut new_pairs_created_count = 0;
    let mut entities_in_new_pairs: HashSet<EntityId> = HashSet::new();
    let mut confidence_scores_for_stats: Vec<f64> = Vec::new();
    let mut task_processing_errors = 0;

    for join_handle_result in task_results {
        // join_handle_result is Result<Result<Option<NamePairInsertData>, anyhow::Error>, JoinError>
        match join_handle_result {
            Ok(task_logic_result) => {
                // Task's JoinHandle completed without panic
                match task_logic_result {
                    // This is Result<Option<NamePairInsertData>, anyhow::Error>
                    Ok(Some(pair_data)) => {
                        // Task succeeded and returned a new pair
                        new_pairs_created_count += 1;
                        entities_in_new_pairs.insert(pair_data.entity_id_1);
                        entities_in_new_pairs.insert(pair_data.entity_id_2);
                        confidence_scores_for_stats.push(pair_data.final_confidence_score);
                    }
                    Ok(None) => { /* Task succeeded but determined no new pair (skipped, already exists, etc.) */
                    }
                    Err(task_err) => {
                        // Task logic returned an anyhow::Error
                        warn!("A name matching task logic failed: {:?}", task_err);
                        task_processing_errors += 1;
                    }
                }
            }
            Err(join_err) => {
                // Tokio task failed to execute (e.g., panic) - this is JoinError
                warn!("A name matching Tokio task failed to join: {:?}", join_err);
                task_processing_errors += 1;
            }
        }
    }

    if task_processing_errors > 0 {
        warn!(
            "Encountered {} errors during name pair processing tasks.",
            task_processing_errors
        );
        // Depending on policy, might want to return an error for the whole strategy
        // return Err(anyhow::anyhow!("{} name pair processing tasks failed", task_processing_errors));
    }

    let avg_confidence: f64 = if !confidence_scores_for_stats.is_empty() {
        confidence_scores_for_stats.iter().sum::<f64>() / confidence_scores_for_stats.len() as f64
    } else {
        0.0
    };

    let method_stats = MatchMethodStats {
        method_type: MatchMethodType::Name,
        groups_created: new_pairs_created_count,
        entities_matched: entities_in_new_pairs.len(),
        avg_confidence,
        avg_group_size: if new_pairs_created_count > 0 {
            2.0
        } else {
            0.0
        }, // Always 2.0 for pairwise
    };

    let elapsed_total = start_time.elapsed();
    info!(
        "Pairwise name matching completed in {:.2?}: created {} new pairs, involving {} unique entities. {} task errors.",
        elapsed_total,
        method_stats.groups_created,
        method_stats.entities_matched,
        task_processing_errors
    );

    let name_specific_result = NameMatchResult {
        groups_created: method_stats.groups_created,
        stats: method_stats,
    };

    Ok(AnyMatchResult::Name(name_specific_result))
}

/// Fetches all entities that have a non-null and non-empty name.
/// Accepts a GenericClient, which can be a direct connection or transaction.
async fn get_all_entities_with_names(
    conn: &impl tokio_postgres::GenericClient,
) -> Result<Vec<Entity>> {
    let query = "
        SELECT e.id, e.organization_id, e.name, e.created_at, e.updated_at, e.source_system, e.source_id
        FROM entity e
        WHERE e.name IS NOT NULL AND e.name != ''
    ";
    let rows = conn
        .query(query, &[])
        .await
        .context("Failed to query all entities with names")?;

    rows.iter()
        .map(|row| {
            Ok(Entity {
                id: EntityId(row.try_get("id").context("Failed to get 'id' for entity")?),
                organization_id: OrganizationId(
                    row.try_get("organization_id")
                        .context("Failed to get 'organization_id' for entity")?,
                ),
                name: row
                    .try_get("name")
                    .context("Failed to get 'name' for entity")?,
                created_at: row
                    .try_get("created_at")
                    .context("Failed to get 'created_at' for entity")?,
                updated_at: row
                    .try_get("updated_at")
                    .context("Failed to get 'updated_at' for entity")?,
                source_system: row.try_get("source_system").ok(),
                source_id: row.try_get("source_id").ok(),
            })
        })
        .collect::<Result<Vec<Entity>>>()
}

/// Get organization embeddings for a set of entities.
/// Accepts a GenericClient.
async fn get_organization_embeddings(
    conn: &impl tokio_postgres::GenericClient,
    entities: &[Entity],
) -> Result<HashMap<EntityId, Option<Vec<f32>>>> {
    let mut embeddings = HashMap::new();
    let org_ids: HashSet<String> = entities
        .iter()
        .map(|e| e.organization_id.0.clone())
        .collect();

    if org_ids.is_empty() {
        return Ok(embeddings);
    }
    let org_ids_vec: Vec<String> = org_ids.into_iter().collect();

    let batch_size = 100; // Or a configurable value
    debug!(
        "Fetching embeddings for {} unique organizations in batches of {}...",
        org_ids_vec.len(),
        batch_size
    );

    // Chunk processing for fetching embeddings
    for batch_org_ids_str_slice in org_ids_vec.chunks(batch_size) {
        // Convert slice of String to slice of &str for the query parameter
        let batch_org_ids_refs: Vec<&str> =
            batch_org_ids_str_slice.iter().map(AsRef::as_ref).collect();

        let query = "SELECT id, embedding FROM organization WHERE id = ANY($1::TEXT[]) AND embedding IS NOT NULL";

        let rows = conn
            .query(query, &[&batch_org_ids_refs]) // Pass as slice of &str
            .await
            .with_context(|| {
                format!(
                    "Failed to query org embeddings for batch: {:?}",
                    batch_org_ids_refs
                )
            })?;

        for row in rows {
            let org_id_str: String = row
                .try_get("id")
                .context("Failed to get 'id' for organization embedding row")?;
            let embedding_vec_opt: Option<Vec<f32>> = row.try_get("embedding").ok(); // .ok() converts Result to Option

            if let Some(embedding_vec) = embedding_vec_opt {
                // Find all entities associated with this org_id and store the embedding
                for entity in entities
                    .iter()
                    .filter(|e| e.organization_id.0 == org_id_str)
                {
                    embeddings.insert(entity.id.clone(), Some(embedding_vec.clone()));
                }
            }
        }
    }
    // Ensure all entities from the input list have an entry in the map, even if None
    for entity in entities {
        embeddings.entry(entity.id.clone()).or_insert(None);
    }
    debug!(
        "Finished fetching embeddings. Found embeddings for {} entity IDs (out of {} total entities).",
        embeddings.values().filter(|v| v.is_some()).count(),
        entities.len()
    );
    Ok(embeddings)
}

/// Normalize an organization name for matching.
fn normalize_name(name: &str) -> String {
    let mut normalized = name.to_lowercase();
    // Common suffixes to remove (ensure spaces for whole word, or handle carefully)
    let suffixes = [
        " incorporated",
        " inc.",
        " inc",
        " corporation",
        " corp.",
        " corp",
        " limited liability company",
        " llc.",
        " llc",
        " limited",
        " ltd.",
        " ltd",
        " limited partnership",
        " lp.",
        " lp",
        " limited liability partnership",
        " llp.",
        " llp",
        " foundation",
        " trust",
        " charitable trust",
        " company",
        " co.",
        " co",
        " non-profit",
        " nonprofit",
        " nfp",
        " association",
        " assn.",
        " assn",
        " coop",
        " co-op",
        " cooperative",
        " npo",
        " organisation",
        " organization",
        " org.",
        " org",
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
        " gmbh",
        " ag",
        " sarl",
        " bv",
        " spa",
        " pty",
        " plc",
        " p.c.",
        " pc",
    ];

    for suffix in suffixes {
        if normalized.ends_with(suffix) {
            normalized = normalized[..normalized.len() - suffix.len()]
                .trim_end()
                .to_string();
        }
    }
    // Trim again in case the original name was just a suffix or became empty
    normalized = normalized.trim().to_string();

    // Regex-based replacements for common abbreviations within the name
    let replacements = [
        (r"\b(ctr|cntr|cent|cen)\b", "center"),
        (r"\b(assoc|assn)\b", "association"),
        (r"\b(dept|dpt)\b", "department"),
        (r"\b(intl|int'l)\b", "international"),
        (r"\b(nat'l|natl)\b", "national"),
        (r"\b(comm|cmty)\b", "community"),
        (r"\b(srv|svcs|serv|svc)\b", "service"),
        (r"\b(univ)\b", "university"),
        (r"\b(coll)\b", "college"),
        (r"\b(inst)\b", "institute"),
        (r"\b(mfg)\b", "manufacturing"),
        (r"\b(tech)\b", "technology"),
        // Remove standalone legal terms if not caught by suffix logic and they are actual words
        (r"\binc\b", ""),
        (r"\bcorp\b", ""),
        (r"\bllc\b", ""),
        (r"\bltd\b", ""),
    ];

    for (pattern, replacement) in &replacements {
        // Compile regex for each pattern. Cache if performance becomes an issue.
        // For simplicity here, compiling on each call.
        match Regex::new(pattern) {
            Ok(re) => {
                normalized = re.replace_all(&normalized, *replacement).into_owned();
            }
            Err(e) => {
                warn!("Invalid regex pattern: '{}'. Error: {}", pattern, e);
            }
        }
    }

    // Remove all non-alphanumeric characters (except spaces)
    normalized = normalized
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect();

    // Normalize whitespace (multiple spaces to single, trim)
    normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");

    normalized.trim().to_string()
}

/// Calculate cosine similarity between two vectors (slice references).
fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    if vec1.len() != vec2.len() || vec1.is_empty() {
        return 0.0; // Or handle error appropriately
    }
    let dot_product: f64 = vec1
        .iter()
        .zip(vec2.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let norm1_sq: f64 = vec1.iter().map(|a| (*a as f64) * (*a as f64)).sum();
    let norm2_sq: f64 = vec2.iter().map(|b| (*b as f64) * (*b as f64)).sum();

    if norm1_sq == 0.0 || norm2_sq == 0.0 {
        // Avoid division by zero if a vector is all zeros
        return 0.0;
    }

    let magnitude = (norm1_sq * norm2_sq).sqrt();
    if magnitude == 0.0 {
        0.0
    } else {
        (dot_product / magnitude) as f32
    }
}
