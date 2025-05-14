// src/matching/name.rs

use anyhow::{Context, Result};
use chrono::Utc;
use futures::stream::{self, StreamExt}; // Added for stream processing
use log::{debug, info, trace, warn};
use regex::Regex;
use std::{
    collections::{BinaryHeap, HashMap, HashSet},
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

// Updated configuration for stricter name matching
const MIN_FUZZY_SIMILARITY_THRESHOLD: f32 = 0.92; // Increased from 0.85
const MIN_SEMANTIC_SIMILARITY_THRESHOLD: f32 = 0.94; // Increased from 0.88
const COMBINED_SIMILARITY_THRESHOLD: f32 = 0.93; // Increased from 0.86
const FUZZY_WEIGHT: f32 = 0.3; // Decreased from 0.4
const SEMANTIC_WEIGHT: f32 = 0.7; // Increased from 0.6
const INTERNAL_WORKERS_NAME_STRATEGY: usize = 2; // Number of concurrent tasks for pair processing

/// Number of discriminative tokens to use per entity
const TOP_TOKENS_PER_ENTITY: usize = 10;
/// Minimum token overlap required for candidate pair consideration
const MIN_TOKEN_OVERLAP: usize = 2;
const MAX_CANDIDATES_PER_ENTITY: usize = 500; // Safety limit to prevent excessive comparisons
const MIN_TOKEN_LENGTH: usize = 2; // Ignore tokens shorter than this length

// Organizational types that should never match even with high similarity
const INCOMPATIBLE_ORG_TYPES: [(&str, &str); 10] = [
    ("fire", "police"),
    ("fire", "school"),
    ("police", "school"),
    ("hospital", "school"),
    ("hospital", "church"),
    ("health", "education"),
    ("medical", "food"),
    ("college", "hospital"),
    ("prison", "school"),
    ("shelter", "college"),
];

// Known prefix noise patterns to clean
const NOISE_PREFIXES: [&str; 2] = ["zzz - ", "re-add "];

// Common prefixes that need special handling
const LOCATION_PREFIXES: [&str; 8] = [
    "north", "south", "east", "west", "sound", "valley", "pacific", "olympic",
];

// Expanded stopwords with more domain-specific irrelevant words
const STOPWORDS: [&str; 132] = [
    // Common English articles, conjunctions, and prepositions
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "nor",
    "for",
    "yet",
    "so",
    "in",
    "on",
    "at",
    "by",
    "to",
    "with",
    "from",
    "of",
    "as",
    "into",
    "about",
    "before",
    "after",
    "during",
    "until",
    "since",
    "unless",
    // Common business terms with little discriminative value
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "llc",
    "ltd",
    "limited",
    "company",
    "co",
    "group",
    "holdings",
    "enterprises",
    "international",
    "global",
    "worldwide",
    "national",
    "american",
    "usa",
    "us",
    "service",
    "services",
    "solutions",
    "systems",
    "associates",
    "partners",
    "partnership",
    // Generic organizational terms
    "organization",
    "organisation",
    "foundation",
    "institute",
    "association",
    "society",
    "council",
    "committee",
    "center",
    "centre",
    "department",
    "division",
    "unit",
    "office",
    "bureau",
    "agency",
    "authority",
    "board",
    // Common descriptive terms
    "new",
    "old",
    "great",
    "greater",
    "best",
    "better",
    "first",
    "second",
    "third",
    "primary",
    "main",
    "central",
    "local",
    "regional",
    "official",
    // Additional stopwords
    "this",
    "that",
    "these",
    "those",
    "it",
    "they",
    "them",
    "their",
    "our",
    "your",
    "all",
    "any",
    "each",
    "every",
    "some",
    "such",
    "no",
    "not",
    "only",
    "very",
    // Specific to your dataset
    "program",
    "community",
    "resource",
    "resources",
    "support",
    "help",
    "health",
    "care",
    "management",
    "professional",
    "public",
    "private",
    "general",
    "federal",
    "state",
    "county",
    "regional",
    "district",
    "area",
    "branch",
    "program",
    "provider",
    "member",
    "directory",
    "guide",
    "network",
];

// SQL query for inserting into entity_group
const INSERT_ENTITY_GROUP_SQL: &str = "
    INSERT INTO public.entity_group
    (id, entity_id_1, entity_id_2, method_type, match_values, confidence_score, pre_rl_confidence_score, created_at, updated_at, version)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 1)";

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

/// Enhanced data structure to hold entity information with organization type detection
struct EntityNameData {
    entity: Entity,
    normalized_name: String,
    entity_type: Option<&'static str>,
    tokens: HashSet<String>,             // Tokens for this entity's name
    weighted_tokens: Vec<(String, f32)>, // Tokens with their context-based weights
}

/// Calculates and stores token statistics for efficient filtering
#[derive(Debug)]
struct TokenStats {
    token: String,
    frequency: usize, // Number of entities containing this token
    idf: f32,         // Inverse Document Frequency
}

/// Main function to find name-based matches with improved algorithmic complexity
pub async fn find_matches(
    pool: &PgPool,
    reinforcement_orchestrator_arc_mutex: Option<Arc<Mutex<reinforcement::MatchingOrchestrator>>>,
    pipeline_run_id: &str,
) -> Result<AnyMatchResult> {
    info!(
        "Starting high-performance pairwise name-based entity matching (run ID: {}){}...",
        pipeline_run_id,
        if reinforcement_orchestrator_arc_mutex.is_some() {
            " with ML guidance"
        } else {
            ""
        }
    );
    let start_time = Instant::now();

    // Get a single connection for initial data fetching
    let mut initial_conn = pool
        .get()
        .await
        .context("Failed to get DB connection for name matching initial reads")?;

    info!("Fetching all entities with names...");
    let all_entities_with_names_vec = get_all_entities_with_names(&*initial_conn).await?;
    let total_entities = all_entities_with_names_vec.len();
    info!("Found {} entities with non-empty names.", total_entities);

    if total_entities < 2 {
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
        "Found {} existing name-matched pairs to skip.",
        existing_processed_pairs_set.len()
    );

    // Create stopwords HashSet for faster lookups
    let stopwords: HashSet<String> = STOPWORDS.iter().map(|&s| s.to_string()).collect();

    // Step 1: Normalize and tokenize entity names
    info!("Normalizing and tokenizing entity names...");
    let tokenization_start = Instant::now();

    let mut entity_data: Vec<EntityNameData> = Vec::with_capacity(total_entities);
    let mut token_frequency: HashMap<String, usize> = HashMap::new();

    // First pass: Normalize, tokenize, and count token frequencies
    for entity in &all_entities_with_names_vec {
        if let Some(name) = &entity.name {
            // Enhanced normalization with entity type detection
            let (normalized_name, entity_type) = normalize_name(name);

            if !normalized_name.is_empty() {
                // Enhanced tokenization with context-aware weights
                let (tokens, weighted_tokens) =
                    tokenize_name(&normalized_name, &stopwords, entity_type);

                // Update token frequency counts
                for token in &tokens {
                    *token_frequency.entry(token.clone()).or_insert(0) += 1;
                }

                // Store the entity data with enhanced information
                entity_data.push(EntityNameData {
                    entity: entity.clone(),
                    normalized_name,
                    entity_type,
                    tokens,
                    weighted_tokens,
                });
            }
        }
    }

    // Calculate IDF for each token
    let total_docs = entity_data.len() as f32;
    let mut token_stats: HashMap<String, TokenStats> = HashMap::new();

    for (token, frequency) in token_frequency {
        let idf = (total_docs / (frequency as f32)).ln();
        token_stats.insert(
            token.clone(),
            TokenStats {
                token,
                frequency,
                idf,
            },
        );
    }

    // Step 2: Build optimized index using only discriminative tokens
    info!("Building optimized token index...");

    // Create a filtered index with only discriminative tokens
    // We'll use different token sets based on entity size for better balancing
    let mut token_to_entities: HashMap<String, Vec<usize>> = HashMap::new();

    // Define token frequency thresholds (can be tuned based on dataset size)
    let max_common_token_freq = (total_entities as f32 * 0.05) as usize; // 5% of entities is the max allowed frequency

    for (idx, entity_datum) in entity_data.iter().enumerate() {
        // Get tokens sorted by weight and IDF (most discriminative first)
        let mut token_scores: Vec<(String, f32)> = entity_datum
            .weighted_tokens
            .iter()
            .filter_map(|(token, weight)| {
                token_stats.get(token).map(|stats| {
                    // Only include tokens that don't appear in too many entities
                    if stats.frequency <= max_common_token_freq {
                        // Combine IDF and context-based weight for better discrimination
                        Some((token.clone(), stats.idf * weight))
                    } else {
                        None
                    }
                })
            })
            .flatten() // Remove None values
            .collect();

        // Sort by combined score (descending order)
        token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top tokens (number depends on entity size)
        let tokens_to_use = token_scores
            .iter()
            .take(TOP_TOKENS_PER_ENTITY) // Take top N most discriminative tokens
            .map(|(token, _)| token.clone())
            .collect::<Vec<String>>();

        // Add to inverted index
        for token in tokens_to_use {
            token_to_entities
                .entry(token)
                .or_insert_with(Vec::new)
                .push(idx);
        }
    }

    info!(
        "Normalized and indexed {} entities with {} discriminative tokens in {:.2?}",
        entity_data.len(),
        token_to_entities.len(),
        tokenization_start.elapsed()
    );

    // Step 3: Generate candidate pairs using logarithmic lookup
    info!("Generating candidate pairs using optimized logarithmic lookup...");
    let candidates_start = Instant::now();

    // For each entity, use a priority queue to find most similar candidates
    let mut candidate_pairs: Vec<(usize, usize)> = Vec::new();
    let mut total_possible_pairs = 0;

    // Process entities in parallel for better performance
    let entity_indices: Vec<usize> = (0..entity_data.len()).collect();
    let chunk_size =
        (entity_data.len() + INTERNAL_WORKERS_NAME_STRATEGY - 1) / INTERNAL_WORKERS_NAME_STRATEGY;

    // Use multiple worker threads to generate candidate pairs
    let chunked_indices: Vec<Vec<usize>> = entity_indices
        .chunks(chunk_size)
        .map(|chunk| chunk.to_vec())
        .collect();

    let entity_data_arc = Arc::new(entity_data);
    let token_to_entities_arc = Arc::new(token_to_entities);
    let token_stats_arc = Arc::new(token_stats);

    let mut candidate_tasks = Vec::new();

    for chunk in chunked_indices {
        let entity_data_chunk = entity_data_arc.clone();
        let token_to_entities_chunk = token_to_entities_arc.clone();
        let token_stats_chunk = token_stats_arc.clone();

        let task = tokio::spawn(async move {
            let mut chunk_candidates = Vec::new();

            for &i in &chunk {
                // Use a BinaryHeap for efficient top-k extraction (O(log n) operations)
                let mut candidates_heap: BinaryHeap<(usize, usize)> = BinaryHeap::new();
                let mut seen_candidates = HashSet::new();

                // Get all candidate entities from discriminative tokens
                for token in &entity_data_chunk[i].tokens {
                    if let Some(stats) = token_stats_chunk.get(token) {
                        // Skip very common tokens
                        if stats.frequency > max_common_token_freq {
                            continue;
                        }

                        if let Some(entity_indices) = token_to_entities_chunk.get(token) {
                            for &j in entity_indices {
                                if i != j && i < j && !seen_candidates.contains(&j) {
                                    seen_candidates.insert(j);

                                    // Skip incompatible organization types early
                                    if let (Some(t1), Some(t2)) = (
                                        entity_data_chunk[i].entity_type,
                                        entity_data_chunk[j].entity_type,
                                    ) {
                                        let mut should_skip = false;
                                        for (incompatible1, incompatible2) in
                                            &INCOMPATIBLE_ORG_TYPES
                                        {
                                            if (t1 == *incompatible1 && t2 == *incompatible2)
                                                || (t1 == *incompatible2 && t2 == *incompatible1)
                                            {
                                                should_skip = true;
                                                break;
                                            }
                                        }
                                        if should_skip {
                                            continue;
                                        }
                                    }

                                    // Calculate token overlap with weights as a quick similarity metric
                                    let token_overlap = calculate_weighted_token_overlap(
                                        &entity_data_chunk[i].tokens,
                                        &entity_data_chunk[j].tokens,
                                        &entity_data_chunk[i].weighted_tokens,
                                        &entity_data_chunk[j].weighted_tokens,
                                    );

                                    if token_overlap >= MIN_TOKEN_OVERLAP {
                                        candidates_heap.push((token_overlap, j));

                                        // Keep heap at most MAX_CANDIDATES_PER_ENTITY size
                                        if candidates_heap.len() > MAX_CANDIDATES_PER_ENTITY {
                                            candidates_heap.pop(); // Remove lowest overlap
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Extract candidates from heap (will be sorted by token overlap, highest first)
                for (_, j) in candidates_heap.into_sorted_vec() {
                    chunk_candidates.push((i, j));
                }
            }

            chunk_candidates
        });

        candidate_tasks.push(task);
    }

    // Collect results from all candidate generation tasks
    let candidate_results = futures::future::join_all(candidate_tasks).await;
    for result in candidate_results {
        match result {
            Ok(chunk_candidates) => {
                candidate_pairs.extend(chunk_candidates);
            }
            Err(e) => {
                warn!("Error in candidate generation task: {:?}", e);
            }
        }
    }

    total_possible_pairs = total_entities * (total_entities - 1) / 2;

    info!(
        "Generated {} candidate pairs for comparison ({:.2}% of total possible {} pairs) in {:.2?}",
        candidate_pairs.len(),
        100.0 * candidate_pairs.len() as f64 / total_possible_pairs as f64,
        total_possible_pairs,
        candidates_start.elapsed()
    );

    // Step 4: Fetch embeddings for semantic similarity comparison
    info!("Fetching embeddings for entity name semantic comparison...");
    let org_embeddings_map =
        get_organization_embeddings(&*initial_conn, &all_entities_with_names_vec).await?;

    drop(initial_conn); // Release the initial connection

    // Prepare data for sharing with concurrent tasks
    let existing_processed_pairs_arc = Arc::new(existing_processed_pairs_set);
    let pipeline_run_id_arc = Arc::new(pipeline_run_id.to_string());
    let org_embeddings_arc = Arc::new(org_embeddings_map);

    info!(
        "Processing {} candidate pairs for name similarity using {} concurrent workers...",
        candidate_pairs.len(),
        INTERNAL_WORKERS_NAME_STRATEGY
    );

    // Stream processing for concurrent pair evaluation
    let task_results: Vec<Result<Result<Option<NamePairInsertData>, anyhow::Error>, JoinError>> = stream::iter(
        candidate_pairs,
    )
    .map(|(i, j)| {
        // Clone Arcs and other necessary data for each spawned task
        let entity_data_task_arc = entity_data_arc.clone();
        let embeddings_task_arc = org_embeddings_arc.clone();
        let existing_pairs_task_arc = existing_processed_pairs_arc.clone();
        let pool_task_clone = pool.clone();
        let ro_task_arc_mutex_clone = reinforcement_orchestrator_arc_mutex.clone();
        let run_id_task_arc = pipeline_run_id_arc.clone();
        // Spawn a Tokio task for each pair
        tokio::spawn(async move {
            let entity1 = &entity_data_task_arc[i].entity;
            let entity2 = &entity_data_task_arc[j].entity;
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
            let normalized_name1 = entity_data_task_arc[i].normalized_name.clone();
            let normalized_name2 = entity_data_task_arc[j].normalized_name.clone();
            let entity_type1 = entity_data_task_arc[i].entity_type;
            let entity_type2 = entity_data_task_arc[j].entity_type;
            if normalized_name1.is_empty() || normalized_name2.is_empty() {
                trace!(
                    "Skipping pair ({}, {}) due to empty normalized name(s).",
                    e1_id.0,
                    e2_id.0
                );
                return Ok(None);
            }
            // Check for incompatible organization types - strict rejection
            if let (Some(t1), Some(t2)) = (entity_type1, entity_type2) {
                if t1 != t2 {
                    for (incompatible1, incompatible2) in &INCOMPATIBLE_ORG_TYPES {
                        if (t1 == *incompatible1 && t2 == *incompatible2) ||
                           (t1 == *incompatible2 && t2 == *incompatible1) {
                            trace!(
                                "Skipping pair ({}, {}) due to incompatible types: {} vs {}",
                                e1_id.0, e2_id.0, t1, t2
                            );
                            return Ok(None);
                        }
                    }
                }
            }
            // CPU-bound calculations: Jaro-Winkler similarity
            let fuzzy_score = jaro_winkler(&normalized_name1, &normalized_name2) as f32;
            // Only proceed with semantic similarity if fuzzy score is promising
            // Using stricter early filtering threshold
            if fuzzy_score < (MIN_FUZZY_SIMILARITY_THRESHOLD * 0.9) {
                trace!(
                    "Fuzzy score {:.2} for pair ({}, {}) below quick filter threshold. Skipping.",
                    fuzzy_score, e1_id.0, e2_id.0
                );
                return Ok(None);
            }
            // Get embeddings and calculate semantic similarity
            let embedding1_opt = embeddings_task_arc.get(&entity1.id).and_then(|opt_emb| opt_emb.as_ref());
            let embedding2_opt = embeddings_task_arc.get(&entity2.id).and_then(|opt_emb| opt_emb.as_ref());
            let semantic_score = match (embedding1_opt, embedding2_opt) {
                (Some(emb1), Some(emb2)) => cosine_similarity(emb1, emb2),
                _ => 0.0,
            };
            // Calculate pre-RL score and match type
            let pre_rl_score;
            let pre_rl_match_type;
            // Using stricter thresholds
            if semantic_score >= MIN_SEMANTIC_SIMILARITY_THRESHOLD {
                pre_rl_score = (fuzzy_score * FUZZY_WEIGHT) + (semantic_score * SEMANTIC_WEIGHT);
                pre_rl_match_type = "combined".to_string();
            } else if fuzzy_score >= MIN_FUZZY_SIMILARITY_THRESHOLD {
                pre_rl_score = fuzzy_score;
                pre_rl_match_type = "fuzzy".to_string();
            } else {
                trace!("Pair ({}, {}) did not meet pre-RL thresholds (fuzzy: {:.2}, semantic: {:.2}). Skipping.", 
                      e1_id.0, e2_id.0, fuzzy_score, semantic_score);
                return Ok(None);
            }
            // Apply domain-specific rules to adjust the score
            let adjusted_pre_rl_score = apply_domain_rules(&normalized_name1, &normalized_name2, pre_rl_score, entity_type1, entity_type2);
            if adjusted_pre_rl_score < COMBINED_SIMILARITY_THRESHOLD {
                trace!(
                    "Pair ({}, {}) adjusted pre-RL score {:.2} below combined threshold {}. Skipping RL.",
                    e1_id.0, e2_id.0, adjusted_pre_rl_score, COMBINED_SIMILARITY_THRESHOLD
                );
                return Ok(None);
            }
            // Default confidence is adjusted pre-RL score
            let mut final_confidence_score = adjusted_pre_rl_score as f64;
            let mut predicted_method_type_from_ml = MatchMethodType::Name; // Default if RL not used or fails
            let mut features_for_logging: Option<Vec<f64>> = None;
            // Reinforcement Learning Orchestrator Interaction (Async)
            if let Some(orchestrator_arc_mutex_ref) = ro_task_arc_mutex_clone.as_ref() {
                match MatchingOrchestrator::extract_pair_context(&pool_task_clone, &e1_id, &e2_id).await {
                    Ok(features) => {
                        features_for_logging = Some(features.clone());
                        let orchestrator_guard = orchestrator_arc_mutex_ref.lock().await;
                        match orchestrator_guard.predict_method_with_context(&features) {
                            Ok((predicted_method, rl_conf)) => {
                                predicted_method_type_from_ml = predicted_method;
                                final_confidence_score = rl_conf;
                                info!("ML guidance for name pair ({}, {}): Predicted Method: {:?}, RL Confidence: {:.4}. Pre-RL score: {:.2} ({})", 
                                     e1_id.0, e2_id.0, predicted_method_type_from_ml, final_confidence_score, pre_rl_score, pre_rl_match_type);
                            }
                            Err(e) => {
                                warn!("ML prediction failed for name pair ({}, {}): {}. Using pre-RL score {:.2} as confidence.", 
                                     e1_id.0, e2_id.0, e, pre_rl_score);
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Context extraction failed for name pair ({}, {}): {}. Using pre-RL score {:.2} as confidence.", 
                             e1_id.0, e2_id.0, e, pre_rl_score);
                    }
                }
            } else {
                info!("No ML orchestrator. Using pre-RL score {:.2} ({}) as confidence for name pair ({}, {}).", 
                     pre_rl_score, pre_rl_match_type, e1_id.0, e2_id.0);
            }
            // Construct MatchValues for database insertion
            let match_values = MatchValues::Name(NameMatchValue {
                original_name1: original_name1.clone(),
                original_name2: original_name2.clone(),
                normalized_name1: normalized_name1.clone(),
                normalized_name2: normalized_name2.clone(),
                pre_rl_match_type: Some(pre_rl_match_type.clone()),
            });
            let match_values_json = serde_json::to_value(&match_values)
                .with_context(|| format!("Failed to serialize NameMatchValue for pair ({}, {})", e1_id.0, e2_id.0))?;
            // Generate new entity group ID and current timestamp
            let new_entity_group_id = EntityGroupId(Uuid::new_v4().to_string());
            let now_utc = Utc::now().naive_utc();
            // Database Insert (Async, requires its own connection)
            let conn = pool_task_clone.get().await.context("Failed to get DB connection for name pair insert task")?;
            let insert_result = conn.execute(
                INSERT_ENTITY_GROUP_SQL,
                &[
                    &new_entity_group_id.0,
                    &e1_id.0,
                    &e2_id.0,
                    &MatchMethodType::Name.as_str(),
                    &match_values_json,
                    &final_confidence_score,
                    &(adjusted_pre_rl_score as f64),
                    &now_utc,
                    &now_utc,
                ],
            ).await;
            match insert_result {
                Ok(_) => {
                    info!(
                        "DB INSERT: Created new name pair group {} for ({}, {}) with adjusted pre-RL score {:.2} ({}), RL confidence: {:.4}",
                        new_entity_group_id.0, e1_id.0, e2_id.0, adjusted_pre_rl_score, pre_rl_match_type, final_confidence_score
                    );
                    // Log ML feedback if orchestrator was used
                    if let Some(orchestrator_arc_mutex_feedback) = ro_task_arc_mutex_clone {
                         let mut orchestrator_guard_feedback = orchestrator_arc_mutex_feedback.lock().await;
                         if let Err(e) = orchestrator_guard_feedback.log_match_result(
                             &pool_task_clone,
                             &e1_id,
                             &e2_id,
                             &predicted_method_type_from_ml,
                             final_confidence_score,
                             true,
                             features_for_logging.as_ref(),
                             Some(&MatchMethodType::Name),
                             Some(final_confidence_score),
                         ).await {
                             warn!("Failed to log name match result to entity_match_pairs for ({},{}): {}", e1_id.0, e2_id.0, e);
                         }
                    }
                    // Log suggestion if confidence is low
                    if final_confidence_score < config::MODERATE_LOW_SUGGESTION_THRESHOLD {
                        let priority = if final_confidence_score < config::CRITICALLY_LOW_SUGGESTION_THRESHOLD { 2 } else { 1 };
                        let details_json = serde_json::json!({
                            "method_type": MatchMethodType::Name.as_str(),
                            "original_name1": original_name1, "normalized_name1": normalized_name1,
                            "original_name2": original_name2, "normalized_name2": normalized_name2,
                            "pre_rl_score": adjusted_pre_rl_score, "pre_rl_match_type": pre_rl_match_type.clone(),
                            "entity_group_id": &new_entity_group_id.0,
                            "rl_predicted_method": predicted_method_type_from_ml.as_str(),
                        });
                        let reason_message = format!(
                            "Pair ({}, {}) matched by Name with low RL confidence ({:.4}). Pre-RL: {:.2} ({}). RL Predicted: {:?}.",
                            e1_id.0, e2_id.0, final_confidence_score, adjusted_pre_rl_score, pre_rl_match_type, predicted_method_type_from_ml
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
                        if let Err(e) = insert_suggestion(&*conn, &suggestion).await {
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
                        pre_rl_score: adjusted_pre_rl_score,
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
        match join_handle_result {
            Ok(task_logic_result) => {
                match task_logic_result {
                    Ok(Some(pair_data)) => {
                        new_pairs_created_count += 1;
                        entities_in_new_pairs.insert(pair_data.entity_id_1);
                        entities_in_new_pairs.insert(pair_data.entity_id_2);
                        confidence_scores_for_stats.push(pair_data.final_confidence_score);
                    }
                    Ok(None) => { /* Task succeeded but determined no new pair */ }
                    Err(task_err) => {
                        warn!("A name matching task logic failed: {:?}", task_err);
                        task_processing_errors += 1;
                    }
                }
            }
            Err(join_err) => {
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
    }

    // Calculate final statistics
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
        },
    };

    let elapsed_total = start_time.elapsed();
    info!(
        "Optimized pairwise name matching completed in {:.2?}: created {} new pairs, involving {} unique entities. {} task errors.",
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
        FROM public.entity e
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

        let query = "SELECT id, embedding FROM public.organization WHERE id = ANY($1::TEXT[]) AND embedding IS NOT NULL";

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

/// Enhanced normalize_name function with entity type detection
fn normalize_name(name: &str) -> (String, Option<&'static str>) {
    let mut normalized = name.to_lowercase();

    // Clean noise prefixes
    for prefix in &NOISE_PREFIXES {
        if normalized.starts_with(prefix) {
            normalized = normalized[prefix.len()..].trim().to_string();
        }
    }

    // Handle special character substitutions
    let char_substitutions = [
        ("&", " and "),
        ("+", " plus "),
        ("/", " "),
        ("-", " "),
        (".", " "),
        ("'", ""),
        ("(", " "),
        (")", " "),
        (",", " "),
    ];

    for (pattern, replacement) in &char_substitutions {
        normalized = normalized.replace(pattern, replacement);
    }

    // Detect organization type early for specialized handling
    let entity_type = detect_entity_type(&normalized);

    // Apply specialized normalization based on entity type
    if let Some(etype) = entity_type {
        match etype {
            "city" => {
                // For cities, preserve the city name part carefully
                if let Some(stripped) = normalized.strip_prefix("city of ") {
                    let city_name = stripped.trim();
                    // Don't remove suffixes from city names
                    return (format!("city of {}", city_name), Some(etype));
                }
            }
            "police" => {
                // Standardize police department names
                normalized = normalized.replace("pd", "police department");
                normalized = normalized.replace("police dept", "police department");
            }
            "fire" => {
                // Standardize fire department names
                normalized = normalized.replace("fd", "fire department");
                normalized = normalized.replace("fire dept", "fire department");
            }
            "hospital" => {
                // Standardize medical facility terms
                normalized = normalized.replace("medical center", "hospital");
                normalized = normalized.replace("med ctr", "hospital");
                normalized = normalized.replace("med center", "hospital");
            }
            "education" => {
                // Standardize educational institution terms
                normalized = normalized.replace("school dist", "school district");
                normalized = normalized.replace("sd", "school district");
            }
            _ => {}
        }
    }

    // Common prefixes to handle specially
    let prefixes = ["the ", "a ", "an "];

    for prefix in prefixes {
        if normalized.starts_with(prefix) {
            normalized = normalized[prefix.len()..].to_string();
        }
    }

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

    // Skip suffix removal for city names
    if entity_type != Some("city") {
        for suffix in suffixes {
            if normalized.ends_with(suffix) {
                normalized = normalized[..normalized.len() - suffix.len()]
                    .trim_end()
                    .to_string();
            }
        }
    }

    // Extract location in parentheses for special handling (e.g., "YWCA (Seattle)")
    let mut location_suffix = None;
    let paren_regex = Regex::new(r"\s*\((.*?)\)\s*$").unwrap_or_else(|_| Regex::new(r"").unwrap());
    if let Some(captures) = paren_regex.captures(&normalized) {
        if let Some(location_match) = captures.get(1) {
            location_suffix = Some(location_match.as_str().to_string());
            normalized = paren_regex.replace(&normalized, "").to_string();
        }
    }

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
        (r"\b(st)\b", "saint"),
        (r"\bwa\b", "washington"),
        // Specific organization types for better matching
        (r"\b(fd)\b", "fire department"),
        (r"\b(pd)\b", "police department"),
        // Remove standalone legal terms
        (r"\binc\b", ""),
        (r"\bcorp\b", ""),
        (r"\bllc\b", ""),
        (r"\bltd\b", ""),
    ];

    for (pattern, replacement) in &replacements {
        match Regex::new(pattern) {
            Ok(re) => {
                normalized = re.replace_all(&normalized, *replacement).into_owned();
            }
            Err(e) => {
                warn!("Invalid regex pattern: '{}'. Error: {}", pattern, e);
            }
        }
    }

    // Remove all non-alphanumeric characters except spaces
    normalized = normalized
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect();

    // Normalize whitespace
    normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");

    // Reattach location suffix for branch/location awareness
    if let Some(location) = location_suffix {
        if !location.is_empty() {
            // Only reattach if not empty after normalization
            normalized = format!("{} {}", normalized.trim(), location.trim());
        }
    }

    (normalized.trim().to_string(), entity_type)
}

/// Detect organization type based on name patterns
fn detect_entity_type(normalized_name: &str) -> Option<&'static str> {
    let type_indicators = [
        ("city of", "city"),
        ("county", "county"),
        ("police department", "police"),
        ("fire department", "fire"),
        ("hospital", "hospital"),
        ("medical center", "hospital"),
        ("health", "health"),
        ("school district", "education"),
        ("school", "education"),
        ("college", "education"),
        ("university", "education"),
        ("church", "religious"),
        ("food bank", "food"),
        ("library", "library"),
        ("senior center", "senior"),
        ("ymca", "recreation"),
        ("community center", "community"),
    ];

    for (indicator, entity_type) in &type_indicators {
        if normalized_name.contains(indicator) {
            return Some(entity_type);
        }
    }

    None
}

/// Calculate context-aware token weight based on token importance
fn get_token_weight(token: &str, entity_type: Option<&str>) -> f32 {
    // Common words get lower weight
    let common_words = [
        "the",
        "and",
        "of",
        "in",
        "for",
        "at",
        "with",
        "by",
        "on",
        "to",
        "service",
        "services",
        "center",
        "association",
        "organization",
    ];

    if common_words.contains(&token) {
        return 0.5; // Lower weight for common words
    }

    // Location names get medium weight
    if LOCATION_PREFIXES.contains(&token) {
        return 0.8; // Medium weight for location prefixes
    }

    // Entity type indicators get high weight
    let type_indicators = [
        "police", "fire", "hospital", "school", "college", "church", "bank", "library",
    ];

    if type_indicators.contains(&token) {
        return 2.0; // Higher weight for entity type indicators
    }

    // Adjust weight based on entity type if available
    if let Some(etype) = entity_type {
        match etype {
            "city" => {
                if token == "city" {
                    0.5
                } else {
                    1.5
                }
            } // City name more important than "city"
            "education" => {
                if token == "school" || token == "college" {
                    1.5
                } else {
                    1.2
                }
            }
            _ => 1.0,
        }
    } else {
        1.0 // Default weight
    }
}

/// Enhanced tokenization with context-aware weighting
fn tokenize_name(
    normalized_name: &str,
    stopwords: &HashSet<String>,
    entity_type: Option<&'static str>,
) -> (HashSet<String>, Vec<(String, f32)>) {
    let mut tokens = HashSet::new();
    let mut weighted_tokens = Vec::new();

    // Process words
    let words: Vec<&str> = normalized_name.split_whitespace().collect();

    // First pass: process individual words
    for word in &words {
        let token = word.to_lowercase();

        // Skip stopwords and very short tokens
        if !stopwords.contains(&token) && token.len() >= MIN_TOKEN_LENGTH {
            tokens.insert(token.clone());

            // Get context-aware weight for this token
            let weight = get_token_weight(&token, entity_type);
            weighted_tokens.push((token, weight));
        }
    }

    // Second pass: add bigrams for better phrase matching
    for i in 0..words.len().saturating_sub(1) {
        let bigram = format!("{} {}", words[i], words[i + 1]).to_lowercase();
        if bigram.len() >= 2 * MIN_TOKEN_LENGTH {
            tokens.insert(bigram.clone());

            // Bigrams get higher weight because they're more specific
            let weight = 1.5;
            weighted_tokens.push((bigram, weight));
        }
    }

    // Special case for cities - make the city name extremely important
    if entity_type == Some("city") && normalized_name.starts_with("city of ") {
        if let Some(city_name) = normalized_name.strip_prefix("city of ") {
            let city = city_name.trim().to_string();
            // Add the full city name as a special token with very high weight
            if !city.is_empty() {
                tokens.insert(city.clone());
                weighted_tokens.push((city, 3.0)); // Very high weight for city name
            }
        }
    }

    (tokens, weighted_tokens)
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

/// Calculate weight-based token overlap for better candidate selection
fn calculate_weighted_token_overlap(
    tokens1: &HashSet<String>,
    tokens2: &HashSet<String>,
    weighted_tokens1: &[(String, f32)],
    weighted_tokens2: &[(String, f32)],
) -> usize {
    // Create maps of token to weight for fast lookup
    let mut token_weights1: HashMap<&str, f32> = HashMap::new();
    let mut token_weights2: HashMap<&str, f32> = HashMap::new();

    for (token, weight) in weighted_tokens1 {
        token_weights1.insert(token, *weight);
    }

    for (token, weight) in weighted_tokens2 {
        token_weights2.insert(token, *weight);
    }

    // Calculate weighted overlap
    let mut total_weight = 0.0;

    for token in tokens1.intersection(tokens2) {
        let weight1 = token_weights1.get(token.as_str()).unwrap_or(&1.0);
        let weight2 = token_weights2.get(token.as_str()).unwrap_or(&1.0);

        // Use the product of weights to prioritize important token matches
        total_weight += weight1 * weight2;
    }

    // Convert to usize for compatibility with existing code
    // Adding 0.5 for proper rounding
    (total_weight + 0.5) as usize
}

/// Domain-specific rule application for name matching
fn apply_domain_rules(
    normalized_name1: &str,
    normalized_name2: &str,
    pre_rl_score: f32,
    entity_type1: Option<&'static str>,
    entity_type2: Option<&'static str>,
) -> f32 {
    // Rule 1: Special handling for city matching
    let is_city1 = normalized_name1.starts_with("city of ");
    let is_city2 = normalized_name2.starts_with("city of ");

    if is_city1 && is_city2 {
        let city_name1 = normalized_name1
            .strip_prefix("city of ")
            .unwrap_or(normalized_name1)
            .trim();
        let city_name2 = normalized_name2
            .strip_prefix("city of ")
            .unwrap_or(normalized_name2)
            .trim();

        // If city names differ at all, apply severe penalty based on how different they are
        if city_name1 != city_name2 {
            // Calculate edit distance between city names
            let edit_distance = levenshtein_distance(city_name1, city_name2);

            // If edit distance is more than 3 characters or >30% of length, severely penalize
            let max_length = std::cmp::max(city_name1.len(), city_name2.len());
            if edit_distance > 3
                || (max_length > 0 && edit_distance as f32 / max_length as f32 > 0.3)
            {
                return pre_rl_score * 0.6; // 40% penalty
            } else {
                // Less penalty for very similar city names
                return pre_rl_score * 0.8; // 20% penalty
            }
        }
    }

    // Rule 2: Organization type compatibility
    if let (Some(t1), Some(t2)) = (entity_type1, entity_type2) {
        if t1 != t2 {
            // Already checked for incompatible types in the main function
            // Here we just apply a penalty for different but not explicitly incompatible types
            return pre_rl_score * 0.85; // 15% penalty
        }
    }

    // Rule 3: Special handling for department names
    if normalized_name1.contains("department") || normalized_name2.contains("department") {
        // If only one contains "department" or they contain different department types
        let dept_types = [
            "police",
            "fire",
            "health",
            "public",
            "revenue",
            "transportation",
        ];

        for dept_type in &dept_types {
            let type1_present = normalized_name1.contains(dept_type);
            let type2_present = normalized_name2.contains(dept_type);

            if type1_present != type2_present {
                // Different department types
                return pre_rl_score * 0.75; // 25% penalty
            }
        }
    }

    // Rule 4: Names that share only common words/location prefixes
    let words1: Vec<&str> = normalized_name1.split_whitespace().collect();
    let words2: Vec<&str> = normalized_name2.split_whitespace().collect();

    // Find common prefix words
    let mut common_prefix_len = 0;
    for i in 0..std::cmp::min(words1.len(), words2.len()) {
        if words1[i] == words2[i] {
            common_prefix_len += 1;
        } else {
            break;
        }
    }

    // If names only share common prefix but are otherwise different
    if common_prefix_len > 0 && common_prefix_len < words1.len() && common_prefix_len < words2.len()
    {
        // Calculate what percentage of words are in the common prefix
        let total_words = words1.len() + words2.len();
        let common_prefix_percentage = (2 * common_prefix_len) as f32 / total_words as f32;

        // If common prefix is a significant portion of the names
        if common_prefix_percentage > 0.3 {
            // Check if the remaining words are distinctly different
            let remaining_similarity = jaro_winkler(
                &words1[common_prefix_len..].join(" "),
                &words2[common_prefix_len..].join(" "),
            );

            // If the remaining parts are significantly different
            if remaining_similarity < 0.7 {
                return pre_rl_score * (0.9 - (common_prefix_percentage * 0.2)); // Penalty based on prefix ratio
            }
        }
    }

    // Rule 5: Handle common location prefixes
    for prefix in &LOCATION_PREFIXES {
        if normalized_name1.starts_with(prefix) && normalized_name2.starts_with(prefix) {
            let suffix1 = normalized_name1[prefix.len()..].trim();
            let suffix2 = normalized_name2[prefix.len()..].trim();

            if !suffix1.is_empty() && !suffix2.is_empty() && jaro_winkler(suffix1, suffix2) < 0.8 {
                return pre_rl_score * 0.8; // 20% penalty for different suffixes after common prefix
            }
        }
    }

    // Rule 6: Special handling for branches/locations in parentheses
    let location_regex = Regex::new(r"\b([a-z]+)$").unwrap_or_else(|_| Regex::new(r"").unwrap());
    if let (Some(loc1_match), Some(loc2_match)) = (
        location_regex.find(normalized_name1),
        location_regex.find(normalized_name2),
    ) {
        let loc1 = loc1_match.as_str();
        let loc2 = loc2_match.as_str();

        // Check if names differ only by location suffix
        let name1_without_loc = &normalized_name1[0..loc1_match.start()].trim();
        let name2_without_loc = &normalized_name2[0..loc2_match.start()].trim();

        if name1_without_loc == name2_without_loc && loc1 != loc2 {
            // Same organization but different locations
            return pre_rl_score * 0.7; // 30% penalty
        }
    }

    // Rule 7: Abbreviated names vs full names
    // For known abbreviations like "WSU", "DSHS"
    let abbrev_regex =
        Regex::new(r"^([A-Za-z]{2,5})\s").unwrap_or_else(|_| Regex::new(r"").unwrap());
    if let (Some(abbr1_match), Some(abbr2_match)) = (
        abbrev_regex.find(normalized_name1),
        abbrev_regex.find(normalized_name2),
    ) {
        let abbr1 = abbr1_match.as_str().trim();
        let abbr2 = abbr2_match.as_str().trim();

        if abbr1 == abbr2 {
            // Same abbreviation but check the rest of the name
            let rest1 = &normalized_name1[abbr1_match.end()..].trim();
            let rest2 = &normalized_name2[abbr2_match.end()..].trim();

            if rest1 != rest2 && jaro_winkler(rest1, rest2) < 0.85 {
                // Same abbreviation but different contexts
                return pre_rl_score * 0.75; // 25% penalty
            }
        }
    }

    // If no rules triggered, return the original score
    pre_rl_score
}

/// Calculate Levenshtein edit distance for better city name comparison
fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    for i in 0..=len1 {
        matrix[i][0] = i;
    }

    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                0
            } else {
                1
            };

            matrix[i][j] = std::cmp::min(
                std::cmp::min(
                    matrix[i - 1][j] + 1, // deletion
                    matrix[i][j - 1] + 1, // insertion
                ),
                matrix[i - 1][j - 1] + cost, // substitution
            );
        }
    }

    matrix[len1][len2]
}
