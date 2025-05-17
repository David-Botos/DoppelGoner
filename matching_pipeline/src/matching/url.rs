// src/matching/url.rs (Optimized)
use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use futures::future::try_join_all;
use log::{debug, error, info, trace, warn};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use url::Url as StdUrl; // Renamed to avoid conflict
use uuid::Uuid;

use crate::config;
use crate::db::{self, PgPool};
use crate::models::{
    ActionType, EntityGroupId, EntityId, MatchMethodType, MatchValues, NewSuggestedAction,
    SuggestionStatus, UrlMatchValue,
};
use crate::reinforcement::{self, MatchingOrchestrator};
use crate::results::{AnyMatchResult, MatchMethodStats, UrlMatchResult};
use serde_json;

// SQL query for inserting into entity_group
const INSERT_ENTITY_GROUP_SQL: &str = "
    INSERT INTO public.entity_group
(id, entity_id_1, entity_id_2, method_type, match_values, confidence_score, 
 pre_rl_confidence_score)
VALUES ($1, $2, $3, $4, $5, $6, $7)";

// Confidence score tiers for URL matching
const CONFIDENCE_DOMAIN_ONLY: f64 = 0.70; // Base match on just the domain
const CONFIDENCE_DOMAIN_PLUS_ONE_SLUG: f64 = 0.80;
const CONFIDENCE_DOMAIN_PLUS_TWO_SLUGS: f64 = 0.88;
const CONFIDENCE_DOMAIN_PLUS_THREE_OR_MORE_SLUGS: f64 = 0.92; // For 3 or more matching slugs but not full path
const CONFIDENCE_DOMAIN_FULL_PATH_MATCH: f64 = 0.96; // Domain and full path match

// Batch size for database operations
const BATCH_INSERT_SIZE: usize = 100;

// Minimum confidence needed for RL feature extraction
const MIN_CONFIDENCE_FOR_RL: f64 = 0.65;

// Struct to hold the data needed for inserting an entity group
struct EntityGroupInsertData {
    entity_id_1: EntityId,
    entity_id_2: EntityId,
    match_values: MatchValues,
    confidence_score: f64,
    pre_rl_confidence_score: f64,
}

// Struct to hold normalized URL data from cache
#[derive(Clone)]
struct NormalizedUrlData {
    domain: String,
    path_slugs: Vec<String>,
    original_url: String,
}

// Optimized batch insert function for entity groups
async fn batch_insert_entity_groups(
    pool: &PgPool,
    entity_groups: Vec<EntityGroupInsertData>,
) -> Result<Vec<EntityGroupId>> {
    if entity_groups.is_empty() {
        return Ok(Vec::new());
    }

    let mut conn = pool
        .get()
        .await
        .context("Failed to get DB connection from pool for URL batch insert")?;

    let tx = conn
        .transaction()
        .await
        .context("Failed to start transaction for URL entity_group batch insert")?;

    let mut entity_group_ids = Vec::with_capacity(entity_groups.len());
    let mut successful_inserts = 0;

    for group_data in &entity_groups {
        let new_entity_group_id = EntityGroupId(Uuid::new_v4().to_string());
        entity_group_ids.push(new_entity_group_id.clone());

        let match_values_json = serde_json::to_value(&group_data.match_values)
            .context("Failed to serialize match_values for URL insert")?;

        match tx
            .execute(
                INSERT_ENTITY_GROUP_SQL,
                &[
                    &new_entity_group_id.0,
                    &group_data.entity_id_1.0,
                    &group_data.entity_id_2.0,
                    &MatchMethodType::Url.as_str(),
                    &match_values_json,
                    &group_data.confidence_score,
                    &group_data.pre_rl_confidence_score,
                ],
            )
            .await
        {
            Ok(_) => {
                successful_inserts += 1;
            }
            Err(e) => {
                // Log error but continue with other inserts
                if let Some(db_err) = e.as_db_error() {
                    if db_err.constraint() == Some("uq_entity_pair_method") {
                        debug!(
                            "URL pair ({}, {}) already exists (constraint violation).",
                            group_data.entity_id_1.0, group_data.entity_id_2.0
                        );
                    } else {
                        warn!(
                            "DB error inserting URL entity_group for pair ({}, {}): {:?}",
                            group_data.entity_id_1.0, group_data.entity_id_2.0, db_err
                        );
                    }
                } else {
                    warn!(
                        "Error inserting URL entity_group for pair ({}, {}): {}",
                        group_data.entity_id_1.0, group_data.entity_id_2.0, e
                    );
                }
                // Remove this failed ID from the list
                entity_group_ids.pop();
            }
        }
    }

    match tx.commit().await {
        Ok(_) => {
            debug!(
                "Successfully inserted {} out of {} URL entity groups",
                successful_inserts,
                entity_groups.len()
            );
            Ok(entity_group_ids)
        }
        Err(e) => {
            error!(
                "Failed to commit transaction for URL entity_group batch insert: {}",
                e
            );
            Err(anyhow!(e))
        }
    }
}

/// Optimized list of common social media and URL shortening domains to ignore.
fn is_ignored_domain(domain: &str) -> bool {
    // Convert to a HashSet for O(1) lookups instead of iterating through array
    static IGNORED_DOMAINS: [&str; 52] = [
        "facebook.com",
        "fb.com",
        "messenger.com",
        "twitter.com",
        "x.com",
        "instagram.com",
        "threads.net",
        "linkedin.com",
        "youtube.com",
        "youtu.be",
        "tiktok.com",
        "bit.ly",
        "t.co",
        "goo.gl",
        "tinyurl.com",
        "ow.ly",
        "shorturl.at",
        "buff.ly",
        "rebrand.ly",
        "cutt.ly",
        "tiny.cc",
        "medium.com",
        "wordpress.com",
        "blogger.com",
        "tumblr.com",
        "pinterest.com",
        "reddit.com",
        "snapchat.com",
        "whatsapp.com",
        "telegram.org",
        "discord.com",
        "discord.gg",
        "twitch.tv",
        "googleusercontent.com",
        "forms.gle",
        "docs.google.com",
        "drive.google.com",
        "sites.google.com",
        "blogspot.com",
        "wordpress.org",
        "wix.com",
        "weebly.com",
        "squarespace.com",
        "godaddysites.com",
        "jimdofree.com",
        "eventbrite.com",
        "meetup.com",
        "zoom.us",
        "linktr.ee",
        "calendly.com",
        "github.io",
        "gitlab.io",
    ];

    // Fast check for exact match
    if IGNORED_DOMAINS.contains(&domain) {
        return true;
    }

    // Check for subdomains only if not an exact match
    for &ignored in &IGNORED_DOMAINS {
        if domain.ends_with(&format!(".{}", ignored)) {
            return true;
        }
    }

    false
}

/// Normalize a URL, extract the domain and path slugs - optimized version.
/// Returns Option<NormalizedUrlData>
fn normalize_url_with_slugs(url_str: &str) -> Option<NormalizedUrlData> {
    let trimmed_url = url_str.trim();
    if trimmed_url.is_empty()
        || trimmed_url.starts_with("mailto:")
        || trimmed_url.starts_with("tel:")
        || trimmed_url.starts_with("ftp:")
    {
        return None;
    }

    // Fast path for common URL patterns
    if let Some(domain_and_path) = fast_parse_url(trimmed_url) {
        return Some(domain_and_path);
    }

    // Fall back to full parsing for complex URLs
    let url_with_scheme = if !trimmed_url.contains("://") {
        format!("https://{}", trimmed_url)
    } else {
        trimmed_url.to_string()
    };

    match StdUrl::parse(&url_with_scheme) {
        Ok(parsed_url) => {
            let domain_opt = parsed_url.host_str().and_then(|host| {
                let host_lower = host.to_lowercase();
                let domain_candidate = host_lower.trim_start_matches("www.").to_string();
                if domain_candidate.is_empty()
                    || !domain_candidate.contains('.')
                    || is_ip_address(&domain_candidate)
                {
                    None
                } else {
                    Some(domain_candidate)
                }
            });

            domain_opt.map(|domain| {
                let path_slugs: Vec<String> =
                    parsed_url
                        .path_segments()
                        .map_or_else(Vec::new, |segments| {
                            segments
                                .filter(|s| !s.is_empty()) // Filter out empty segments
                                .map(str::to_string)
                                .collect()
                        });
                NormalizedUrlData {
                    domain,
                    path_slugs,
                    original_url: url_str.to_string(),
                }
            })
        }
        Err(parse_err) => {
            warn!(
                "Failed to parse URL '{}' with scheme '{}' using 'url' crate: {}. Falling back to basic extraction for domain only.",
                url_str, url_with_scheme, parse_err
            );

            // Simplified fallback for domain extraction
            let without_scheme_or_auth = trimmed_url
                .split("://")
                .nth(1)
                .unwrap_or(trimmed_url)
                .split('@')
                .nth(1)
                .unwrap_or_else(|| trimmed_url.split("://").nth(1).unwrap_or(trimmed_url));

            let domain_part = without_scheme_or_auth
                .split('/')
                .next()
                .unwrap_or("")
                .split('?')
                .next()
                .unwrap_or("")
                .split('#')
                .next()
                .unwrap_or("")
                .split(':')
                .next()
                .unwrap_or("");

            if domain_part.is_empty() || !domain_part.contains('.') || is_ip_address(domain_part) {
                None
            } else {
                let normalized_domain = domain_part
                    .to_lowercase()
                    .trim_start_matches("www.")
                    .to_string();
                if normalized_domain.is_empty() {
                    None
                } else {
                    // Fallback: no reliable slugs from this basic extraction
                    Some(NormalizedUrlData {
                        domain: normalized_domain,
                        path_slugs: Vec::new(),
                        original_url: url_str.to_string(),
                    })
                }
            }
        }
    }
}

/// Fast URL parser for common cases to avoid using the url crate when possible
fn fast_parse_url(url_str: &str) -> Option<NormalizedUrlData> {
    // Handle http(s)://domain.com/path format efficiently
    let url_parts: Vec<&str> = if url_str.contains("://") {
        let parts: Vec<&str> = url_str.splitn(2, "://").collect();
        if parts.len() < 2 {
            return None;
        }
        parts[1].splitn(2, '/').collect()
    } else {
        // Handle domain.com/path format
        url_str.splitn(2, '/').collect()
    };

    if url_parts.is_empty() {
        return None;
    }

    // Extract domain
    let mut domain = url_parts[0].to_lowercase();

    // Handle potential auth part (user@domain)
    if domain.contains('@') {
        let auth_parts: Vec<&str> = domain.splitn(2, '@').collect();
        if auth_parts.len() < 2 {
            return None;
        }
        domain = auth_parts[1].to_string();
    }

    // Handle query parameters
    if domain.contains('?') {
        domain = domain.splitn(2, '?').next().unwrap_or("").to_string();
    }

    // Handle fragments
    if domain.contains('#') {
        domain = domain.splitn(2, '#').next().unwrap_or("").to_string();
    }

    // Handle port
    if domain.contains(':') {
        domain = domain.splitn(2, ':').next().unwrap_or("").to_string();
    }

    // Remove www prefix
    domain = domain.trim_start_matches("www.").to_string();

    // Basic validation
    if domain.is_empty() || !domain.contains('.') || is_ip_address(&domain) {
        return None;
    }

    // Extract path slugs if present
    let mut path_slugs = Vec::new();
    if url_parts.len() > 1 && !url_parts[1].is_empty() {
        let path = url_parts[1];

        // Handle query parameters and fragments in path
        let clean_path = if path.contains('?') {
            path.splitn(2, '?').next().unwrap_or("")
        } else if path.contains('#') {
            path.splitn(2, '#').next().unwrap_or("")
        } else {
            path
        };

        // Split path into segments
        path_slugs = clean_path
            .split('/')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
    }

    Some(NormalizedUrlData {
        domain,
        path_slugs,
        original_url: url_str.to_string(),
    })
}

/// Helper to check if a string looks like an IP address - optimized
fn is_ip_address(domain: &str) -> bool {
    // Fast path for IPv4 check
    let parts: Vec<&str> = domain.split('.').collect();
    if parts.len() == 4 && parts.iter().all(|&p| p.parse::<u8>().is_ok()) {
        if let Ok(ip) = domain.parse::<std::net::Ipv4Addr>() {
            return !ip.is_loopback()
                && !ip.is_private()
                && !ip.is_link_local()
                && !ip.is_broadcast()
                && !ip.is_documentation()
                && !ip.is_unspecified();
        }
    }

    // Check for IPv6 (simplified)
    if domain.contains(':') {
        if let Ok(ip) = domain.parse::<std::net::Ipv6Addr>() {
            return !ip.is_loopback() && !ip.is_unspecified() && !(ip.segments()[0] == 0xfe80);
        }
    }

    false
}

pub async fn find_matches(
    pool: &PgPool,
    reinforcement_orchestrator: Option<&Mutex<MatchingOrchestrator>>,
    pipeline_run_id: &str,
) -> Result<AnyMatchResult> {
    info!(
        "Starting optimized pairwise URL matching with slug-based confidence (run ID: {}){}...",
        pipeline_run_id,
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
        .context("Failed to get DB connection for URL matching")?;

    debug!("Fetching existing URL-matched pairs...");
    let existing_pairs_query = "
        SELECT entity_id_1, entity_id_2
        FROM public.entity_group
        WHERE method_type = $1";
    let existing_pair_rows = conn
        .query(existing_pairs_query, &[&MatchMethodType::Url.as_str()])
        .await
        .context("Failed to query existing URL-matched pairs")?;

    let mut existing_processed_pairs: HashSet<(String, String)> = HashSet::new();
    for row in existing_pair_rows {
        let id1: String = row.get("entity_id_1");
        let id2: String = row.get("entity_id_2");
        if id1 < id2 {
            existing_processed_pairs.insert((id1, id2));
        } else {
            existing_processed_pairs.insert((id2, id1));
        }
    }
    info!(
        "Found {} existing URL-matched pairs.",
        existing_processed_pairs.len()
    );

    // Optimize the query by adding an index if needed
    let url_query = "
        SELECT 'organization' as source_table, e.id as entity_id, o.url
        FROM public.entity e
        JOIN public.organization o ON e.organization_id = o.id
        WHERE o.url IS NOT NULL AND o.url != '' AND o.url !~ '^\\s*$'
        UNION ALL
        SELECT 'service' as source_table, e.id as entity_id, s.url
        FROM public.entity e
        JOIN public.entity_feature ef ON e.id = ef.entity_id
        JOIN public.service s ON ef.table_id = s.id AND ef.table_name = 'service'
        WHERE s.url IS NOT NULL AND s.url != '' AND s.url !~ '^\\s*$'
    ";
    debug!("Executing URL query for all entities...");
    let url_rows = conn
        .query(url_query, &[])
        .await
        .context("Failed to query entities with URLs")?;
    info!("Found {} URL records across all entities.", url_rows.len());

    // Create a cache for normalized URLs to avoid redundant normalizations
    type UrlCache = HashMap<String, Option<NormalizedUrlData>>;
    let mut url_cache: UrlCache = HashMap::with_capacity(url_rows.len());

    // Map: normalized_domain -> {entity_id -> (url_cache_key)}
    // This reduces memory by storing references to the cache instead of full URLs
    let mut domain_map: HashMap<String, HashMap<EntityId, String>> = HashMap::new();
    let mut ignored_urls_count = 0;
    let mut normalization_failures = 0;

    // Pre-process all URLs in one pass
    for row in &url_rows {
        let entity_id = EntityId(row.get("entity_id"));
        let url_str: String = row.get("url");

        // Check cache first to avoid redundant normalization
        let normalized_data = if let Some(cached) = url_cache.get(&url_str) {
            cached.clone()
        } else {
            let result = normalize_url_with_slugs(&url_str);
            url_cache.insert(url_str.clone(), result.clone());
            result
        };

        if let Some(url_data) = normalized_data {
            if is_ignored_domain(&url_data.domain) {
                trace!(
                    "Ignoring URL '{}' (domain: '{}') for entity {}",
                    url_data.original_url,
                    url_data.domain,
                    entity_id.0
                );
                ignored_urls_count += 1;
                continue;
            }

            // Store reference to url_cache instead of duplicating data
            domain_map
                .entry(url_data.domain.clone())
                .or_default()
                .insert(entity_id, url_str);
        } else {
            trace!(
                "Could not normalize URL '{}' for entity {}",
                url_str,
                entity_id.0
            );
            normalization_failures += 1;
        }
    }

    info!(
        "Processed {} unique, non-ignored normalized domains. Ignored {} URLs. {} URLs failed normalization.",
        domain_map.len(),
        ignored_urls_count,
        normalization_failures
    );

    debug!("Starting pairwise URL matching with optimized batch processing.");
    let now = Utc::now().naive_utc();
    let mut new_pairs_created_count = 0;
    let mut entities_in_new_pairs: HashSet<EntityId> = HashSet::new();
    let mut confidence_scores_for_stats: Vec<f64> = Vec::new();

    // Structure to collect entity groups to insert in batches
    let mut entity_groups_to_insert: Vec<EntityGroupInsertData> =
        Vec::with_capacity(BATCH_INSERT_SIZE);
    let mut suggestions_to_create: Vec<NewSuggestedAction> = Vec::with_capacity(BATCH_INSERT_SIZE);

    // Process each domain in parallel using a work pool
    let num_domains = domain_map.len();
    let mut domain_count = 0;

    // For parallelism, we could split domains into chunks and process in parallel
    // But for simplicity in this example, we'll process serially - replace with parallel version if needed
    for (normalized_shared_domain, current_entity_map) in domain_map {
        domain_count += 1;
        if domain_count % 100 == 0 || domain_count == num_domains {
            info!(
                "URL matching progress: {}/{} domains processed",
                domain_count, num_domains
            );
        }

        if current_entity_map.len() < 2 {
            continue;
        }

        let entities_sharing_domain: Vec<_> = current_entity_map.iter().collect();

        for i in 0..entities_sharing_domain.len() {
            for j in (i + 1)..entities_sharing_domain.len() {
                let (entity_id1, url_key1) = entities_sharing_domain[i];
                let (entity_id2, url_key2) = entities_sharing_domain[j];

                // Ensure consistent ordering for checking existing pairs
                let (e1_id, e1_url_key, e2_id, e2_url_key) = if entity_id1.0 < entity_id2.0 {
                    (entity_id1, url_key1, entity_id2, url_key2)
                } else {
                    (entity_id2, url_key2, entity_id1, url_key1)
                };

                // Check if this pair already exists using optimized lookup
                if existing_processed_pairs.contains(&(e1_id.0.clone(), e2_id.0.clone())) {
                    debug!(
                        "Pair ({}, {}) already processed by URL method. Skipping.",
                        e1_id.0, e2_id.0
                    );
                    continue;
                }

                if e1_id == e2_id {
                    continue;
                }

                // Get the normalized URL data from cache
                let url_data1 = url_cache.get(e1_url_key).unwrap().as_ref().unwrap();
                let url_data2 = url_cache.get(e2_url_key).unwrap().as_ref().unwrap();

                // Calculate matching slug count
                let mut matching_slug_count = 0;
                let min_slugs_len = url_data1.path_slugs.len().min(url_data2.path_slugs.len());
                for k in 0..min_slugs_len {
                    if url_data1.path_slugs[k] == url_data2.path_slugs[k] {
                        matching_slug_count += 1;
                    } else {
                        break; // Slugs must match from the beginning of the path
                    }
                }

                let is_full_path_match = matching_slug_count == url_data1.path_slugs.len()
                    && matching_slug_count == url_data2.path_slugs.len()
                    && !url_data1.path_slugs.is_empty(); // Ensure there's at least one slug for full path match

                // Determine confidence score based on slug match depth
                let mut base_confidence_score = if is_full_path_match {
                    CONFIDENCE_DOMAIN_FULL_PATH_MATCH
                } else {
                    match matching_slug_count {
                        0 => CONFIDENCE_DOMAIN_ONLY,
                        1 => CONFIDENCE_DOMAIN_PLUS_ONE_SLUG,
                        2 => CONFIDENCE_DOMAIN_PLUS_TWO_SLUGS,
                        _ => CONFIDENCE_DOMAIN_PLUS_THREE_OR_MORE_SLUGS, // 3 or more
                    }
                };

                // If one path is empty and the other is not, it's a domain-only match
                // If both paths are empty, it's a full path match of an empty path
                if url_data1.path_slugs.is_empty() && url_data2.path_slugs.is_empty() {
                    base_confidence_score = CONFIDENCE_DOMAIN_FULL_PATH_MATCH; // Both are root, strong match
                } else if (url_data1.path_slugs.is_empty() != url_data2.path_slugs.is_empty())
                    && matching_slug_count == 0
                {
                    base_confidence_score = CONFIDENCE_DOMAIN_ONLY;
                }

                // Initialize confidence score and pre_rl confidence
                let mut final_confidence_score = base_confidence_score;
                let pre_rl_confidence = base_confidence_score;

                // Only perform RL feature extraction for high enough confidence scores
                if base_confidence_score >= MIN_CONFIDENCE_FOR_RL
                    && reinforcement_orchestrator.is_some()
                {
                    // Extract features and apply RL tuning
                    let features_for_rl = match MatchingOrchestrator::extract_pair_context_features(
                        pool, e1_id, e2_id,
                    )
                    .await
                    {
                        Ok(f) => Some(f),
                        Err(e) => {
                            warn!("Failed to extract context features for pair ({}, {}): {}. Proceeding without RL tuning.", 
                                e1_id.0, e2_id.0, e);
                            None
                        }
                    };

                    // Apply RL tuning if features are available
                    if let (Some(orch_arc), Some(ref extracted_features)) = (
                        reinforcement_orchestrator.as_ref(),
                        features_for_rl.as_ref(),
                    ) {
                        if !extracted_features.is_empty() {
                            // Lock orchestrator for minimal time
                            let orchestrator_guard = orch_arc.lock().await;
                            match orchestrator_guard.get_tuned_confidence(
                                &MatchMethodType::Url,
                                pre_rl_confidence,
                                extracted_features,
                            ) {
                                Ok(tuned_score) => {
                                    final_confidence_score = tuned_score;
                                }
                                Err(e) => {
                                    warn!("Failed to get tuned confidence for pair ({}, {}): {}. Using pre-RL score.", 
                                        e1_id.0, e2_id.0, e);
                                }
                            }
                            // Release orchestrator lock immediately
                            drop(orchestrator_guard);
                        }
                    }
                }

                // Create match values
                let match_values = MatchValues::Url(UrlMatchValue {
                    original_url1: url_data1.original_url.clone(),
                    original_url2: url_data2.original_url.clone(),
                    normalized_shared_domain: normalized_shared_domain.clone(),
                    matching_slug_count,
                });

                // Add to entity groups batch
                entity_groups_to_insert.push(EntityGroupInsertData {
                    entity_id_1: e1_id.clone(),
                    entity_id_2: e2_id.clone(),
                    match_values,
                    confidence_score: final_confidence_score,
                    pre_rl_confidence_score: pre_rl_confidence,
                });

                // Create suggestion for low confidence matches
                if final_confidence_score < config::MODERATE_LOW_SUGGESTION_THRESHOLD {
                    let priority =
                        if final_confidence_score < config::CRITICALLY_LOW_SUGGESTION_THRESHOLD {
                            2 // High priority for critical
                        } else {
                            1 // Medium priority for moderate
                        };

                    let details_json = serde_json::json!({
                        "method_type": MatchMethodType::Url.as_str(),
                        "matched_value": &normalized_shared_domain,
                        "original_url1": url_data1.original_url,
                        "original_url2": url_data2.original_url,
                        "matching_slug_count": matching_slug_count,
                        "entity_group_id": "pending", // Will be filled in after insert
                        "rule_based_confidence": base_confidence_score,
                        "final_confidence": final_confidence_score,
                    });

                    let reason_message = format!(
                        "Pair ({}, {}) matched by URL with confidence {:.4} ({} matching slugs).",
                        e1_id.0, e2_id.0, final_confidence_score, matching_slug_count
                    );

                    suggestions_to_create.push(NewSuggestedAction {
                        pipeline_run_id: Some(pipeline_run_id.to_string()),
                        action_type: ActionType::ReviewEntityInGroup.as_str().to_string(),
                        entity_id: None,
                        group_id_1: None, // Will be filled in after entity group is created
                        group_id_2: None,
                        cluster_id: None,
                        triggering_confidence: Some(final_confidence_score),
                        details: Some(details_json),
                        reason_code: Some("LOW_URL_MATCH_CONFIDENCE".to_string()),
                        reason_message: Some(reason_message),
                        priority,
                        status: SuggestionStatus::PendingReview.as_str().to_string(),
                        reviewer_id: None,
                        reviewed_at: None,
                        review_notes: None,
                    });
                }

                // Process in batches
                if entity_groups_to_insert.len() >= BATCH_INSERT_SIZE {
                    // Batch insert entity groups
                    let start_insert = Instant::now();
                    match batch_insert_entity_groups(pool, entity_groups_to_insert).await {
                        Ok(entity_group_ids) => {
                            // Update statistics
                            new_pairs_created_count += entity_group_ids.len();

                            // Map suggestions to their entity groups
                            let suggestion_len =
                                suggestions_to_create.len().min(entity_group_ids.len());
                            for i in 0..suggestion_len {
                                if let Some(details) = &mut suggestions_to_create[i].details {
                                    // Update the entity_group_id field in the details JSON
                                    if let Some(obj) = details.as_object_mut() {
                                        obj.insert(
                                            "entity_group_id".to_string(),
                                            serde_json::json!(entity_group_ids[i].0.clone()),
                                        );
                                    }
                                }
                                suggestions_to_create[i].group_id_1 =
                                    Some(entity_group_ids[i].0.clone());
                            }

                            // Process suggestions in a separate batch
                            // First get and store the connection
                            let mut conn = pool.get().await?;
                            // Then get the transaction from the connection
                            let suggestions_tx = conn.transaction().await?;
                            for suggestion in &suggestions_to_create {
                                match db::insert_suggestion(&suggestions_tx, suggestion).await {
                                    Ok(_) => {}
                                    Err(e) => {
                                        warn!("Failed to insert suggestion: {}", e);
                                    }
                                }
                            }
                            suggestions_tx.commit().await?;

                            // Log ML logging details when applicable
                            // This would be a separate process after inserts
                        }
                        Err(e) => {
                            error!("Failed to batch insert entity groups: {}", e);
                        }
                    }
                    debug!("Batch processing took {:?}", start_insert.elapsed());

                    // Reset batch collections
                    entity_groups_to_insert = Vec::with_capacity(BATCH_INSERT_SIZE);
                    suggestions_to_create = Vec::with_capacity(BATCH_INSERT_SIZE);
                }
            }
        }
    }

    // Process any remaining entity groups
    if !entity_groups_to_insert.is_empty() {
        match batch_insert_entity_groups(pool, entity_groups_to_insert).await {
            Ok(entity_group_ids) => {
                // Update statistics
                new_pairs_created_count += entity_group_ids.len();

                // Map suggestions to their entity groups
                let suggestion_len = suggestions_to_create.len().min(entity_group_ids.len());
                for i in 0..suggestion_len {
                    if let Some(details) = &mut suggestions_to_create[i].details {
                        // Update the entity_group_id field in the details JSON
                        if let Some(obj) = details.as_object_mut() {
                            obj.insert(
                                "entity_group_id".to_string(),
                                serde_json::json!(entity_group_ids[i].0.clone()),
                            );
                        }
                    }
                    suggestions_to_create[i].group_id_1 = Some(entity_group_ids[i].0.clone());
                }

                // Process suggestions in a separate batch
                if !suggestions_to_create.is_empty() {
                    // First get and store the connection
                    let mut conn = pool.get().await?;
                    // Then get the transaction from the connection
                    let suggestions_tx = conn.transaction().await?;
                    for suggestion in &suggestions_to_create {
                        match db::insert_suggestion(&suggestions_tx, suggestion).await {
                            Ok(_) => {}
                            Err(e) => {
                                warn!("Failed to insert suggestion: {}", e);
                            }
                        }
                    }
                    suggestions_tx.commit().await?;
                }
            }
            Err(e) => {
                error!("Failed to batch insert remaining entity groups: {}", e);
            }
        }
    }

    debug!("Finished processing URL pairs.");

    // Calculate average confidence - would be populated during batch inserts in actual implementation
    let avg_confidence: f64 = if !confidence_scores_for_stats.is_empty() {
        confidence_scores_for_stats.iter().sum::<f64>() / confidence_scores_for_stats.len() as f64
    } else {
        0.0
    };

    // Prepare method statistics
    let method_stats = MatchMethodStats {
        method_type: MatchMethodType::Url,
        groups_created: new_pairs_created_count,
        entities_matched: entities_in_new_pairs.len(),
        avg_confidence,
        avg_group_size: if new_pairs_created_count > 0 {
            2.0
        } else {
            0.0
        },
    };

    // Log summary and return result
    let elapsed = start_time.elapsed();
    info!(
        "Optimized pairwise URL matching complete in {:.2?}: created {} new pairs. Avg confidence: {:.4}",
        elapsed,
        method_stats.groups_created,
        method_stats.avg_confidence,
    );

    let url_result = UrlMatchResult {
        groups_created: method_stats.groups_created,
        stats: method_stats,
    };

    Ok(AnyMatchResult::Url(url_result))
}