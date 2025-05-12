// src/matching/url.rs
use anyhow::{Context, Result};
use chrono::Utc;
use log::{debug, info, trace, warn};
use std::collections::{HashMap, HashSet};
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
use crate::results::{AnyMatchResult, MatchMethodStats, UrlMatchResult}; // Removed PairMlResult as it's not used
use serde_json;

// SQL query for inserting into entity_group
const INSERT_ENTITY_GROUP_SQL: &str = "
    INSERT INTO public.entity_group
    (id, entity_id_1, entity_id_2, method_type, match_values, confidence_score, created_at, updated_at, version)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 1)";

/// List of common social media and URL shortening domains to ignore.
fn is_ignored_domain(domain: &str) -> bool {
    let ignored_domains = [
        "facebook.com",
        "fb.com",
        "messenger.com",
        "twitter.com",
        "x.com",
        "instagram.com",
        "threads.net",
        "linkedin.com",
        // Simplified youtube - consider more robust checking if subdomains like music.youtube.com are relevant
        "youtube.com", // Covers www.youtube.com, m.youtube.com etc. after normalization
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
        "googleusercontent.com", // Catch-all for user content
        // Add other generic platform domains if necessary
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
    ];
    // Check if the domain exactly matches or is a subdomain of an ignored TLD+1
    ignored_domains
        .iter()
        .any(|&ignored| domain == ignored || domain.ends_with(&format!(".{}", ignored)))
}

/// Normalize a URL by extracting and cleaning the domain.
/// Returns Option<(normalized_domain, original_url_str_if_valid_host)>
fn normalize_url(url_str: &str) -> Option<(String, String)> {
    let trimmed_url = url_str.trim();
    if trimmed_url.is_empty()
        || trimmed_url.starts_with("mailto:")
        || trimmed_url.starts_with("tel:")
        || trimmed_url.starts_with("ftp:")
    {
        return None;
    }

    // Prepend scheme if missing, default to https for robust parsing
    let url_with_scheme = if !trimmed_url.contains("://") {
        format!("https://{}", trimmed_url)
    } else {
        trimmed_url.to_string()
    };

    match StdUrl::parse(&url_with_scheme) {
        Ok(parsed_url) => {
            parsed_url.host_str().and_then(|host| {
                let host_lower = host.to_lowercase();
                // Remove www. prefix
                let domain_candidate = host_lower.trim_start_matches("www.").to_string();

                // Basic validation: must contain a dot, not be empty, and not be an IP address
                if domain_candidate.is_empty()
                    || !domain_candidate.contains('.')
                    || is_ip_address(&domain_candidate)
                {
                    None
                } else {
                    Some((domain_candidate, url_str.to_string())) // Return original URL for UrlMatchValue
                }
            })
        }
        Err(parse_err) => {
            warn!("Failed to parse URL '{}' with scheme '{}' using 'url' crate: {}. Falling back to basic extraction.", url_str, url_with_scheme, parse_err);
            // Fallback for URLs the `url` crate struggles with, but try to be stricter
            let without_scheme_or_auth = trimmed_url
                .split("://")
                .nth(1)
                .unwrap_or(trimmed_url) // Remove scheme if present
                .split('@')
                .nth(1)
                .unwrap_or_else(|| trimmed_url.split("://").nth(1).unwrap_or(trimmed_url)); // Remove user:pass@ if present

            let domain_part = without_scheme_or_auth
                .split('/')
                .next()
                .unwrap_or("") // Get part before first '/'
                .split('?')
                .next()
                .unwrap_or("") // Get part before first '?'
                .split('#')
                .next()
                .unwrap_or("") // Get part before first '#'
                .split(':')
                .next()
                .unwrap_or(""); // Get part before first ':' (port)

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
                    Some((normalized_domain, url_str.to_string()))
                }
            }
        }
    }
}

/// Helper to check if a string looks like an IP address.
fn is_ip_address(domain: &str) -> bool {
    // Check for IPv4
    if domain.split('.').count() == 4 && domain.chars().all(|c| c.is_ascii_digit() || c == '.') {
        if let Ok(ip) = domain.parse::<std::net::Ipv4Addr>() {
            return !ip.is_loopback()
                && !ip.is_private()
                && !ip.is_link_local()
                && !ip.is_broadcast()
                && !ip.is_documentation()
                && !ip.is_unspecified();
        }
    }
    // Check for IPv6 (simplified: contains ':')
    if domain.contains(':') {
        if let Ok(ip) = domain.parse::<std::net::Ipv6Addr>() {
            return !ip.is_loopback() && !ip.is_unspecified() && !(ip.segments()[0] == 0xfe80);
            // Link-local
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
        "Starting pairwise URL matching (run ID: {}){}...",
        pipeline_run_id,
        if reinforcement_orchestrator.is_some() {
            " with ML guidance"
        } else {
            ""
        }
    );
    let start_time = Instant::now();

    // Get a connection from the pool. This connection will be used for all operations.
    // No single overarching transaction will be used for the entire process.
    let mut conn = pool
        .get()
        .await
        .context("Failed to get DB connection for URL matching")?;

    // 1. Fetch existing URL-matched pairs
    debug!("Fetching existing URL-matched pairs...");
    let existing_pairs_query = "
        SELECT entity_id_1, entity_id_2
        FROM public.entity_group
        WHERE method_type = $1";
    let existing_pair_rows = conn
        .query(existing_pairs_query, &[&MatchMethodType::Url.as_str()])
        .await
        .context("Failed to query existing URL-matched pairs")?;

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
        "Found {} existing URL-matched pairs.",
        existing_processed_pairs.len()
    );

    // 2. Fetch URL Data for all entities
    // Fetches from both 'organization' table and 'service' table via 'entity_feature'
    let url_query = "
        SELECT 'organization' as source_table, e.id as entity_id, o.url
        FROM entity e
        JOIN organization o ON e.organization_id = o.id
        WHERE o.url IS NOT NULL AND o.url != '' AND o.url !~ '^\\s*$'
        UNION ALL
        SELECT 'service' as source_table, e.id as entity_id, s.url
        FROM entity e
        JOIN entity_feature ef ON e.id = ef.entity_id
        JOIN service s ON ef.table_id = s.id AND ef.table_name = 'service'
        WHERE s.url IS NOT NULL AND s.url != '' AND s.url !~ '^\\s*$'
    ";
    debug!("Executing URL query for all entities...");
    let url_rows = conn
        .query(url_query, &[])
        .await
        .context("Failed to query entities with URLs")?;
    info!("Found {} URL records across all entities.", url_rows.len());

    // Map: normalized_domain -> {entity_id -> original_url_str}
    let mut domain_map: HashMap<String, HashMap<EntityId, String>> = HashMap::new();
    let mut ignored_urls_count = 0;
    let mut normalization_failures = 0;

    for row in &url_rows {
        let entity_id = EntityId(row.get("entity_id"));
        let url_str: String = row.get("url");

        if let Some((normalized_domain, original_url_for_map)) = normalize_url(&url_str) {
            if is_ignored_domain(&normalized_domain) {
                trace!(
                    "Ignoring URL '{}' (domain: '{}') for entity {}",
                    original_url_for_map,
                    normalized_domain,
                    entity_id.0
                );
                ignored_urls_count += 1;
                continue;
            }
            domain_map
                .entry(normalized_domain)
                .or_default()
                .insert(entity_id, original_url_for_map);
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

    // 3. Database Operations (No transaction spanning all inserts)
    debug!("Starting pairwise URL matching inserts (non-transactional for the whole batch).");

    let now = Utc::now().naive_utc();
    let mut new_pairs_created_count = 0;
    let mut entities_in_new_pairs: HashSet<EntityId> = HashSet::new();
    let mut confidence_scores_for_stats: Vec<f64> = Vec::new();

    // 4. Process domains and form pairs
    for (normalized_shared_domain, current_entity_map) in domain_map {
        if current_entity_map.len() < 2 {
            continue;
        }

        let entities_sharing_domain: Vec<_> = current_entity_map.iter().collect();

        for i in 0..entities_sharing_domain.len() {
            for j in (i + 1)..entities_sharing_domain.len() {
                let (entity_id1_obj, original_url1) = entities_sharing_domain[i];
                let (entity_id2_obj, original_url2) = entities_sharing_domain[j];

                let (e1_id, e1_orig_url, e2_id, e2_orig_url) =
                    if entity_id1_obj.0 < entity_id2_obj.0 {
                        (entity_id1_obj, original_url1, entity_id2_obj, original_url2)
                    } else {
                        (entity_id2_obj, original_url2, entity_id1_obj, original_url1)
                    };

                if existing_processed_pairs.contains(&(e1_id.clone(), e2_id.clone())) {
                    debug!(
                        "Pair ({}, {}) already processed by URL method. Skipping.",
                        e1_id.0, e2_id.0
                    );
                    continue;
                }
                // Prevent self-matching if somehow an entity has multiple URLs leading to the same normalized domain
                // This check is more robustly handled by the i < j loop and distinct entity_ids in current_entity_map
                if e1_id == e2_id {
                    continue;
                }

                let mut final_confidence_score = 0.9; // Default for URL domain match
                let mut predicted_method_type_from_ml = MatchMethodType::Url;
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
                                    info!("ML guidance for URL pair ({}, {}), domain '{}': Predicted Method: {:?}, Confidence: {:.4}", e1_id.0, e2_id.0, normalized_shared_domain, predicted_method_type_from_ml, final_confidence_score);
                                }
                                Err(e) => {
                                    warn!("ML prediction failed for URL pair ({}, {}), domain '{}': {}. Using default confidence {:.2}.", e1_id.0, e2_id.0, normalized_shared_domain, e, final_confidence_score);
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Context extraction failed for URL pair ({}, {}), domain '{}': {}. Using default confidence {:.2}.", e1_id.0, e2_id.0, normalized_shared_domain, e, final_confidence_score);
                        }
                    }
                }

                let match_values = MatchValues::Url(UrlMatchValue {
                    original_url1: e1_orig_url.clone(),
                    original_url2: e2_orig_url.clone(),
                    normalized_shared_domain: normalized_shared_domain.clone(),
                });
                let match_values_json = serde_json::to_value(&match_values).with_context(|| {
                    format!(
                        "Failed to serialize URL MatchValues for pair ({}, {})",
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
                            &MatchMethodType::Url.as_str(),
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
                            "Created new URL pair group {} for ({}, {}) with shared domain '{}', confidence: {:.4}",
                            new_entity_group_id.0, e1_id.0, e2_id.0, normalized_shared_domain, final_confidence_score
                        );

                        if let Some(orchestrator_mutex) = reinforcement_orchestrator {
                            let mut orchestrator_guard = orchestrator_mutex.lock().await;
                            if let Err(e) = orchestrator_guard
                                .log_match_result(
                                    pool,
                                    e1_id, // Assuming EntityId is Clone or Copy
                                    e2_id,
                                    &predicted_method_type_from_ml, // This is the method predicted by ML
                                    final_confidence_score, // This is the confidence from ML (or default if ML failed)
                                    true, // is_match (since we are creating a pair)
                                    features_for_logging.as_ref(),
                                    Some(&MatchMethodType::Url), // actual_method_type (the method that made the decision)
                                    Some(final_confidence_score), // actual_confidence (the confidence for this method)
                                )
                                .await
                            {
                                warn!("Failed to log URL match result to entity_match_pairs for ({},{}): {}", e1_id.0, e2_id.0, e);
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
                                "method_type": MatchMethodType::Url.as_str(),
                                "matched_value": &normalized_shared_domain,
                                "original_url1": e1_orig_url,
                                "original_url2": e2_orig_url,
                                "entity_group_id": &new_entity_group_id.0,
                                "rl_predicted_method": predicted_method_type_from_ml.as_str(), // Log what RL predicted
                            });
                            let reason_message = format!(
                                "Pair ({}, {}) matched by URL with low RL confidence ({:.4}). RL predicted: {:?}.",
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
                                reason_code: Some("LOW_RL_CONFIDENCE_PAIR".to_string()), // Or "LOW_URL_CONFIDENCE_PAIR"
                                reason_message: Some(reason_message),
                                priority,
                                status: SuggestionStatus::PendingReview.as_str().to_string(),
                                reviewer_id: None,
                                reviewed_at: None,
                                review_notes: None,
                            };
                            // Pass the connection `conn` to insert_suggestion
                            if let Err(e) = db::insert_suggestion(&*conn, &suggestion).await {
                                warn!("Failed to log suggestion for low confidence URL pair ({}, {}): {}. This operation was attempted on the main connection.", e1_id.0, e2_id.0, e);
                            }
                        }
                    }
                    Err(e) => {
                        // Detailed error logging for the specific insert failure
                        let error_message = format!(
                            "Failed to insert URL pair group for ({}, {}) with shared domain '{}'",
                            e1_id.0, e2_id.0, normalized_shared_domain
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
    debug!("Finished processing URL pairs.");

    // 5. Calculate Statistics and Return
    let avg_confidence: f64 = if !confidence_scores_for_stats.is_empty() {
        confidence_scores_for_stats.iter().sum::<f64>() / confidence_scores_for_stats.len() as f64
    } else {
        0.0
    };

    let method_stats = MatchMethodStats {
        method_type: MatchMethodType::Url,
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
        "Pairwise URL matching complete in {:.2?}: created {} new pairs, involving {} unique entities. Errors for individual pairs (if any) were logged above.",
        elapsed,
        method_stats.groups_created,
        method_stats.entities_matched
    );
    let url_result = UrlMatchResult {
        groups_created: method_stats.groups_created,
        stats: method_stats,
    };

    Ok(AnyMatchResult::Url(url_result))
}
