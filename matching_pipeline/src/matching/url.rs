// src/matching/url.rs
use anyhow::{Context, Result};
use chrono::Utc;
use log::{debug, info};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use url::Url;
use uuid::Uuid;

use crate::db::PgPool;
use crate::models::{
    EntityGroup, EntityGroupId, EntityId, MatchMethodType, MatchValues, UrlMatchValue,
};
use crate::results::{MatchMethodStats, UrlMatchResult};

/// List of common social media and URL shortening domains to ignore
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
    ];

    ignored_domains
        .iter()
        .any(|&ignored| domain == ignored || domain.ends_with(&format!(".{}", ignored)))
}

pub async fn find_matches(pool: &PgPool) -> Result<UrlMatchResult> {
    info!("Starting URL matching...");
    let start_time = Instant::now();

    let mut conn = pool
        .get()
        .await
        .context("Failed to get DB connection for URL matching")?;

    // First, get all entities that are already part of a URL match group
    debug!("Finding entities already processed by URL matching");
    let processed_query = "
        SELECT DISTINCT ge.entity_id
        FROM group_entity ge
        JOIN group_method gm ON ge.entity_group_id = gm.entity_group_id
        WHERE gm.method_type = 'url'
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
        "Found {} entities already processed by URL matching",
        processed_entities.len()
    );

    // Get existing URL groups and their normalized domains
    debug!("Finding existing URL groups");
    let existing_groups_query = "
        SELECT 
            eg.id AS group_id,
            gm.id AS method_id,
            gm.match_values
        FROM 
            entity_group eg
            JOIN group_method gm ON eg.id = gm.entity_group_id
        WHERE 
            gm.method_type = 'url'
    ";

    let existing_groups_rows = conn
        .query(existing_groups_query, &[])
        .await
        .context("Failed to query existing groups")?;

    // Map of normalized domain -> (group_id, method_id)
    let mut existing_domain_groups: HashMap<String, (String, String)> = HashMap::new();

    for row in &existing_groups_rows {
        let group_id: String = row.get("group_id");
        let method_id: String = row.get("method_id");
        let match_values_json: serde_json::Value = row.get("match_values");

        // Extract domains from the match_values JSON
        if let Ok(match_values) = serde_json::from_value::<MatchValues>(match_values_json.clone()) {
            if let MatchValues::Url(url_values) = match_values {
                // The normalized domain is not directly stored in UrlMatchValue
                // We need to extract it from the domain field after removing www.
                for url_value in &url_values {
                    let normalized_domain = url_value.domain.trim_start_matches("www.").to_string();
                    // Store mapping from normalized domain to group/method ids
                    existing_domain_groups
                        .insert(normalized_domain, (group_id.clone(), method_id.clone()));
                }
            }
        }
    }

    info!(
        "Found {} unique normalized domains in existing groups",
        existing_domain_groups.len()
    );

    // Combine both queries into a single query with a source indicator
    // Also filter out already processed entities
    let url_query = "
        SELECT 'organization' as source, e.id as entity_id, o.url 
        FROM entity e 
        JOIN organization o ON e.organization_id = o.id 
        WHERE o.url IS NOT NULL AND o.url != ''
        AND e.id NOT IN (
            SELECT ge.entity_id 
            FROM group_entity ge
            JOIN group_method gm ON ge.entity_group_id = gm.entity_group_id
            WHERE gm.method_type = 'url'
        )
        
        UNION ALL
        
        SELECT 'service' as source, e.id as entity_id, s.url 
        FROM entity e 
        JOIN entity_feature ef ON e.id = ef.entity_id 
        JOIN service s ON ef.table_id = s.id 
        WHERE ef.table_name = 'service' 
        AND s.url IS NOT NULL AND s.url != ''
        AND e.id NOT IN (
            SELECT ge.entity_id 
            FROM group_entity ge
            JOIN group_method gm ON ge.entity_group_id = gm.entity_group_id
            WHERE gm.method_type = 'url'
        )
    ";

    debug!("Executing combined URL query");
    let url_rows = conn
        .query(url_query, &[])
        .await
        .context("Failed to query URLs")?;

    info!("Found {} URL records to process", url_rows.len());

    // Map to store normalized domains -> map of entities with matching values
    // Using HashMap<normalized_domain, HashMap<entity_id, (original_url, domain)>> to avoid duplicates
    let mut domain_map: HashMap<String, HashMap<EntityId, (String, String)>> = HashMap::new();
    let mut processed_count = 0;
    let mut ignored_count = 0;

    // Process all URLs in a single pass
    for row in &url_rows {
        let entity_id: String = row.get("entity_id");
        let entity_id = EntityId(entity_id);

        // Skip if this entity has already been processed
        if processed_entities.contains(&entity_id) {
            continue;
        }

        let url_str: String = row.get("url");

        if let Some((normalized, domain)) = normalize_url(&url_str) {
            // Skip if this is a domain we want to ignore
            if is_ignored_domain(&normalized) {
                ignored_count += 1;
                continue;
            }

            // Get or create the inner entity map for this normalized domain
            let entity_map = domain_map.entry(normalized).or_default();

            // Only add if this entity isn't already in the map for this domain
            entity_map.entry(entity_id).or_insert((url_str, domain));
        }

        processed_count += 1;
        if processed_count % 1000 == 0 {
            debug!("Processed {} URL records so far", processed_count);
        }
    }

    let unique_domains = domain_map.len();
    info!(
        "Found {} unique normalized domains from unprocessed records (ignored {} social media/shortener domains)",
        unique_domains, ignored_count
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

    // Process domains
    for (normalized_domain, entity_map) in domain_map {
        // Skip if this is a domain we want to ignore (double check)
        if is_ignored_domain(&normalized_domain) {
            continue;
        }

        // Store value for logging
        let entity_map_length = entity_map.len();

        // Check if this domain already has a group
        if let Some((group_id, method_id)) = existing_domain_groups.get(&normalized_domain) {
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

            // Extract the URL values
            let mut url_values = if let MatchValues::Url(values) = current_match_values {
                values
            } else {
                Vec::new()
            };

            // Track which entities we've actually added
            let mut entities_added = 0;

            // Add the new entities to the group
            for (entity_id, (original, domain)) in entity_map {
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
                url_values.push(UrlMatchValue {
                    original,
                    domain,
                    entity_id,
                });

                entities_added += 1;
            }

            // Update the match values
            let updated_match_values = MatchValues::Url(url_values);
            let updated_json = serde_json::to_value(updated_match_values)
                .context("Failed to serialize updated match values")?;

            tx.execute(&update_method_stmt, &[&updated_json, &method_id])
                .await
                .context("Failed to update match values")?;

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

            // URL matches have good confidence
            confidence_scores.push(0.9);

            info!(
                "Added {} entities to existing URL group for {}",
                entities_added, normalized_domain
            );
        } else if entity_map.len() >= 2 {
            // Create a new group for this domain (only if multiple entities)

            // Convert the entity map to a vector of (EntityId, String, String) tuples
            let entities: Vec<(EntityId, String, String)> = entity_map
                .into_iter()
                .map(|(entity_id, (url_str, domain))| (entity_id, url_str, domain))
                .collect();

            let entity_count = entities.len() as i32;

            // Create a new entity group
            let group_id = EntityGroupId(Uuid::new_v4().to_string());

            let group = EntityGroup {
                id: group_id.clone(),
                name: Some(format!("URL match on {}", normalized_domain)),
                group_cluster_id: None,
                created_at: now,
                updated_at: now,
                confidence_score: 0.9, // High confidence for domain matches
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

            // Add all entities to the batch inserts
            for (entity_id, original, domain) in &entities {
                // Add to group_entity
                tx.execute(
                    &entity_stmt,
                    &[&Uuid::new_v4().to_string(), &group_id.0, &entity_id.0, &now],
                )
                .await
                .context("Failed to insert group entity")?;

                // Create match value for this entity
                match_values.push(UrlMatchValue {
                    original: original.clone(),
                    domain: domain.clone(),
                    entity_id: entity_id.clone(),
                });
            }

            // Serialize match values to JSON
            let method_values = MatchValues::Url(match_values);
            let match_values_json =
                serde_json::to_value(&method_values).context("Failed to serialize match values")?;

            // Insert group method
            tx.execute(
                &method_stmt,
                &[
                    &Uuid::new_v4().to_string(),
                    &group_id.0,
                    &MatchMethodType::Url.as_str(),
                    &format!("Matched on domain: {}", normalized_domain),
                    &match_values_json,
                    &0.9f64,
                    &now,
                ],
            )
            .await
            .context("Failed to insert group method")?;

            total_groups_created += 1;
            total_entities_matched += entity_count as usize;

            // URL matches have good confidence
            confidence_scores.push(0.9);
            group_sizes.push(entity_count as f32);

            info!(
                "Created new URL group for {} with {} entities",
                normalized_domain, entity_count
            );
        }
    }

    // Commit the transaction
    tx.commit().await.context("Failed to commit transaction")?;

    // Calculate average confidence score and group size
    let avg_confidence: f64 = if !confidence_scores.is_empty() {
        (confidence_scores.iter().sum::<f32>() / confidence_scores.len() as f32).into()
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
        method_type: MatchMethodType::Url,
        groups_created: total_groups_created,
        entities_matched: total_entities_matched,
        avg_confidence,
        avg_group_size,
    };

    let elapsed = start_time.elapsed();
    info!(
        "URL matching complete: created {} new entity groups and added {} entities to existing groups in {:.2?}",
        total_groups_created, total_entities_added, elapsed
    );

    Ok(UrlMatchResult {
        groups_created: total_groups_created,
        stats: method_stats,
    })
}

/// Normalize a URL by:
/// - Extracting the domain name
/// - Removing "www." prefix
/// - Converting to lowercase
///
/// Returns (normalized domain, original domain) tuple
fn normalize_url(url_str: &str) -> Option<(String, String)> {
    // Ensure URL has a scheme
    let url_with_scheme = if !url_str.starts_with("http://") && !url_str.starts_with("https://") {
        format!("https://{}", url_str)
    } else {
        url_str.to_string()
    };

    // Parse the URL
    match Url::parse(&url_with_scheme) {
        Ok(url) => {
            // Extract the host/domain
            if let Some(host) = url.host_str() {
                // Store original domain
                let original_domain = host.to_string();

                // Normalize domain: lowercase and remove www prefix
                let normalized = host.to_lowercase().trim_start_matches("www.").to_string();

                // Skip empty domains after normalization
                if !normalized.is_empty() {
                    Some((normalized, original_domain))
                } else {
                    None
                }
            } else {
                None
            }
        }
        Err(_) => {
            // If parsing fails, try a basic domain extraction
            basic_domain_extraction(url_str)
        }
    }
}

/// Fallback domain extraction for URLs that the URL crate can't parse
fn basic_domain_extraction(url_str: &str) -> Option<(String, String)> {
    let lower = url_str.to_lowercase();

    // Strip common schemes
    let without_scheme = lower
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_start_matches("ftp://");

    // Extract everything before first slash or querystring
    let domain_part = without_scheme.split('/').next()?.split('?').next()?;

    // Original domain without normalization
    let original_domain = domain_part.to_string();

    // Remove www prefix
    let normalized = domain_part.trim_start_matches("www.").to_string();

    if !normalized.is_empty() && normalized.contains('.') {
        Some((normalized, original_domain))
    } else {
        None
    }
}
