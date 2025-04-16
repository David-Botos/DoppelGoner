// src/matching/email.rs
use anyhow::{Context, Result};
use chrono::Utc;
use log::{debug, info};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use crate::db::PgPool;
use crate::models::{
    EmailMatchValue, EntityGroup, EntityGroupId, EntityId, MatchMethodType, MatchValues,
};
use crate::results::{EmailMatchResult, MatchMethodStats};

pub async fn find_matches(pool: &PgPool) -> Result<EmailMatchResult> {
    info!("Starting email matching process...");
    let start_time = std::time::Instant::now();

    let mut conn = pool
        .get()
        .await
        .context("Failed to get DB connection for email matching")?;

    // First, get all entities that are already part of an email match group
    // These entities have already been checked and don't need to be processed again
    debug!("Finding entities already processed by email matching");
    let processed_query = "
        SELECT DISTINCT ge.entity_id
        FROM group_entity ge
        JOIN group_method gm ON ge.entity_group_id = gm.entity_group_id
        WHERE gm.method_type = 'email'
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
        "Found {} entities already processed by email matching",
        processed_entities.len()
    );

    // Get existing email groups and their normalized emails
    debug!("Finding existing email groups");
    let existing_groups_query = "
        SELECT 
            eg.id AS group_id,
            gm.id AS method_id,
            gm.match_values
        FROM 
            entity_group eg
            JOIN group_method gm ON eg.id = gm.entity_group_id
        WHERE 
            gm.method_type = 'email'
    ";

    let existing_groups_rows = conn
        .query(existing_groups_query, &[])
        .await
        .context("Failed to query existing groups")?;

    // Map of normalized email -> (group_id, method_id)
    let mut existing_email_groups: HashMap<String, (String, String)> = HashMap::new();

    for row in &existing_groups_rows {
        let group_id: String = row.get("group_id");
        let method_id: String = row.get("method_id");
        let match_values_json: serde_json::Value = row.get("match_values");

        // Extract emails from the match_values JSON
        if let Ok(match_values) = serde_json::from_value::<MatchValues>(match_values_json.clone()) {
            if let MatchValues::Email(email_values) = match_values {
                for email_value in &email_values {
                    // Store mapping from normalized email to group/method ids
                    existing_email_groups.insert(
                        email_value.normalized.clone(),
                        (group_id.clone(), method_id.clone()),
                    );
                }
            }
        }
    }

    info!(
        "Found {} unique normalized emails in existing groups",
        existing_email_groups.len()
    );

    // Modified query to get all email data in a single query
    // Filter out already processed entities
    let email_query = "
    SELECT 'organization' as source, e.id as entity_id, o.email 
    FROM entity e 
    JOIN organization o ON e.id = o.id 
    WHERE o.email IS NOT NULL AND o.email != ''
    AND e.id NOT IN (
        SELECT ge.entity_id 
        FROM group_entity ge
        JOIN group_method gm ON ge.entity_group_id = gm.entity_group_id
        WHERE gm.method_type = 'email'
    )
    
    UNION ALL
    
    SELECT 'service' as source, e.id as entity_id, s.email 
    FROM entity e 
    JOIN entity_feature ef ON e.id = ef.entity_id 
    JOIN service s ON ef.table_id = s.id 
    WHERE ef.table_name = 'service' 
    AND s.email IS NOT NULL AND s.email != ''
    AND e.id NOT IN (
        SELECT ge.entity_id 
        FROM group_entity ge
        JOIN group_method gm ON ge.entity_group_id = gm.entity_group_id
        WHERE gm.method_type = 'email'
    )";

    debug!("Executing combined email query");
    let email_rows = conn
        .query(email_query, &[])
        .await
        .context("Failed to query entities with emails")?;

    info!("Found {} total email records to process", email_rows.len());

    // Map to store normalized emails -> list of entities with matching values
    // Using HashMap<normalized_email, HashMap<entity_id, original_email>> to avoid duplicates
    let mut email_map: HashMap<String, HashMap<EntityId, String>> = HashMap::new();
    let mut processed_count = 0;

    // Process all emails in a single pass
    for row in &email_rows {
        let entity_id: String = row.get("entity_id");
        let entity_id = EntityId(entity_id);

        // Skip if this entity has already been processed
        if processed_entities.contains(&entity_id) {
            continue;
        }

        let email: String = row.get("email");
        let normalized = normalize_email(&email);

        if !normalized.is_empty() {
            // Get or create the inner entity map for this normalized email
            let entity_map = email_map.entry(normalized).or_default();

            // Only add if this entity isn't already in the map for this email
            entity_map.entry(entity_id).or_insert(email);
        }

        processed_count += 1;
        if processed_count % 1000 == 0 {
            debug!("Processed {} email records so far", processed_count);
        }
    }

    let unique_emails = email_map.len();
    info!(
        "Found {} unique normalized emails from unprocessed records",
        unique_emails
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

    // Process emails
    for (normalized_email, entity_map) in email_map {
        // Store value for logging
        let entity_map_length = entity_map.len();

        // Check if this email already has a group
        if let Some((group_id, method_id)) = existing_email_groups.get(&normalized_email) {
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

            // Extract the email values
            let mut email_values = if let MatchValues::Email(values) = current_match_values {
                values
            } else {
                Vec::new()
            };

            // Track which entities we've actually added
            let mut entities_added = 0;

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
                let domain = extract_domain(&original);
                email_values.push(EmailMatchValue {
                    original,
                    normalized: normalized_email.clone(),
                    domain,
                    entity_id,
                });

                entities_added += 1;
            }

            // Update the match values
            let updated_match_values = MatchValues::Email(email_values);
            let updated_json = serde_json::to_value(updated_match_values)
                .context("Failed to serialize updated match values")?;

            tx.execute(&update_method_stmt, &[&updated_json, &method_id])
                .await
                .context("Failed to update match values")?;

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

            // Email matches have high confidence
            confidence_scores.push(1.0);

            info!(
                "Added {} entities to existing email group for {}",
                entities_added, normalized_email
            );
        } else if entity_map.len() >= 2 {
            // Create a new group for this email (only if multiple entities)

            // Convert the entity map to a vector of (EntityId, String) pairs
            let entities: Vec<(EntityId, String)> = entity_map.into_iter().collect();
            let entity_count = entities.len() as i32;

            // Create a new entity group
            let group_id = EntityGroupId(Uuid::new_v4().to_string());

            let group = EntityGroup {
                id: group_id.clone(),
                name: Some(format!("Email match on {}", normalized_email)),
                group_cluster_id: None,
                created_at: now,
                updated_at: now,
                confidence_score: 1.0,
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
            for (entity_id, original) in &entities {
                // Add to group_entity
                tx.execute(
                    &entity_stmt,
                    &[&Uuid::new_v4().to_string(), &group_id.0, &entity_id.0, &now],
                )
                .await
                .context("Failed to insert group entity")?;

                // Create match value for this entity
                let domain = extract_domain(original);
                match_values.push(EmailMatchValue {
                    original: original.clone(),
                    normalized: normalized_email.clone(),
                    domain,
                    entity_id: entity_id.clone(),
                });
            }

            // Serialize match values to JSON
            let method_values = MatchValues::Email(match_values);
            let match_values_json =
                serde_json::to_value(&method_values).context("Failed to serialize match values")?;

            // Insert group method
            tx.execute(
                &method_stmt,
                &[
                    &Uuid::new_v4().to_string(),
                    &group_id.0,
                    &MatchMethodType::Email.as_str(),
                    &format!("Matched on email: {}", normalized_email),
                    &match_values_json,
                    &1.0f64,
                    &now,
                ],
            )
            .await
            .context("Failed to insert group method")?;

            total_groups_created += 1;
            total_entities_matched += entity_count as usize;
            total_entities_added += entity_count as usize;

            // Email matches have high confidence
            confidence_scores.push(1.0);
            group_sizes.push(entity_count as f32);

            info!(
                "Created new email group for {} with {} entities",
                normalized_email, entity_count
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
        (confidence_scores.iter().sum::<f32>() / confidence_scores.len() as f32).into()
    } else {
        0.0
    };

    // Create method stats
    let method_stats = MatchMethodStats {
        method_type: MatchMethodType::Email,
        groups_created: total_groups_created,
        entities_matched: total_entities_matched,
        avg_confidence,
        avg_group_size,
    };

    let elapsed = start_time.elapsed();
    info!(
        "Email matching complete: created {} new entity groups and added {} entities to existing groups in {:.2?}",
        total_groups_created, total_entities_added, elapsed
    );

    Ok(EmailMatchResult {
        groups_created: total_groups_created,
        stats: method_stats,
    })
}
/// Normalize an email address by:
/// - Converting to lowercase
/// - Trimming whitespace
/// - Removing dots before @ in Gmail addresses
/// - Removing plus addressing (everything between + and @ is removed)
fn normalize_email(email: &str) -> String {
    // Basic normalization
    let email = email.trim().to_lowercase();

    // If it doesn't contain an @ symbol, just return the trimmed lowercase version
    if !email.contains('@') {
        return email;
    }

    // Split the email into local part and domain
    let parts: Vec<&str> = email.split('@').collect();
    if parts.len() != 2 {
        return email; // Return original if multiple @ symbols
    }

    let (local_part, domain) = (parts[0], parts[1]);

    // Normalize the local part based on domain-specific rules
    let normalized_local = if domain == "gmail.com" || domain == "googlemail.com" {
        // For Gmail: remove dots and anything after a plus sign
        let without_dots = local_part.replace('.', "");
        match without_dots.split('+').next() {
            Some(username) => username.to_string(),
            None => local_part.to_string(),
        }
    } else {
        // For other domains: just remove anything after a plus sign
        match local_part.split('+').next() {
            Some(username) => username.to_string(),
            None => local_part.to_string(),
        }
    };

    // Normalize domain
    let normalized_domain = match domain {
        "googlemail.com" => "gmail.com",
        "hotmail.com" | "live.com" | "outlook.com" => "outlook.com",
        "ymail.com" | "rocketmail.com" => "yahoo.com",
        _ => domain,
    };

    format!("{}@{}", normalized_local, normalized_domain)
}

/// Extract the domain part from an email address
fn extract_domain(email: &str) -> String {
    if let Some(index) = email.find('@') {
        if index < email.len() - 1 {
            return email[index + 1..].to_lowercase();
        }
    }

    // Default fallback if no @ or @ is the last character
    String::new()
}
