// src/matching/geospatial/service_utils.rs
//
// Utility functions for comparing service similarity

use anyhow::{Context, Result};
use log::{debug, trace, warn};
use std::collections::HashSet;
use tokio_postgres::Transaction;

use crate::models::EntityId;

/// Struct to hold service details for semantic comparison
pub struct ServiceDetails {
    pub service_id: String,
    pub name: String,
    pub description: Option<String>,
    pub short_description: Option<String>,
}

/// Similarity threshold for services
pub const SERVICE_SIMILARITY_THRESHOLD: f64 = 0.3; // Relatively low threshold for initial matching

/// Get all services for an entity
pub async fn get_services_for_entity(
    tx: &Transaction<'_>,
    entity_id: &EntityId,
) -> Result<Vec<ServiceDetails>> {
    let query = r#"
        SELECT 
            s.id as service_id,
            s.name,
            s.description,
            s.short_description
        FROM 
            entity_feature ef
        JOIN 
            service s ON ef.table_id = s.id
        WHERE 
            ef.entity_id = $1
            AND ef.table_name = 'service'
    "#;

    let rows = tx.query(query, &[&entity_id.0]).await.context(format!(
        "Failed to query services for entity {}",
        entity_id.0
    ))?;

    let mut services = Vec::with_capacity(rows.len());
    for row in rows {
        let service_id: String = row.get("service_id");
        let name: String = row.get("name");
        let description: Option<String> = row.get("description");
        let short_description: Option<String> = row.get("short_description");

        services.push(ServiceDetails {
            service_id,
            name,
            description,
            short_description,
        });
    }

    trace!(
        "Found {} services for entity {}",
        services.len(),
        entity_id.0
    );
    Ok(services)
}

/// Calculate semantic similarity between services of two entities
pub async fn compare_entity_services(
    tx: &Transaction<'_>,
    entity1: &EntityId,
    entity2: &EntityId,
) -> Result<f64> {
    let services1 = get_services_for_entity(tx, entity1).await?;
    let services2 = get_services_for_entity(tx, entity2).await?;

    if services1.is_empty() && services2.is_empty() {
        debug!("Both entities have no services, defaulting to neutral similarity");
        let similarity = 0.5; // Default to neutral if both have no services

        // Log this comparison
        log_service_similarity(
            tx,
            similarity,
            similarity < SERVICE_SIMILARITY_THRESHOLD,
            &entity1.0,
            &entity2.0,
        )
        .await?;

        return Ok(similarity);
    }

    if services1.is_empty() || services2.is_empty() {
        debug!("One entity has no services, defaulting to low similarity");
        let similarity = 0.2; // Low similarity if one has services and the other doesn't

        // Log this comparison
        log_service_similarity(
            tx,
            similarity,
            similarity < SERVICE_SIMILARITY_THRESHOLD,
            &entity1.0,
            &entity2.0,
        )
        .await?;

        return Ok(similarity);
    }

    let similarity = calculate_service_similarity(&services1, &services2);
    trace!(
        "Calculated similarity {:.4} between entity {} ({} services) and entity {} ({} services)",
        similarity,
        entity1.0,
        services1.len(),
        entity2.0,
        services2.len()
    );

    // Log the comparison result
    log_service_similarity(
        tx,
        similarity,
        similarity < SERVICE_SIMILARITY_THRESHOLD,
        &entity1.0,
        &entity2.0,
    )
    .await?;

    Ok(similarity)
}
/// Calculate similarity between two sets of services
pub fn calculate_service_similarity(
    services1: &[ServiceDetails],
    services2: &[ServiceDetails],
) -> f64 {
    // Collect all text from services
    let text1 = collect_service_text(services1);
    let text2 = collect_service_text(services2);

    // Calculate token-based similarity (using Jaccard similarity)
    calculate_token_similarity(&text1, &text2)
}

/// Collect all text from a set of services
fn collect_service_text(services: &[ServiceDetails]) -> String {
    let mut text = String::new();

    for service in services {
        text.push_str(&service.name);
        text.push_str(" ");

        if let Some(desc) = &service.short_description {
            text.push_str(desc);
            text.push_str(" ");
        }

        if let Some(desc) = &service.description {
            text.push_str(desc);
            text.push_str(" ");
        }
    }

    text
}

/// Calculate token-based Jaccard similarity between two texts
fn calculate_token_similarity(text1: &str, text2: &str) -> f64 {
    // Convert to lowercase and tokenize
    let tokens1: HashSet<String> = text1
        .to_lowercase()
        .split_whitespace()
        .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|s| !s.is_empty() && s.len() > 2) // Filter out short tokens and empty strings
        .collect();

    let tokens2: HashSet<String> = text2
        .to_lowercase()
        .split_whitespace()
        .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|s| !s.is_empty() && s.len() > 2) // Filter out short tokens and empty strings
        .collect();

    if tokens1.is_empty() || tokens2.is_empty() {
        return 0.5; // Default to moderate similarity if no meaningful tokens
    }

    // Calculate Jaccard similarity
    let intersection_size = tokens1.intersection(&tokens2).count() as f64;
    let union_size = tokens1.union(&tokens2).count() as f64;

    intersection_size / union_size
}

/// Log service similarity scores and rejections to the database for statistics
pub async fn log_service_similarity(
    tx: &Transaction<'_>,
    similarity: f64,
    is_rejected: bool,
    entity1_id: &str,
    entity2_id: &str,
) -> Result<()> {
    // Create a metadata table if it doesn't exist
    let create_table = "
        CREATE TABLE IF NOT EXISTS matching_metadata (
            id SERIAL PRIMARY KEY,
            type TEXT NOT NULL,
            entity1_id TEXT,
            entity2_id TEXT,
            value DOUBLE PRECISION,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    ";

    match tx.execute(create_table, &[]).await {
        Ok(_) => {}
        Err(e) => {
            warn!("Failed to create matching_metadata table: {}", e);
            return Ok(()); // Continue execution even if logging fails
        }
    }

    // Insert the similarity score
    let insert_stmt = "
        INSERT INTO matching_metadata (type, entity1_id, entity2_id, value, created_at)
        VALUES ($1, $2, $3, $4, NOW())
    ";

    let record_type = if is_rejected {
        "service_similarity_reject"
    } else {
        "service_similarity"
    };

    match tx
        .execute(
            insert_stmt,
            &[&record_type, &entity1_id, &entity2_id, &similarity],
        )
        .await
    {
        Ok(_) => {
            trace!(
                "Logged service similarity score {:.4} between entities {} and {} (rejected: {})",
                similarity, entity1_id, entity2_id, is_rejected
            );
            Ok(())
        }
        Err(e) => {
            warn!("Failed to log service similarity: {}", e);
            Ok(()) // Continue execution even if logging fails
        }
    }
}

/// For performance reasons, we might want to test a sample of entities in a group
/// Returns a list of entity IDs to test (either all if small, or a sample if large)
pub fn sample_group_entities(entity_ids: &[EntityId], max_sample: usize) -> Vec<EntityId> {
    if entity_ids.len() <= max_sample {
        return entity_ids.to_vec();
    }

    // Simple sampling - take first, last, and some from middle
    let mut sample = Vec::with_capacity(max_sample);

    // Always include first entity
    if !entity_ids.is_empty() {
        sample.push(entity_ids[0].clone());
    }

    // Sample middle entities
    let middle_samples = max_sample - 2; // Minus first and last
    if middle_samples > 0 && entity_ids.len() > 2 {
        let step = (entity_ids.len() - 2) as f64 / (middle_samples + 1) as f64;

        for i in 1..=middle_samples {
            let index = 1 + (i as f64 * step).round() as usize;
            if index < entity_ids.len() - 1 {
                sample.push(entity_ids[index].clone());
            }
        }
    }

    // Always include last entity
    if entity_ids.len() > 1 {
        sample.push(entity_ids[entity_ids.len() - 1].clone());
    }

    sample
}
