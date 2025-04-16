// src/entity_organization.rs
use anyhow::{Context, Result};
use chrono::Utc;
use log::{info, warn};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use crate::db::PgPool;
use crate::models::{Entity, EntityFeature, EntityId, OrganizationId};

/// Extracts entities from the organization table
/// Creates an entity record for each organization with its metadata
/// Only creates entities that don't already exist
///
/// This involves:
/// 1. Querying the entity table to get existing entities
/// 2. Querying the organization table to get all organizations
/// 3. Creating an entity record for each organization that doesn't already have one
/// 4. Setting entity.organization_id to the original organization.id
/// 5. Populating other entity fields (name, source info, etc.)
/// 6. Batch inserting the new entity records into the entity table
pub async fn extract_entities(pool: &PgPool) -> Result<Vec<Entity>> {
    info!("Extracting entities from organizations...");

    let conn = pool.get().await.context("Failed to get DB connection")?;

    // First, get all existing entities to avoid duplicates
    let existing_rows = conn
        .query("SELECT id, organization_id FROM entity", &[])
        .await
        .context("Failed to query existing entities")?;

    // Create a set of organization_ids that already have entities
    let mut existing_org_ids = HashSet::new();
    let mut existing_entities = Vec::with_capacity(existing_rows.len());

    for row in &existing_rows {
        let entity_id: String = row.get("id");
        let org_id: String = row.get("organization_id");

        existing_org_ids.insert(org_id.clone());
        existing_entities.push(Entity {
            id: EntityId(entity_id),
            organization_id: OrganizationId(org_id),
            name: None, // We don't need to populate all fields for existing entities
            created_at: Utc::now().naive_utc(), // Placeholder
            updated_at: Utc::now().naive_utc(), // Placeholder
            source_system: None,
            source_id: None,
        });
    }

    info!("Found {} existing entities", existing_org_ids.len());

    // Query the organization table to get all organizations
    let org_rows = conn
        .query("SELECT id, name FROM organization", &[])
        .await
        .context("Failed to query organization table")?;

    info!("Found {} total organizations", org_rows.len());

    let now = Utc::now().naive_utc();
    let mut new_entities = Vec::new();

    // Create an entity for each organization that doesn't already have one
    for (i, row) in org_rows.iter().enumerate() {
        let org_id: String = row.get("id");

        // Skip if this organization already has an entity
        if existing_org_ids.contains(&org_id) {
            continue;
        }

        let name: Option<String> = row.try_get("name").ok();
        let source_system = Some("Snowflake - feature not fully implemented in pipeline".into());
        let source_id = Some("Feature not implemented in pipeline".into());

        let entity = Entity {
            id: EntityId(Uuid::new_v4().to_string()),
            organization_id: OrganizationId(org_id),
            name,
            created_at: now,
            updated_at: now,
            source_system,
            source_id,
        };

        new_entities.push(entity);

        // Log progress at regular intervals
        if (i + 1) % 1000 == 0 || i + 1 == org_rows.len() {
            info!(
                "Processed {}/{} organizations ({:.1}%)",
                i + 1,
                org_rows.len(),
                (i + 1) as f32 / org_rows.len() as f32 * 100.0
            );
        }
    }

    info!("Found {} new entities to create", new_entities.len());

    // If there are no new entities, return the existing ones
    if new_entities.is_empty() {
        info!("No new entities to insert");
        return Ok(existing_entities);
    }

    // Insert the new entities into the database in batches
    let batch_size = 100;
    let total_batches = (new_entities.len() + batch_size - 1) / batch_size;
    info!("Starting entity insertion in {} batches", total_batches);
    let mut inserted = 0;

    for (batch_idx, chunk) in new_entities.chunks(batch_size).enumerate() {
        let mut batch_query = String::from(
            "INSERT INTO entity (id, organization_id, name, created_at, updated_at, source_system, source_id) VALUES ",
        );

        for (i, entity) in chunk.iter().enumerate() {
            if i > 0 {
                batch_query.push_str(", ");
            }
            batch_query.push_str(&format!(
                "('{}', '{}', {}, '{}', '{}', {}, {})",
                entity.id.0,
                entity.organization_id.0,
                match &entity.name {
                    Some(name) => format!("'{}'", name.replace('\'', "''")),
                    None => "NULL".to_string(),
                },
                entity.created_at,
                entity.updated_at,
                match &entity.source_system {
                    Some(src) => format!("'{}'", src.replace('\'', "''")),
                    None => "NULL".to_string(),
                },
                match &entity.source_id {
                    Some(id) => format!("'{}'", id.replace('\'', "''")),
                    None => "NULL".to_string(),
                }
            ));
        }

        // Execute the batch insert
        let result = conn.execute(&batch_query, &[]).await;
        match result {
            Ok(count) => {
                inserted += count as usize;
            }
            Err(e) => {
                warn!("Error inserting batch of entities: {}", e);
                // Continue with other batches even if one fails
            }
        }

        // Log batch progress
        if (batch_idx + 1) % 10 == 0 || batch_idx + 1 == total_batches {
            info!(
                "Processed batch {}/{} ({:.1}%)",
                batch_idx + 1,
                total_batches,
                (batch_idx + 1) as f32 / total_batches as f32 * 100.0
            );
        }
    }

    info!("Inserted {} new entities into the database", inserted);

    // Combine existing and new entities for return
    let mut all_entities = existing_entities;
    all_entities.extend(new_entities);

    Ok(all_entities)
}

/// Links entities to their features
/// Finds all records related to each entity and creates entity_feature records
/// Only creates features that don't already exist
///
/// This involves:
/// 1. Querying existing entity_feature records to avoid duplicates
/// 2. For each entity, find all references to its organization_id in other tables:
///    - services provided by the organization
///    - phones associated with the organization
///    - locations operated by the organization
///    - contacts linked to the organization
/// 3. For each reference found, create an entity_feature record if it doesn't exist with:
///    - entity_id pointing to the entity
///    - table_name indicating the source table (e.g., "service", "phone")
///    - table_id containing the ID of the referenced record
/// 4. Batch insert the new entity_feature records
pub async fn link_entity_features(pool: &PgPool, entities: &[Entity]) -> Result<usize> {
    info!("Linking entities to their features...");

    let conn = pool.get().await.context("Failed to get DB connection")?;

    // Create a map of organization_id to entity_id for faster lookups
    let mut org_to_entity = HashMap::new();
    for entity in entities {
        org_to_entity.insert(entity.organization_id.0.clone(), entity.id.0.clone());
    }

    // First, get all existing entity_feature records to avoid duplicates
    let existing_rows = conn
        .query(
            "SELECT entity_id, table_name, table_id FROM entity_feature",
            &[],
        )
        .await
        .context("Failed to query existing entity_features")?;

    // Create a set of (entity_id, table_name, table_id) tuples that already exist
    let mut existing_features = HashSet::new();

    for row in &existing_rows {
        let entity_id: String = row.get("entity_id");
        let table_name: String = row.get("table_name");
        let table_id: String = row.get("table_id");

        existing_features.insert((entity_id, table_name, table_id));
    }

    info!(
        "Found {} existing entity_feature records",
        existing_features.len()
    );

    let mut new_features = Vec::new();
    let now = Utc::now().naive_utc();

    // 1. Link services
    info!("Linking services to entities...");
    let service_rows = conn
        .query(
            "SELECT id, organization_id FROM service WHERE organization_id IS NOT NULL",
            &[],
        )
        .await
        .context("Failed to query services")?;

    for row in &service_rows {
        let service_id: String = row.get("id");
        let org_id: String = row.get("organization_id");

        if let Some(entity_id) = org_to_entity.get(&org_id) {
            let table_name = "service".to_string();

            // Skip if this feature already exists
            if existing_features.contains(&(
                entity_id.clone(),
                table_name.clone(),
                service_id.clone(),
            )) {
                continue;
            }

            let feature = EntityFeature {
                id: Uuid::new_v4().to_string(),
                entity_id: EntityId(entity_id.clone()),
                table_name,
                table_id: service_id,
                created_at: now,
            };
            new_features.push(feature);
        }
    }

    // 2. Link phones
    info!("Linking phones to entities...");
    let phone_rows = conn
        .query(
            "SELECT id, organization_id FROM phone WHERE organization_id IS NOT NULL",
            &[],
        )
        .await
        .context("Failed to query phones")?;

    for row in &phone_rows {
        let phone_id: String = row.get("id");
        let org_id: String = row.get("organization_id");

        if let Some(entity_id) = org_to_entity.get(&org_id) {
            let table_name = "phone".to_string();

            // Skip if this feature already exists
            if existing_features.contains(&(
                entity_id.clone(),
                table_name.clone(),
                phone_id.clone(),
            )) {
                continue;
            }

            let feature = EntityFeature {
                id: Uuid::new_v4().to_string(),
                entity_id: EntityId(entity_id.clone()),
                table_name,
                table_id: phone_id,
                created_at: now,
            };
            new_features.push(feature);
        }
    }

    // 3. Link locations
    info!("Linking locations to entities...");
    let location_rows = conn
        .query(
            "SELECT id, organization_id FROM location WHERE organization_id IS NOT NULL",
            &[],
        )
        .await
        .context("Failed to query locations")?;

    for row in &location_rows {
        let location_id: String = row.get("id");
        let org_id: String = row.get("organization_id");

        if let Some(entity_id) = org_to_entity.get(&org_id) {
            let table_name = "location".to_string();

            // Skip if this feature already exists
            if existing_features.contains(&(
                entity_id.clone(),
                table_name.clone(),
                location_id.clone(),
            )) {
                continue;
            }

            let feature = EntityFeature {
                id: Uuid::new_v4().to_string(),
                entity_id: EntityId(entity_id.clone()),
                table_name,
                table_id: location_id,
                created_at: now,
            };
            new_features.push(feature);
        }
    }

    // 4. Link contacts
    info!("Linking contacts to entities...");
    let contact_rows = conn
        .query(
            "SELECT id, organization_id FROM contact WHERE organization_id IS NOT NULL",
            &[],
        )
        .await
        .context("Failed to query contacts")?;

    for row in &contact_rows {
        let contact_id: String = row.get("id");
        let org_id: String = row.get("organization_id");

        if let Some(entity_id) = org_to_entity.get(&org_id) {
            let table_name = "contact".to_string();

            // Skip if this feature already exists
            if existing_features.contains(&(
                entity_id.clone(),
                table_name.clone(),
                contact_id.clone(),
            )) {
                continue;
            }

            let feature = EntityFeature {
                id: Uuid::new_v4().to_string(),
                entity_id: EntityId(entity_id.clone()),
                table_name,
                table_id: contact_id,
                created_at: now,
            };
            new_features.push(feature);
        }
    }

    info!("Found {} new features to link", new_features.len());

    // If there are no new features, return 0
    if new_features.is_empty() {
        info!("No new features to insert");
        return Ok(0);
    }

    // Insert the new features into the database in batches
    let batch_size = 100;
    let total_batches = (new_features.len() + batch_size - 1) / batch_size;

    info!(
        "Inserting {} new features in {} batches...",
        new_features.len(),
        total_batches
    );

    let mut inserted = 0;

    for (batch_idx, chunk) in new_features.chunks(batch_size).enumerate() {
        let mut batch_query = String::from(
            "INSERT INTO entity_feature (id, entity_id, table_name, table_id, created_at) VALUES ",
        );

        for (i, feature) in chunk.iter().enumerate() {
            if i > 0 {
                batch_query.push_str(", ");
            }
            batch_query.push_str(&format!(
                "('{}', '{}', '{}', '{}', '{}')",
                feature.id,
                feature.entity_id.0,
                feature.table_name,
                feature.table_id,
                feature.created_at
            ));
        }

        // Execute the batch insert
        let result = conn.execute(&batch_query, &[]).await;
        match result {
            Ok(count) => {
                inserted += count as usize;
            }
            Err(e) => {
                warn!("Error inserting batch of features: {}", e);
                // Continue with other batches even if one fails
            }
        }

        // Log batch progress
        if (batch_idx + 1) % 10 == 0 || batch_idx + 1 == total_batches {
            info!(
                "Processed batch {}/{} ({:.1}%)",
                batch_idx + 1,
                total_batches,
                (batch_idx + 1) as f32 / total_batches as f32 * 100.0
            );
        }
    }

    info!(
        "Inserted {} new entity features into the database",
        inserted
    );

    // Return the total number of features (existing + new)
    Ok(existing_features.len() + inserted)
}
