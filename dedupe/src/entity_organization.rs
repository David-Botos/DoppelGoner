// src/entity_organization.rs
use crate::db::PgPool;
use crate::models::{MatchGroup, ClusterEntity, EntityFeature, HierarchicalMatchGroup};
use anyhow::{Result, Context};
use std::collections::HashMap;
use uuid::Uuid;

/// Organizes flat record IDs into hierarchical entities
pub async fn organize_entities(pool: &PgPool, groups: &[MatchGroup]) -> Result<Vec<HierarchicalMatchGroup>> {
    let mut result = Vec::with_capacity(groups.len());
    
    for group in groups {
        // Skip groups with only one record
        if group.record_ids.len() <= 1 {
            continue;
        }
        
        // Extract the raw IDs from the prefixed record_ids
        let ids: Vec<(String, String)> = group.record_ids.iter()
            .filter_map(|id| {
                let parts: Vec<&str> = id.split(':').collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();
            
        // Fetch all the related records for these IDs
        let related_entities = fetch_related_entities(pool, &ids).await?;
        
        // Create the hierarchical group
        let hierarchical_group = HierarchicalMatchGroup {
            base_group: group.clone(),
            entities: related_entities,
        };
        
        result.push(hierarchical_group);
    }
    
    Ok(result)
}

/// Fetches all related entity data for a collection of IDs
async fn fetch_related_entities(pool: &PgPool, ids: &[(String, String)]) -> Result<Vec<ClusterEntity>> {
    let mut entities = Vec::new();
    let client = pool.get().await.context("Failed to get DB connection")?;
    
    // Group IDs by entity type
    let mut org_ids = Vec::new();
    let mut service_ids = Vec::new();
    let mut location_ids = Vec::new();
    
    for (entity_type, id) in ids {
        match entity_type.as_str() {
            "organization" => org_ids.push(id.clone()),
            "service" => service_ids.push(id.clone()),
            "location" => location_ids.push(id.clone()),
            _ => continue,
        }
    }
    
    // Create a map to track which entities we've processed
    let mut processed_entities = HashMap::new();
    
    // 1. Start with organizations (usually the primary entities)
    if !org_ids.is_empty() {
        for org_id in &org_ids {
            // Generate entity ID upfront
            let entity_id = Uuid::new_v4().to_string();
            
            let mut entity = ClusterEntity {
                id: Some(entity_id.clone()),
                is_primary: true, // Organizations are primary entities
                features: vec![
                    EntityFeature {
                        table_name: "organization".to_string(),
                        table_id: org_id.to_string(),
                        feature_type: Some("primary".to_string()),
                        weight: Some(1.0),
                    }
                ],
            };
            
            // Fetch and add all related services
            let services = fetch_services_for_org(&client, org_id).await?;
            for service_id in services {
                entity.features.push(EntityFeature {
                    table_name: "service".to_string(),
                    table_id: service_id.clone(),
                    feature_type: Some("service".to_string()),
                    weight: Some(0.9),
                });
                
                // Mark this service as processed with the stored entity ID
                processed_entities.insert(format!("service:{}", service_id), entity_id.clone());
            }
            
            // Fetch and add locations
            let locations = fetch_locations_for_org(&client, org_id).await?;
            for location_id in locations {
                entity.features.push(EntityFeature {
                    table_name: "location".to_string(),
                    table_id: location_id.clone(),
                    feature_type: Some("location".to_string()),
                    weight: Some(0.8),
                });
                
                // Mark this location as processed with the stored entity ID
                processed_entities.insert(format!("location:{}", location_id), entity_id.clone());
                
                // Fetch and add addresses for this location
                let addresses = fetch_addresses_for_location(&client, &location_id).await?;
                for address_id in addresses {
                    entity.features.push(EntityFeature {
                        table_name: "address".to_string(),
                        table_id: address_id,
                        feature_type: Some("address".to_string()),
                        weight: Some(0.7),
                    });
                }
                
                // Fetch and add phones for this location
                let phones = fetch_phones_for_location(&client, &location_id).await?;
                for phone_id in phones {
                    entity.features.push(EntityFeature {
                        table_name: "phone".to_string(),
                        table_id: phone_id,
                        feature_type: Some("contact".to_string()),
                        weight: Some(0.6),
                    });
                }
            }
            
            // Add entity to collection
            entities.push(entity);
            
            // Record this org as processed
            processed_entities.insert(format!("organization:{}", org_id), entity_id);
        }
    }
    
    // 2. Process services not already linked to an organization
    for service_id in &service_ids {
        let key = format!("service:{}", service_id);
        if processed_entities.contains_key(&key) {
            continue; // Already processed
        }
        
        // Generate entity ID upfront
        let entity_id = Uuid::new_v4().to_string();
        
        let mut entity = ClusterEntity {
            id: Some(entity_id.clone()),
            is_primary: org_ids.is_empty(), // Services are primary only if no orgs exist
            features: vec![
                EntityFeature {
                    table_name: "service".to_string(),
                    table_id: service_id.to_string(),
                    feature_type: Some("primary".to_string()),
                    weight: Some(1.0),
                }
            ],
        };
        
        // Fetch service_at_location records
        let service_locations = fetch_locations_for_service(&client, service_id).await?;
        for location_id in service_locations {
            entity.features.push(EntityFeature {
                table_name: "location".to_string(),
                table_id: location_id.clone(),
                feature_type: Some("location".to_string()),
                weight: Some(0.8),
            });
            
            // Mark this location as processed with the stored entity ID
            processed_entities.insert(format!("location:{}", location_id), entity_id.clone());
            
            // Add addresses and phones as we did with orgs
            let addresses = fetch_addresses_for_location(&client, &location_id).await?;
            for address_id in addresses {
                entity.features.push(EntityFeature {
                    table_name: "address".to_string(),
                    table_id: address_id,
                    feature_type: Some("address".to_string()),
                    weight: Some(0.7),
                });
            }
            
            let phones = fetch_phones_for_location(&client, &location_id).await?;
            for phone_id in phones {
                entity.features.push(EntityFeature {
                    table_name: "phone".to_string(),
                    table_id: phone_id,
                    feature_type: Some("contact".to_string()),
                    weight: Some(0.6),
                });
            }
        }
        
        // Add phones directly linked to the service
        let service_phones = fetch_phones_for_service(&client, service_id).await?;
        for phone_id in service_phones {
            entity.features.push(EntityFeature {
                table_name: "phone".to_string(),
                table_id: phone_id,
                feature_type: Some("contact".to_string()),
                weight: Some(0.6),
            });
        }
        
        entities.push(entity);
        processed_entities.insert(key, entity_id);
    }
    
    // 3. Process remaining locations not already linked
    for location_id in &location_ids {
        let key = format!("location:{}", location_id);
        if processed_entities.contains_key(&key) {
            continue; // Already processed
        }
        
        // Generate entity ID upfront
        let entity_id = Uuid::new_v4().to_string();
        
        let mut entity = ClusterEntity {
            id: Some(entity_id.clone()),
            is_primary: org_ids.is_empty() && service_ids.is_empty(), // Primary only if no orgs/services
            features: vec![
                EntityFeature {
                    table_name: "location".to_string(),
                    table_id: location_id.to_string(),
                    feature_type: Some("primary".to_string()),
                    weight: Some(1.0),
                }
            ],
        };
        
        // Add addresses and phones
        let addresses = fetch_addresses_for_location(&client, location_id).await?;
        for address_id in addresses {
            entity.features.push(EntityFeature {
                table_name: "address".to_string(),
                table_id: address_id,
                feature_type: Some("address".to_string()),
                weight: Some(0.7),
            });
        }
        
        let phones = fetch_phones_for_location(&client, location_id).await?;
        for phone_id in phones {
            entity.features.push(EntityFeature {
                table_name: "phone".to_string(),
                table_id: phone_id,
                feature_type: Some("contact".to_string()),
                weight: Some(0.6),
            });
        }
        
        entities.push(entity);
        processed_entities.insert(key, entity_id);
    }
    
    Ok(entities)
}

// Helper functions to fetch related records
async fn fetch_services_for_org(client: &tokio_postgres::Client, org_id: &str) -> Result<Vec<String>> {
    let rows = client
        .query(
            "SELECT id FROM service WHERE organization_id = $1",
            &[&org_id],
        )
        .await?;
    
    Ok(rows.iter().map(|row| row.get::<_, String>(0)).collect())
}

async fn fetch_locations_for_org(client: &tokio_postgres::Client, org_id: &str) -> Result<Vec<String>> {
    let rows = client
        .query(
            "SELECT id FROM location WHERE organization_id = $1",
            &[&org_id],
        )
        .await?;
    
    Ok(rows.iter().map(|row| row.get::<_, String>(0)).collect())
}

async fn fetch_locations_for_service(client: &tokio_postgres::Client, service_id: &str) -> Result<Vec<String>> {
    let rows = client
        .query(
            "SELECT location_id FROM service_at_location WHERE service_id = $1",
            &[&service_id],
        )
        .await?;
    
    Ok(rows.iter().map(|row| row.get::<_, String>(0)).collect())
}

async fn fetch_addresses_for_location(client: &tokio_postgres::Client, location_id: &str) -> Result<Vec<String>> {
    let rows = client
        .query(
            "SELECT id FROM address WHERE location_id = $1",
            &[&location_id],
        )
        .await?;
    
    Ok(rows.iter().map(|row| row.get::<_, String>(0)).collect())
}

async fn fetch_phones_for_location(client: &tokio_postgres::Client, location_id: &str) -> Result<Vec<String>> {
    let rows = client
        .query(
            "SELECT id FROM phone WHERE location_id = $1",
            &[&location_id],
        )
        .await?;
    
    Ok(rows.iter().map(|row| row.get::<_, String>(0)).collect())
}

async fn fetch_phones_for_service(client: &tokio_postgres::Client, service_id: &str) -> Result<Vec<String>> {
    let rows = client
        .query(
            "SELECT id FROM phone WHERE service_id = $1",
            &[&service_id],
        )
        .await?;
    
    Ok(rows.iter().map(|row| row.get::<_, String>(0)).collect())
}