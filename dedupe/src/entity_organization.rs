// src/entity_organization.rs
use crate::db::PgPool;
use crate::models::{MatchGroup, ClusterEntity, EntityFeature, HierarchicalMatchGroup};
use anyhow::{Result, Context};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use futures::{stream, StreamExt};
use uuid::Uuid;

/// Constants for performance tuning
const BATCH_SIZE: usize = 25;
const MAX_CONCURRENT_BATCHES: usize = 8;

/// Organizes flat record IDs into hierarchical entities with optimized batch processing
pub async fn organize_entities(pool: &PgPool, groups: &[MatchGroup]) -> Result<Vec<HierarchicalMatchGroup>> {
    let start_time = Instant::now();
    println!("🏗️ Organizing {} groups into hierarchical entities...", groups.len());

    // Filter out groups with only one record to avoid unnecessary processing
    let valid_groups: Vec<&MatchGroup> = groups.iter()
        .filter(|g| g.record_ids.len() > 1)
        .collect();
    
    if valid_groups.is_empty() {
        println!("   No valid groups to organize, skipping organization step");
        return Ok(Vec::new());
    }
    
    println!("   Processing {} valid groups in batches", valid_groups.len());
    
    // Create an Arc for sharing the pool across tasks
    let pool = Arc::new(pool.clone());
    
    // Use semaphore to limit concurrent operations
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_BATCHES));
    
    // Calculate number of batches
    let total_batches = (valid_groups.len() + BATCH_SIZE - 1) / BATCH_SIZE;
    println!("   Using {} batches with max {} concurrent workers", total_batches, MAX_CONCURRENT_BATCHES);
    
    // Process batches concurrently but with controlled parallelism
    let results = stream::iter(0..total_batches)
        .map(|batch_idx| {
            let start = batch_idx * BATCH_SIZE;
            let end = std::cmp::min(start + BATCH_SIZE, valid_groups.len());
            let batch_groups = &valid_groups[start..end];
            
            let sem_clone = Arc::clone(&semaphore);
            let pool_clone = Arc::clone(&pool);
            
            async move {
                // Acquire semaphore permit (will wait if max concurrency reached)
                let _permit = sem_clone.acquire().await.expect("Semaphore was closed");
                
                let batch_start = Instant::now();
                println!("   Starting batch {}/{} ({} groups)...", 
                         batch_idx + 1, total_batches, batch_groups.len());
                
                let result = process_batch(pool_clone.as_ref(), batch_groups).await;
                
                let elapsed = batch_start.elapsed();
                if let Ok(ref groups) = result {
                    println!("   Completed batch {}/{} in {:.2}s: {} entities created", 
                             batch_idx + 1, total_batches, elapsed.as_secs_f32(), groups.len());
                } else {
                    eprintln!("   Failed batch {}/{}: {:?}", batch_idx + 1, total_batches, result);
                }
                
                result
            }
        })
        .buffer_unordered(MAX_CONCURRENT_BATCHES) // Process up to MAX_CONCURRENT_BATCHES at once
        .collect::<Vec<Result<Vec<HierarchicalMatchGroup>>>>()
        .await;
    
    // Combine results from all batches
    let mut combined_results = Vec::new();
    let mut error_count = 0;
    
    for batch_result in results {
        match batch_result {
            Ok(batch_groups) => {
                combined_results.extend(batch_groups);
            },
            Err(e) => {
                eprintln!("Error processing batch: {}", e);
                error_count += 1;
            }
        }
    }
    
    let total_time = start_time.elapsed();
    println!("   Hierarchical organization complete in {:.2}s", total_time.as_secs_f32());
    println!("   Created {} hierarchical groups, {} batches had errors", 
             combined_results.len(), error_count);
    
    Ok(combined_results)
}

/// Process a batch of groups in parallel
async fn process_batch(pool: &PgPool, groups: &[&MatchGroup]) -> Result<Vec<HierarchicalMatchGroup>> {
    let client = pool.get().await.context("Failed to get DB connection for batch")?;
    let mut result = Vec::with_capacity(groups.len());
    
    // Prepare statements for better performance
    let services_stmt = client.prepare(
        "SELECT s.id, s.organization_id FROM service s WHERE s.organization_id = ANY($1)"
    ).await?;
    
    let locations_stmt = client.prepare(
        "SELECT l.id, l.organization_id FROM location l WHERE l.organization_id = ANY($1)"
    ).await?;
    
    let service_locations_stmt = client.prepare(
        "SELECT sal.location_id, sal.service_id 
         FROM service_at_location sal 
         WHERE sal.service_id = ANY($1)"
    ).await?;
    
    let addresses_stmt = client.prepare(
        "SELECT a.id, a.location_id FROM address a WHERE a.location_id = ANY($1)"
    ).await?;
    
    let phones_loc_stmt = client.prepare(
        "SELECT p.id, p.location_id FROM phone p WHERE p.location_id = ANY($1)"
    ).await?;
    
    let phones_svc_stmt = client.prepare(
        "SELECT p.id, p.service_id FROM phone p WHERE p.service_id = ANY($1)"
    ).await?;
    
    // Process each group in the batch
    for group in groups {
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
        
        // Skip if no valid IDs
        if ids.is_empty() {
            continue;
        }
            
        // Use optimized fetch function with prepared statements
        let related_entities = fetch_related_entities_optimized(
            &client, 
            &ids,
            &services_stmt,
            &locations_stmt,
            &service_locations_stmt,
            &addresses_stmt,
            &phones_loc_stmt,
            &phones_svc_stmt
        ).await?;
        
        // Create the hierarchical group
        let hierarchical_group = HierarchicalMatchGroup {
            base_group: (**group).clone(),
            entities: related_entities,
        };
        
        result.push(hierarchical_group);
    }
    
    Ok(result)
}

/// Optimized version of fetch_related_entities that reduces database queries
/// by using prepared statements and batch operations
async fn fetch_related_entities_optimized(
    client: &tokio_postgres::Client,
    ids: &[(String, String)],
    services_stmt: &tokio_postgres::Statement,
    locations_stmt: &tokio_postgres::Statement,
    service_locations_stmt: &tokio_postgres::Statement,
    addresses_stmt: &tokio_postgres::Statement,
    phones_loc_stmt: &tokio_postgres::Statement,
    phones_svc_stmt: &tokio_postgres::Statement,
) -> Result<Vec<ClusterEntity>> {
    let mut entities = Vec::new();
    
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
    
    // Pre-fetch all relationships at once to reduce round-trips
    // 1. Get all services for all organizations in a single query
    let mut org_to_services: HashMap<String, Vec<String>> = HashMap::new();
    if !org_ids.is_empty() {
        let rows = client.query(services_stmt, &[&org_ids]).await?;
        for row in rows {
            let service_id: String = row.get(0);
            let org_id: String = row.get(1);
            org_to_services.entry(org_id).or_insert_with(Vec::new).push(service_id);
        }
    }
    
    // 2. Get all locations for all organizations in a single query
    let mut org_to_locations: HashMap<String, Vec<String>> = HashMap::new();
    if !org_ids.is_empty() {
        let rows = client.query(locations_stmt, &[&org_ids]).await?;
        for row in rows {
            let location_id: String = row.get(0);
            let org_id: String = row.get(1);
            org_to_locations.entry(org_id).or_insert_with(Vec::new).push(location_id);
        }
    }
    
    // 3. Get all service_at_location for all services
    let mut service_to_locations: HashMap<String, Vec<String>> = HashMap::new();
    if !service_ids.is_empty() {
        let rows = client.query(service_locations_stmt, &[&service_ids]).await?;
        for row in rows {
            let location_id: String = row.get(0);
            let service_id: String = row.get(1);
            service_to_locations.entry(service_id).or_insert_with(Vec::new).push(location_id);
        }
    }
    
    // Collect all location IDs (from organizations, services, and direct)
    let mut all_location_ids = HashSet::new();
    for locations in org_to_locations.values() {
        all_location_ids.extend(locations.iter().cloned());
    }
    for locations in service_to_locations.values() {
        all_location_ids.extend(locations.iter().cloned());
    }
    all_location_ids.extend(location_ids.iter().cloned());
    
    let all_location_ids: Vec<String> = all_location_ids.into_iter().collect();
    
    // 4. Get all addresses for all locations
    let mut location_to_addresses: HashMap<String, Vec<String>> = HashMap::new();
    if !all_location_ids.is_empty() {
        let rows = client.query(addresses_stmt, &[&all_location_ids]).await?;
        for row in rows {
            let address_id: String = row.get(0);
            let location_id: String = row.get(1);
            location_to_addresses.entry(location_id).or_insert_with(Vec::new).push(address_id);
        }
    }
    
    // 5. Get all phones for all locations
    let mut location_to_phones: HashMap<String, Vec<String>> = HashMap::new();
    if !all_location_ids.is_empty() {
        let rows = client.query(phones_loc_stmt, &[&all_location_ids]).await?;
        for row in rows {
            let phone_id: String = row.get(0);
            let location_id: String = row.get(1);
            location_to_phones.entry(location_id).or_insert_with(Vec::new).push(phone_id);
        }
    }
    
    // 6. Get all phones for all services
    let mut service_to_phones: HashMap<String, Vec<String>> = HashMap::new();
    if !service_ids.is_empty() {
        let rows = client.query(phones_svc_stmt, &[&service_ids]).await?;
        for row in rows {
            let phone_id: String = row.get(0);
            let service_id: String = row.get(1);
            service_to_phones.entry(service_id).or_insert_with(Vec::new).push(phone_id);
        }
    }
    
    // Now that we have all the data, build the entity hierarchy
    
    // 1. Start with organizations
    if !org_ids.is_empty() {
        for org_id in &org_ids {
            let entity_id = Uuid::new_v4().to_string();
            
            let mut entity = ClusterEntity {
                id: Some(entity_id.clone()),
                is_primary: true,
                features: vec![
                    EntityFeature {
                        table_name: "organization".to_string(),
                        table_id: org_id.to_string(),
                        feature_type: Some("primary".to_string()),
                        weight: Some(1.0),
                    }
                ],
            };
            
            // Add services for this org
            if let Some(services) = org_to_services.get(org_id) {
                for service_id in services {
                    entity.features.push(EntityFeature {
                        table_name: "service".to_string(),
                        table_id: service_id.clone(),
                        feature_type: Some("service".to_string()),
                        weight: Some(0.9),
                    });
                    
                    processed_entities.insert(format!("service:{}", service_id), entity_id.clone());
                }
            }
            
            // Add locations for this org
            if let Some(locations) = org_to_locations.get(org_id) {
                for location_id in locations {
                    entity.features.push(EntityFeature {
                        table_name: "location".to_string(),
                        table_id: location_id.clone(),
                        feature_type: Some("location".to_string()),
                        weight: Some(0.8),
                    });
                    
                    processed_entities.insert(format!("location:{}", location_id), entity_id.clone());
                    
                    // Add addresses for this location
                    if let Some(addresses) = location_to_addresses.get(location_id) {
                        for address_id in addresses {
                            entity.features.push(EntityFeature {
                                table_name: "address".to_string(),
                                table_id: address_id.clone(),
                                feature_type: Some("address".to_string()),
                                weight: Some(0.7),
                            });
                        }
                    }
                    
                    // Add phones for this location
                    if let Some(phones) = location_to_phones.get(location_id) {
                        for phone_id in phones {
                            entity.features.push(EntityFeature {
                                table_name: "phone".to_string(),
                                table_id: phone_id.clone(),
                                feature_type: Some("contact".to_string()),
                                weight: Some(0.6),
                            });
                        }
                    }
                }
            }
            
            entities.push(entity);
            
            processed_entities.insert(format!("organization:{}", org_id), entity_id);
        }
    }
    
    // 2. Process services not already linked to an organization
    for service_id in &service_ids {
        let key = format!("service:{}", service_id);
        if processed_entities.contains_key(&key) {
            continue; // Already processed
        }
        
        let entity_id = Uuid::new_v4().to_string();
        
        let mut entity = ClusterEntity {
            id: Some(entity_id.clone()),
            is_primary: org_ids.is_empty(),
            features: vec![
                EntityFeature {
                    table_name: "service".to_string(),
                    table_id: service_id.to_string(),
                    feature_type: Some("primary".to_string()),
                    weight: Some(1.0),
                }
            ],
        };
        
        // Add locations for this service
        if let Some(locations) = service_to_locations.get(service_id) {
            for location_id in locations {
                entity.features.push(EntityFeature {
                    table_name: "location".to_string(),
                    table_id: location_id.clone(),
                    feature_type: Some("location".to_string()),
                    weight: Some(0.8),
                });
                
                processed_entities.insert(format!("location:{}", location_id), entity_id.clone());
                
                // Add addresses for this location
                if let Some(addresses) = location_to_addresses.get(location_id) {
                    for address_id in addresses {
                        entity.features.push(EntityFeature {
                            table_name: "address".to_string(),
                            table_id: address_id.clone(),
                            feature_type: Some("address".to_string()),
                            weight: Some(0.7),
                        });
                    }
                }
                
                // Add phones for this location
                if let Some(phones) = location_to_phones.get(location_id) {
                    for phone_id in phones {
                        entity.features.push(EntityFeature {
                            table_name: "phone".to_string(),
                            table_id: phone_id.clone(),
                            feature_type: Some("contact".to_string()),
                            weight: Some(0.6),
                        });
                    }
                }
            }
        }
        
        // Add phones directly linked to the service
        if let Some(phones) = service_to_phones.get(service_id) {
            for phone_id in phones {
                entity.features.push(EntityFeature {
                    table_name: "phone".to_string(),
                    table_id: phone_id.clone(),
                    feature_type: Some("contact".to_string()),
                    weight: Some(0.6),
                });
            }
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
        
        let entity_id = Uuid::new_v4().to_string();
        
        let mut entity = ClusterEntity {
            id: Some(entity_id.clone()),
            is_primary: org_ids.is_empty() && service_ids.is_empty(),
            features: vec![
                EntityFeature {
                    table_name: "location".to_string(),
                    table_id: location_id.to_string(),
                    feature_type: Some("primary".to_string()),
                    weight: Some(1.0),
                }
            ],
        };
        
        // Add addresses for this location
        if let Some(addresses) = location_to_addresses.get(location_id) {
            for address_id in addresses {
                entity.features.push(EntityFeature {
                    table_name: "address".to_string(),
                    table_id: address_id.clone(),
                    feature_type: Some("address".to_string()),
                    weight: Some(0.7),
                });
            }
        }
        
        // Add phones for this location
        if let Some(phones) = location_to_phones.get(location_id) {
            for phone_id in phones {
                entity.features.push(EntityFeature {
                    table_name: "phone".to_string(),
                    table_id: phone_id.clone(),
                    feature_type: Some("contact".to_string()),
                    weight: Some(0.6),
                });
            }
        }
        
        entities.push(entity);
        processed_entities.insert(key, entity_id);
    }
    
    Ok(entities)
}

// Remove the individual fetch functions as they're no longer needed
// They're replaced by the batch fetching in fetch_related_entities_optimized