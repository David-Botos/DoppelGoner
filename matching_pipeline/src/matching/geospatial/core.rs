// src/matching/geospatial/core.rs
//
// Core matching functionality for geospatial entities

use anyhow::{Context, Result};
use chrono::NaiveDateTime;
use log::{debug, error, info, trace, warn};
use std::collections::{HashMap, HashSet};
use tokio_postgres::Transaction;
use uuid::Uuid;

use super::config::THRESHOLDS;
use super::database::DbStatements;
use super::postgis;
use super::utils::{
    calculate_centroid, calculate_centroid_from_full, calculate_haversine_distance,
    distance_to_centroid,
};
use crate::models::{
    EntityGroupId, EntityId, GeospatialMatchValue, GroupResults, MatchMethodType, MatchResult,
    MatchValues,
};

/// Try to match new entities to existing groups
pub async fn match_to_existing_groups(
    tx: &Transaction<'_>,
    db_stmts: &DbStatements,
    locations: &[(EntityId, f64, f64)],
    group_results: &GroupResults,
    has_postgis: bool,
    now: &NaiveDateTime,
) -> Result<MatchResult> {
    info!(
        "Starting match_to_existing_groups with {} locations to process",
        locations.len()
    );
    debug!("PostGIS available: {}", has_postgis);

    // Track which entities have been processed in this run
    let mut newly_processed = HashSet::new();
    let mut total_entities_added = 0;
    let mut skipped_entities = 0;

    // Set maximum distance to consider (largest threshold)
    let max_distance = THRESHOLDS.iter().map(|(d, _)| *d).fold(0.0, f64::max);
    debug!("Maximum distance threshold: {}m", max_distance);

    // Track progress
    let total_locations = locations.len();

    // For each new location
    for (idx, (new_entity_id, new_lat, new_lon)) in locations.iter().enumerate() {
        // Log progress every 10 entities or at the beginning/end
        if idx % 10 == 0 || idx == total_locations - 1 {
            info!(
                "Processing location {}/{} ({:.1}%)",
                idx + 1,
                total_locations,
                (idx + 1) as f64 / total_locations as f64 * 100.0
            );
        }

        trace!(
            "Processing entity {} at coordinates: lat={}, lon={}",
            new_entity_id.0, new_lat, new_lon
        );

        if newly_processed.contains(new_entity_id) {
            debug!(
                "Entity {} already processed in this run, skipping",
                new_entity_id.0
            );
            continue; // Skip if already processed in this run
        }

        // Check if this entity is already in a geospatial group (double-check)
        let check_result = tx
            .query_one(&db_stmts.consistency_check_stmt, &[&new_entity_id.0])
            .await
            .context("Failed to perform consistency check")?;

        let count: i64 = check_result.get(0);
        if count > 0 {
            // Entity is already in a geospatial group, skip to avoid duplication
            warn!(
                "Entity {} already exists in a geospatial group, skipping",
                new_entity_id.0
            );
            skipped_entities += 1;
            newly_processed.insert(new_entity_id.clone());
            continue;
        }

        // For PostGIS-enabled databases, use spatial filtering to find nearby groups
        let candidate_group_ids = if has_postgis {
            // Use PostGIS to find nearby groups efficiently
            debug!(
                "Using PostGIS to find nearby groups for entity {}",
                new_entity_id.0
            );
            postgis::find_nearby_groups(tx, *new_lat, *new_lon, max_distance).await?
        } else {
            // Fallback to checking all groups
            debug!(
                "PostGIS not available, checking all {} existing groups for entity {}",
                group_results.groups.len(),
                new_entity_id.0
            );
            group_results.groups.keys().cloned().collect::<Vec<_>>()
        };

        // For each candidate group
        let mut matched_group = None;

        info!(
            "Checking {} candidate groups for entity {}",
            candidate_group_ids.len(),
            new_entity_id.0
        );

        for group_id in &candidate_group_ids {
            if !group_results.groups.contains_key(group_id) {
                trace!("Group {} not found in group_results, skipping", group_id);
                continue; // Skip if no group data available
            }

            let (method_id, centroid_lat, centroid_lon, _version) = &group_results.groups[group_id];

            trace!(
                "Evaluating group {} with centroid at lat={}, lon={} for entity {}",
                group_id, centroid_lat, centroid_lon, new_entity_id.0
            );

            // Calculate distance using PostGIS if available
            let min_distance = if has_postgis && db_stmts.distance_stmt.is_some() {
                // Use PostGIS for distance calculation
                let mut min_dist = f64::MAX;

                // Distance to centroid
                if let Some(ref stmt) = db_stmts.distance_stmt {
                    let cent_row = tx
                        .query_one(stmt, &[&new_lon, &new_lat, &centroid_lon, &centroid_lat])
                        .await
                        .context("Failed to calculate PostGIS distance to centroid")?;

                    min_dist = cent_row.get::<_, f64>("distance");
                    trace!(
                        "PostGIS distance to centroid of group {}: {}m",
                        group_id, min_dist
                    );

                    // Only check individual entities if centroid is close enough
                    if min_dist <= max_distance * 1.5
                        && group_results.group_entities.contains_key(group_id)
                    {
                        // Check distances to group entities, but limit the number checked
                        let mut entities_checked = 0;
                        let max_entities_to_check = 5; // Limit to first 5 entities

                        debug!(
                            "Centroid of group {} is close, checking up to {} individual entities",
                            group_id, max_entities_to_check
                        );

                        for (entity_id, loc_lat, loc_lon) in &group_results.group_entities[group_id]
                        {
                            if entities_checked >= max_entities_to_check {
                                break; // Limit the number of entities we check
                            }

                            let ent_row = tx
                                .query_one(stmt, &[&new_lon, &new_lat, &loc_lon, &loc_lat])
                                .await
                                .context("Failed to calculate PostGIS distance to entity")?;

                            let dist: f64 = ent_row.get("distance");
                            trace!(
                                "Distance from entity {} to group entity {}: {}m",
                                new_entity_id.0, entity_id.0, dist
                            );

                            if dist < min_dist {
                                trace!(
                                    "Found closer entity in group {}: {}m (previous min: {}m)",
                                    group_id, dist, min_dist
                                );
                                min_dist = dist;
                            }

                            entities_checked += 1;

                            // Early exit if we found a close match
                            if min_dist <= THRESHOLDS[0].0 {
                                debug!(
                                    "Found very close match ({}m) to entity {} in group {}, stopping search",
                                    min_dist, entity_id.0, group_id
                                );
                                break;
                            }
                        }
                    }
                }

                min_dist
            } else {
                // Fallback to Haversine calculation
                let mut min_dist =
                    calculate_haversine_distance(*new_lat, *new_lon, *centroid_lat, *centroid_lon);

                trace!(
                    "Haversine distance to centroid of group {}: {}m",
                    group_id, min_dist
                );

                // Only check individual entities if centroid is close enough
                if min_dist <= max_distance * 1.5
                    && group_results.group_entities.contains_key(group_id)
                {
                    // Limit the number of entities we check
                    let mut entities_checked = 0;
                    let max_entities_to_check = 5;

                    debug!(
                        "Centroid of group {} is close, checking up to {} individual entities using Haversine",
                        group_id, max_entities_to_check
                    );

                    for (entity_id, loc_lat, loc_lon) in &group_results.group_entities[group_id] {
                        if entities_checked >= max_entities_to_check {
                            break;
                        }

                        let dist =
                            calculate_haversine_distance(*new_lat, *new_lon, *loc_lat, *loc_lon);

                        trace!(
                            "Haversine distance from entity {} to group entity {}: {}m",
                            new_entity_id.0, entity_id.0, dist
                        );

                        if dist < min_dist {
                            trace!(
                                "Found closer entity in group {}: {}m (previous min: {}m)",
                                group_id, dist, min_dist
                            );
                            min_dist = dist;
                        }

                        entities_checked += 1;

                        // Early exit if we found a close match
                        if min_dist <= THRESHOLDS[0].0 {
                            debug!(
                                "Found very close match ({}m) to entity {} in group {}, stopping search",
                                min_dist, entity_id.0, group_id
                            );
                            break;
                        }
                    }
                }

                min_dist
            };

            // Check if within any threshold
            for &(threshold, confidence) in &THRESHOLDS {
                if min_distance <= threshold {
                    debug!(
                        "Entity {} matched to group {} with distance {}m (threshold: {}m, confidence: {})",
                        new_entity_id.0, group_id, min_distance, threshold, confidence
                    );

                    // Get sample of entities from the group to compare services
                    let group_entity_ids =
                        if let Some(group_entities) = group_results.group_entities.get(group_id) {
                            group_entities
                                .iter()
                                .map(|(entity_id, _, _)| entity_id.clone())
                                .collect::<Vec<_>>()
                        } else {
                            vec![]
                        };

                    // Sample a few entities from the group for comparison (max 3 to keep performance reasonable)
                    let entity_sample =
                        super::service_utils::sample_group_entities(&group_entity_ids, 3);

                    if entity_sample.is_empty() {
                        debug!(
                            "No entities found in group {} for service comparison, proceeding with match based on distance only",
                            group_id
                        );
                        matched_group =
                            Some((group_id.clone(), method_id.clone(), threshold, confidence));
                        break;
                    }

                    // Check if services are semantically similar
                    let mut total_similarity = 0.0;
                    let mut valid_comparisons = 0;

                    for group_entity_id in &entity_sample {
                        match super::service_utils::compare_entity_services(
                            tx,
                            new_entity_id,
                            group_entity_id,
                        )
                        .await
                        {
                            Ok(similarity) => {
                                debug!(
                                    "Service similarity between entity {} and group entity {}: {:.4}",
                                    new_entity_id.0, group_entity_id.0, similarity
                                );
                                total_similarity += similarity;
                                valid_comparisons += 1;
                            }
                            Err(e) => {
                                warn!(
                                    "Failed to compare services for entities {} and {}: {}",
                                    new_entity_id.0, group_entity_id.0, e
                                );
                            }
                        }
                    }

                    // Calculate average similarity if any valid comparisons were made
                    if valid_comparisons > 0 {
                        let avg_similarity = total_similarity / valid_comparisons as f64;

                        // Only match if services are similar enough
                        if avg_similarity >= super::service_utils::SERVICE_SIMILARITY_THRESHOLD {
                            debug!(
                                "Services of entity {} are similar to those in group {} (score: {:.4}), proceeding with match",
                                new_entity_id.0, group_id, avg_similarity
                            );
                            matched_group =
                                Some((group_id.clone(), method_id.clone(), threshold, confidence));
                            break;
                        } else {
                            debug!(
                                "Services of entity {} are not similar enough to those in group {} (score: {:.4}), skipping despite geographic proximity",
                                new_entity_id.0, group_id, avg_similarity
                            );
                            // Continue to next group, as this one doesn't have similar services
                            continue;
                        }
                    } else {
                        // If no valid service comparisons, proceed with match based on distance only
                        debug!(
                            "No valid service comparisons made between entity {} and group {}, proceeding with match based on distance only",
                            new_entity_id.0, group_id
                        );
                        matched_group =
                            Some((group_id.clone(), method_id.clone(), threshold, confidence));
                        break;
                    }
                }
            }

            if matched_group.is_some() {
                break;
            }
        }

        // If a matching group was found, add the entity to it
        if let Some((group_id, method_id, threshold, confidence)) = matched_group {
            info!(
                "Adding entity {} to existing group {} (distance: <={}m, confidence: {})",
                new_entity_id.0, group_id, threshold, confidence
            );

            // Fetch the current match values for this group
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

            // Extract the geospatial values
            let mut geo_values = if let MatchValues::Geospatial(values) = current_match_values {
                debug!(
                    "Retrieved {} existing geospatial match values for group {}",
                    values.len(),
                    group_id
                );
                values
            } else {
                warn!(
                    "Unexpected match values type for geospatial group {}",
                    group_id
                );
                Vec::new()
            };

            // Get the centroid for calculating distance
            let (centroid_lat, centroid_lon) =
                if let Some((_, cent_lat, cent_lon, _)) = group_results.groups.get(&group_id) {
                    trace!(
                        "Using existing centroid for group {}: lat={}, lon={}",
                        group_id, cent_lat, cent_lon
                    );
                    (*cent_lat, *cent_lon)
                } else {
                    // Fallback to calculating from values
                    trace!(
                        "Recalculating centroid for group {} from {} match values",
                        group_id,
                        geo_values.len()
                    );
                    let locations: Vec<(EntityId, f64, f64)> = geo_values
                        .iter()
                        .map(|v| (v.entity_id.clone(), v.latitude, v.longitude))
                        .collect();
                    calculate_centroid(&locations)
                };

            // Add the new entity to the group
            debug!(
                "Inserting entity {} into group {}",
                new_entity_id.0, group_id
            );
            tx.execute(
                &db_stmts.entity_stmt,
                &[
                    &Uuid::new_v4().to_string(),
                    &group_id,
                    &new_entity_id.0,
                    &now,
                ],
            )
            .await
            .context("Failed to insert group entity")?;

            // Update entity count and version for the group
            debug!("Updating entity count and version for group {}", group_id);
            tx.execute(&db_stmts.update_group_count_stmt, &[&now, &group_id])
                .await
                .context("Failed to update group count")?;

            // Calculate distance to center using either PostGIS or Haversine
            let distance_to_center = if has_postgis && db_stmts.distance_stmt.is_some() {
                if let Some(ref stmt) = db_stmts.distance_stmt {
                    let row = tx
                        .query_one(stmt, &[&new_lon, &new_lat, &centroid_lon, &centroid_lat])
                        .await
                        .context("Failed to calculate distance to center")?;

                    let dist = row.get::<_, f64>("distance");
                    trace!(
                        "PostGIS distance from entity {} to group {} centroid: {}m",
                        new_entity_id.0, group_id, dist
                    );
                    Some(dist)
                } else {
                    let dist = calculate_haversine_distance(
                        *new_lat,
                        *new_lon,
                        centroid_lat,
                        centroid_lon,
                    );
                    trace!(
                        "Haversine distance from entity {} to group {} centroid: {}m",
                        new_entity_id.0, group_id, dist
                    );
                    Some(dist)
                }
            } else {
                let dist =
                    calculate_haversine_distance(*new_lat, *new_lon, centroid_lat, centroid_lon);
                trace!(
                    "Haversine distance from entity {} to group {} centroid: {}m",
                    new_entity_id.0, group_id, dist
                );
                Some(dist)
            };

            // Add to match values
            debug!(
                "Adding entity {} to geospatial match values of group {}",
                new_entity_id.0, group_id
            );
            geo_values.push(GeospatialMatchValue {
                latitude: *new_lat,
                longitude: *new_lon,
                distance_to_center,
                entity_id: new_entity_id.clone(),
            });

            // Update the match values
            let updated_match_values = MatchValues::Geospatial(geo_values);
            let updated_json = serde_json::to_value(updated_match_values)
                .context("Failed to serialize updated match values")?;

            debug!("Updating match values for method {}", method_id);
            tx.execute(&db_stmts.update_method_stmt, &[&updated_json, &method_id])
                .await
                .context("Failed to update match values")?;

            // Add to processed entities
            newly_processed.insert(new_entity_id.clone());
            total_entities_added += 1;

            info!(
                "Successfully added entity {} to existing group {} (within {}m)",
                new_entity_id.0, group_id, threshold
            );
        } else {
            debug!("No matching group found for entity {}", new_entity_id.0);
        }
    }

    info!(
        "Completed match_to_existing_groups: {} entities added to existing groups, {} entities skipped",
        total_entities_added, skipped_entities
    );

    Ok(MatchResult {
        entities_added: total_entities_added,
        entities_skipped: skipped_entities,
        groups_created: 0,
        processed_entities: newly_processed,
    })
}

/// Create new groups for entities that couldn't be matched to existing groups
pub async fn create_new_groups(
    tx: &Transaction<'_>,
    db_stmts: &DbStatements,
    locations: &[(EntityId, f64, f64)],
    has_postgis: bool,
    now: &NaiveDateTime,
) -> Result<MatchResult> {
    info!(
        "Starting create_new_groups for {} remaining unprocessed entities",
        locations.len()
    );
    debug!("PostGIS available: {}", has_postgis);

    // Track entities processed in new groups
    let mut newly_processed = HashSet::new();
    let mut total_groups_created = 0;
    let mut total_entities_added = 0;

    let start_time = std::time::Instant::now();

    if has_postgis && db_stmts.distance_stmt.is_some() {
        // Use PostGIS-based clustering for better performance and accuracy
        info!("Using PostGIS-based clustering algorithm for group creation");
        debug!(
            "Starting PostGIS clustering with {} entities",
            locations.len()
        );

        let result =
            create_groups_with_postgis(tx, db_stmts, locations, &mut newly_processed, now).await?;

        total_groups_created = result.groups_created;
        total_entities_added = result.entities_added;

        info!(
            "PostGIS clustering completed: created {} new groups with {} entities",
            result.groups_created, result.entities_added
        );
        trace!(
            "PostGIS clustering processed {} entities total",
            result.processed_entities.len()
        );
    } else {
        // Use fallback algorithm for proximity detection
        if has_postgis {
            warn!(
                "PostGIS is available but distance statement is missing, falling back to Haversine algorithm"
            );
        } else {
            info!("PostGIS not available, using Haversine-based clustering algorithm");
        }

        debug!(
            "Starting Haversine clustering with {} entities",
            locations.len()
        );

        let result =
            create_groups_with_haversine(tx, db_stmts, locations, &mut newly_processed, now)
                .await?;

        total_groups_created = result.groups_created;
        total_entities_added = result.entities_added;

        info!(
            "Haversine clustering completed: created {} new groups with {} entities",
            result.groups_created, result.entities_added
        );
        trace!(
            "Haversine clustering processed {} entities total",
            result.processed_entities.len()
        );
    }

    let elapsed = start_time.elapsed();
    info!(
        "Group creation completed in {:.2?}: created {} new groups with {} entities",
        elapsed, total_groups_created, total_entities_added
    );

    if total_entities_added < locations.len() {
        debug!(
            "{} entities were not assigned to any group",
            locations.len() - total_entities_added
        );
    }

    if let Some(skipped) = locations.len().checked_sub(newly_processed.len()) {
        if skipped > 0 {
            debug!("{} entities were skipped during processing", skipped);
        }
    }

    Ok(MatchResult {
        entities_added: total_entities_added,
        entities_skipped: 0,
        groups_created: total_groups_created,
        processed_entities: newly_processed,
    })
}

/// Create new groups using PostGIS clustering
/// Create new groups using PostGIS clustering
async fn create_groups_with_postgis(
    tx: &Transaction<'_>,
    db_stmts: &DbStatements,
    locations: &[(EntityId, f64, f64)],
    processed_entities: &mut HashSet<EntityId>,
    now: &NaiveDateTime,
) -> Result<MatchResult> {
    info!(
        "Starting PostGIS clustering with {} total locations",
        locations.len()
    );
    let start_time = std::time::Instant::now();

    let mut total_groups_created = 0;
    let mut total_entities_added = 0;
    let initially_processed = processed_entities.len();

    debug!(
        "Initially {} entities are already processed",
        initially_processed
    );

    // Create temporary table for PostGIS operations - do this once
    info!("Creating temporary PostGIS tables for clustering");
    let table_start = std::time::Instant::now();
    postgis::create_temp_location_table(tx).await?;
    debug!(
        "Temporary PostGIS tables created in {:.2?}",
        table_start.elapsed()
    );

    // For each threshold, find groups of nearby locations
    for (threshold_idx, (threshold, confidence)) in THRESHOLDS.iter().enumerate() {
        info!(
            "Processing proximity threshold {}/{}: {}m with confidence {}",
            threshold_idx + 1,
            THRESHOLDS.len(),
            threshold,
            confidence
        );

        let threshold_start = std::time::Instant::now();

        // Skip locations that are already processed in higher confidence thresholds
        let candidate_locations: Vec<(EntityId, f64, f64)> = locations
            .iter()
            .filter(|(entity_id, _, _)| !processed_entities.contains(entity_id))
            .map(|(e, lat, lon)| (e.clone(), *lat, *lon))
            .collect();

        if candidate_locations.is_empty() {
            info!(
                "No unprocessed locations for threshold {}m, skipping",
                threshold
            );
            continue;
        }

        info!(
            "Found {} candidate locations for {}m threshold",
            candidate_locations.len(),
            threshold
        );

        trace!(
            "Skipping {} already processed entities for this threshold",
            locations.len() - candidate_locations.len()
        );

        // Insert candidate locations (clears and inserts new data)
        debug!(
            "Inserting {} locations into temporary PostGIS table",
            candidate_locations.len()
        );
        let insert_start = std::time::Instant::now();
        postgis::insert_location_data(tx, &candidate_locations).await?;
        debug!("Location data inserted in {:.2?}", insert_start.elapsed());

        // Find clusters using PostGIS
        debug!(
            "Finding location clusters using PostGIS with threshold {}m",
            threshold
        );
        let cluster_start = std::time::Instant::now();
        let clusters = postgis::find_location_clusters(tx, *threshold).await?;
        debug!(
            "Cluster identification completed in {:.2?}: found {} clusters",
            cluster_start.elapsed(),
            clusters.len()
        );

        info!(
            "Found {} spatial clusters within {}m threshold",
            clusters.len(),
            threshold
        );

        if clusters.is_empty() {
            debug!(
                "No clusters found for threshold {}m, continuing to next threshold",
                threshold
            );
            continue;
        }

        // Calculate total entities in all clusters for validation
        let total_cluster_entities: usize = clusters.iter().map(|c| c.entity_ids.len()).sum();
        debug!(
            "Total entities in all clusters: {} (may include duplicates across clusters)",
            total_cluster_entities
        );

        // Process each cluster
        let clusters_start = std::time::Instant::now();
        let mut threshold_groups_created = 0;
        let mut threshold_entities_added = 0;

        for (cluster_idx, cluster) in clusters.iter().enumerate() {
            // Log progress for large cluster sets
            if clusters.len() > 10 && (cluster_idx % 10 == 0 || cluster_idx == clusters.len() - 1) {
                info!(
                    "Processing cluster {}/{} for {}m threshold",
                    cluster_idx + 1,
                    clusters.len(),
                    threshold
                );
            }

            trace!(
                "Cluster {}/{} contains {} entities with centroid at lat={}, lon={}",
                cluster_idx + 1,
                clusters.len(),
                cluster.entity_ids.len(),
                cluster.centroid.latitude,
                cluster.centroid.longitude
            );
            
            // Verify that entities in the cluster have semantically similar services
            debug!("Verifying service similarity among {} entities in cluster", cluster.entity_ids.len());
            
            // Convert string IDs to EntityId structs for the filtering
            let entity_ids: Vec<EntityId> = cluster.entity_ids.iter()
                .map(|id| EntityId(id.clone()))
                .collect();
            
            // Create a new filtered list based on service similarity
            let mut filtered_entity_ids = Vec::new();
            let mut entity_similarity_matrix = HashMap::new();
            
            // Only process up to 5 entities for performance reasons
            let sample_size = 5.min(entity_ids.len());
            
            // First entity is our reference point
            if !entity_ids.is_empty() {
                filtered_entity_ids.push(entity_ids[0].clone());
                
                // Compare other entities to the first one
                for i in 1..sample_size {
                    let entity_id = &entity_ids[i];
                    let similarity = match super::service_utils::compare_entity_services(tx, &entity_ids[0], entity_id).await {
                        Ok(sim) => {
                            debug!(
                                "Service similarity between entities {} and {}: {:.4}",
                                entity_ids[0].0, entity_id.0, sim
                            );
                            sim
                        },
                        Err(e) => {
                            warn!(
                                "Failed to compare services for entities {} and {}: {}. Defaulting to neutral similarity.",
                                entity_ids[0].0, entity_id.0, e
                            );
                            0.5 // Default if comparison fails
                        }
                    };
                    
                    entity_similarity_matrix.insert(entity_id.clone(), similarity);
                    
                    // Only add entities with similar enough services
                    if similarity >= super::service_utils::SERVICE_SIMILARITY_THRESHOLD {
                        filtered_entity_ids.push(entity_id.clone());
                    } else {
                        debug!(
                            "Entity {} excluded from cluster due to low service similarity ({:.4})",
                            entity_id.0, similarity
                        );
                    }
                }
                
                // If we have too few entities after filtering, skip creating this group
                if filtered_entity_ids.len() < 2 {
                    debug!(
                        "Cluster {}/{} has insufficient entities with similar services after filtering, skipping",
                        cluster_idx + 1, clusters.len()
                    );
                    continue;
                }
                
                debug!(
                    "Cluster {}/{} has {} entities with similar services after filtering",
                    cluster_idx + 1, clusters.len(), filtered_entity_ids.len()
                );
            } else {
                debug!("Cluster {}/{} has no entities, skipping", cluster_idx + 1, clusters.len());
                continue;
            }

            let group_id = EntityGroupId(postgis::generate_id());
            debug!("Creating new group with ID: {}", group_id.0);

            // Insert the group
            debug!(
                "Inserting group {} with description 'Geospatial match within {}m with similar services'",
                group_id.0, threshold
            );

            tx.execute(
                &db_stmts.group_stmt,
                &[
                    &group_id.0,
                    &format!("Geospatial match within {}m with similar services", threshold),
                    &now,
                    &now,
                    &confidence,
                    &(filtered_entity_ids.len() as i32),
                ],
            )
            .await
            .context("Failed to insert entity group")?;

            // Create a HashSet to track which entities we've processed in this group
            let mut processed_group_entities = HashSet::<EntityId>::new();
            let mut match_values = Vec::new();
            let mut cluster_entities_added = 0;
            let mut skipped_entities = 0;

            for entity_id_str in &cluster.entity_ids {
                let entity_id = EntityId(entity_id_str.clone());

                // Skip if we've already processed this entity in this group
                if processed_group_entities.contains(&entity_id) {
                    debug!(
                        "Entity {} already processed for group {}, skipping duplicate",
                        entity_id.0, group_id.0
                    );
                    continue;
                }

                // Add to processed set
                processed_group_entities.insert(entity_id.clone());

                trace!(
                    "Processing entity {} in cluster {}",
                    entity_id.0,
                    cluster_idx + 1
                );

                if let Some(&(_, lat, lon)) = candidate_locations
                    .iter()
                    .find(|(e, _, _)| e == &entity_id)
                {
                    let distance = calculate_haversine_distance(
                        lat,
                        lon,
                        cluster.centroid.latitude,
                        cluster.centroid.longitude,
                    );

                    trace!(
                        "Entity {} at lat={}, lon={}, distance to centroid: {}m",
                        entity_id.0, lat, lon, distance
                    );

                    match_values.push(GeospatialMatchValue {
                        latitude: lat,
                        longitude: lon,
                        distance_to_center: Some(distance),
                        entity_id: entity_id.clone(),
                    });

                    // Add detailed error logging here
                    debug!(
                        "Attempting to insert entity {} into group {} with UUID {}",
                        entity_id.0,
                        group_id.0,
                        Uuid::new_v4().to_string()
                    );

                    // Add try/catch around database operation
                    let group_entity_uuid = Uuid::new_v4().to_string();
                    match tx
                        .execute(
                            &db_stmts.entity_stmt,
                            &[&group_entity_uuid, &group_id.0, &entity_id.0, &now],
                        )
                        .await
                    {
                        Ok(_) => {
                            debug!(
                                "Successfully inserted entity {} into group {}",
                                entity_id.0, group_id.0
                            );
                            processed_entities.insert(entity_id);
                            total_entities_added += 1;
                            threshold_entities_added += 1;
                            cluster_entities_added += 1;
                        }
                        Err(e) => {
                            // Log detailed error information
                            error!(
                                "Failed to insert entity {} into group {}: {:?}",
                                entity_id.0, group_id.0, e
                            );

                            // Log parameter values for debugging
                            error!(
                                "Insert parameters: group_entity_uuid={}, group_id={}, entity_id={}, timestamp={:?}",
                                group_entity_uuid, group_id.0, entity_id.0, now
                            );

                            // Re-throw the error
                            return Err(e).context(format!(
                                "Failed to insert entity {} into group {}",
                                entity_id.0, group_id.0
                            ))?;
                        }
                    }
                } else {
                    warn!(
                        "Entity {} from cluster not found in candidate locations, skipping",
                        entity_id.0
                    );
                    skipped_entities += 1;
                }
            }

            if skipped_entities > 0 {
                warn!(
                    "Skipped {} entities when creating group {} due to missing location data",
                    skipped_entities, group_id.0
                );
            }

            debug!("Creating match method entry for group {}", group_id.0);
            let method_values = MatchValues::Geospatial(match_values);
            let match_values_json =
                serde_json::to_value(&method_values).context("Failed to serialize match values")?;

            tx.execute(
                &db_stmts.method_stmt,
                &[
                    &Uuid::new_v4().to_string(),
                    &group_id.0,
                    &MatchMethodType::Geospatial.as_str(),
                    &format!("Matched on geospatial proximity within {}m", threshold),
                    &match_values_json,
                    &confidence,
                    &now,
                ],
            )
            .await
            .context("Failed to insert group method")?;

            total_groups_created += 1;
            threshold_groups_created += 1;

            info!(
                "Created group {} with {} entities within {}m threshold",
                group_id.0, cluster_entities_added, threshold
            );
        }

        debug!(
            "Processed all clusters for {}m threshold in {:.2?}",
            threshold,
            clusters_start.elapsed()
        );

        info!(
            "Completed threshold {}m in {:.2?}: created {} groups with {} entities",
            threshold,
            threshold_start.elapsed(),
            threshold_groups_created,
            threshold_entities_added
        );
    }

    // Clean up temporary table at the end of all processing
    info!("Cleaning up temporary PostGIS tables");
    let cleanup_start = std::time::Instant::now();
    postgis::cleanup_temp_tables(tx).await?;
    debug!(
        "Temporary tables cleanup completed in {:.2?}",
        cleanup_start.elapsed()
    );

    let newly_processed = processed_entities.len() - initially_processed;

    info!(
        "PostGIS clustering completed in {:.2?}: created {} groups containing {} entities",
        start_time.elapsed(),
        total_groups_created,
        total_entities_added
    );

    if newly_processed != total_entities_added {
        warn!(
            "Discrepancy in processed entities: added {} to groups but processed set grew by {}",
            total_entities_added, newly_processed
        );
    }

    Ok(MatchResult {
        entities_added: total_entities_added,
        entities_skipped: 0,
        groups_created: total_groups_created,
        processed_entities: processed_entities.clone(),
    })
}

/// Create new groups using Haversine distance calculations
async fn create_groups_with_haversine(
    tx: &Transaction<'_>,
    db_stmts: &DbStatements,
    locations: &[(EntityId, f64, f64)],
    processed_entities: &mut HashSet<EntityId>,
    now: &NaiveDateTime,
) -> Result<MatchResult> {
    info!(
        "Starting Haversine-based clustering with {} total locations",
        locations.len()
    );
    let start_time = std::time::Instant::now();

    let mut total_groups_created = 0;
    let mut total_entities_added = 0;
    let initially_processed = processed_entities.len();

    debug!(
        "Initially {} entities are already processed",
        initially_processed
    );

    // For each threshold, find groups of nearby locations
    for (threshold_idx, (threshold, confidence)) in THRESHOLDS.iter().enumerate() {
        info!(
            "Processing proximity threshold {}/{}: {}m with confidence {}",
            threshold_idx + 1,
            THRESHOLDS.len(),
            threshold,
            confidence
        );

        let threshold_start = std::time::Instant::now();
        let mut proximity_groups = Vec::new();
        let mut threshold_entities_added = 0;

        // Skip locations that are already processed in higher confidence thresholds
        let candidate_locations: Vec<_> = locations
            .iter()
            .filter(|(entity_id, _, _)| !processed_entities.contains(entity_id))
            .collect();

        if candidate_locations.is_empty() {
            info!(
                "No unprocessed locations for threshold {}m, skipping",
                threshold
            );
            continue;
        }

        info!(
            "Found {} candidate locations for {}m threshold",
            candidate_locations.len(),
            threshold
        );

        trace!(
            "Skipping {} already processed entities for this threshold",
            locations.len() - candidate_locations.len()
        );

        // Process in chunks for better progress tracking
        let chunk_size = 50;
        let total_candidates = candidate_locations.len();
        let mut processed_candidates = 0;

        debug!(
            "Starting proximity search with O(n²) comparisons for {} candidates",
            total_candidates
        );
        let proximity_start = std::time::Instant::now();
        let mut total_comparisons = 0;
        let mut total_matches_found = 0;

        // For each unprocessed location
        for (i, (entity_id1, lat1, lon1)) in candidate_locations.iter().enumerate() {
            // Skip if already processed
            if processed_entities.contains(entity_id1) {
                trace!(
                    "Skipping already processed entity {} during proximity search",
                    entity_id1.0
                );
                continue;
            }

            trace!(
                "Evaluating entity {} at lat={}, lon={} as potential group center",
                entity_id1.0, lat1, lon1
            );

            // Store nearby entities
            let mut nearby_entities_map: HashMap<EntityId, (f64, f64, f64)> = HashMap::new();

            // Add this entity as the center point (0.0 distance to itself)
            nearby_entities_map.insert(entity_id1.clone(), (*lat1, *lon1, 0.0));

            // Compare with all other locations
            for (entity_id2, lat2, lon2) in candidate_locations.iter().skip(i + 1) {
                total_comparisons += 1;

                // Skip if already processed
                if processed_entities.contains(entity_id2) {
                    trace!(
                        "Skipping already processed entity {} during comparison",
                        entity_id2.0
                    );
                    continue;
                }

                // Calculate distance using fallback Haversine
                let distance = calculate_haversine_distance(*lat1, *lon1, *lat2, *lon2);

                trace!(
                    "Distance between entities {} and {}: {}m",
                    entity_id1.0, entity_id2.0, distance
                );

                if distance <= *threshold {
                    trace!(
                        "Found nearby entity {} within {}m (actual: {}m)",
                        entity_id2.0, threshold, distance
                    );

                    // Check if services are semantically similar
                    let service_similarity = match super::service_utils::compare_entity_services(
                        tx, entity_id1, entity_id2,
                    )
                    .await
                    {
                        Ok(similarity) => {
                            debug!(
                                "Service similarity between entities {} and {}: {:.4}",
                                entity_id1.0, entity_id2.0, similarity
                            );
                            similarity
                        }
                        Err(e) => {
                            warn!(
                                "Failed to compare services for entities {} and {}: {}. Defaulting to neutral similarity.",
                                entity_id1.0, entity_id2.0, e
                            );
                            0.5 // Default to moderate similarity if comparison fails
                        }
                    };

                    // Only group entities if their services are similar enough
                    if service_similarity >= super::service_utils::SERVICE_SIMILARITY_THRESHOLD {
                        debug!(
                            "Services of entities {} and {} are similar (score: {:.4}), adding to potential group",
                            entity_id1.0, entity_id2.0, service_similarity
                        );
                        nearby_entities_map.insert(entity_id2.clone(), (*lat2, *lon2, distance));
                        total_matches_found += 1;
                    } else {
                        debug!(
                            "Services of entities {} and {} are not similar enough (score: {:.4}), not grouping despite geographic proximity",
                            entity_id1.0, entity_id2.0, service_similarity
                        );
                    }
                }
            }

            // Convert to vector for further processing
            let nearby_entities: Vec<(EntityId, f64, f64, f64)> = nearby_entities_map
                .into_iter()
                .map(|(entity_id, (lat, lon, distance))| (entity_id, lat, lon, distance))
                .collect();

            // If we found multiple entities, create a group
            if nearby_entities.len() > 1 {
                debug!(
                    "Found potential group with {} entities centered around entity {}",
                    nearby_entities.len(),
                    entity_id1.0
                );

                // Mark all entities as processed
                for (entity_id, _, _, _) in &nearby_entities {
                    processed_entities.insert(entity_id.clone());
                    trace!("Marking entity {} as processed", entity_id.0);
                }

                // Then add to proximity_groups
                proximity_groups.push(nearby_entities);
            } else {
                trace!(
                    "No group formed for entity {} (insufficient nearby entities)",
                    entity_id1.0
                );
            }

            processed_candidates += 1;
            if processed_candidates % chunk_size == 0 || processed_candidates == total_candidates {
                debug!(
                    "Processed {}/{} candidates ({:.1}%) for {}m threshold",
                    processed_candidates,
                    total_candidates,
                    (processed_candidates as f64 / total_candidates as f64) * 100.0,
                    threshold
                );
            }
        }

        debug!(
            "Proximity search completed in {:.2?}: made {} comparisons, found {} matches",
            proximity_start.elapsed(),
            total_comparisons,
            total_matches_found
        );

        let groups_in_threshold = proximity_groups.len();
        info!(
            "Found {} proximity groups within {}m threshold",
            groups_in_threshold, threshold
        );

        if groups_in_threshold == 0 {
            debug!(
                "No groups formed for threshold {}m, continuing to next threshold",
                threshold
            );
            continue;
        }

        // Calculate total entities across all groups (for validation)
        let total_in_groups: usize = proximity_groups.iter().map(|g| g.len()).sum();
        debug!(
            "Total entities in all groups: {} (may include duplicates due to overlapping groups)",
            total_in_groups
        );

        debug!(
            "Starting database operations for {} groups",
            groups_in_threshold
        );
        let db_start = std::time::Instant::now();
        let mut threshold_groups_created = 0;

        // Create database records for each proximity group
        for (group_idx, nearby_entities) in proximity_groups.iter().enumerate() {
            if group_idx % 50 == 0 || group_idx == groups_in_threshold - 1 {
                info!(
                    "Processing group {}/{} ({:.1}%) for {}m threshold",
                    group_idx + 1,
                    groups_in_threshold,
                    (group_idx + 1) as f64 / groups_in_threshold as f64 * 100.0,
                    threshold
                );
            }

            // Create a new entity group
            let group_id = EntityGroupId(Uuid::new_v4().to_string());
            let entity_count = nearby_entities.len() as i32;

            debug!(
                "Creating new group {} with {} entities for {}m threshold",
                group_id.0, entity_count, threshold
            );

            // Calculate centroid for the group
            let (centroid_lat, centroid_lon) = calculate_centroid_from_full(&nearby_entities);

            debug!(
                "Calculated centroid for group {}: lat={}, lon={}",
                group_id.0, centroid_lat, centroid_lon
            );

            // Insert the group with initial version = 1
            trace!("Inserting group {} record into database", group_id.0);
            tx.execute(
                &db_stmts.group_stmt,
                &[
                    &group_id.0,
                    &format!("Geospatial match within {}m", threshold),
                    &now,
                    &now,
                    &confidence,
                    &entity_count,
                ],
            )
            .await
            .context("Failed to insert entity group")?;

            // Create the group_entity records and match values
            let mut match_values = Vec::new();
            let mut group_entities_added = 0;

            for (entity_id, lat, lon, distance_from_center) in nearby_entities {
                // Calculate distance to actual centroid (may differ from original center entity)
                let centroid_distance =
                    distance_to_centroid(*lat, *lon, centroid_lat, centroid_lon);

                trace!(
                    "Entity {} in group {}: distance to centroid: {}m, distance to center entity: {}m",
                    entity_id.0, group_id.0, centroid_distance, distance_from_center
                );

                // Add to match values
                match_values.push(GeospatialMatchValue {
                    latitude: *lat,
                    longitude: *lon,
                    distance_to_center: Some(centroid_distance),
                    entity_id: entity_id.clone(),
                });

                // Insert group entity record with improved error handling
                let group_entity_uuid = Uuid::new_v4().to_string();
                debug!(
                    "Attempting to insert entity {} into group {} with UUID {}",
                    entity_id.0, group_id.0, group_entity_uuid
                );

                match tx
                    .execute(
                        &db_stmts.entity_stmt,
                        &[&group_entity_uuid, &group_id.0, &entity_id.0, &now],
                    )
                    .await
                {
                    Ok(_) => {
                        debug!(
                            "Successfully inserted entity {} into group {}",
                            entity_id.0, group_id.0
                        );
                        total_entities_added += 1;
                        threshold_entities_added += 1;
                        group_entities_added += 1;
                    }
                    Err(e) => {
                        // Log detailed error information
                        error!(
                            "Failed to insert entity {} into group {}: {:?}",
                            entity_id.0, group_id.0, e
                        );

                        // Log parameter values for debugging
                        error!(
                            "Insert parameters: group_entity_uuid={}, group_id={}, entity_id={}, timestamp={:?}",
                            group_entity_uuid, group_id.0, entity_id.0, now
                        );

                        // Re-throw the error
                        return Err(e).context(format!(
                            "Failed to insert entity {} into group {}",
                            entity_id.0, group_id.0
                        ))?;
                    }
                }
            }

            debug!(
                "Added {} entities to group {}",
                group_entities_added, group_id.0
            );

            // Serialize match_values to JSON
            trace!("Creating match method entry for group {}", group_id.0);
            let method_values = MatchValues::Geospatial(match_values);
            let match_values_json =
                serde_json::to_value(&method_values).context("Failed to serialize match values")?;

            // Insert group method record
            tx.execute(
                &db_stmts.method_stmt,
                &[
                    &Uuid::new_v4().to_string(),
                    &group_id.0,
                    &MatchMethodType::Geospatial.as_str(),
                    &format!("Matched on geospatial proximity within {}m", threshold),
                    &match_values_json,
                    &confidence,
                    &now,
                ],
            )
            .await
            .context("Failed to insert group method")?;

            total_groups_created += 1;
            threshold_groups_created += 1;

            debug!(
                "Successfully created group {} with {} entities within {}m",
                group_id.0, group_entities_added, threshold
            );
        }

        debug!(
            "Database operations for {} groups completed in {:.2?}",
            groups_in_threshold,
            db_start.elapsed()
        );

        info!(
            "Completed threshold {}m in {:.2?}: created {} groups with {} entities",
            threshold,
            threshold_start.elapsed(),
            threshold_groups_created,
            threshold_entities_added
        );
    }

    let newly_processed = processed_entities.len() - initially_processed;

    info!(
        "Haversine clustering completed in {:.2?}: created {} groups containing {} entities",
        start_time.elapsed(),
        total_groups_created,
        total_entities_added
    );

    if newly_processed != total_entities_added {
        warn!(
            "Discrepancy in processed entities: added {} to groups but processed set grew by {}",
            total_entities_added, newly_processed
        );
    }

    Ok(MatchResult {
        entities_added: total_entities_added,
        entities_skipped: 0,
        groups_created: total_groups_created,
        processed_entities: processed_entities.clone(),
    })
}
