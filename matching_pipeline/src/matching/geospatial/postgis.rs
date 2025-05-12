// src/matching/geospatial/postgis.rs

use std::collections::HashMap;

use anyhow::{Context, Result};
use log::{debug, error, info, warn}; // Added error
use tokio_postgres::Transaction;
use uuid::Uuid; // Not directly used here but good for consistency if IDs generated

use super::database::ExistingGeoPairsMap;
use crate::models::{
    Centroid, EntityGroupId, EntityId, GeospatialMatchValue, MatchValues, SpatialCluster,
}; // For the return type alias

// create_temp_location_table, insert_location_data, find_location_clusters,
// calculate_distance, cleanup_temp_tables can remain largely as they are in your current files,
// as they deal with raw location data or are generic utilities.
// The crucial change is how their outputs (like SpatialCluster) are consumed by core.rs
// and how existing pairs are queried.

/// Finds existing geospatial *pairs* where at least one constituent entity
/// is within `max_distance` of the `new_lat`, `new_lon`.
pub async fn find_nearby_geospatial_pairs_with_details(
    tx: &Transaction<'_>,
    new_lat: f64,
    new_lon: f64,
    max_distance: f64,
) -> Result<ExistingGeoPairsMap> {
    // The match_values for geospatial pairs is `Geospatial(GeospatialMatchValue)`
    // where GeospatialMatchValue is `{ latitude1, longitude1, latitude2, longitude2, distance }`
    let query = "
        SELECT
            id AS pair_id,
            entity_id_1,
            entity_id_2,
            match_values,
            confidence_score
        FROM
            entity_group
        WHERE
            method_type = 'geospatial'
            AND (
                ST_DWithin(
                    ST_SetSRID(ST_MakePoint(
                        (match_values->'values'->>'longitude1')::FLOAT,
                        (match_values->'values'->>'latitude1')::FLOAT
                    ), 4326)::geography,
                    ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography,
                    $3
                ) OR
                ST_DWithin(
                    ST_SetSRID(ST_MakePoint(
                        (match_values->'values'->>'longitude2')::FLOAT,
                        (match_values->'values'->>'latitude2')::FLOAT
                    ), 4326)::geography,
                    ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography,
                    $3
                )
            )
    ";

    let rows = tx
        .query(query, &[&new_lon, &new_lat, &max_distance])
        .await
        .context("Failed to find nearby geospatial pairs using PostGIS ST_DWithin")?;

    let mut candidate_pairs: ExistingGeoPairsMap = HashMap::new();
    for row in &rows {
        let pair_id_str: String = row.get("pair_id");
        let entity_id_1_str: String = row.get("entity_id_1");
        let entity_id_2_str: String = row.get("entity_id_2");
        let match_values_json: serde_json::Value = row.get("match_values");
        let confidence_score: Option<f64> = row.get("confidence_score");

        match serde_json::from_value::<MatchValues>(match_values_json.clone()) {
            Ok(MatchValues::Geospatial(gv)) => {
                // Expecting the refactored GeospatialMatchValue
                candidate_pairs.insert(
                    EntityGroupId(pair_id_str),
                    (
                        EntityId(entity_id_1_str),
                        EntityId(entity_id_2_str),
                        gv,
                        confidence_score,
                    ),
                );
            }
            _ => {
                warn!(
                    "PostGIS: Could not deserialize GeospatialMatchValue for pair_id {} or it was wrong type. JSON: {}",
                     pair_id_str, match_values_json);
            }
        }
    }
    debug!(
        "PostGIS: Found {} nearby existing geospatial pairs within {}m of ({}, {})",
        candidate_pairs.len(),
        max_distance,
        new_lat,
        new_lon
    );
    Ok(candidate_pairs)
}

// --- The following functions are assumed to be mostly unchanged from your provided files ---
// --- as they operate on raw locations or are generic utilities. ---

pub async fn create_temp_location_table(tx: &Transaction<'_>) -> Result<()> {
    tx.execute(
        "CREATE TEMPORARY TABLE IF NOT EXISTS temp_candidates (
            id SERIAL PRIMARY KEY, entity_id TEXT NOT NULL,
            geom GEOGRAPHY(Point, 4326), latitude DOUBLE PRECISION, longitude DOUBLE PRECISION
        ) ON COMMIT DROP",
        &[],
    )
    .await
    .context("Create temp_candidates table")?;
    tx.execute(
        "CREATE INDEX IF NOT EXISTS idx_temp_candidates_geom ON temp_candidates USING GIST (geom)",
        &[],
    )
    .await
    .context("Create geom index on temp_candidates")?;
    tx.execute(
        "CREATE INDEX IF NOT EXISTS idx_temp_candidates_entity_id ON temp_candidates (entity_id)",
        &[],
    )
    .await
    .context("Create entity_id index on temp_candidates")?;
    Ok(())
}

pub async fn insert_location_data(
    tx: &Transaction<'_>,
    locations: &[(EntityId, f64, f64)],
) -> Result<()> {
    if locations.is_empty() {
        return Ok(());
    }
    debug!(
        "Inserting {} locations into temp_candidates table",
        locations.len()
    );
    tx.execute("TRUNCATE TABLE temp_candidates", &[])
        .await
        .context("Truncate temp_candidates")?; // Ensure it's empty

    let mut errors = 0;
    for chunk in locations.chunks(1000) {
        // Process in larger chunks if possible
        for (entity_id, lat, lon) in chunk {
            if !(-90.0..=90.0).contains(lat) || !(-180.0..=180.0).contains(lon) {
                warn!(
                    "Invalid coordinates for entity {}: lat={}, lon={}. Skipping insert.",
                    entity_id.0, lat, lon
                );
                errors += 1;
                continue;
            }
            if let Err(e) = tx
                .execute(
                    "INSERT INTO temp_candidates (entity_id, geom, latitude, longitude)
                 VALUES ($1, ST_SetSRID(ST_MakePoint($2, $3), 4326)::geography, $4, $5)",
                    &[&entity_id.0, lon, lat, lat, lon],
                )
                .await
            {
                error!(
                    "Failed to insert entity {} ({},{}): {}",
                    entity_id.0, lat, lon, e
                );
                errors += 1;
                // Decide on error handling: continue or return error?
                // For now, logging and continuing.
            }
        }
    }
    if errors > 0 {
        warn!("Completed insert_location_data with {} errors.", errors);
    }
    info!(
        "Finished inserting locations into temp_candidates ({} successful, {} errors).",
        locations.len() - errors,
        errors
    );
    Ok(())
}

pub async fn find_location_clusters(
    tx: &Transaction<'_>,
    threshold_distance: f64,
) -> Result<Vec<SpatialCluster>> {
    debug!(
        "Finding location clusters using PostGIS ST_ClusterDBSCAN (eps={}, min_points=2)",
        threshold_distance
    );
    let cluster_query = format!(
        "WITH clusters AS (
            SELECT entity_id, ST_ClusterDBSCAN(geom::geometry, {}, 1) OVER () AS cluster_id, latitude, longitude
            FROM temp_candidates
        ), unique_entities_in_cluster AS (
            SELECT DISTINCT ON (cluster_id, entity_id) cluster_id, entity_id, latitude, longitude
            FROM clusters WHERE cluster_id IS NOT NULL ORDER BY cluster_id, entity_id
        )
        SELECT cluster_id, array_agg(entity_id) AS entity_ids,
               ST_Y(ST_Centroid(ST_Collect(ST_MakePoint(longitude, latitude)))) AS centroid_lat,
               ST_X(ST_Centroid(ST_Collect(ST_MakePoint(longitude, latitude)))) AS centroid_lon,
               COUNT(DISTINCT entity_id) AS entity_count
        FROM unique_entities_in_cluster GROUP BY cluster_id HAVING COUNT(DISTINCT entity_id) >= 2", // Ensure clusters have at least 2 entities
        threshold_distance
    );

    let cluster_rows = tx
        .query(&cluster_query, &[])
        .await
        .context("PostGIS ST_ClusterDBSCAN query failed")?;
    let mut spatial_clusters = Vec::new();
    for row in &cluster_rows {
        let entity_ids_str: Vec<String> = row.get("entity_ids");
        let entity_ids_obj: Vec<EntityId> = entity_ids_str.into_iter().map(EntityId).collect();
        spatial_clusters.push(SpatialCluster {
            cluster_id: row.get("cluster_id"),
            entity_ids: entity_ids_obj,
            centroid: Centroid {
                latitude: row.get("centroid_lat"),
                longitude: row.get("centroid_lon"),
            },
            entity_count: row.get("entity_count"),
        });
    }
    debug!(
        "Found {} raw spatial clusters from PostGIS.",
        spatial_clusters.len()
    );
    Ok(spatial_clusters)
}

pub async fn cleanup_temp_tables(tx: &Transaction<'_>) -> Result<()> {
    tx.execute("DROP TABLE IF EXISTS temp_candidates", &[])
        .await
        .context("Drop temp_candidates table")?;
    Ok(())
}
