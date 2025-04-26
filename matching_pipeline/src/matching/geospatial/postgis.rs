// src/matching/geospatial/postgis.rs
//
// PostGIS-specific operations for geospatial matching

use anyhow::{Context, Result};
use log::{debug, error, info, warn};
use tokio_postgres::Transaction;
use uuid::Uuid;

use crate::models::{Centroid, EntityId, SpatialCluster};

/// Create a temporary table for storing locations to process with PostGIS
pub async fn create_temp_location_table(tx: &Transaction<'_>) -> Result<()> {
    tx.execute(
        "
        CREATE TEMPORARY TABLE IF NOT EXISTS temp_candidates (
            id SERIAL PRIMARY KEY,
            entity_id TEXT NOT NULL,
            geom GEOGRAPHY(Point, 4326),
            latitude DOUBLE PRECISION,
            longitude DOUBLE PRECISION
        ) ON COMMIT DROP
    ",
        &[],
    )
    .await
    .context("Failed to create temp candidates table")?;

    // Create spatial index on temp table for faster queries
    // Now indexing on geom rather than entity_id
    tx.execute(
        "
        CREATE INDEX IF NOT EXISTS idx_temp_candidates_geom 
        ON temp_candidates 
        USING GIST (geom)
    ",
        &[],
    )
    .await
    .context("Failed to create spatial index on temp table")?;

    // Create an additional index on entity_id for efficient lookups
    tx.execute(
        "
        CREATE INDEX IF NOT EXISTS idx_temp_candidates_entity_id
        ON temp_candidates (entity_id)
    ",
        &[],
    )
    .await
    .context("Failed to create entity_id index on temp table")?;

    Ok(())
}

/// Insert location data into the temporary table
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

    // First truncate the table to ensure it's empty
    match tx.execute("TRUNCATE TABLE temp_candidates", &[]).await {
        Ok(_) => debug!("Successfully truncated temp_candidates table"),
        Err(e) => {
            warn!("Failed to truncate temp_candidates table: {}", e);
            // Continue execution since this might be the first run
        }
    };

    // Batch in chunks for better performance
    let chunk_size = 100;
    let mut inserted = 0;
    let mut errors = 0;

    for (chunk_idx, chunk) in locations.chunks(chunk_size).enumerate() {
        debug!("Processing chunk {}, size {}", chunk_idx + 1, chunk.len());

        for (entity_id, lat, lon) in chunk {
            // Validate coordinates
            if !(-90.0..=90.0).contains(lat) || !(-180.0..=180.0).contains(lon) {
                warn!(
                    "Invalid coordinates for entity {}: lat={}, lon={}",
                    entity_id.0, lat, lon
                );
                errors += 1;
                continue;
            }

            match tx
                .execute(
                    "
                INSERT INTO temp_candidates (entity_id, geom, latitude, longitude) 
                VALUES ($1, ST_SetSRID(ST_MakePoint($2, $3), 4326)::geography, $4, $5)
                ",
                    &[&entity_id.0, &lon, &lat, &lat, &lon],
                )
                .await
            {
                Ok(_) => {
                    inserted += 1;
                    if inserted % 10 == 0 {
                        debug!("Inserted {} locations so far", inserted);
                    }
                }
                Err(e) => {
                    errors += 1;
                    // Log detailed error information for troubleshooting
                    error!(
                        "Failed to insert entity {}: lat={}, lon={}, error: {}",
                        entity_id.0, lat, lon, e
                    );

                    // Stop after too many errors to prevent excessive logging
                    if errors >= 5 {
                        return Err(anyhow::anyhow!(
                            "Too many insertion errors ({}). Last error: {}",
                            errors,
                            e
                        ));
                    }
                }
            }
        }
    }

    // Summary logging
    info!(
        "Finished inserting locations: {} successful, {} errors",
        inserted, errors
    );

    if errors > 0 && inserted == 0 {
        return Err(anyhow::anyhow!(
            "Failed to insert any locations ({} errors)",
            errors
        ));
    } else if errors > 0 {
        warn!("Completed with {} insertion errors", errors);
    }

    Ok(())
}

/// Find nearby groups for a new location using spatial indexing
/// This is a key optimization to avoid O(n*m) comparisons
pub async fn find_nearby_groups(
    tx: &Transaction<'_>,
    new_lat: f64,
    new_lon: f64,
    max_distance: f64,
) -> Result<Vec<String>> {
    let query = "
        WITH group_points AS (
            SELECT 
                gm.entity_group_id,
                ST_SetSRID(ST_MakePoint(
                    CAST(value->>'longitude' AS FLOAT),
                    CAST(value->>'latitude' AS FLOAT)
                ), 4326)::geography AS point
            FROM 
                group_method gm,
                jsonb_array_elements(gm.match_values->'values') as value
            WHERE 
                gm.method_type = 'geospatial'
        )
        SELECT DISTINCT entity_group_id 
        FROM group_points
        WHERE ST_DWithin(
            point,
            ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography,
            $3
        )
    ";

    let rows = tx
        .query(query, &[&new_lon, &new_lat, &max_distance])
        .await
        .context("Failed to find nearby groups")?;

    let mut group_ids = Vec::new();
    for row in &rows {
        group_ids.push(row.get::<_, String>("entity_group_id"));
    }

    debug!(
        "Found {} nearby groups within {}m of ({}, {})",
        group_ids.len(),
        max_distance,
        new_lat,
        new_lon
    );

    Ok(group_ids)
}

/// Find clusters of locations using PostGIS
pub async fn find_location_clusters(
    tx: &Transaction<'_>,
    threshold: f64,
) -> Result<Vec<SpatialCluster>> {
    debug!(
        "Finding clusters with PostGIS using threshold {}m",
        threshold
    );

    // Modified query to ensure each entity appears only once per cluster
    let cluster_query = format!(
        "
        WITH clusters AS (
            SELECT 
                entity_id,
                ST_ClusterDBSCAN(geom::geometry, {}, 1) OVER () AS cluster_id,
                latitude,
                longitude
            FROM temp_candidates
        ),
        -- Get one representative location per entity per cluster
        unique_entities AS (
            SELECT DISTINCT ON (cluster_id, entity_id)
                cluster_id,
                entity_id,
                latitude,
                longitude
            FROM clusters
            WHERE cluster_id IS NOT NULL
            ORDER BY cluster_id, entity_id, latitude
        )
        SELECT 
            cluster_id,
            array_agg(entity_id) AS entity_ids,
            ST_Y(ST_Centroid(ST_Collect(ST_MakePoint(longitude, latitude)))) AS centroid_lat,
            ST_X(ST_Centroid(ST_Collect(ST_MakePoint(longitude, latitude)))) AS centroid_lon,
            COUNT(*) AS entity_count
        FROM unique_entities
        GROUP BY cluster_id
        ",
        threshold
    );

    let cluster_rows = tx
        .query(&cluster_query, &[])
        .await
        .context("Failed to execute PostGIS clustering")?;

    let mut clusters = Vec::new();

    for row in &cluster_rows {
        let cluster_id: i32 = row.get("cluster_id");
        let entity_ids: Vec<String> = row.get("entity_ids");
        let centroid_lat: f64 = row.get("centroid_lat");
        let centroid_lon: f64 = row.get("centroid_lon");
        let entity_count: i64 = row.get("entity_count");

        // Skip single-entity clusters
        if entity_count < 2 {
            continue;
        }

        clusters.push(SpatialCluster {
            cluster_id,
            entity_ids,
            centroid: Centroid {
                latitude: centroid_lat,
                longitude: centroid_lon,
            },
            entity_count,
        });
    }

    Ok(clusters)
}

/// Calculate distance between points using PostGIS
pub async fn calculate_distance(
    tx: &Transaction<'_>,
    lat1: f64,
    lon1: f64,
    lat2: f64,
    lon2: f64,
) -> Result<f64> {
    let distance_query = "
        SELECT ST_Distance(
            ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography,
            ST_SetSRID(ST_MakePoint($3, $4), 4326)::geography
        ) AS distance
    ";

    let row = tx
        .query_one(distance_query, &[&lon1, &lat1, &lon2, &lat2])
        .await
        .context("Failed to calculate PostGIS distance")?;

    let distance: f64 = row.get("distance");
    Ok(distance)
}

/// Generate a new unique ID
pub fn generate_id() -> String {
    Uuid::new_v4().to_string()
}

/// Clean up temporary tables
pub async fn cleanup_temp_tables(tx: &Transaction<'_>) -> Result<()> {
    tx.execute("DROP TABLE IF EXISTS temp_candidates", &[])
        .await
        .context("Failed to drop temp candidates table")?;

    Ok(())
}
