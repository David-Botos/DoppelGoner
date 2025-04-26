// src/matching/geospatial/database.rs
//
// Database interaction functions for geospatial matching

use anyhow::{Context, Result};
use log::debug;
use std::collections::{HashMap, HashSet};
use tokio_postgres::{Client, Statement, Transaction};

use crate::models::{EntityId, GroupResults, LocationResults, MatchValues};

/// Struct to hold prepared database statements
pub struct DbStatements {
    pub group_stmt: Statement,
    pub entity_stmt: Statement,
    pub method_stmt: Statement,
    pub update_group_count_stmt: Statement,
    pub update_method_stmt: Statement,
    pub consistency_check_stmt: Statement,
    pub distance_stmt: Option<Statement>,
}

/// Prepare database statements for reuse
pub async fn prepare_statements(tx: &Transaction<'_>) -> Result<DbStatements> {
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

    let consistency_check_stmt = tx
        .prepare(
            "
        SELECT COUNT(*) 
        FROM group_entity ge
        JOIN group_method gm ON ge.entity_group_id = gm.entity_group_id
        WHERE ge.entity_id = $1
        AND gm.method_type = 'geospatial'
    ",
        )
        .await
        .context("Failed to prepare consistency check statement")?;

    // Prepare PostGIS-specific statement if available
    let distance_stmt = tx
        .prepare(
            "
        SELECT ST_Distance(
            ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography,
            ST_SetSRID(ST_MakePoint($3, $4), 4326)::geography
        ) AS distance
    ",
        )
        .await
        .ok();

    Ok(DbStatements {
        group_stmt,
        entity_stmt,
        method_stmt,
        update_group_count_stmt,
        update_method_stmt,
        consistency_check_stmt,
        distance_stmt,
    })
}

/// Get all entities that are already part of a geospatial match group
pub async fn get_processed_entities(conn: &Client) -> Result<HashSet<EntityId>> {
    let processed_query = "
        SELECT DISTINCT ge.entity_id
        FROM group_entity ge
        JOIN group_method gm ON ge.entity_group_id = gm.entity_group_id
        WHERE gm.method_type = 'geospatial'
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

    Ok(processed_entities)
}

/// Query for unprocessed locations
pub async fn get_unprocessed_locations(
    conn: &Client,
    processed_entities: &HashSet<EntityId>,
) -> Result<LocationResults> {
    // Query for unprocessed locations, detecting if geography column exists
    let location_query = "
        SELECT 
            e.id AS entity_id,
            l.id AS location_id,
            l.latitude,
            l.longitude,
            CASE 
                WHEN l.geom IS NOT NULL THEN TRUE
                ELSE FALSE
            END AS has_geom
        FROM 
            entity e
            JOIN entity_feature ef ON e.id = ef.entity_id
            JOIN location l ON ef.table_id = l.id AND ef.table_name = 'location'
        WHERE 
            l.latitude IS NOT NULL 
            AND l.longitude IS NOT NULL
            AND e.id NOT IN (
                SELECT DISTINCT ge.entity_id
                FROM group_entity ge
                JOIN group_method gm ON ge.entity_group_id = gm.entity_group_id
                WHERE gm.method_type = 'geospatial'
            )
    ";

    debug!("Executing location query for unprocessed entities");
    let location_rows = conn
        .query(location_query, &[])
        .await
        .context("Failed to query locations")?;

    // Check if the location table has the geom column by checking the first row
    let has_postgis = if !location_rows.is_empty() {
        let has_geom: bool = location_rows[0].get("has_geom");
        has_geom
    } else {
        false
    };

    // Store all unprocessed locations
    let mut new_locations: Vec<(EntityId, f64, f64)> = Vec::new();

    for row in &location_rows {
        let entity_id: String = row.get("entity_id");
        let entity_id = EntityId(entity_id);

        if processed_entities.contains(&entity_id) {
            continue; // Skip if already processed
        }

        let latitude: f64 = row.get("latitude");
        let longitude: f64 = row.get("longitude");

        // Add to new locations
        new_locations.push((entity_id, latitude, longitude));
    }

    Ok(LocationResults {
        locations: new_locations,
        has_postgis,
    })
}

/// Load existing groups with their centroids
pub async fn get_existing_groups(conn: &Client, has_postgis: bool) -> Result<GroupResults> {
    // Load existing groups with their centroids using PostGIS if available
    let existing_groups_query = if has_postgis {
        "SELECT 
            eg.id AS group_id,
            gm.id AS method_id,
            gm.match_values,
            ST_Y(ST_Centroid(ST_Collect(ST_MakePoint(
                CAST(value->>'longitude' AS FLOAT), 
                CAST(value->>'latitude' AS FLOAT)
            )))) AS centroid_lat,
            ST_X(ST_Centroid(ST_Collect(ST_MakePoint(
                CAST(value->>'longitude' AS FLOAT), 
                CAST(value->>'latitude' AS FLOAT)
            )))) AS centroid_lon,
            COALESCE(eg.version, 1) as version
        FROM 
            entity_group eg
            JOIN group_method gm ON eg.id = gm.entity_group_id,
            jsonb_array_elements(gm.match_values->'values') as value
        WHERE 
            gm.method_type = 'geospatial'
        GROUP BY
            eg.id, gm.id, gm.match_values, eg.version"
    } else {
        "SELECT 
            eg.id AS group_id,
            gm.id AS method_id,
            gm.match_values,
            (
                SELECT AVG(CAST(value->>'latitude' AS double precision))
                FROM jsonb_array_elements(gm.match_values->'values') as value
            ) AS centroid_lat,
            (
                SELECT AVG(CAST(value->>'longitude' AS double precision))
                FROM jsonb_array_elements(gm.match_values->'values') as value
            ) AS centroid_lon,
            COALESCE(eg.version, 1) as version
        FROM 
            entity_group eg
            JOIN group_method gm ON eg.id = gm.entity_group_id
        WHERE 
            gm.method_type = 'geospatial'"
    };

    let existing_groups_rows = conn
        .query(existing_groups_query, &[])
        .await
        .context("Failed to query existing groups")?;

    // Structure to store group info: (group_id, method_id, centroid_lat, centroid_lon, version)
    let mut existing_groups: HashMap<String, (String, f64, f64, i32)> = HashMap::new();

    // Also store entities for each group to use in consistency checks
    let mut group_entities: HashMap<String, Vec<(EntityId, f64, f64)>> = HashMap::new();

    for row in &existing_groups_rows {
        let group_id: String = row.get("group_id");
        let method_id: String = row.get("method_id");
        let centroid_lat: f64 = row.get("centroid_lat");
        let centroid_lon: f64 = row.get("centroid_lon");
        let version: i32 = row.get("version");

        // Store group info
        existing_groups.insert(
            group_id.clone(),
            (method_id, centroid_lat, centroid_lon, version),
        );

        // Parse match values to extract entities
        let match_values_json: serde_json::Value = row.get("match_values");
        if let Ok(match_values) = serde_json::from_value::<MatchValues>(match_values_json.clone()) {
            if let MatchValues::Geospatial(values) = match_values {
                let entities = values
                    .into_iter()
                    .map(|v| (v.entity_id, v.latitude, v.longitude))
                    .collect();
                group_entities.insert(group_id, entities);
            }
        }
    }

    Ok(GroupResults {
        groups: existing_groups,
        group_entities,
    })
}
