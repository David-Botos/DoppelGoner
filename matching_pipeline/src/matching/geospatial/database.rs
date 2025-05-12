// src/matching/geospatial/database.rs

use anyhow::{Context, Result};
use log::debug;
use std::collections::{HashMap, HashSet};
use tokio_postgres::Client; // Removed Statement and Transaction as they are not used directly for prepare here

use crate::models::{EntityGroupId, EntityId, GeospatialMatchValue, LocationResults, MatchValues};

// Type alias for clarity when fetching existing geospatial pairs
pub type ExistingGeoPairsMap =
    HashMap<EntityGroupId, (EntityId, EntityId, GeospatialMatchValue, Option<f64>)>;

/// Struct to hold SQL query strings for pairwise geospatial matching.
/// These are no longer prepared statements but direct query strings.
#[derive(Debug, Clone)]
pub struct DbStatements {
    /// SQL to insert a new pairwise entity_group for a geospatial match.
    pub new_geospatial_pair_sql: &'static str,

    /// SQL for consistency check: if an entity is already in ANY geospatial pair.
    pub consistency_check_sql: &'static str,

    /// PostGIS distance calculation SQL (point-to-point).
    pub distance_sql: &'static str,
}

/// Returns a struct containing the SQL query strings.
/// This function no longer prepares statements or requires a transaction.
pub fn get_statements() -> DbStatements {
    DbStatements {
        new_geospatial_pair_sql: "
            INSERT INTO entity_group 
            (id, entity_id_1, entity_id_2, method_type, match_values, confidence_score, created_at, updated_at, version) 
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 1)
        ",
        consistency_check_sql: "
            SELECT COUNT(*) 
            FROM entity_group
            WHERE (entity_id_1 = $1 OR entity_id_2 = $1)
            AND method_type = 'geospatial'
        ",
        distance_sql: "
            SELECT ST_Distance(
                ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography,
                ST_SetSRID(ST_MakePoint($3, $4), 4326)::geography
            ) AS distance
        ",
    }
}

/// Get all entities that are already part of *any* pair in the entity_group table.
/// Uses a direct connection client.
pub async fn get_globally_paired_entities(conn: &Client) -> Result<HashSet<EntityId>> {
    const QUERY: &str = "
        SELECT entity_id_1 AS entity_id FROM entity_group
        UNION
        SELECT entity_id_2 AS entity_id FROM entity_group
    ";
    let rows = conn
        .query(QUERY, &[])
        .await
        .context("Failed to query globally paired entities")?;
    Ok(rows
        .iter()
        .map(|row| EntityId(row.get("entity_id")))
        .collect())
}

/// Query for locations of entities that are not yet part of *any* pair.
/// Uses a direct connection client.
pub async fn get_locations_for_unpaired_entities(conn: &Client) -> Result<LocationResults> {
    const LOCATION_QUERY: &str = "
        SELECT 
            e.id AS entity_id,
            l.latitude,
            l.longitude,
            (l.geom IS NOT NULL) AS has_geom 
        FROM 
            entity e
            INNER JOIN entity_feature ef ON e.id = ef.entity_id AND ef.table_name = 'location'
            INNER JOIN location l ON ef.table_id = l.id
        WHERE 
            l.latitude IS NOT NULL AND l.longitude IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM entity_group eg
                WHERE eg.entity_id_1 = e.id OR eg.entity_id_2 = e.id
            )
    ";
    debug!("Executing query for locations of unpaired entities.");
    let location_rows = conn
        .query(LOCATION_QUERY, &[])
        .await
        .context("Failed to query locations for unpaired entities")?;

    let has_postgis = if !location_rows.is_empty() {
        // Check the first row to determine if PostGIS geometries are generally available.
        // This assumes consistency in the 'has_geom' column for the dataset.
        location_rows[0].get("has_geom")
    } else {
        false // If no rows, assume no PostGIS or no data to infer from.
    };
    let new_locations: Vec<(EntityId, f64, f64)> = location_rows
        .iter()
        .map(|row| {
            (
                EntityId(row.get("entity_id")),
                row.get("latitude"),
                row.get("longitude"),
            )
        })
        .collect();
    Ok(LocationResults {
        locations: new_locations,
        has_postgis,
    })
}

/// Load existing *geospatial pairs* from the entity_group table.
/// Uses a direct connection client.
pub async fn get_existing_geospatial_pairs(conn: &Client) -> Result<ExistingGeoPairsMap> {
    const QUERY: &str = "
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
    ";
    let rows = conn
        .query(QUERY, &[])
        .await
        .context("Failed to query existing geospatial pairs from entity_group")?;

    let mut existing_pairs: ExistingGeoPairsMap = HashMap::new();
    for row in &rows {
        let pair_id_str: String = row.get("pair_id");
        let entity_id_1_str: String = row.get("entity_id_1");
        let entity_id_2_str: String = row.get("entity_id_2");
        let confidence_score: Option<f64> = row.get("confidence_score");
        let match_values_json: serde_json::Value = row.get("match_values");

        // Deserialize the match_values JSON into the MatchValues enum,
        // then extract the GeospatialMatchValue variant.
        match serde_json::from_value::<MatchValues>(match_values_json.clone()) {
            Ok(MatchValues::Geospatial(geospatial_value)) => {
                existing_pairs.insert(
                    EntityGroupId(pair_id_str),
                    (
                        EntityId(entity_id_1_str),
                        EntityId(entity_id_2_str),
                        geospatial_value, // This is the GeospatialMatchValue struct
                        confidence_score,
                    ),
                );
            }
            Ok(other_type) => {
                // Log if the match_values are not of the expected Geospatial type.
                log::warn!(
                    "Parsed match_values for geospatial pair_id {} but was not Geospatial type as expected, found: {:?}",
                    pair_id_str, other_type
                );
            }
            Err(e) => {
                // Log if deserialization fails.
                log::warn!(
                    "Failed to parse GeospatialMatchValue for entity_group_id {}. Error: {}. JSON: {}",
                    pair_id_str, e, match_values_json
                );
            }
        }
    }
    debug!(
        "Fetched {} existing geospatial pairs.",
        existing_pairs.len()
    );
    Ok(existing_pairs)
}
