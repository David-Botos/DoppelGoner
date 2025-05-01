use crate::{
    db::PgPool,
    models::{MatchMethodType, ServiceMatchStatus},
    results::ServiceMatchResult,
};
use anyhow::{Context, Result};
use log::{info, warn};
use std::collections::{HashMap, HashSet};
use tokio_postgres::types::Type;
use uuid::Uuid;

pub async fn match_services(pool: &PgPool) -> Result<ServiceMatchResult> {
    let mut conn = pool.get().await.context("Failed to get DB connection")?;
    let mut total_matches = 0;
    let mut clusters_with_matches = HashSet::new();

    // For tracking similarity statistics
    let mut similarity_sum = 0.0;
    let mut high_similarity_matches = 0; // >= 0.9
    let mut medium_similarity_matches = 0; // >= 0.85 and < 0.9
    let mut low_similarity_matches = 0; // < 0.85 (shouldn't happen with our threshold, but included for completeness)

    info!("Starting service matching based on semantic and geospatial similarity");

    // Process each cluster
    let clusters_query = "SELECT id FROM group_cluster";
    let clusters_rows = conn
        .query(clusters_query, &[])
        .await
        .context("Failed to fetch clusters")?;

    info!(
        "Processing {} entity clusters for service matching",
        clusters_rows.len()
    );

    for (idx, row) in clusters_rows.iter().enumerate() {
        let cluster_id: String = row.get(0);

        // Get all services associated with entities in this cluster
        let services_query = r#"
            SELECT DISTINCT s.id 
            FROM service s
            JOIN entity_feature ef ON s.id = ef.table_id
            JOIN entity e ON ef.entity_id = e.id
            JOIN group_entity ge ON e.id = ge.entity_id
            JOIN entity_group eg ON ge.entity_group_id = eg.id
            WHERE eg.group_cluster_id = $1
              AND ef.table_name = 'service'
              AND s.embedding_v2 IS NOT NULL
        "#;

        let services_rows = conn
            .query(services_query, &[&cluster_id])
            .await
            .context("Failed to fetch services for cluster")?;

        if services_rows.is_empty() {
            continue;
        }

        let mut services = Vec::new();
        for row in &services_rows {
            services.push(row.get::<_, String>(0));
        }

        info!(
            "Cluster {}/{}: Processing {} services",
            idx + 1,
            clusters_rows.len(),
            services.len()
        );
        let mut cluster_had_matches = false;

        // Compare each service with other services in the same cluster
        for i in 0..services.len() {
            let service_id = &services[i];

            // Find semantically similar services using pgvector
            let similar_services_query = r#"
                WITH cluster_services AS (
                    SELECT DISTINCT s.id
                    FROM service s
                    JOIN entity_feature ef ON s.id = ef.table_id
                    JOIN entity e ON ef.entity_id = e.id
                    JOIN group_entity ge ON e.id = ge.entity_id
                    JOIN entity_group eg ON ge.entity_group_id = eg.id
                    WHERE eg.group_cluster_id = $1
                      AND ef.table_name = 'service'
                      AND s.embedding_v2 IS NOT NULL
                      AND s.id != $2
                )
                SELECT 
                    s2.id,
                    1 - (s1.embedding_v2 <=> s2.embedding_v2) AS similarity
                FROM 
                    service s1,
                    service s2
                WHERE 
                    s1.id = $2
                    AND s2.id IN (SELECT id FROM cluster_services)
                    AND s1.embedding_v2 IS NOT NULL
                    AND s2.embedding_v2 IS NOT NULL
                    AND 1 - (s1.embedding_v2 <=> s2.embedding_v2) >= 0.85
                ORDER BY 
                    s1.embedding_v2 <=> s2.embedding_v2
                LIMIT 20
            "#;

            let similar_services_rows = conn
                .query(similar_services_query, &[&cluster_id, &service_id])
                .await
                .context("Failed to find similar services")?;

            // For each semantically similar service, check geospatial proximity
            for similar_row in similar_services_rows {
                let similar_id: String = similar_row.get(0);
                let similarity: f64 = similar_row.get(1);

                // Only process pairs where service_id < similar_id to avoid duplicates
                if service_id >= &similar_id {
                    continue;
                }

                // Track similarity statistics
                similarity_sum += similarity;
                if similarity >= 0.9 {
                    high_similarity_matches += 1;
                } else if similarity >= 0.85 {
                    medium_similarity_matches += 1;
                } else {
                    low_similarity_matches += 1;
                }

                // Check if both services have locations and they are near each other
                let location_check_query = r#"
                    WITH service1_locations AS (
                        SELECT 
                            l.id,
                            l.latitude,
                            l.longitude
                        FROM 
                            service_at_location sal
                        JOIN 
                            location l ON sal.location_id = l.id
                        WHERE 
                            sal.service_id = $1
                            AND l.latitude IS NOT NULL
                            AND l.longitude IS NOT NULL
                    ),
                    service2_locations AS (
                        SELECT 
                            l.id,
                            l.latitude,
                            l.longitude
                        FROM 
                            service_at_location sal
                        JOIN 
                            location l ON sal.location_id = l.id
                        WHERE 
                            sal.service_id = $2
                            AND l.latitude IS NOT NULL
                            AND l.longitude IS NOT NULL
                    ),
                    proximity_check AS (
                        SELECT 
                            COUNT(*) AS nearby_count
                        FROM 
                            service1_locations s1l,
                            service2_locations s2l
                        WHERE 
                            ST_DWithin(
                                ST_SetSRID(ST_MakePoint(s1l.longitude, s1l.latitude), 4326)::geography,
                                ST_SetSRID(ST_MakePoint(s2l.longitude, s2l.latitude), 4326)::geography,
                                2000  -- 2km threshold
                            )
                    )
                    SELECT
                        (SELECT COUNT(*) FROM service1_locations) AS s1_loc_count,
                        (SELECT COUNT(*) FROM service2_locations) AS s2_loc_count,
                        (SELECT nearby_count FROM proximity_check) AS nearby_count
                "#;

                let location_check_row = conn
                    .query_one(location_check_query, &[&service_id, &similar_id])
                    .await
                    .context("Failed to check service locations")?;

                let s1_loc_count: i64 = location_check_row.get(0);
                let s2_loc_count: i64 = location_check_row.get(1);
                let nearby_count: i64 = location_check_row.get(2);

                let s1_has_locations = s1_loc_count > 0;
                let s2_has_locations = s2_loc_count > 0;
                let has_nearby_locations = nearby_count > 0;

                // Create a match if:
                // 1. Both services have locations and at least one pair is nearby, or
                // 2. At least one service doesn't have location data (can't verify proximity)
                let should_match = if s1_has_locations && s2_has_locations {
                    // Both have locations, check if any are nearby
                    has_nearby_locations
                } else {
                    // At least one doesn't have locations, match based on semantics only
                    true
                };

                if should_match {
                    // Check if the match already exists
                    let existing_match_query = r#"
                        SELECT id 
                        FROM service_match 
                        WHERE 
                            (service_id_1 = $1 AND service_id_2 = $2)
                            OR (service_id_1 = $2 AND service_id_2 = $1)
                    "#;

                    let existing_match_rows = conn
                        .query(existing_match_query, &[&service_id, &similar_id])
                        .await
                        .context("Failed to check for existing match")?;

                    if existing_match_rows.is_empty() {
                        // Create the match record
                        let match_id = Uuid::new_v4().to_string();
                        let insert_match_query = r#"
                            INSERT INTO service_match (
                                id, 
                                group_cluster_id, 
                                service_id_1, 
                                service_id_2, 
                                similarity_score, 
                                match_reasons,
                                status,
                                created_at
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, 'potential', CURRENT_TIMESTAMP)
                        "#;

                        conn.execute(
                            insert_match_query,
                            &[
                                &match_id,
                                &cluster_id,
                                &service_id,
                                &similar_id,
                                &(similarity as f32),
                                &"Semantic and geospatial similarity",
                            ],
                        )
                        .await
                        .context("Failed to insert service match")?;

                        total_matches += 1;
                        cluster_had_matches = true;
                    }
                }
            }
        }

        if cluster_had_matches {
            clusters_with_matches.insert(cluster_id);
        }
    }

    // Calculate average similarity score
    let avg_similarity = if total_matches > 0 {
        similarity_sum / (total_matches as f64)
    } else {
        0.0
    };

    // Update the service_match_stats table
    if total_matches > 0 {
        let stats_id = Uuid::new_v4().to_string();
        let pipeline_run_id = "current_run"; // This should be passed in or retrieved from the database

        let insert_stats_query = r#"
            INSERT INTO clustering_metadata.service_match_stats (
                id,
                pipeline_run_id,
                total_matches,
                avg_similarity,
                high_similarity_matches,
                medium_similarity_matches,
                low_similarity_matches,
                clusters_with_matches,
                created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
        "#;

        conn.execute(
            insert_stats_query,
            &[
                &stats_id,
                &pipeline_run_id,
                &(total_matches as i64),
                &avg_similarity,
                &(high_similarity_matches as i64),
                &(medium_similarity_matches as i64),
                &(low_similarity_matches as i64),
                &(clusters_with_matches.len() as i64),
            ],
        )
        .await
        .context("Failed to insert service match stats")?;
    }

    info!(
        "Service matching completed: {} new potential matches found across {} clusters",
        total_matches,
        clusters_with_matches.len()
    );

    // Return ServiceMatchResult
    Ok(ServiceMatchResult {
        groups_created: total_matches,
        stats: crate::results::MatchMethodStats {
            method_type: MatchMethodType::Custom("semantic_geospatial".to_string()),
            groups_created: total_matches,
            entities_matched: total_matches * 2, // Each match connects two services
            avg_confidence: avg_similarity,
            avg_group_size: 2.0, // Each match always involves 2 services
        },
    })
}
