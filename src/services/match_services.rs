// src/services/match_services.rs
use crate::db::PgPool;
use anyhow::{Context, Result};
use log::{debug, info};

/// Minimum similarity threshold (1 - cosine distance)
/// A lower threshold (e.g., 0.85) might be more appropriate for catching more potential matches
const SIMILARITY_THRESHOLD: f32 = 0.85;

/// Entry point for vector-based service matching
pub async fn find_service_matches(pool: &PgPool) -> Result<usize> {
    let conn = pool.get().await.context("Failed to get DB connection")?;

    // First, ensure the service_cluster table has been properly populated
    let cluster_count = conn
        .query_one(
            "SELECT COUNT(DISTINCT cluster_id) FROM service_cluster",
            &[],
        )
        .await
        .context("Failed to count clusters")?;

    let cluster_count: i64 = cluster_count.get(0);
    info!(
        "Found {} clusters with services for matching",
        cluster_count
    );

    // Get distinct cluster IDs
    let cluster_rows = conn
        .query("SELECT DISTINCT cluster_id FROM service_cluster", &[])
        .await
        .context("Failed to fetch cluster IDs")?;

    let mut total_matches = 0;
    let total_clusters = cluster_rows.len();
    info!(
        "Processing {} clusters for service matching",
        total_clusters
    );

    for (idx, row) in cluster_rows.iter().enumerate() {
        let cluster_id: String = row.get("cluster_id");

        // Get count of services in this cluster
        let service_count = conn
            .query_one(
                "SELECT COUNT(*) FROM service_cluster WHERE cluster_id = $1",
                &[&cluster_id],
            )
            .await
            .context(format!(
                "Failed to count services in cluster {}",
                cluster_id
            ))?;

        let service_count: i64 = service_count.get(0);
        debug!("Cluster {} has {} services", cluster_id, service_count);

        // Only process clusters with at least 2 services
        if service_count < 2 {
            debug!(
                "Skipping cluster {} with only {} service",
                cluster_id, service_count
            );
            continue;
        }

        // Run vector search for this cluster
        let matches = match_cluster_services(&conn, &cluster_id)
            .await
            .context(format!(
                "Failed to match services in cluster {}",
                cluster_id
            ))?;

        total_matches += matches;

        if (idx + 1) % 10 == 0 || idx + 1 == total_clusters {
            info!(
                "Processed {}/{} clusters ({:.1}%), found {} matches so far",
                idx + 1,
                total_clusters,
                (idx + 1) as f32 / total_clusters as f32 * 100.0,
                total_matches
            );
        }
    }

    info!(
        "Service matching complete. Created {} new service matches",
        total_matches
    );
    Ok(total_matches)
}

async fn match_cluster_services(conn: &tokio_postgres::Client, cluster_id: &str) -> Result<usize> {
    // Query all service IDs in this cluster
    let services_rows = conn
        .query(
            r#"
            SELECT s.id, s.embedding
            FROM service s
            JOIN service_cluster sc ON s.id = sc.service_id
            WHERE sc.cluster_id = $1 AND s.embedding IS NOT NULL
            "#,
            &[&cluster_id],
        )
        .await
        .context("Failed to fetch services with embeddings")?;

    let services: Vec<(String, Vec<f32>)> = services_rows
        .into_iter()
        .map(|row| {
            let id: String = row.get("id");
            let embedding: Vec<f32> = row.get("embedding");
            (id, embedding)
        })
        .collect();

    let service_count = services.len();
    if service_count == 0 {
        debug!(
            "No services with embeddings found in cluster {}",
            cluster_id
        );
        return Ok(0);
    }

    debug!(
        "Processing {} services in cluster {}",
        service_count, cluster_id
    );
    let mut inserted = 0;

    for (service_id, embedding) in &services {
        // Find nearest neighbors for this service (within same cluster)
        // We're using the PostgreSQL vector operators (<=> is cosine distance)
        let neighbors = conn
            .query(
                r#"
                SELECT s.id, 1 - (s.embedding <=> $1) AS similarity
                FROM service s
                JOIN service_cluster sc ON s.id = sc.service_id
                WHERE sc.cluster_id = $2
                  AND s.id != $3
                  AND s.embedding IS NOT NULL
                ORDER BY s.embedding <=> $1
                LIMIT 20  -- Increased from 10 to catch more potential matches
                "#,
                &[&embedding, &cluster_id, &service_id],
            )
            .await
            .context("Failed ANN query")?;

        for row in neighbors {
            let other_id: String = row.get("id");
            let similarity: f32 = row.get("similarity");

            if similarity >= SIMILARITY_THRESHOLD {
                // Ensure ordered IDs to prevent duplicates
                let (id1, id2) = if service_id < &other_id {
                    (service_id, &other_id)
                } else {
                    (&other_id, service_id)
                };

                // Insert the match with the specified match_reason of "ANN ML"
                let result = conn
                    .execute(
                        r#"
                    INSERT INTO service_match 
                    VALUES (
                        gen_random_uuid(), $1, $2, $3, $4, 
                        ARRAY['ANN ML'], NOW(), 'potential'
                    )
                    ON CONFLICT (service_id_1, service_id_2) DO NOTHING
                    "#,
                        &[&cluster_id, id1, id2, &similarity],
                    )
                    .await
                    .context("Failed to insert service match")?;

                inserted += result as usize;
            }
        }
    }

    if inserted > 0 {
        debug!(
            "Created {} service matches in cluster {}",
            inserted, cluster_id
        );
    }

    Ok(inserted)
}
