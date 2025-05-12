use crate::{
    db::PgPool,
    models::{GroupClusterId, MatchMethodType, ServiceId, ServiceMatchStatus},
    results::{MatchMethodStats, ServiceMatchResult},
};
use anyhow::{Context, Result};
use candle_core::{Device, IndexOp, Tensor, D};
use log::{debug, info, warn};
use pgvector::Vector; // Assuming you use the pgvector crate for fetching vectors
use std::collections::HashSet;
use uuid::Uuid;

// Helper function to select Candle device (GPU if available, else CPU)
fn candle_device() -> Result<Device> {
    // Try Metal first (for Apple Silicon), then CUDA, then fallback to CPU
    // Adjust order or add specific checks based on your target hardware
    if candle_core::utils::metal_is_available() {
        return Device::new_metal(0).context("Failed to initialize Metal device");
    }
    if candle_core::utils::cuda_is_available() {
        return Device::new_cuda(0).context("Failed to initialize CUDA device");
    }
    info!("No GPU backend (Metal or CUDA) found/enabled for Candle. Using CPU.");
    Ok(Device::Cpu)
}

pub async fn match_services(pool: &PgPool) -> Result<ServiceMatchResult> {
    let mut conn = pool
        .get()
        .await
        .context("Failed to get DB connection for service matching")?;
    let mut total_matches_created = 0;
    let mut clusters_with_new_matches = HashSet::new();

    let mut new_match_similarity_sum = 0.0;
    let mut new_match_count_for_avg = 0;

    // Initialize Candle device
    let device = candle_device().context("Failed to select Candle device")?;
    info!("Using Candle device: {:?}", device.location());

    info!("Starting service matching based on semantic (batched) and geospatial similarity for consolidated clusters");

    let clusters_query = "SELECT id FROM public.group_cluster";
    let cluster_rows = conn
        .query(clusters_query, &[])
        .await
        .context("Failed to fetch group_cluster IDs")?;

    if cluster_rows.is_empty() {
        info!("No consolidated clusters found to process for service matching.");
        return Ok(ServiceMatchResult {
            groups_created: 0,
            stats: MatchMethodStats {
                method_type: MatchMethodType::Custom("semantic_geospatial_candle".to_string()),
                groups_created: 0,
                entities_matched: 0,
                avg_confidence: 0.0,
                avg_group_size: 2.0,
            },
        });
    }

    info!(
        "Processing {} consolidated entity clusters for service matching using Candle on device {:?}.",
        cluster_rows.len(),
        device.location()
    );

    for (idx, cluster_row) in cluster_rows.iter().enumerate() {
        let cluster_id_str: String = cluster_row.get(0);
        let group_cluster_id = GroupClusterId(cluster_id_str.clone());

        // Fetch service IDs and their embeddings for the current cluster
        let services_with_embeddings_query = r#"
            SELECT s.id, s.embedding_v2
            FROM public.service s
            JOIN public.entity_feature ef ON s.id = ef.table_id AND ef.table_name = 'service'
            WHERE ef.entity_id IN (
                SELECT entity_id_1 FROM public.entity_group WHERE group_cluster_id = $1
                UNION
                SELECT entity_id_2 FROM public.entity_group WHERE group_cluster_id = $1
            )
            AND s.embedding_v2 IS NOT NULL
        "#;

        let service_embedding_rows = conn
            .query(services_with_embeddings_query, &[&group_cluster_id.0])
            .await
            .context(format!(
                "Failed to fetch services and embeddings for cluster {}",
                group_cluster_id.0
            ))?;

        if service_embedding_rows.is_empty() {
            debug!(
                "No services with embeddings found for cluster {}",
                group_cluster_id.0
            );
            continue;
        }

        let mut service_data: Vec<(ServiceId, Vec<f32>)> = Vec::new();
        let mut expected_embedding_dim: Option<usize> = None;

        for row in service_embedding_rows {
            let service_id_str: String = row.get("id");
            let pg_embedding: Vector = row.get("embedding_v2");
            let embedding_vec = pg_embedding.to_vec(); // Consumes pgvector::Vector

            if embedding_vec.is_empty() {
                debug!(
                    "Service {} in cluster {} has an empty embedding. Skipping.",
                    service_id_str, group_cluster_id.0
                );
                continue;
            }

            if expected_embedding_dim.is_none() {
                expected_embedding_dim = Some(embedding_vec.len());
            } else if expected_embedding_dim.unwrap() != embedding_vec.len() {
                warn!(
                    "Service {} in cluster {} has an inconsistent embedding dimension. Expected {}, got {}. Skipping.",
                    service_id_str, group_cluster_id.0, expected_embedding_dim.unwrap(), embedding_vec.len()
                );
                continue;
            }
            service_data.push((ServiceId(service_id_str), embedding_vec));
        }

        if service_data.len() < 2 {
            debug!(
                "Cluster {} has fewer than 2 services with valid, consistent embeddings. Skipping pairwise comparison.",
                group_cluster_id.0
            );
            continue;
        }

        info!(
            "Cluster {}/{} (ID: {}): Processing {} services with embeddings for batch similarity.",
            idx + 1,
            cluster_rows.len(),
            group_cluster_id.0,
            service_data.len()
        );

        let num_services = service_data.len();
        let embedding_dim = expected_embedding_dim.unwrap(); // Safe due to len < 2 check

        // Prepare data for Candle tensor
        let mut flat_embeddings_data = Vec::with_capacity(num_services * embedding_dim);
        for (_, embedding_vec) in &service_data {
            flat_embeddings_data.extend_from_slice(embedding_vec);
        }

        let embeddings_tensor =
            Tensor::from_vec(flat_embeddings_data, (num_services, embedding_dim), &device)
                .context("Failed to create Candle tensor from embeddings")?;

        // Calculate cosine similarity matrix
        // 1. L2 normalize each embedding vector (row)
        // norms = sqrt(sum(embeddings_tensor^2, dim=1, keepdim=True))
        let norms = embeddings_tensor
            .sqr()?
            .sum_keepdim(D::Minus1)? // Sum along the embedding dimension
            .sqrt()?;
        // normalized_embeddings = embeddings_tensor / norms
        let normalized_embeddings = embeddings_tensor.broadcast_div(&norms)?;

        // 2. similarity_matrix = normalized_embeddings @ normalized_embeddings.T
        let similarity_matrix = normalized_embeddings
            .matmul(&normalized_embeddings.transpose(D::Minus2, D::Minus1)?)?;

        let mut cluster_had_new_matches = false;

        for i in 0..num_services {
            for j in (i + 1)..num_services {
                // Iterate upper triangle to avoid duplicates and self-comparison
                let service1_id = &service_data[i].0;
                let service2_id = &service_data[j].0;

                // Retrieve similarity from the Candle tensor
                let similarity_val = similarity_matrix
                    .i((i, j))? // Access element at (row_i, col_j)
                    .to_scalar::<f32>()
                    .context(format!(
                        "Failed to get similarity score from tensor for pair ({}, {})",
                        i, j
                    ))?;
                let similarity = similarity_val as f64;

                if similarity < 0.85 {
                    // Semantic threshold
                    continue;
                }

                // Geospatial proximity check (remains the same)
                let location_check_query = r#"
                    WITH service1_locations AS (
                        SELECT l.latitude, l.longitude FROM public.service_at_location sal
                        JOIN public.location l ON sal.location_id = l.id
                        WHERE sal.service_id = $1 AND l.latitude IS NOT NULL AND l.longitude IS NOT NULL
                    ),
                    service2_locations AS (
                        SELECT l.latitude, l.longitude FROM public.service_at_location sal
                        JOIN public.location l ON sal.location_id = l.id
                        WHERE sal.service_id = $2 AND l.latitude IS NOT NULL AND l.longitude IS NOT NULL
                    )
                    SELECT
                        (SELECT COUNT(*) FROM service1_locations) > 0 AS s1_has_loc,
                        (SELECT COUNT(*) FROM service2_locations) > 0 AS s2_has_loc,
                        EXISTS (
                            SELECT 1 FROM service1_locations s1l, service2_locations s2l
                            WHERE ST_DWithin(
                                ST_SetSRID(ST_MakePoint(s1l.longitude, s1l.latitude), 4326)::geography,
                                ST_SetSRID(ST_MakePoint(s2l.longitude, s2l.latitude), 4326)::geography,
                                2000  -- 2km threshold
                            )
                        ) AS nearby
                "#;
                let loc_check_row = conn
                    .query_one(location_check_query, &[&service1_id.0, &service2_id.0])
                    .await
                    .context(format!(
                        "Failed to check service locations for {} and {}",
                        service1_id.0, service2_id.0
                    ))?;

                let s1_has_locations: bool = loc_check_row.get("s1_has_loc");
                let s2_has_locations: bool = loc_check_row.get("s2_has_loc");
                let has_nearby_locations: bool = loc_check_row.get("nearby");

                let should_match = if s1_has_locations && s2_has_locations {
                    has_nearby_locations
                } else {
                    true // Match on semantics if one or both lack location
                };

                if should_match {
                    let existing_match_query = r#"
                        SELECT id FROM public.service_match
                        WHERE (service_id_1 = $1 AND service_id_2 = $2) OR (service_id_1 = $2 AND service_id_2 = $1)
                    "#;
                    let existing_match = conn
                        .query_opt(existing_match_query, &[&service1_id.0, &service2_id.0])
                        .await
                        .context("Failed to check for existing service match")?;

                    if existing_match.is_none() {
                        let match_id = Uuid::new_v4().to_string();
                        let insert_match_query = r#"
                            INSERT INTO public.service_match
                            (id, group_cluster_id, service_id_1, service_id_2, similarity_score, match_reasons, status, created_at)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)
                        "#;
                        conn.execute(
                            insert_match_query,
                            &[
                                &match_id,
                                &group_cluster_id.0,
                                &service1_id.0,
                                &service2_id.0,
                                &(similarity as f32), // Store the calculated similarity
                                &"Semantic (Candle) and geospatial similarity",
                                &ServiceMatchStatus::Potential.as_str(),
                            ],
                        )
                        .await
                        .context("Failed to insert service match")?;

                        total_matches_created += 1;
                        new_match_similarity_sum += similarity;
                        new_match_count_for_avg += 1;
                        cluster_had_new_matches = true;
                    }
                }
            }
        }
        if cluster_had_new_matches {
            clusters_with_new_matches.insert(group_cluster_id.0.clone());
        }
    }

    let avg_similarity_of_new_matches = if new_match_count_for_avg > 0 {
        new_match_similarity_sum / (new_match_count_for_avg as f64)
    } else {
        0.0
    };

    info!(
        "Service matching completed: {} new potential service matches found across {} clusters using Candle.",
        total_matches_created,
        clusters_with_new_matches.len()
    );

    Ok(ServiceMatchResult {
        groups_created: total_matches_created,
        stats: MatchMethodStats {
            method_type: MatchMethodType::Custom("semantic_geospatial_candle".to_string()),
            groups_created: total_matches_created,
            entities_matched: total_matches_created * 2,
            avg_confidence: avg_similarity_of_new_matches,
            avg_group_size: 2.0,
        },
    })
}
