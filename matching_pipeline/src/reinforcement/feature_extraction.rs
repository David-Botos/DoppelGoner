use anyhow::Result;
use log::{debug, info, warn};
use tokio_postgres::Client as PgConnection;
use uuid::Uuid;

use super::types::FeatureMetadata;
use crate::db::PgPool;
use crate::models::EntityId;

// Feature metadata for documentation
pub fn get_feature_metadata() -> Vec<FeatureMetadata> {
    vec![
        // Basic features
        FeatureMetadata {
            name: "name_complexity".to_string(),
            description: "Complexity of the organization name (based on length, word count, etc.)"
                .to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "data_completeness".to_string(),
            description: "Ratio of populated fields to total fields for this entity".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "has_email".to_string(),
            description: "Whether the entity has an email (0 = no, 1 = yes)".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "has_phone".to_string(),
            description: "Whether the entity has a phone number (0 = no, 1 = yes)".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "has_url".to_string(),
            description: "Whether the entity has a URL (0 = no, 1 = yes)".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "has_address".to_string(),
            description: "Whether the entity has an address (0 = no, 1 = yes)".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "has_location".to_string(),
            description: "Whether the entity has location coordinates (0 = no, 1 = yes)"
                .to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "organization_size".to_string(),
            description: "Estimated size of the organization based on metadata".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "service_count".to_string(),
            description: "Normalized count of services provided by the entity".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        // Enhanced vector-based features
        FeatureMetadata {
            name: "embedding_centroid_distance".to_string(),
            description: "Distance from this organization's embedding to the domain centroid"
                .to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "service_semantic_coherence".to_string(),
            description: "Semantic similarity between services offered by this entity".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "embedding_quality".to_string(),
            description:
                "Estimated quality of the embedding based on description length and content"
                    .to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        // Pairwise features (only in context_for_pair)
        FeatureMetadata {
            name: "name_similarity".to_string(),
            description: "String similarity between entity names".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "embedding_similarity".to_string(),
            description: "Cosine similarity between entity embeddings".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "max_service_similarity".to_string(),
            description: "Maximum similarity between any services from the two entities"
                .to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "geographic_distance".to_string(),
            description: "Normalized geographic distance between entities".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "shared_domain".to_string(),
            description: "Whether entities share the same domain (0 = no, 1 = yes)".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "shared_phone".to_string(),
            description: "Whether entities share a phone number (0 = no, 1 = yes)".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "service_geo_semantic_score".to_string(),
            description: "Combined score of service similarity and location proximity".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
    ]
}

// Main function to extract all features for a single entity
pub async fn extract_entity_features(
    conn: &PgConnection,
    entity_id: &EntityId,
) -> Result<Vec<f64>> {
    // Extract features from entity data
    let features = vec![
        // Basic features
        extract_name_complexity(&conn, entity_id).await?,
        extract_data_completeness(&conn, entity_id).await?,
        extract_has_email(&conn, entity_id).await?,
        extract_has_phone(&conn, entity_id).await?,
        extract_has_url(&conn, entity_id).await?,
        extract_has_address(&conn, entity_id).await?,
        extract_has_location(&conn, entity_id).await?,
        extract_organization_size(&conn, entity_id).await?,
        extract_service_count(&conn, entity_id).await?,
        // Enhanced vector-based features
        extract_embedding_centroid_distance(&conn, entity_id).await?,
        extract_service_semantic_coherence(&conn, entity_id).await?,
        extract_embedding_quality_score(&conn, entity_id).await?,
    ];

    // Store features for later use
    store_entity_features(&conn, entity_id, &features).await?;

    Ok(features)
}

// Function to extract context features for a pair of entities
pub async fn extract_context_for_pair(
    pool: &PgPool,
    entity1: &EntityId,
    entity2: &EntityId,
) -> Result<Vec<f64>> {
    let conn = pool.get().await?;

    // Extract features about the pair relationship
    let pair_features = vec![
        calculate_name_similarity(&conn, entity1, entity2).await?,
        calculate_embedding_similarity(&conn, entity1, entity2).await?,
        calculate_max_service_similarity(&conn, entity1, entity2).await?,
        calculate_geographic_distance(&conn, entity1, entity2).await?,
        check_shared_domain(&conn, entity1, entity2).await?,
        check_shared_phone(&conn, entity1, entity2).await?,
        calculate_service_geo_semantic_score(&conn, entity1, entity2).await?,
    ];

    // Combine with individual entity features
    let entity1_features = get_stored_entity_features(&conn, entity1).await?;
    let entity2_features = get_stored_entity_features(&conn, entity2).await?;

    // Combine all features
    let mut context = Vec::new();
    context.extend(entity1_features);
    context.extend(entity2_features);
    context.extend(pair_features);

    Ok(context)
}

// Implementation of basic entity feature extraction functions
async fn extract_name_complexity(conn: &PgConnection, entity_id: &EntityId) -> Result<f64> {
    let row = conn
        .query_one(
            "SELECT o.name 
         FROM entity e
         JOIN organization o ON e.organization_id = o.id
         WHERE e.id = $1",
            &[&entity_id],
        )
        .await?;

    let name: Option<String> = row.get(0);

    if let Some(name) = name {
        // Calculate complexity based on length and word count
        let length = name.len() as f64;
        let word_count = name.split_whitespace().count() as f64;

        // Normalize: longer names with more words are more complex
        let complexity = (length / 100.0).min(1.0) * 0.5 + (word_count / 10.0).min(1.0) * 0.5;

        Ok(complexity)
    } else {
        Ok(0.0)
    }
}

async fn extract_data_completeness(conn: &PgConnection, entity_id: &EntityId) -> Result<f64> {
    let row = conn
        .query_one(
            "SELECT 
            (CASE WHEN o.name IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN o.description IS NOT NULL AND LENGTH(o.description) > 5 THEN 1 ELSE 0 END +
             CASE WHEN o.email IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN o.url IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN o.tax_id IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN o.legal_status IS NOT NULL THEN 1 ELSE 0 END)::float / 6 as completeness
         FROM entity e
         JOIN organization o ON e.organization_id = o.id
         WHERE e.id = $1",
            &[&entity_id],
        )
        .await?;

    let completeness: f64 = row.get(0);
    Ok(completeness)
}

async fn extract_has_email(conn: &PgConnection, entity_id: &EntityId) -> Result<f64> {
    let row = conn
        .query_one(
            "SELECT CASE WHEN o.email IS NOT NULL THEN 1.0 ELSE 0.0 END
         FROM entity e
         JOIN organization o ON e.organization_id = o.id
         WHERE e.id = $1",
            &[&entity_id],
        )
        .await?;

    let has_email: f64 = row.get(0);
    Ok(has_email)
}

async fn extract_has_phone(conn: &PgConnection, entity_id: &EntityId) -> Result<f64> {
    let row = conn
        .query_one(
            "SELECT CASE WHEN EXISTS (
            SELECT 1 FROM phone p
            JOIN entity_feature ef ON ef.table_id = p.id
            WHERE ef.entity_id = $1 AND ef.table_name = 'phone'
        ) THEN 1.0 ELSE 0.0 END",
            &[&entity_id],
        )
        .await?;

    let has_phone: f64 = row.get(0);
    Ok(has_phone)
}

async fn extract_has_url(conn: &PgConnection, entity_id: &EntityId) -> Result<f64> {
    let row = conn
        .query_one(
            "SELECT CASE WHEN o.url IS NOT NULL AND LENGTH(o.url) > 0 THEN 1.0 ELSE 0.0 END
         FROM entity e
         JOIN organization o ON e.organization_id = o.id
         WHERE e.id = $1",
            &[&entity_id],
        )
        .await?;

    let has_url: f64 = row.get(0);
    Ok(has_url)
}

async fn extract_has_address(conn: &PgConnection, entity_id: &EntityId) -> Result<f64> {
    let row = conn
        .query_one(
            "SELECT CASE WHEN EXISTS (
            SELECT 1 FROM address a
            JOIN location l ON a.location_id = l.id
            JOIN entity_feature ef ON ef.table_id = l.id
            WHERE ef.entity_id = $1 AND ef.table_name = 'location'
        ) THEN 1.0 ELSE 0.0 END",
            &[&entity_id],
        )
        .await?;

    let has_address: f64 = row.get(0);
    Ok(has_address)
}

async fn extract_has_location(conn: &PgConnection, entity_id: &EntityId) -> Result<f64> {
    let row = conn
        .query_one(
            "SELECT CASE WHEN EXISTS (
            SELECT 1 FROM location l
            JOIN entity_feature ef ON ef.table_id = l.id
            WHERE ef.entity_id = $1 
            AND ef.table_name = 'location'
            AND l.latitude IS NOT NULL 
            AND l.longitude IS NOT NULL
        ) THEN 1.0 ELSE 0.0 END",
            &[&entity_id],
        )
        .await?;

    let has_location: f64 = row.get(0);
    Ok(has_location)
}

async fn extract_organization_size(conn: &PgConnection, entity_id: &EntityId) -> Result<f64> {
    // Estimate organization size based on available data
    let row = conn
        .query_one(
            "SELECT 
            (SELECT COUNT(*) FROM service s
             JOIN entity_feature ef ON ef.table_id = s.id
             WHERE ef.entity_id = $1 AND ef.table_name = 'service') as service_count,
            (SELECT COUNT(*) FROM location l
             JOIN entity_feature ef ON ef.table_id = l.id
             WHERE ef.entity_id = $1 AND ef.table_name = 'location') as location_count,
            (SELECT COUNT(*) FROM phone p
             JOIN entity_feature ef ON ef.table_id = p.id
             WHERE ef.entity_id = $1 AND ef.table_name = 'phone') as phone_count
        ",
            &[&entity_id],
        )
        .await?;

    let service_count: i64 = row.get(0);
    let location_count: i64 = row.get(1);
    let phone_count: i64 = row.get(2);

    // Combine metrics into a single size score (0-1)
    let size_score = ((service_count as f64 / 10.0).min(1.0) * 0.5
        + (location_count as f64 / 5.0).min(1.0) * 0.3
        + (phone_count as f64 / 3.0).min(1.0) * 0.2);

    Ok(size_score)
}

async fn extract_service_count(conn: &PgConnection, entity_id: &EntityId) -> Result<f64> {
    let row = conn
        .query_one(
            "SELECT COUNT(*) FROM service s
         JOIN entity_feature ef ON ef.table_id = s.id
         WHERE ef.entity_id = $1 AND ef.table_name = 'service'",
            &[&entity_id],
        )
        .await?;

    let count: i64 = row.get(0);

    // Normalize: 0-10 services, with 10+ giving 1.0
    let normalized = (count as f64 / 10.0).min(1.0);

    Ok(normalized)
}

// Implementation of enhanced vector-based features
async fn extract_embedding_centroid_distance(
    conn: &PgConnection,
    entity_id: &EntityId,
) -> Result<f64> {
    // Get organization embedding
    let row = conn
        .query_opt(
            "SELECT o.embedding FROM organization o
         JOIN entity e ON e.organization_id = o.id
         WHERE e.id = $1 AND o.embedding IS NOT NULL",
            &[&entity_id],
        )
        .await?;

    if let Some(row) = row {
        // Note: pgvector returns Vec<f32> not Vec<f64>
        let embedding: Option<Vec<f32>> = row.get(0);

        if let Some(embedding) = embedding {
            // Get domain centroid using pgvector's avg function
            let centroid_row = conn
                .query_one(
                    "SELECT avg(embedding) FROM organization 
                 WHERE embedding IS NOT NULL",
                    &[],
                )
                .await?;

            let centroid: Option<Vec<f32>> = centroid_row.get(0);

            if let Some(centroid) = centroid {
                // Calculate distance to domain centroid
                // Lower means more "typical" organization, higher means more unique
                let distance_query = conn
                    .query_one(
                        "SELECT 1 - ($1::vector <=> $2::vector) as similarity",
                        &[&embedding, &centroid],
                    )
                    .await?;

                return Ok(distance_query.get(0));
            }
        }
    }

    Ok(0.0) // Default if no embedding
}

async fn extract_service_semantic_coherence(
    conn: &PgConnection,
    entity_id: &EntityId,
) -> Result<f64> {
    // Get count of services for this entity
    let count_row = conn
        .query_one(
            "SELECT COUNT(*) FROM service s
         JOIN entity_feature ef ON ef.table_id = s.id
         WHERE ef.entity_id = $1 AND ef.table_name = 'service' 
         AND s.embedding_v2 IS NOT NULL",
            &[&entity_id],
        )
        .await?;

    let service_count: i64 = count_row.get(0);
    if service_count < 2 {
        return Ok(0.0); // Not enough services to calculate coherence
    }

    // Using pgvector to calculate average pairwise similarity of services
    // Higher value means services are more semantically related
    let result = conn
        .query_opt(
            "WITH service_pairs AS (
            SELECT s1.embedding_v2 as emb1, s2.embedding_v2 as emb2
            FROM service s1
            JOIN entity_feature ef1 ON ef1.table_id = s1.id
            JOIN service s2 ON s2.id <> s1.id
            JOIN entity_feature ef2 ON ef2.table_id = s2.id
            WHERE ef1.entity_id = $1 AND ef1.table_name = 'service'
            AND ef2.entity_id = $1 AND ef2.table_name = 'service'
            AND s1.embedding_v2 IS NOT NULL AND s2.embedding_v2 IS NOT NULL
        )
        SELECT AVG(1 - (emb1 <=> emb2)) FROM service_pairs",
            &[&entity_id],
        )
        .await?;

    Ok(result.map_or(0.0, |r| r.get(0)))
}

async fn extract_embedding_quality_score(conn: &PgConnection, entity_id: &EntityId) -> Result<f64> {
    // Estimate embedding quality based on description length/completeness
    let row = conn
        .query_opt(
            "SELECT 
            CASE 
                WHEN o.embedding IS NULL THEN 0
                WHEN LENGTH(o.description) < 20 THEN 0.3
                WHEN LENGTH(o.description) < 100 THEN 0.6
                ELSE 0.9
            END as embedding_quality
         FROM entity e
         JOIN organization o ON e.organization_id = o.id
         WHERE e.id = $1",
            &[&entity_id],
        )
        .await?;

    Ok(row.map_or(0.0, |r| r.get(0)))
}

// Implementation of pairwise feature extraction
async fn calculate_name_similarity(
    conn: &PgConnection,
    entity1: &EntityId,
    entity2: &EntityId,
) -> Result<f64> {
    // This would ideally use a proper string similarity algorithm
    // For now, we'll use a simple implementation based on LOWER +
    // calculating similarity using a PostgreSQL function
    let row = conn
        .query_opt(
            "SELECT similarity(
            LOWER(o1.name), 
            LOWER(o2.name)
        ) as name_similarity
         FROM entity e1
         JOIN organization o1 ON e1.organization_id = o1.id
         JOIN entity e2 ON e2.id = $2
         JOIN organization o2 ON e2.organization_id = o2.id
         WHERE e1.id = $1",
            &[&entity1, &entity2],
        )
        .await?;

    Ok(row.map_or(0.0, |r| r.get(0)))
}

async fn calculate_embedding_similarity(
    conn: &PgConnection,
    entity1: &EntityId,
    entity2: &EntityId,
) -> Result<f64> {
    // Direct organization embedding similarity using pgvector
    let row = conn
        .query_opt(
            "SELECT 1 - (o1.embedding <=> o2.embedding) as similarity
         FROM entity e1
         JOIN organization o1 ON e1.organization_id = o1.id
         JOIN entity e2 ON e2.id = $2
         JOIN organization o2 ON e2.organization_id = o2.id
         WHERE e1.id = $1 
         AND o1.embedding IS NOT NULL 
         AND o2.embedding IS NOT NULL",
            &[&entity1, &entity2],
        )
        .await?;

    Ok(row.map_or(0.0, |r| r.get(0)))
}

async fn calculate_max_service_similarity(
    conn: &PgConnection,
    entity1: &EntityId,
    entity2: &EntityId,
) -> Result<f64> {
    // Find the highest similarity between any services from these entities
    let row = conn
        .query_opt(
            "SELECT MAX(1 - (s1.embedding_v2 <=> s2.embedding_v2)) as max_similarity
         FROM service s1
         JOIN entity_feature ef1 ON ef1.table_id = s1.id
         JOIN service s2
         JOIN entity_feature ef2 ON ef2.table_id = s2.id
         WHERE ef1.entity_id = $1 AND ef1.table_name = 'service'
         AND ef2.entity_id = $2 AND ef2.table_name = 'service'
         AND s1.embedding_v2 IS NOT NULL 
         AND s2.embedding_v2 IS NOT NULL",
            &[&entity1, &entity2],
        )
        .await?;

    Ok(row.map_or(0.0, |r| r.get(0)))
}

async fn calculate_geographic_distance(
    conn: &PgConnection,
    entity1: &EntityId,
    entity2: &EntityId,
) -> Result<f64> {
    // Find the minimum distance between any locations for these entities
    // Returns a normalized value where 0 = very far, 1 = same location
    let row = conn
        .query_opt(
            "WITH loc_distances AS (
            SELECT ST_Distance(l1.geom, l2.geom) as distance
            FROM location l1
            JOIN entity_feature ef1 ON ef1.table_id = l1.id
            JOIN location l2
            JOIN entity_feature ef2 ON ef2.table_id = l2.id
            WHERE ef1.entity_id = $1 AND ef1.table_name = 'location'
            AND ef2.entity_id = $2 AND ef2.table_name = 'location'
            AND l1.geom IS NOT NULL AND l2.geom IS NOT NULL
        )
        SELECT CASE 
            WHEN MIN(distance) IS NULL THEN 0 -- No locations to compare
            WHEN MIN(distance) > 10000 THEN 0 -- Too far
            ELSE 1 - (MIN(distance) / 10000)  -- Normalize 0-10km to 0-1 scale
        END as normalized_proximity
        FROM loc_distances",
            &[&entity1, &entity2],
        )
        .await?;

    Ok(row.map_or(0.0, |r| r.get(0)))
}

async fn check_shared_domain(
    conn: &PgConnection,
    entity1: &EntityId,
    entity2: &EntityId,
) -> Result<f64> {
    // Check if organizations share the same domain in URL
    let row = conn
        .query_opt(
            "WITH domain_extract AS (
            SELECT 
                regexp_replace(
                    regexp_replace(LOWER(o1.url), '^https?://(www\\.)?', ''), 
                    '/.*$', ''
                ) as domain1,
                regexp_replace(
                    regexp_replace(LOWER(o2.url), '^https?://(www\\.)?', ''), 
                    '/.*$', ''
                ) as domain2
            FROM entity e1
            JOIN organization o1 ON e1.organization_id = o1.id
            JOIN entity e2 ON e2.id = $2
            JOIN organization o2 ON e2.organization_id = o2.id
            WHERE e1.id = $1
            AND o1.url IS NOT NULL AND LENGTH(o1.url) > 0
            AND o2.url IS NOT NULL AND LENGTH(o2.url) > 0
        )
        SELECT CASE 
            WHEN domain1 = domain2 AND domain1 <> '' THEN 1.0
            ELSE 0.0
        END as shared_domain
        FROM domain_extract",
            &[&entity1, &entity2],
        )
        .await?;

    Ok(row.map_or(0.0, |r| r.get(0)))
}

async fn check_shared_phone(
    conn: &PgConnection,
    entity1: &EntityId,
    entity2: &EntityId,
) -> Result<f64> {
    // Check if organizations share any phone numbers
    let row = conn
        .query_opt(
            "SELECT CASE WHEN EXISTS (
            SELECT 1 
            FROM phone p1
            JOIN entity_feature ef1 ON ef1.table_id = p1.id
            JOIN phone p2
            JOIN entity_feature ef2 ON ef2.table_id = p2.id
            WHERE ef1.entity_id = $1 AND ef1.table_name = 'phone'
            AND ef2.entity_id = $2 AND ef2.table_name = 'phone'
            AND regexp_replace(p1.number, '[^0-9]', '', 'g') = 
                regexp_replace(p2.number, '[^0-9]', '', 'g')
            AND regexp_replace(p1.number, '[^0-9]', '', 'g') <> ''
        ) THEN 1.0 ELSE 0.0 END as shared_phone",
            &[&entity1, &entity2],
        )
        .await?;

    Ok(row.map_or(0.0, |r| r.get(0)))
}

async fn calculate_service_geo_semantic_score(
    conn: &PgConnection,
    entity1: &EntityId,
    entity2: &EntityId,
) -> Result<f64> {
    // Combines semantic similarity with geographical proximity
    let row = conn
        .query_opt(
            "WITH service_location_pairs AS (
            SELECT 
                1 - (s1.embedding_v2 <=> s2.embedding_v2) as semantic_sim,
                ST_Distance(l1.geom, l2.geom) as geo_distance
            FROM service s1
            JOIN entity_feature ef1 ON ef1.table_id = s1.id
            JOIN service_at_location sal1 ON sal1.service_id = s1.id
            JOIN location l1 ON l1.id = sal1.location_id
            JOIN service s2
            JOIN entity_feature ef2 ON ef2.table_id = s2.id
            JOIN service_at_location sal2 ON sal2.service_id = s2.id
            JOIN location l2 ON l2.id = sal2.location_id
            WHERE ef1.entity_id = $1 AND ef1.table_name = 'service'
            AND ef2.entity_id = $2 AND ef2.table_name = 'service'
            AND s1.embedding_v2 IS NOT NULL 
            AND s2.embedding_v2 IS NOT NULL
            AND l1.geom IS NOT NULL 
            AND l2.geom IS NOT NULL
        )
        SELECT 
            CASE 
                WHEN MIN(geo_distance) > 10000 THEN 0  -- Too far
                ELSE AVG(semantic_sim * (1 - LEAST(geo_distance/10000, 1)))  -- Weighted score
            END as hybrid_score
        FROM service_location_pairs",
            &[&entity1, &entity2],
        )
        .await?;

    Ok(row.map_or(0.0, |r| r.get(0)))
}

// Feature storage functions
async fn store_entity_features(
    conn: &PgConnection,
    entity_id: &EntityId,
    features: &[f64],
) -> Result<()> {
    // Get feature metadata names
    let metadata = get_feature_metadata();
    let entity_features = metadata
        .iter()
        .take(features.len())
        .enumerate()
        .map(|(i, meta)| (meta.name.clone(), features[i]));

    // Store each feature individually
    for (name, value) in entity_features {
        let id = Uuid::new_v4().to_string();

        // Check if feature already exists
        let existing = conn
            .query_opt(
                "SELECT id FROM clustering_metadata.entity_context_features
             WHERE entity_id = $1 AND feature_name = $2",
                &[&entity_id, &name],
            )
            .await?;

        if let Some(row) = existing {
            // Update existing
            let existing_id: String = row.get(0);
            conn.execute(
                "UPDATE clustering_metadata.entity_context_features
                 SET feature_value = $1
                 WHERE id = $2",
                &[&value, &existing_id],
            )
            .await?;
        } else {
            // Insert new
            conn.execute(
                "INSERT INTO clustering_metadata.entity_context_features
                 (id, entity_id, feature_name, feature_value)
                 VALUES ($1, $2, $3, $4)",
                &[&id, &entity_id, &name, &value],
            )
            .await?;
        }
    }

    Ok(())
}

async fn get_stored_entity_features(conn: &PgConnection, entity_id: &EntityId) -> Result<Vec<f64>> {
    // Get all features for this entity
    let rows = conn
        .query(
            "SELECT feature_name, feature_value
         FROM clustering_metadata.entity_context_features
         WHERE entity_id = $1
         ORDER BY feature_name",
            &[&entity_id],
        )
        .await?;

    // If no features stored, extract them now
    if rows.is_empty() {
        debug!(
            "No stored features found for entity {:?}, extracting now",
            entity_id
        );
        // Create a new pool or use a different approach to get entity features
        let features = extract_entity_features(conn, entity_id).await?;
        return Ok(features);
    }

    // Get metadata for expected ordering
    let metadata = get_feature_metadata();
    let entity_feature_names: Vec<String> = metadata
        .iter()
        .take(12) // Only individual entity features, not pair features
        .map(|m| m.name.clone())
        .collect();

    // Convert to map for easier lookup
    let mut feature_map = std::collections::HashMap::new();
    for row in rows {
        let name: String = row.get(0);
        let value: f64 = row.get(1);
        feature_map.insert(name, value);
    }

    // Reconstruct vector in correct order, with defaults for missing values
    let features: Vec<f64> = entity_feature_names
        .iter()
        .map(|name| feature_map.get(name).copied().unwrap_or(0.0))
        .collect();

    Ok(features)
}
