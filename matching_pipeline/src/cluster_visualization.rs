// src/cluster_visualization.rs
use anyhow::{Context, Result};
use chrono::Utc;
use log::{debug, info, warn};
use serde_json::json;
use std::time::Instant;
use tokio_postgres::{GenericClient, Transaction};
use uuid::Uuid;

// Local imports
use crate::db::PgPool;
use crate::models::{EntityId, GroupClusterId};

/// Simple struct to hold entity_group record details
struct EntityGroupRecord {
    id: String,
    method_type: String,
    confidence_score: Option<f64>,
    pre_rl_confidence_score: Option<f64>,
    match_values: Option<serde_json::Value>,
}

/// Simple struct to hold group_cluster record details
struct GroupClusterRecord {
    id: String,
    name: Option<String>,
    entity_count: Option<i32>,
    average_coherence_score: Option<f64>,
}

/// Ensures the tables needed for visualization edge weights exist
pub async fn ensure_visualization_tables_exist(pool: &PgPool) -> Result<()> {
    let client = pool.get().await.context("Failed to get DB connection")?;

    // Create indices for performance
    let index_sqls = [
        "CREATE INDEX IF NOT EXISTS idx_cluster_entity_edges_cluster_id 
         ON clustering_metadata.cluster_entity_edges(cluster_id)",
        "CREATE INDEX IF NOT EXISTS idx_cluster_entity_edges_entity_pairs 
         ON clustering_metadata.cluster_entity_edges(entity_id_1, entity_id_2)",
    ];

    for sql in &index_sqls {
        client
            .execute(*sql, &[])
            .await
            .context(format!("Failed to create index with SQL: {}", sql))?;
    }

    Ok(())
}

/// Main function to calculate edge weights between entities within each cluster
/// These edge weights will be used for frontend visualization
pub async fn calculate_visualization_edges(pool: &PgPool, pipeline_run_id: &str) -> Result<usize> {
    info!(
        "Starting entity edge weight calculation for cluster visualization (run ID: {})...",
        pipeline_run_id
    );
    let start_time = Instant::now();

    let mut conn = pool
        .get()
        .await
        .context("Failed to get DB connection for calculate_visualization_edges")?;
    let transaction = conn
        .transaction()
        .await
        .context("Failed to start transaction for calculate_visualization_edges")?;

    // Clean up existing edges for this run
    transaction
        .execute(
            "DELETE FROM public.visualization_entity_edges WHERE pipeline_run_id = $1",
            &[&pipeline_run_id],
        )
        .await
        .context("Failed to clean up existing edges")?;

    // Fetch all clusters
    let clusters = fetch_clusters(&transaction).await?;
    info!(
        "Processing {} clusters for visualization edge calculation",
        clusters.len()
    );

    let mut total_edges = 0;

    for cluster in &clusters {
        let edges_count = process_cluster_for_visualization(
            &transaction,
            &GroupClusterId(cluster.id.clone()),
            pipeline_run_id,
        )
        .await?;

        total_edges += edges_count;
    }

    transaction
        .commit()
        .await
        .context("Failed to commit cluster visualization edge transaction")?;

    info!(
        "Cluster visualization edge calculation finished in {:.2?}. {} edges created.",
        start_time.elapsed(),
        total_edges
    );

    Ok(total_edges)
}

/// Process a single cluster to calculate visualization edges between its entities
async fn process_cluster_for_visualization(
    transaction: &Transaction<'_>,
    cluster_id: &GroupClusterId,
    pipeline_run_id: &str,
) -> Result<usize> {
    debug!(
        "Processing cluster {} for visualization edges",
        cluster_id.0
    );

    // 1. Get all unique entities in this cluster
    let entities = fetch_entities_in_cluster(transaction, cluster_id).await?;
    if entities.len() < 2 {
        debug!(
            "Cluster {} has fewer than 2 entities, skipping visualization edge calculation",
            cluster_id.0
        );
        return Ok(0);
    }

    debug!(
        "Found {} entities in cluster {}",
        entities.len(),
        cluster_id.0
    );

    // 2. Get RL weight based on human feedback
    let rl_weight = get_rl_weight_from_feedback(transaction).await?;
    debug!(
        "Using RL confidence weight of {:.2} based on human feedback",
        rl_weight
    );

    // 3. For each entity pair, calculate edge weight
    let mut edges_created = 0;
    for i in 0..entities.len() {
        for j in (i + 1)..entities.len() {
            let entity1_id = &entities[i];
            let entity2_id = &entities[j];

            // Ensure consistent ordering (entity_id_1 < entity_id_2)
            let (source_id, target_id) = if entity1_id.0 < entity2_id.0 {
                (entity1_id, entity2_id)
            } else {
                (entity2_id, entity1_id)
            };

            // Find all matching methods between these entities
            let matching_methods =
                fetch_entity_matching_methods(transaction, source_id, target_id).await?;

            if !matching_methods.is_empty() {
                let edge_id = create_visualization_edge(
                    transaction,
                    cluster_id,
                    source_id,
                    target_id,
                    &matching_methods,
                    rl_weight,
                    pipeline_run_id,
                )
                .await?;

                if !edge_id.is_empty() {
                    edges_created += 1;
                }
            }
        }
    }

    debug!(
        "Created {} visualization edges for cluster {}",
        edges_created, cluster_id.0
    );
    Ok(edges_created)
}

/// Fetch all clusters from the database
async fn fetch_clusters(transaction: &Transaction<'_>) -> Result<Vec<GroupClusterRecord>> {
    let query = "SELECT id, name, entity_count, average_coherence_score FROM public.group_cluster";

    let rows = transaction
        .query(query, &[])
        .await
        .context("Failed to fetch clusters")?;

    let mut clusters = Vec::with_capacity(rows.len());
    for row in rows {
        clusters.push(GroupClusterRecord {
            id: row.get("id"),
            name: row.get("name"),
            entity_count: row.get("entity_count"),
            average_coherence_score: row.get("average_coherence_score"),
        });
    }

    Ok(clusters)
}

/// Fetch all unique entities that belong to a specific cluster
async fn fetch_entities_in_cluster(
    transaction: &Transaction<'_>,
    cluster_id: &GroupClusterId,
) -> Result<Vec<EntityId>> {
    // Query to get all unique entities in a cluster from entity_group records
    let query = "
        SELECT DISTINCT entity_id_1 as entity_id FROM public.entity_group 
        WHERE group_cluster_id = $1
        UNION
        SELECT DISTINCT entity_id_2 as entity_id FROM public.entity_group 
        WHERE group_cluster_id = $1
    ";

    let rows = transaction
        .query(query, &[&cluster_id.0])
        .await
        .context("Failed to fetch entities in cluster")?;

    let mut entities = Vec::with_capacity(rows.len());
    for row in rows {
        let entity_id: String = row.get("entity_id");
        entities.push(EntityId(entity_id));
    }

    Ok(entities)
}

/// Fetch all matching methods that connect two specific entities
async fn fetch_entity_matching_methods(
    transaction: &Transaction<'_>,
    entity1_id: &EntityId,
    entity2_id: &EntityId,
) -> Result<Vec<EntityGroupRecord>> {
    // We already ensure entity1_id < entity2_id before calling this
    let query = "
        SELECT id, method_type, confidence_score, pre_rl_confidence_score, match_values
        FROM public.entity_group
        WHERE entity_id_1 = $1 AND entity_id_2 = $2
    ";

    let rows = transaction
        .query(query, &[&entity1_id.0, &entity2_id.0])
        .await
        .context("Failed to fetch entity matching methods")?;

    let mut methods = Vec::with_capacity(rows.len());
    for row in rows {
        methods.push(EntityGroupRecord {
            id: row.get("id"),
            method_type: row.get("method_type"),
            confidence_score: row.get("confidence_score"),
            pre_rl_confidence_score: row.get("pre_rl_confidence_score"),
            match_values: row.get("match_values"),
        });
    }

    Ok(methods)
}

/// Determine the proper weighting between RL and pre-RL confidence scores
/// based on historical human feedback
async fn get_rl_weight_from_feedback(transaction: &Transaction<'_>) -> Result<f64> {
    // Query human review feedback to see how often the RL model has been correct
    let query = "
        SELECT 
            COUNT(CASE WHEN was_correct = true THEN 1 END) as correct_count,
            COUNT(CASE WHEN was_correct = false THEN 1 END) as incorrect_count
        FROM clustering_metadata.human_review_method_feedback
        WHERE reviewer_id != 'ml_system'
    ";

    // This might fail if the table doesn't exist yet, so we need to handle that
    let result = transaction.query_opt(query, &[]).await;
    match result {
        Ok(Some(row)) => {
            let correct_count: i64 = row.get("correct_count");
            let incorrect_count: i64 = row.get("incorrect_count");

            // Calculate weight based on ML system accuracy
            if correct_count + incorrect_count > 0 {
                let accuracy = correct_count as f64 / (correct_count + incorrect_count) as f64;
                // Scale from 0.4 (poor accuracy) to 0.8 (excellent accuracy)
                let rl_weight = 0.4 + (0.4 * accuracy);
                Ok(rl_weight)
            } else {
                // No feedback yet
                Ok(0.6) // Balanced default
            }
        }
        _ => {
            // Table might not exist or query had an issue
            debug!("Could not query human feedback, using default RL weight");
            Ok(0.6) // Balanced default
        }
    }
}

/// Calculate and store an edge between two entities for visualization
async fn create_visualization_edge(
    transaction: &Transaction<'_>,
    cluster_id: &GroupClusterId,
    entity1_id: &EntityId,
    entity2_id: &EntityId,
    matching_methods: &[EntityGroupRecord],
    rl_weight: f64,
    pipeline_run_id: &str,
) -> Result<String> {
    // Calculate edge weight
    let pre_rl_weight = 1.0 - rl_weight;

    // For each matching method, calculate its contribution
    let mut method_details = Vec::new();
    let mut method_confidences = Vec::new();

    for method in matching_methods {
        // Default to the RL confidence if pre_rl is missing
        let pre_rl_conf = method
            .pre_rl_confidence_score
            .unwrap_or_else(|| method.confidence_score.unwrap_or(0.0));

        // Default to pre_rl if RL is missing
        let rl_conf = method
            .confidence_score
            .unwrap_or_else(|| method.pre_rl_confidence_score.unwrap_or(0.0));

        // Combined confidence
        let combined_confidence = (pre_rl_weight * pre_rl_conf) + (rl_weight * rl_conf);
        method_confidences.push(combined_confidence);

        // Store details for JSONB
        method_details.push(json!({
            "method_type": method.method_type,
            "pre_rl_confidence": pre_rl_conf,
            "rl_confidence": rl_conf,
            "combined_confidence": combined_confidence
        }));
    }

    // Calculate aggregate edge weight using probabilistic combination
    // This ensures multiple weak methods don't produce a stronger edge than one strong method
    let edge_weight = if method_confidences.is_empty() {
        0.0
    } else {
        // 1 - product of (1 - confidence)
        // Similar approach as used in consolidate_clusters.rs for edge weight calculation
        1.0 - method_confidences
            .iter()
            .fold(1.0, |acc, &conf| acc * (1.0 - conf.max(0.0).min(1.0)))
    };

    // Store the edge
    let edge_id = Uuid::new_v4().to_string();
    let details_json = json!({
        "methods": method_details,
        "rl_weight_factor": rl_weight,
        "method_count": matching_methods.len()
    });

    transaction
        .execute(
            "INSERT INTO public.visualization_entity_edges
         (id, cluster_id, entity_id_1, entity_id_2, edge_weight, details, pipeline_run_id)
         VALUES ($1, $2, $3, $4, $5, $6, $7)",
            &[
                &edge_id,
                &cluster_id.0,
                &entity1_id.0,
                &entity2_id.0,
                &edge_weight,
                &details_json,
                &pipeline_run_id,
            ],
        )
        .await
        .context("Failed to insert visualization entity edge")?;

    Ok(edge_id)
}
