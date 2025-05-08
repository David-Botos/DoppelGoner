// src/consolidate_clusters.rs
use anyhow::{Context, Result};
use chrono::Utc;
use log::{debug, info, warn};
use petgraph::algo::kosaraju_scc;
use petgraph::prelude::*;
use tokio::sync::Mutex;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tokio_postgres::Transaction;
use uuid::Uuid;

// Use consistent crate-relative paths
use crate::db::PgPool;
use crate::models::{EntityGroupId, EntityId, GroupClusterId};
// Import the specific type instead of the whole module
use crate::reinforcement::MatchingOrchestrator;

/// Builds an undirected graph where:
/// - Nodes are entity groups
/// - Edges connect groups that share at least one entity
async fn build_group_graph(
    pool: &PgPool,
) -> Result<(
    UnGraph<EntityGroupId, ()>,
    HashMap<EntityGroupId, HashSet<EntityId>>,
)> {
    let conn = pool.get().await.context("Failed to get DB connection")?;

    // First, fetch all entity groups that don't already have a cluster assigned
    let groups_query = "SELECT id FROM entity_group WHERE group_cluster_id IS NULL";
    let groups_rows = conn
        .query(groups_query, &[])
        .await
        .context("Failed to fetch entity groups")?;

    // Create nodes for each group
    let mut graph = Graph::new_undirected();
    let mut group_nodes = HashMap::new();

    for row in &groups_rows {
        let group_id: String = row.get(0);
        let entity_group_id = EntityGroupId(group_id);
        let node_idx = graph.add_node(entity_group_id.clone());
        group_nodes.insert(entity_group_id, node_idx);
    }

    debug!("Created {} nodes in the graph", group_nodes.len());

    // Now fetch all entities for each group
    let group_entities_query = "
        SELECT entity_group_id, entity_id 
        FROM group_entity 
        WHERE entity_group_id = ANY($1)
    ";

    let group_ids: Vec<String> = groups_rows
        .iter()
        .map(|row| row.get::<_, String>(0))
        .collect();

    let entities_rows = conn
        .query(group_entities_query, &[&group_ids])
        .await
        .context("Failed to fetch group entities")?;

    // Create a mapping of groups to their entities
    let mut group_entities: HashMap<EntityGroupId, HashSet<EntityId>> = HashMap::new();

    for row in &entities_rows {
        let group_id: String = row.get(0);
        let entity_id: String = row.get(1);

        let entity_group_id = EntityGroupId(group_id);
        let entity_id = EntityId(entity_id);

        group_entities
            .entry(entity_group_id.clone())
            .or_insert_with(HashSet::new)
            .insert(entity_id);
    }

    debug!("Loaded entities for {} groups", group_entities.len());

    // Build a reverse index: entity -> groups
    let mut entity_groups: HashMap<EntityId, Vec<EntityGroupId>> = HashMap::new();

    for (group_id, entities) in &group_entities {
        for entity_id in entities {
            entity_groups
                .entry(entity_id.clone())
                .or_insert_with(Vec::new)
                .push(group_id.clone());
        }
    }

    // Add edges between groups that share entities
    for (_, groups) in entity_groups {
        // For each pair of groups that share this entity, add an edge
        for i in 0..groups.len() {
            for j in (i + 1)..groups.len() {
                if let (Some(node_i), Some(node_j)) =
                    (group_nodes.get(&groups[i]), group_nodes.get(&groups[j]))
                {
                    // Only add edge if it doesn't already exist
                    if !graph.contains_edge(*node_i, *node_j) {
                        graph.add_edge(*node_i, *node_j, ());
                    }
                }
            }
        }
    }

    debug!("Added {} edges to the graph", graph.edge_count());

    Ok((graph, group_entities))
}

/// Creates a new group cluster record
async fn create_cluster(
    transaction: &Transaction<'_>,
    cluster_id: &GroupClusterId,
    group_count: i32,
    entity_count: i32,
) -> Result<()> {
    let now = Utc::now().naive_utc();

    let insert_query = "
        INSERT INTO group_cluster (
            id, name, description, created_at, updated_at, entity_count, group_count
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
    ";

    let cluster_name = format!("Cluster {}", &cluster_id.0[..8]);
    let description = format!(
        "Automatically generated cluster with {} groups and {} unique entities",
        group_count, entity_count
    );

    transaction
        .execute(
            insert_query,
            &[
                &cluster_id.0,
                &cluster_name,
                &description,
                &now,
                &now,
                &entity_count,
                &group_count,
            ],
        )
        .await
        .context("Failed to insert group cluster")?;

    Ok(())
}

/// Updates entity groups to link them to their parent cluster
async fn update_groups(
    transaction: &Transaction<'_>,
    cluster_id: &GroupClusterId,
    group_ids: &[String],
) -> Result<()> {
    let now = Utc::now().naive_utc();

    // Use a prepared statement for better performance
    let update_query = "
        UPDATE entity_group
        SET group_cluster_id = $1, updated_at = $2
        WHERE id = ANY($3)
    ";

    transaction
        .execute(update_query, &[&cluster_id.0, &now, &group_ids])
        .await
        .context("Failed to update entity groups")?;

    Ok(())
}

/// Processes entity groups to form clusters with optional ML verification
///
/// This function:
/// 1. Builds a graph of entity groups connected by shared entities
/// 2. Finds connected components (clusters) in this graph
/// 3. Creates cluster records and updates groups to reference their clusters
/// 4. Optionally verifies clusters using ML if an orchestrator is provided
pub async fn process_clusters(
    pool: &PgPool,
    reinforcement_orchestrator: Option<&Mutex<MatchingOrchestrator>>,
) -> Result<usize> {
    // Start by building the graph
    info!("Building entity group graph...");
    let start_time = Instant::now();
    let (graph, group_entities) = build_group_graph(pool).await?;
    info!(
        "Graph built in {:.2?} with {} nodes and {} edges",
        start_time.elapsed(),
        graph.node_count(),
        graph.edge_count()
    );

    // Find connected components in the graph
    info!("Finding connected components in the graph...");
    let component_start = Instant::now();
    let components = kosaraju_scc(&graph);
    info!(
        "Found {} connected components in {:.2?}",
        components.len(),
        component_start.elapsed()
    );

    // Filter out single-node components
    let valid_components: Vec<_> = components.iter().filter(|comp| comp.len() >= 2).collect();

    info!(
        "Found {} components with multiple groups (potential clusters)",
        valid_components.len()
    );

    // Get a new connection for the transaction
    let mut conn = pool.get().await.context("Failed to get DB connection")?;
    let transaction = conn
        .transaction()
        .await
        .context("Failed to start transaction")?;

    let mut clusters_created = 0;
    let total_components = valid_components.len();
    // Track newly created cluster IDs for ML verification
    let mut new_cluster_ids = Vec::new();

    // Log progress at regular intervals
    let log_interval = std::cmp::max(1, total_components / 20); // Log approximately 20 times

    // Process each connected component
    for (idx, component) in valid_components.iter().enumerate() {
        // Get the group IDs for this component
        let group_ids: Vec<EntityGroupId> =
            component.iter().map(|&idx| graph[idx].clone()).collect();

        // Count unique entities across all groups in this component
        let mut unique_entities = HashSet::new();
        for group_id in &group_ids {
            if let Some(entities) = group_entities.get(group_id) {
                unique_entities.extend(entities.iter().cloned());
            }
        }

        // Create a new cluster
        let cluster_id = GroupClusterId(Uuid::new_v4().to_string());
        create_cluster(
            &transaction,
            &cluster_id,
            group_ids.len() as i32,
            unique_entities.len() as i32,
        )
        .await?;

        // Update all groups in this component to reference the new cluster
        let group_id_strings: Vec<String> = group_ids.iter().map(|id| id.0.clone()).collect();

        update_groups(&transaction, &cluster_id, &group_id_strings).await?;

        // Track this new cluster for verification
        new_cluster_ids.push(cluster_id);

        clusters_created += 1;

        // Log progress at regular intervals
        if (idx + 1) % log_interval == 0 || idx + 1 == total_components {
            info!(
                "Processed {}/{} components ({:.1}%), created {} clusters",
                idx + 1,
                total_components,
                (idx + 1) as f32 / total_components as f32 * 100.0,
                clusters_created
            );
        }
    }

    // Verify clusters using ML if orchestrator is provided
    if let Some(orchestrator_ref) = reinforcement_orchestrator {
        let mut orchestrator= orchestrator_ref.lock().await;
        info!(
            "Performing ML-based verification on {} new clusters",
            new_cluster_ids.len()
        );
        match verify_clusters(&transaction, &new_cluster_ids, pool, &mut *orchestrator).await {
            Ok(_) => info!("ML verification completed successfully"),
            Err(e) => warn!(
                "ML verification encountered errors but will continue: {}",
                e
            ),
        }
    }

    // Commit the transaction
    info!(
        "Committing transaction with {} clusters...",
        clusters_created
    );
    let commit_start = Instant::now();
    transaction
        .commit()
        .await
        .context("Failed to commit transaction")?;
    info!("Transaction committed in {:.2?}", commit_start.elapsed());

    info!("Created {} clusters in total", clusters_created);

    Ok(clusters_created)
}

// Add this function to consolidate_clusters.rs
async fn verify_clusters(
    transaction: &Transaction<'_>,
    new_cluster_ids: &[GroupClusterId],
    pool: &PgPool,
    reinforcement_orchestrator: &mut MatchingOrchestrator,
) -> Result<()> {
    info!(
        "Verifying quality of {} newly formed clusters",
        new_cluster_ids.len()
    );

    for cluster_id in new_cluster_ids {
        // Fetch a sample of entities from this cluster (limit to 5 for performance)
        let entity_query = "
            SELECT DISTINCT e.id 
            FROM entity e
            JOIN group_entity ge ON e.id = ge.entity_id
            JOIN entity_group eg ON ge.entity_group_id = eg.id
            WHERE eg.group_cluster_id = $1
            LIMIT 5
        ";

        let entity_rows = transaction.query(entity_query, &[&cluster_id.0]).await?;
        let entities: Vec<EntityId> = entity_rows.iter().map(|row| EntityId(row.get(0))).collect();

        if entities.len() >= 2 {
            let mut verification_scores = Vec::new();

            // Check pairs of entities to verify cluster quality
            for i in 0..entities.len() {
                for j in (i + 1)..entities.len() {
                    match reinforcement_orchestrator
                        .select_matching_method(pool, &entities[i], &entities[j])
                        .await
                    {
                        Ok((_, confidence)) => {
                            verification_scores.push(confidence);
                        }
                        Err(e) => {
                            // Log error but continue verification
                            warn!("Error during cluster verification: {}", e);
                        }
                    }
                }
            }

            // Calculate average verification score
            if !verification_scores.is_empty() {
                let avg_score =
                    verification_scores.iter().sum::<f64>() / verification_scores.len() as f64;

                // Flag low-quality clusters for review
                if avg_score < 0.7 {
                    info!(
                        "Flagging cluster {} for review (confidence score: {:.2})",
                        cluster_id.0, avg_score
                    );

                    // Add metadata for this low-confidence cluster
                    let metadata_query = "
                        INSERT INTO clustering_metadata.cluster_verification (
                            cluster_id, confidence_score, verified_at, needs_review
                        ) VALUES ($1, $2, NOW(), TRUE)
                    ";

                    transaction
                        .execute(metadata_query, &[&cluster_id.0, &avg_score])
                        .await?;
                }
            }
        }
    }

    Ok(())
}
