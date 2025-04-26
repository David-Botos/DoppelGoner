// src/consolidate_clusters/consolidate_clusters.rs
// 1. Build a graph representation of entity groups:
//    - Nodes are entity_group records
//    - Edges exist between groups that share at least one entity
//    - Can use petgraph crate to build and analyze the graph

// 2. Find connected components in this graph:
//    - Each connected component represents a cluster of related groups
//    - Use depth-first search or petgraph's connected_components algorithm

// 3. For each connected component:
//    - Create a new group_cluster record
//    - Set name (could be derived from the largest group's name)
//    - Set entity_count to the unique count of entities across all groups
//    - Set group_count to the number of groups in this component

// 4. Update the entity_group records:
//    - For each group in a connected component, set group_cluster_id
//    - This links the group to its parent cluster

// 5. Performance considerations:
//    - Use batch operations for database updates
//    - For very large datasets, consider processing in chunks
//    - Ensure proper transaction handling for atomicity

use anyhow::{Context, Result};
use chrono::Utc;
use log::{debug, info};
use petgraph::algo::kosaraju_scc;
use petgraph::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tokio_postgres::Transaction;
use uuid::Uuid;

use crate::db::PgPool;
use crate::models::{EntityGroupId, EntityId, GroupClusterId};

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

/// Processes entity groups to form clusters
///
/// This function:
/// 1. Builds a graph of entity groups connected by shared entities
/// 2. Finds connected components (clusters) in this graph
/// 3. Creates cluster records and updates groups to reference their clusters
pub async fn process_clusters(pool: &PgPool) -> Result<usize> {
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
