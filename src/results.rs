// src/results/results.rs

use anyhow::{Context, Result};
use chrono::{NaiveDateTime, Utc};
use log::info;
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

use crate::db::PgPool;
use crate::models::MatchMethodType;

/// Statistics for each matching method
#[derive(Debug, Clone)]
pub struct MatchMethodStats {
    pub method_type: MatchMethodType,
    pub groups_created: usize,
    pub entities_matched: usize,
    pub avg_confidence: f64,
    pub avg_group_size: f64,
}

/// Cluster statistics
#[derive(Debug, Clone)]
pub struct ClusterStats {
    pub total_clusters: usize,
    pub avg_entities_per_cluster: f64,
    pub avg_groups_per_cluster: f64,
    pub largest_cluster_size: usize,
    pub single_group_clusters: usize,
    pub multi_group_clusters: usize,
}

/// Service match statistics
#[derive(Debug, Clone)]
pub struct ServiceMatchStats {
    pub total_matches: usize,
    pub avg_similarity: f64,
    pub high_similarity_matches: usize,   // similarity > 0.8
    pub medium_similarity_matches: usize, // 0.6 - 0.8
    pub low_similarity_matches: usize,    // < 0.6
    pub clusters_with_matches: usize,
}

/// Complete pipeline run statistics
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub run_id: String,
    pub run_timestamp: NaiveDateTime,
    pub description: Option<String>,

    // Overall counts
    pub total_entities: usize,
    pub total_groups: usize,
    pub total_clusters: usize,
    pub total_service_matches: usize,

    // Timing information (in seconds)
    pub entity_processing_time: f64,
    pub matching_time: f64,
    pub clustering_time: f64,
    pub service_matching_time: f64,
    pub total_processing_time: f64,

    // Detailed statistics
    pub method_stats: Vec<MatchMethodStats>,
    pub cluster_stats: Option<ClusterStats>,
    pub service_stats: Option<ServiceMatchStats>,
}

/// Result struct for email matching process
pub struct EmailMatchResult {
    /// Number of new groups created
    pub groups_created: usize,
    /// Detailed statistics for email matching
    pub stats: MatchMethodStats,
}

/// Result struct for phone matching process
pub struct PhoneMatchResult {
    /// Number of new groups created
    pub groups_created: usize,
    /// Detailed statistics for phone matching
    pub stats: MatchMethodStats,
}

/// Result struct for URL matching process
pub struct UrlMatchResult {
    /// Number of new groups created
    pub groups_created: usize,
    /// Detailed statistics for URL matching
    pub stats: MatchMethodStats,
}

/// Result struct for address matching process
pub struct AddressMatchResult {
    /// Number of new groups created
    pub groups_created: usize,
    /// Detailed statistics for address matching
    pub stats: MatchMethodStats,
}

/// Result struct for geospatial matching process
pub struct GeospatialMatchResult {
    /// Number of new groups created
    pub groups_created: usize,
    /// Detailed statistics for geospatial matching
    pub stats: MatchMethodStats,
}

/// Collect matching method statistics from the database
pub async fn collect_method_stats(pool: &PgPool) -> Result<Vec<MatchMethodStats>> {
    let conn = pool.get().await.context("Failed to get DB connection")?;

    // Query method-specific statistics with explicit casting
    let query = "
        SELECT 
            method_type,
            COUNT(DISTINCT gm.entity_group_id)::bigint as groups_created,
            COUNT(DISTINCT ge.entity_id)::bigint as entities_matched,
            AVG(COALESCE(gm.confidence_score, 0))::double precision as avg_confidence,
            AVG(eg.entity_count)::double precision as avg_group_size
        FROM 
            group_method gm
        JOIN 
            entity_group eg ON gm.entity_group_id = eg.id
        JOIN 
            group_entity ge ON eg.id = ge.entity_group_id
        GROUP BY 
            method_type
    ";

    let rows = conn
        .query(query, &[])
        .await
        .context("Failed to query method statistics")?;

    let mut stats = Vec::with_capacity(rows.len());

    for row in rows {
        let method_type_str: String = row.get(0);
        let method_type = MatchMethodType::from_str(&method_type_str);

        stats.push(MatchMethodStats {
            method_type,
            groups_created: row.get::<_, i64>(1) as usize,
            entities_matched: row.get::<_, i64>(2) as usize,
            avg_confidence: row.get(3),
            avg_group_size: row.get(4),
        });
    }

    Ok(stats)
}

/// Collect cluster statistics from the database
pub async fn collect_cluster_stats(pool: &PgPool) -> Result<Option<ClusterStats>> {
    let conn = pool.get().await.context("Failed to get DB connection")?;

    // Check if we have any clusters
    let count_query = "SELECT COUNT(*)::bigint FROM group_cluster";
    let count_row = conn
        .query_one(count_query, &[])
        .await
        .context("Failed to count clusters")?;
    let cluster_count: i64 = count_row.get(0);

    if cluster_count == 0 {
        return Ok(None);
    }

    // Query cluster statistics with explicit casting
    let query = "
        WITH cluster_sizes AS (
            SELECT 
                gc.id,
                gc.entity_count,
                gc.group_count,
                CASE WHEN gc.group_count = 1 THEN 1 ELSE 0 END as is_single_group
            FROM 
                group_cluster gc
        )
        SELECT 
            COUNT(*)::bigint as total_clusters,
            AVG(entity_count)::double precision as avg_entities_per_cluster,
            AVG(group_count)::double precision as avg_groups_per_cluster,
            MAX(entity_count)::bigint as largest_cluster_size,
            SUM(is_single_group)::bigint as single_group_clusters,
            SUM(CASE WHEN is_single_group = 0 THEN 1 ELSE 0 END)::bigint as multi_group_clusters
        FROM 
            cluster_sizes
    ";

    let row = conn
        .query_one(query, &[])
        .await
        .context("Failed to query cluster statistics")?;

    Ok(Some(ClusterStats {
        total_clusters: row.get::<_, i64>(0) as usize,
        avg_entities_per_cluster: row.get(1),
        avg_groups_per_cluster: row.get(2),
        largest_cluster_size: row.get::<_, i64>(3) as usize,
        single_group_clusters: row.get::<_, i64>(4) as usize,
        multi_group_clusters: row.get::<_, i64>(5) as usize,
    }))
}

/// Collect service match statistics from the database
pub async fn collect_service_stats(pool: &PgPool) -> Result<Option<ServiceMatchStats>> {
    let conn = pool.get().await.context("Failed to get DB connection")?;

    // Check if we have any service matches
    let count_query = "SELECT COUNT(*)::bigint FROM service_match";
    let count_row = conn
        .query_one(count_query, &[])
        .await
        .context("Failed to count service matches")?;
    let match_count: i64 = count_row.get(0);

    if match_count == 0 {
        return Ok(None);
    }

    // Query service match statistics with explicit casting
    let query = "
        SELECT 
            COUNT(*)::bigint as total_matches,
            AVG(similarity_score)::double precision as avg_similarity,
            SUM(CASE WHEN similarity_score > 0.8 THEN 1 ELSE 0 END)::bigint as high_similarity,
            SUM(CASE WHEN similarity_score BETWEEN 0.6 AND 0.8 THEN 1 ELSE 0 END)::bigint as medium_similarity,
            SUM(CASE WHEN similarity_score < 0.6 THEN 1 ELSE 0 END)::bigint as low_similarity,
            COUNT(DISTINCT group_cluster_id)::bigint as clusters_with_matches
        FROM 
            service_match
    ";

    let row = conn
        .query_one(query, &[])
        .await
        .context("Failed to query service match statistics")?;

    Ok(Some(ServiceMatchStats {
        total_matches: row.get::<_, i64>(0) as usize,
        avg_similarity: row.get(1),
        high_similarity_matches: row.get::<_, i64>(2) as usize,
        medium_similarity_matches: row.get::<_, i64>(3) as usize,
        low_similarity_matches: row.get::<_, i64>(4) as usize,
        clusters_with_matches: row.get::<_, i64>(5) as usize,
    }))
}

/// Store pipeline run statistics in the database
async fn store_pipeline_stats(pool: &PgPool, stats: &PipelineStats) -> Result<()> {
    let mut conn = pool.get().await.context("Failed to get DB connection")?;
    let transaction = conn
        .transaction()
        .await
        .context("Failed to start transaction")?;

    // Insert main pipeline run record
    let insert_run = "
        INSERT INTO clustering_metadata.pipeline_run (
            id, run_timestamp, description, 
            total_entities, total_groups, total_clusters, total_service_matches,
            entity_processing_time, matching_time, clustering_time, service_matching_time, total_processing_time
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
    ";

    transaction
        .execute(
            insert_run,
            &[
                &stats.run_id,
                &stats.run_timestamp,
                &stats.description,
                &(stats.total_entities as i64),
                &(stats.total_groups as i64),
                &(stats.total_clusters as i64),
                &(stats.total_service_matches as i64),
                &stats.entity_processing_time,
                &stats.matching_time,
                &stats.clustering_time,
                &stats.service_matching_time,
                &stats.total_processing_time,
            ],
        )
        .await
        .context("Failed to insert pipeline run")?;

    // Insert method stats
    for method_stat in &stats.method_stats {
        let id = Uuid::new_v4().to_string();
        let method_type = method_stat.method_type.as_str();

        let insert_method = "
            INSERT INTO clustering_metadata.matching_method_stats (
                id, pipeline_run_id, method_type, groups_created,
                entities_matched, avg_confidence, avg_group_size
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        ";

        transaction
            .execute(
                insert_method,
                &[
                    &id,
                    &stats.run_id,
                    &method_type,
                    &(method_stat.groups_created as i64),
                    &(method_stat.entities_matched as i64),
                    &method_stat.avg_confidence,
                    &method_stat.avg_group_size,
                ],
            )
            .await
            .context("Failed to insert method stats")?;
    }

    // Insert cluster stats if available
    if let Some(cluster_stat) = &stats.cluster_stats {
        let id = Uuid::new_v4().to_string();

        let insert_cluster = "
            INSERT INTO clustering_metadata.cluster_stats (
                id, pipeline_run_id, total_clusters, avg_entities_per_cluster,
                avg_groups_per_cluster, largest_cluster_size, 
                single_group_clusters, multi_group_clusters
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ";

        transaction
            .execute(
                insert_cluster,
                &[
                    &id,
                    &stats.run_id,
                    &(cluster_stat.total_clusters as i64),
                    &cluster_stat.avg_entities_per_cluster,
                    &cluster_stat.avg_groups_per_cluster,
                    &(cluster_stat.largest_cluster_size as i64),
                    &(cluster_stat.single_group_clusters as i64),
                    &(cluster_stat.multi_group_clusters as i64),
                ],
            )
            .await
            .context("Failed to insert cluster stats")?;
    }

    // Insert service match stats if available
    if let Some(service_stat) = &stats.service_stats {
        let id = Uuid::new_v4().to_string();

        let insert_service = "
            INSERT INTO clustering_metadata.service_match_stats (
                id, pipeline_run_id, total_matches, avg_similarity,
                high_similarity_matches, medium_similarity_matches, 
                low_similarity_matches, clusters_with_matches
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ";

        transaction
            .execute(
                insert_service,
                &[
                    &id,
                    &stats.run_id,
                    &(service_stat.total_matches as i64),
                    &service_stat.avg_similarity,
                    &(service_stat.high_similarity_matches as i64),
                    &(service_stat.medium_similarity_matches as i64),
                    &(service_stat.low_similarity_matches as i64),
                    &(service_stat.clusters_with_matches as i64),
                ],
            )
            .await
            .context("Failed to insert service match stats")?;
    }

    transaction
        .commit()
        .await
        .context("Failed to commit transaction")?;

    info!(
        "Pipeline report stored in clustering_metadata schema with ID: {}",
        stats.run_id
    );

    Ok(())
}

/// Print a formatted report to the console
fn print_report(stats: &PipelineStats) {
    println!("\n========== HSDS ENTITY GROUPING PIPELINE REPORT ==========");
    println!("Run ID: {}", stats.run_id);
    println!("Timestamp: {}", stats.run_timestamp);
    if let Some(desc) = &stats.description {
        println!("Description: {}", desc);
    }

    println!("\n--- GENERAL STATISTICS ---");
    println!("Total entities processed: {}", stats.total_entities);
    println!("Total groups created: {}", stats.total_groups);
    println!("Total clusters formed: {}", stats.total_clusters);
    println!(
        "Total service matches identified: {}",
        stats.total_service_matches
    );

    println!("\n--- TIMING INFORMATION ---");
    println!(
        "Entity processing time: {:.2} seconds",
        stats.entity_processing_time
    );
    println!("Entity matching time: {:.2} seconds", stats.matching_time);
    println!(
        "Cluster consolidation time: {:.2} seconds",
        stats.clustering_time
    );
    println!(
        "Service matching time: {:.2} seconds",
        stats.service_matching_time
    );
    println!(
        "Total processing time: {:.2} seconds",
        stats.total_processing_time
    );

    println!("\n--- MATCHING METHOD STATISTICS ---");
    for method in &stats.method_stats {
        println!("\nMethod: {}", method.method_type.as_str());
        println!("  Groups created: {}", method.groups_created);
        println!("  Entities matched: {}", method.entities_matched);
        println!("  Average confidence: {:.2}", method.avg_confidence);
        println!("  Average group size: {:.2}", method.avg_group_size);
    }

    if let Some(cluster_stats) = &stats.cluster_stats {
        println!("\n--- CLUSTERING STATISTICS ---");
        println!("Total clusters: {}", cluster_stats.total_clusters);
        println!(
            "Average entities per cluster: {:.2}",
            cluster_stats.avg_entities_per_cluster
        );
        println!(
            "Average groups per cluster: {:.2}",
            cluster_stats.avg_groups_per_cluster
        );
        println!(
            "Largest cluster size: {} entities",
            cluster_stats.largest_cluster_size
        );
        println!(
            "Single-group clusters: {}",
            cluster_stats.single_group_clusters
        );
        println!(
            "Multi-group clusters: {}",
            cluster_stats.multi_group_clusters
        );
    }

    if let Some(service_stats) = &stats.service_stats {
        println!("\n--- SERVICE MATCHING STATISTICS ---");
        println!("Total service matches: {}", service_stats.total_matches);
        println!(
            "Average similarity score: {:.2}",
            service_stats.avg_similarity
        );
        println!(
            "High similarity matches (>0.8): {}",
            service_stats.high_similarity_matches
        );
        println!(
            "Medium similarity matches (0.6-0.8): {}",
            service_stats.medium_similarity_matches
        );
        println!(
            "Low similarity matches (<0.6): {}",
            service_stats.low_similarity_matches
        );
        println!(
            "Clusters with service matches: {}",
            service_stats.clusters_with_matches
        );
    }

    println!("\n==========================================================\n");
}

/// Generate a report on the pipeline execution
pub async fn generate_report(
    pool: &PgPool,
    run_stats: PipelineStats,
    phase_times: &HashMap<String, Duration>,
    description: Option<String>,
) -> Result<()> {
    // Collect statistics in parallel
    let (method_stats, cluster_stats, service_stats) = tokio::join!(
        collect_method_stats(pool),
        collect_cluster_stats(pool),
        collect_service_stats(pool)
    );

    let method_stats = method_stats?;
    let cluster_stats = cluster_stats?;
    let service_stats = service_stats?;

    // Create pipeline stats
    let entity_time = phase_times
        .get("entity_identification")
        .map_or(0.0, |d| d.as_secs_f64());
    let matching_time = phase_times
        .get("entity_matching")
        .map_or(0.0, |d| d.as_secs_f64());
    let clustering_time = phase_times
        .get("cluster_consolidation")
        .map_or(0.0, |d| d.as_secs_f64());
    let service_time = phase_times
        .get("service_matching")
        .map_or(0.0, |d| d.as_secs_f64());
    let total_time = entity_time + matching_time + clustering_time + service_time;

    let pipeline_stats = PipelineStats {
        run_id: Uuid::new_v4().to_string(),
        run_timestamp: Utc::now().naive_utc(),
        description,

        total_entities: run_stats.total_entities,
        total_groups: run_stats.total_groups,
        total_clusters: run_stats.total_clusters,
        total_service_matches: run_stats.total_service_matches,

        entity_processing_time: entity_time,
        matching_time,
        clustering_time,
        service_matching_time: service_time,
        total_processing_time: total_time,

        method_stats,
        cluster_stats,
        service_stats,
    };

    // Store statistics in database
    store_pipeline_stats(pool, &pipeline_stats).await?;

    // Print report
    print_report(&pipeline_stats);

    info!(
        "Pipeline report generated and stored with ID: {}",
        pipeline_stats.run_id
    );

    Ok(())
}
