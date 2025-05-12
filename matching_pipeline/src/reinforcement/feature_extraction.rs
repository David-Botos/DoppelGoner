use anyhow::{Context, Result};
use futures::future; // For try_join_all
use log::{debug, info, warn, Level as LogLevel};
use pgvector::Vector as PgVector; // Renamed to avoid conflict if candle ever uses 'Vector'
use tokio_postgres::{Client as PgConnection, GenericClient, Row as PgRow};
use uuid::Uuid;

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// Candle imports
use candle_core::{Device, Error as CandleError, Tensor};

use super::types::FeatureMetadata;
use crate::db::PgPool;
use crate::models::EntityId;
use crate::utils::{self, cosine_similarity_candle};

// Helper to get a Candle device
fn get_candle_device() -> Result<Device, CandleError> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        info!("Using Metal device for Candle operations.");
        return Device::new_metal(0);
    }
    #[cfg(all(feature = "cuda", target_os = "linux"))] // Or windows
    {
        info!("Using CUDA device for Candle operations.");
        return Device::new_cuda(0);
    }
    info!("Using CPU device for Candle operations.");
    Ok(Device::Cpu)
}

// Type alias for boxed futures returning a single f64 feature
type SingleFeatureFuture<'a> =
    Pin<Box<dyn Future<Output = Result<f64, anyhow::Error>> + Send + 'a>>;
// Type alias for boxed futures returning a Vec<f64> of features (still used for enhanced features)
type VecFeatureFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<f64>, anyhow::Error>> + Send + 'a>>;

// Helper function to wrap a future with progress logging (remains the same)
async fn wrap_with_progress<F, T, E>(
    task_future: F,
    counter: Arc<AtomicUsize>,
    total_tasks: usize,
    task_description: String,
    entity_context: String,
    log_level: LogLevel,
) -> Result<T, E>
where
    F: Future<Output = Result<T, E>> + Send,
    T: Send,
    E: std::fmt::Debug + Send,
{
    match task_future.await {
        Ok(result) => {
            let completed_count = counter.fetch_add(1, Ordering::Relaxed) + 1;
            log::log!(
                log_level,
                "{}: Task '{}' completed ({}/{}).",
                entity_context,
                task_description,
                completed_count,
                total_tasks
            );
            Ok(result)
        }
        Err(e) => {
            let completed_count = counter.fetch_add(1, Ordering::Relaxed) + 1;
            log::log!(
                LogLevel::Warn, // Log errors at Warn level
                "{}: Task '{}' failed ({}/{}), error: {:?}",
                entity_context,
                task_description,
                completed_count,
                total_tasks,
                e
            );
            Err(e)
        }
    }
}

// Feature metadata (remains the same)
pub fn get_feature_metadata() -> Vec<FeatureMetadata> {
    vec![
        // Basic features (index 0-8)
        FeatureMetadata {
            name: "name_complexity".to_string(),
            description: "Complexity of the organization name.".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "data_completeness".to_string(),
            description: "Completeness of core organization data fields.".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "has_email".to_string(),
            description: "Boolean indicating if an email exists.".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "has_phone".to_string(),
            description: "Boolean indicating if a phone number exists.".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "has_url".to_string(),
            description: "Boolean indicating if a URL exists.".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "has_address".to_string(),
            description: "Boolean indicating if an address exists.".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "has_location".to_string(),
            description: "Boolean indicating if location coordinates exist.".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "organization_size".to_string(),
            description: "Estimated size of the organization based on features.".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "service_count".to_string(),
            description: "Normalized count of services offered.".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        // Enhanced vector-based features (index 9-11)
        // Note: embedding_centroid_distance will now use Candle
        FeatureMetadata {
            name: "embedding_centroid_distance".to_string(),
            description:
                "Distance from organization embedding to global centroid (Candle calculated)."
                    .to_string(),
            min_value: 0.0, // Distance derived from 1.0 - similarity
            max_value: 2.0, // Max possible cosine distance (1 - (-1))
        },
        FeatureMetadata {
            name: "service_semantic_coherence".to_string(),
            description:
                "Average semantic similarity between an org's services (Candle calculated)."
                    .to_string(),
            min_value: -1.0, // Cosine similarity range
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "embedding_quality".to_string(),
            description: "Quality score of the organization's embedding.".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        // Pairwise features (index 12-18)
        // Note: embedding_similarity and max_service_similarity will now use Candle
        //       service_geo_semantic_score will use Candle for its semantic component
        FeatureMetadata {
            name: "name_similarity".to_string(),
            description: "Similarity between two entity names.".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "embedding_similarity".to_string(),
            description: "Cosine similarity between two entity embeddings (Candle calculated)."
                .to_string(),
            min_value: -1.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "max_service_similarity".to_string(),
            description:
                "Max semantic similarity between services of two entities (Candle calculated)."
                    .to_string(),
            min_value: -1.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "geographic_distance".to_string(),
            description: "Normalized geographic proximity between entities.".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "shared_domain".to_string(),
            description: "Boolean if entities share the same website domain.".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "shared_phone".to_string(),
            description: "Boolean if entities share the same phone number.".to_string(),
            min_value: 0.0,
            max_value: 1.0,
        },
        FeatureMetadata {
            name: "service_geo_semantic_score".to_string(),
            description: "Hybrid score combining service similarity (Candle) and geo-proximity."
                .to_string(),
            min_value: 0.0, // Assuming score is non-negative after combining
            max_value: 1.0, // Assuming normalized
        },
    ]
}

/// Struct to hold raw data fetched for an entity to calculate basic features.
#[derive(Debug)]
struct RawEntityOrgData {
    org_name: Option<String>,
    org_description: Option<String>,
    org_email: Option<String>,
    org_url: Option<String>,
    org_tax_id: Option<String>,
    org_legal_status: Option<String>,
    has_phone_flag: bool,
    has_address_flag: bool,
    has_location_coords_flag: bool,
    service_count_val: i64,
    location_count_val: i64,
}

pub async fn extract_entity_features<'conn, 'id>(
    conn: &'conn PgConnection,
    entity_id: &'id EntityId,
) -> Result<Vec<f64>> {
    let entity_context = format!("Entity {}", entity_id.0);
    let all_feature_metadata = get_feature_metadata();

    info!(
        "{} Starting extraction of 9 basic features (consolidated query)...",
        entity_context
    );
    let basic_features_tasks_completed = Arc::new(AtomicUsize::new(0));
    let total_basic_feature_tasks = 1;

    let consolidated_query_future = async {
        let query = "
            SELECT
                o.name AS org_name,
                o.description AS org_description,
                o.email AS org_email,
                o.url AS org_url,
                o.tax_id AS org_tax_id,
                o.legal_status AS org_legal_status,
                EXISTS (SELECT 1 FROM phone p JOIN entity_feature ef ON ef.table_id = p.id WHERE ef.entity_id = e.id AND ef.table_name = 'phone' AND p.number IS NOT NULL AND p.number <> '') AS has_phone_flag,
                EXISTS (SELECT 1 FROM address a JOIN location l_addr ON a.location_id = l_addr.id JOIN entity_feature ef_addr ON ef_addr.table_id = l_addr.id WHERE ef_addr.entity_id = e.id AND ef_addr.table_name = 'location') AS has_address_flag,
                EXISTS (SELECT 1 FROM location l JOIN entity_feature ef_loc ON ef_loc.table_id = l.id WHERE ef_loc.entity_id = e.id AND ef_loc.table_name = 'location' AND l.latitude IS NOT NULL AND l.longitude IS NOT NULL) AS has_location_coords_flag,
                (SELECT COUNT(*) FROM service s JOIN entity_feature ef_s ON ef_s.table_id = s.id WHERE ef_s.entity_id = e.id AND ef_s.table_name = 'service') AS service_count_val,
                (SELECT COUNT(*) FROM location l_loc_count JOIN entity_feature ef_lc ON ef_lc.table_id = l_loc_count.id WHERE ef_lc.entity_id = e.id AND ef_lc.table_name = 'location') AS location_count_val
            FROM entity e
            JOIN organization o ON e.organization_id = o.id
            WHERE e.id = $1";

        conn.query_one(query, &[&entity_id.0])
            .await
            .map_err(anyhow::Error::from)
            .context(format!(
                "Consolidated query failed for entity {}",
                entity_id.0
            ))
    };

    let row = wrap_with_progress(
        consolidated_query_future,
        basic_features_tasks_completed.clone(),
        total_basic_feature_tasks,
        "Consolidated Basic Features Query".to_string(),
        entity_context.clone(),
        LogLevel::Info,
    )
    .await?;

    let raw_data = RawEntityOrgData {
        org_name: row.try_get("org_name").ok(),
        org_description: row.try_get("org_description").ok(),
        org_email: row.try_get("org_email").ok(),
        org_url: row.try_get("org_url").ok(),
        org_tax_id: row.try_get("org_tax_id").ok(),
        org_legal_status: row.try_get("org_legal_status").ok(),
        has_phone_flag: row.try_get("has_phone_flag").unwrap_or(false),
        has_address_flag: row.try_get("has_address_flag").unwrap_or(false),
        has_location_coords_flag: row.try_get("has_location_coords_flag").unwrap_or(false),
        service_count_val: row.try_get("service_count_val").unwrap_or(0),
        location_count_val: row.try_get("location_count_val").unwrap_or(0),
    };

    let basic_features_vec = vec![
        calculate_name_complexity_from_data(&raw_data),
        calculate_data_completeness_from_data(&raw_data),
        calculate_has_email_from_data(&raw_data),
        calculate_has_phone_from_data(&raw_data),
        calculate_has_url_from_data(&raw_data),
        calculate_has_address_from_data(&raw_data),
        calculate_has_location_from_data(&raw_data),
        calculate_organization_size_from_data(&raw_data),
        calculate_service_count_feature_from_data(&raw_data),
    ];
    info!(
        "{} Completed calculation of 9 basic features from consolidated data.",
        entity_context
    );

    let candle_device =
        get_candle_device().map_err(|e| anyhow::anyhow!("Failed to get candle device: {}", e))?;

    info!(
        "{} Starting extraction of 3 enhanced features (now using Candle)...",
        entity_context
    );
    let enhanced_tasks_completed = Arc::new(AtomicUsize::new(0));
    let num_enhanced_feature_tasks = 3;

    // Clone candle_device for each async block if it's not Sync, or ensure it is. Device is Sync.
    let enhanced_futures: [SingleFeatureFuture<'_>; 3] = [
        Box::pin(wrap_with_progress(
            extract_embedding_centroid_distance(conn, entity_id, candle_device.clone()),
            enhanced_tasks_completed.clone(),
            num_enhanced_feature_tasks,
            all_feature_metadata[9].name.clone(),
            entity_context.clone(),
            LogLevel::Debug,
        )),
        Box::pin(wrap_with_progress(
            extract_service_semantic_coherence(conn, entity_id, candle_device.clone()),
            enhanced_tasks_completed.clone(),
            num_enhanced_feature_tasks,
            all_feature_metadata[10].name.clone(),
            entity_context.clone(),
            LogLevel::Debug,
        )),
        Box::pin(wrap_with_progress(
            extract_embedding_quality_score(conn, entity_id), // This one doesn't use Candle
            enhanced_tasks_completed.clone(),
            num_enhanced_feature_tasks,
            all_feature_metadata[11].name.clone(),
            entity_context.clone(),
            LogLevel::Debug,
        )),
    ];
    let enhanced_results = future::try_join_all(Vec::from(enhanced_futures)).await?;
    info!(
        "{} Completed extraction of enhanced features.",
        entity_context
    );

    let mut features = Vec::with_capacity(basic_features_vec.len() + enhanced_results.len());
    features.extend(basic_features_vec);
    features.extend(enhanced_results);

    debug!(
        "{} Storing {} extracted features...",
        entity_context,
        features.len()
    );
    store_entity_features(conn, entity_id, &features, &all_feature_metadata).await?;
    debug!("{} Successfully stored features.", entity_context);

    Ok(features)
}

// --- Synchronous Calculation Helpers for Basic Features (remain the same) ---

fn calculate_name_complexity_from_data(data: &RawEntityOrgData) -> f64 {
    if let Some(name) = &data.org_name {
        let length = name.len() as f64;
        let word_count = name.split_whitespace().count() as f64;
        ((length / 100.0).min(1.0) * 0.5 + (word_count / 10.0).min(1.0) * 0.5)
    } else {
        0.0
    }
}

fn calculate_data_completeness_from_data(data: &RawEntityOrgData) -> f64 {
    let mut score = 0.0;
    let mut total_fields = 0.0;
    total_fields += 1.0;
    if data.org_name.is_some() {
        score += 1.0;
    }
    total_fields += 1.0;
    if data.org_description.as_ref().map_or(false, |d| d.len() > 5) {
        score += 1.0;
    }
    total_fields += 1.0;
    if data.org_email.is_some() {
        score += 1.0;
    }
    total_fields += 1.0;
    if data.org_url.is_some() {
        score += 1.0;
    }
    total_fields += 1.0;
    if data.org_tax_id.is_some() {
        score += 1.0;
    }
    total_fields += 1.0;
    if data.org_legal_status.is_some() {
        score += 1.0;
    }
    if total_fields == 0.0 {
        0.0
    } else {
        score / total_fields
    }
}

fn calculate_has_email_from_data(data: &RawEntityOrgData) -> f64 {
    if data.org_email.is_some() {
        1.0
    } else {
        0.0
    }
}
fn calculate_has_phone_from_data(data: &RawEntityOrgData) -> f64 {
    if data.has_phone_flag {
        1.0
    } else {
        0.0
    }
}
fn calculate_has_url_from_data(data: &RawEntityOrgData) -> f64 {
    if data.org_url.is_some() {
        1.0
    } else {
        0.0
    }
}
fn calculate_has_address_from_data(data: &RawEntityOrgData) -> f64 {
    if data.has_address_flag {
        1.0
    } else {
        0.0
    }
}
fn calculate_has_location_from_data(data: &RawEntityOrgData) -> f64 {
    if data.has_location_coords_flag {
        1.0
    } else {
        0.0
    }
}

fn calculate_organization_size_from_data(data: &RawEntityOrgData) -> f64 {
    let service_score = (data.service_count_val as f64 / 10.0).min(1.0);
    let location_score = (data.location_count_val as f64 / 5.0).min(1.0);
    let phone_score = if data.has_phone_flag { 1.0 } else { 0.0 };
    (service_score * 0.5 + location_score * 0.3 + phone_score * 0.2)
}

fn calculate_service_count_feature_from_data(data: &RawEntityOrgData) -> f64 {
    (data.service_count_val as f64 / 10.0).min(1.0)
}

pub async fn extract_context_for_pair(
    pool: &PgPool,
    entity1: &EntityId, // This is &'a EntityId, references are Copy
    entity2: &EntityId, // This is &'a EntityId, references are Copy
) -> Result<Vec<f64>> {
    let conn_guard = pool // Original conn_guard declaration
        .get()
        .await
        .context("Failed to get DB connection for pair context extraction")?;

    // Step 1: Create a reference to the connection early.
    let conn_ref: &PgConnection = &*conn_guard;

    let pair_context = format!("Pair ({}, {})", entity1.0, entity2.0); //
    let pair_context_for_outer_wrapper = pair_context.clone(); //

    let all_feature_metadata = get_feature_metadata(); //
    // Assuming all_feature_metadata is a Vec; it will be moved into pair_features_calculation_task_inner. This is fine.

    let context_steps_completed = Arc::new(AtomicUsize::new(0)); //
    let total_context_steps = 3; //

    let candle_device = get_candle_device() //
        .map_err(|e| anyhow::anyhow!("Failed to get candle device for pair context: {}", e))?;

    // Step 2: Use conn_ref for tasks that borrow the connection.
    let entity1_features_task = wrap_with_progress( //
        get_stored_entity_features(conn_ref, entity1), // Use conn_ref
        context_steps_completed.clone(),
        total_context_steps,
        format!("Stored/Extracted Features for entity1 ({})", entity1.0),
        pair_context.clone(),
        LogLevel::Info,
    );

    let entity2_features_task = wrap_with_progress( //
        get_stored_entity_features(conn_ref, entity2), // Use conn_ref
        context_steps_completed.clone(),
        total_context_steps,
        format!("Stored/Extracted Features for entity2 ({})", entity2.0),
        pair_context.clone(),
        LogLevel::Info,
    );

    let device_for_pair_calc = candle_device.clone(); //

    // Step 3: The async move block will now capture `conn_ref` (a reference, which is Copy)
    // and other necessary owned/cloned values. `conn_guard` itself is NOT moved.
    let pair_features_calculation_task_inner = async move { //
        // This async move captures:
        // - conn_ref (by copy, as it's a reference: &'some_lifetime PgConnection)
        // - entity1, entity2 (by copy, as they are references: &EntityId)
        // - device_for_pair_calc (the cloned Device is moved)
        // - all_feature_metadata (the Vec is moved)
        // - pair_context (the String is moved)

        let pair_calc_tasks_completed_inner = Arc::new(AtomicUsize::new(0)); //
        const PAIRWISE_METADATA_OFFSET: usize = 12; //
        const NUM_PAIRWISE_FEATURES: usize = 7; //

        let pair_features_futures_list: [SingleFeatureFuture<'_>; NUM_PAIRWISE_FEATURES] = [ //
            Box::pin(wrap_with_progress( //
                calculate_name_similarity(conn_ref, entity1, entity2), // Use conn_ref
                pair_calc_tasks_completed_inner.clone(),
                NUM_PAIRWISE_FEATURES,
                all_feature_metadata[PAIRWISE_METADATA_OFFSET + 0].name.clone(),
                pair_context.clone(), // Uses the moved pair_context
                LogLevel::Debug,
            )),
            Box::pin(wrap_with_progress( //
                calculate_embedding_similarity(conn_ref, entity1, entity2, device_for_pair_calc.clone()), // Use conn_ref
                pair_calc_tasks_completed_inner.clone(),
                NUM_PAIRWISE_FEATURES,
                all_feature_metadata[PAIRWISE_METADATA_OFFSET + 1].name.clone(),
                pair_context.clone(),
                LogLevel::Debug,
            )),
            Box::pin(wrap_with_progress( //
                calculate_max_service_similarity(conn_ref, entity1, entity2, device_for_pair_calc.clone()), // Use conn_ref
                pair_calc_tasks_completed_inner.clone(),
                NUM_PAIRWISE_FEATURES,
                all_feature_metadata[PAIRWISE_METADATA_OFFSET + 2].name.clone(),
                pair_context.clone(),
                LogLevel::Debug,
            )),
            Box::pin(wrap_with_progress( //
                calculate_geographic_distance(conn_ref, entity1, entity2), // Use conn_ref
                pair_calc_tasks_completed_inner.clone(),
                NUM_PAIRWISE_FEATURES,
                all_feature_metadata[PAIRWISE_METADATA_OFFSET + 3].name.clone(),
                pair_context.clone(),
                LogLevel::Debug,
            )),
            Box::pin(wrap_with_progress( //
                check_shared_domain(conn_ref, entity1, entity2), // Use conn_ref
                pair_calc_tasks_completed_inner.clone(),
                NUM_PAIRWISE_FEATURES,
                all_feature_metadata[PAIRWISE_METADATA_OFFSET + 4].name.clone(),
                pair_context.clone(),
                LogLevel::Debug,
            )),
            Box::pin(wrap_with_progress( //
                check_shared_phone(conn_ref, entity1, entity2), // Use conn_ref
                pair_calc_tasks_completed_inner.clone(),
                NUM_PAIRWISE_FEATURES,
                all_feature_metadata[PAIRWISE_METADATA_OFFSET + 5].name.clone(),
                pair_context.clone(),
                LogLevel::Debug,
            )),
            Box::pin(wrap_with_progress( //
                calculate_service_geo_semantic_score(conn_ref, entity1, entity2, device_for_pair_calc.clone()), // Use conn_ref
                pair_calc_tasks_completed_inner.clone(),
                NUM_PAIRWISE_FEATURES,
                all_feature_metadata[PAIRWISE_METADATA_OFFSET + 6].name.clone(),
                pair_context.clone(),
                LogLevel::Debug,
            )),
        ];
        let results = future::try_join_all(Vec::from(pair_features_futures_list)).await; //
        if results.is_ok() { //
            info!( //
                "{} Completed calculation of {} pair-specific features.",
                pair_context.clone(), // Uses the moved pair_context
                NUM_PAIRWISE_FEATURES
            );
        } else {
            warn!( //
                "{} Failed during calculation of some pair-specific features.",
                pair_context.clone() // Uses the moved pair_context
            );
        }
        results
    };

    let pair_features_task_logged = wrap_with_progress( //
        pair_features_calculation_task_inner,
        context_steps_completed.clone(),
        total_context_steps,
        "All Pair-Specific Calculations".to_string(),
        pair_context_for_outer_wrapper.clone(),
        LogLevel::Info,
    );

    let (entity1_features_vec_res, entity2_features_vec_res, pair_features_vec_res) = tokio::join!( //
        entity1_features_task,
        entity2_features_task,
        pair_features_task_logged
    );

    let entity1_features_vec = entity1_features_vec_res?; //
    let entity2_features_vec = entity2_features_vec_res?; //
    let pair_features_vec = pair_features_vec_res?; //

    info!( //
        "{} Completed overall context assembly.",
        pair_context_for_outer_wrapper
    );

    let mut context = Vec::new(); //
    context.extend(entity1_features_vec); //
    context.extend(entity2_features_vec); //
    context.extend(pair_features_vec); //

    Ok(context)
}

pub async fn get_stored_entity_features<'conn, 'id>(
    conn: &'conn PgConnection,
    entity_id: &'id EntityId,
) -> Result<Vec<f64>> {
    let entity_context = format!("Entity {}", entity_id.0);
    debug!("{} Checking for stored features...", entity_context);
    let rows = conn
        .query(
            "SELECT feature_name, feature_value
         FROM clustering_metadata.entity_context_features
         WHERE entity_id = $1
         ORDER BY feature_name",
            &[&entity_id.0],
        )
        .await?;

    if rows.is_empty() {
        info!(
            "{}: No stored features found, extracting now.",
            entity_context
        );
        return extract_entity_features(conn, entity_id).await;
    }

    debug!(
        "{} Found stored features, reconstructing vector...",
        entity_context
    );
    let metadata = get_feature_metadata();
    const INDIVIDUAL_FEATURE_COUNT: usize = 12;

    let mut feature_map = HashMap::new();
    for row in rows {
        let name: String = row.get(0);
        let value: f64 = row.get(1);
        feature_map.insert(name, value);
    }

    let features_vec: Vec<f64> = metadata
        .iter()
        .take(INDIVIDUAL_FEATURE_COUNT)
        .map(|meta| {
            feature_map.get(&meta.name).copied().unwrap_or_else(|| {
                warn!(
                    "{}: Missing stored feature '{}', defaulting to 0.0.",
                    entity_context, meta.name
                );
                0.0
            })
        })
        .collect();

    if features_vec.len() != INDIVIDUAL_FEATURE_COUNT {
        warn!(
            "{}: Expected {} individual features, but reconstructed {}. Re-extracting.",
            entity_context,
            INDIVIDUAL_FEATURE_COUNT,
            features_vec.len()
        );
        return extract_entity_features(conn, entity_id).await;
    }
    debug!(
        "{} Successfully reconstructed stored features.",
        entity_context
    );
    Ok(features_vec)
}

async fn store_entity_features(
    conn: &PgConnection,
    entity_id: &EntityId,
    features: &[f64],
    all_feature_metadata: &[FeatureMetadata],
) -> Result<()> {
    if features.is_empty() {
        debug!(
            "No features for entity_id: {}. Skipping store.",
            entity_id.0
        );
        return Ok(());
    }
    let num_features_to_store = features.len().min(all_feature_metadata.len()).min(12);
    if num_features_to_store == 0 {
        warn!(
            "Mismatch or empty features/metadata for entity_id: {}. Skipping store.",
            entity_id.0
        );
        return Ok(());
    }
    // ... (rest of store_entity_features remains the same)
    let mut query = String::from(
        "INSERT INTO clustering_metadata.entity_context_features (id, entity_id, feature_name, feature_value, created_at) VALUES ",
    );
    let mut params_data: Vec<(String, String, String, f64, chrono::NaiveDateTime)> =
        Vec::with_capacity(num_features_to_store);
    let now = chrono::Utc::now().naive_utc();

    for i in 0..num_features_to_store {
        let feature_meta = &all_feature_metadata[i];
        let feature_value = features[i];
        let feature_id_str = Uuid::new_v4().to_string();
        params_data.push((
            feature_id_str,
            entity_id.0.clone(),
            feature_meta.name.clone(),
            feature_value,
            now,
        ));
        if i > 0 {
            query.push_str(", ");
        }
        let base = i * 5;
        query.push_str(&format!(
            "(${p1}, ${p2}, ${p3}, ${p4}, ${p5})",
            p1 = base + 1,
            p2 = base + 2,
            p3 = base + 3,
            p4 = base + 4,
            p5 = base + 5
        ));
    }
    query.push_str(" ON CONFLICT (entity_id, feature_name) DO UPDATE SET feature_value = EXCLUDED.feature_value, created_at = EXCLUDED.created_at");

    let mut sql_params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> =
        Vec::with_capacity(num_features_to_store * 5);
    for (id_val, entity_id_val, name_val, value_val, created_at_val) in &params_data {
        sql_params.push(id_val);
        sql_params.push(entity_id_val);
        sql_params.push(name_val);
        sql_params.push(value_val);
        sql_params.push(created_at_val);
    }

    match conn.execute(query.as_str(), &sql_params[..]).await {
        Ok(rows_affected) => {
            debug!(
                "Stored/updated {} features for entity_id: {} ({} rows affected).",
                num_features_to_store, entity_id.0, rows_affected
            );
            Ok(())
        }
        Err(e) => {
            warn!(
                "Error storing features for entity_id: {}. Query: [{}], Error: {}",
                entity_id.0, query, e
            );
            Err(anyhow::Error::from(e).context("Storing entity features failed"))
        }
    }
}

// --- Helper functions for robust row data extraction (remain the same) ---
fn get_optional_f64_from_row(
    row: &PgRow,
    index: usize,
    feature_name: &str,
    entity_id_str: &str,
) -> f64 {
    match row.try_get::<_, Option<f64>>(index) {
        Ok(value_option) => value_option.unwrap_or(0.0),
        Err(e) => {
            warn!(
                "Error deserializing optional f64 for '{}' for '{}': {}. Defaulting to 0.0.",
                feature_name, entity_id_str, e
            );
            0.0
        }
    }
}
fn get_guaranteed_f64_from_row(
    row: &PgRow,
    index: usize,
    feature_name: &str,
    entity_id_str: &str,
) -> f64 {
    match row.try_get::<_, f64>(index) {
        Ok(value) => value,
        Err(e) => {
            warn!(
                "Error deserializing guaranteed f64 for '{}' for '{}': {}. Defaulting to 0.0.",
                feature_name, entity_id_str, e
            );
            0.0
        }
    }
}

// --- Enhanced Feature Extraction Functions (Updated for Candle) ---

async fn extract_embedding_centroid_distance(
    conn: &PgConnection,
    entity_id: &EntityId,
    candle_device: Device,
) -> Result<f64> {
    debug!(
        "Extracting embedding centroid distance for entity {} using Candle",
        entity_id.0
    );
    let org_embedding_row = conn.query_opt(
        "SELECT o.embedding FROM organization o JOIN entity e ON e.organization_id = o.id WHERE e.id = $1 AND o.embedding IS NOT NULL",
        &[&entity_id.0]
    ).await.context("DB query for org_embedding failed")?;

    let centroid_embedding_row = conn.query_opt(
        "SELECT avg(embedding) as centroid_embedding FROM organization WHERE embedding IS NOT NULL", // pgvector can average embeddings
        &[]
    ).await.context("DB query for centroid_embedding failed")?;

    match (org_embedding_row, centroid_embedding_row) {
        (Some(org_row), Some(centroid_row)) => {
            let org_pg_vec: Option<PgVector> = org_row.try_get(0).ok();
            let centroid_pg_vec: Option<PgVector> = centroid_row.try_get("centroid_embedding").ok();

            if let (Some(org_pg_vec_val), Some(centroid_pg_vec_val)) = (org_pg_vec, centroid_pg_vec)
            {
                let org_vec_f32 = org_pg_vec_val.to_vec();
                let centroid_vec_f32 = centroid_pg_vec_val.to_vec();

                if org_vec_f32.is_empty() || centroid_vec_f32.is_empty() {
                    warn!("Empty embedding vector found for entity {} or global centroid. Returning max distance.", entity_id.0);
                    return Ok(2.0); // Max cosine distance
                }
                if org_vec_f32.len() != centroid_vec_f32.len() {
                    warn!("Org embedding and centroid embedding have different dimensions for entity {}. Org dim: {}, Centroid dim: {}. Returning max distance.", 
                           entity_id.0, org_vec_f32.len(), centroid_vec_f32.len());
                    return Ok(2.0);
                }

                let similarity = cosine_similarity_candle(&org_vec_f32, &centroid_vec_f32)
                    .map_err(|e| anyhow::anyhow!("Candle similarity calculation failed: {}", e))?;

                Ok((1.0 - similarity as f64).max(0.0).min(2.0)) // Cosine distance
            } else {
                warn!(
                    "Could not retrieve org or centroid embedding for entity {}.",
                    entity_id.0
                );
                Ok(2.0) // Default to max distance if embeddings are missing
            }
        }
        _ => {
            warn!(
                "Org embedding or centroid not found for entity {}.",
                entity_id.0
            );
            Ok(2.0) // Default to max distance
        }
    }
}

async fn extract_service_semantic_coherence(
    conn: &(impl GenericClient + Sync),
    entity_id: &EntityId,
    candle_device: Device,
) -> Result<f64> {
    debug!(
        "Extracting service semantic coherence for entity {} using Candle",
        entity_id.0
    );
    let service_embeddings_rows = conn.query(
        "SELECT s.embedding_v2 FROM service s JOIN entity_feature ef ON ef.table_id = s.id WHERE ef.entity_id = $1 AND ef.table_name = 'service' AND s.embedding_v2 IS NOT NULL",
        &[&entity_id.0]
    ).await.context("DB query for service_embeddings failed")?;

    let service_embeddings_f32: Vec<Vec<f32>> = service_embeddings_rows
        .into_iter()
        .filter_map(|row| {
            row.get::<_, Option<PgVector>>(0)
                .map(|pg_vec| pg_vec.to_vec())
        })
        .filter(|vec| !vec.is_empty()) // Filter out empty embeddings
        .collect();

    if service_embeddings_f32.len() < 2 {
        return Ok(0.0); // Not enough services to compare
    }

    // Validate all embeddings have the same dimension
    let first_dim = service_embeddings_f32[0].len();
    if !service_embeddings_f32.iter().all(|v| v.len() == first_dim) {
        warn!("Service embeddings for entity {} have inconsistent dimensions. Cannot compute coherence.", entity_id.0);
        return Ok(0.0); // Or handle as error
    }

    let mut total_similarity = 0.0;
    let mut pair_count = 0;

    for i in 0..service_embeddings_f32.len() {
        for j in (i + 1)..service_embeddings_f32.len() {
            let sim =
                cosine_similarity_candle(&service_embeddings_f32[i], &service_embeddings_f32[j])
                    .map_err(|e| {
                        anyhow::anyhow!("Candle similarity for service coherence failed: {}", e)
                    })?;
            total_similarity += sim as f64;
            pair_count += 1;
        }
    }

    if pair_count == 0 {
        Ok(0.0)
    } else {
        Ok(total_similarity / pair_count as f64)
    }
}

// This function does not use embeddings/Candle, so it remains unchanged.
async fn extract_embedding_quality_score(
    conn: &(impl GenericClient + Sync),
    entity_id: &EntityId,
) -> Result<f64> {
    let row_opt = conn
        .query_opt(
            "SELECT CASE
            WHEN o.embedding IS NULL THEN 0.0
            WHEN o.description IS NULL OR LENGTH(o.description) = 0 THEN 0.1
            WHEN LENGTH(o.description) < 20 THEN 0.3
            WHEN LENGTH(o.description) < 100 THEN 0.6
            ELSE 0.9
        END::DOUBLE PRECISION as embedding_quality
        FROM entity e
        JOIN organization o ON e.organization_id = o.id
        WHERE e.id = $1",
            &[&entity_id.0],
        )
        .await
        .context(format!(
            "DB query failed for extract_embedding_quality_score for entity {}",
            entity_id.0
        ))?;
    Ok(row_opt.map_or(0.0, |row| {
        get_guaranteed_f64_from_row(&row, 0, "embedding_quality_score", &entity_id.0)
    }))
}

// --- Pairwise Feature Functions (Updated for Candle where applicable) ---

// Name similarity is string-based, not for Candle. Remains SQL-based or strsim-based.
async fn calculate_name_similarity(
    conn: &(impl GenericClient + Sync),
    entity1: &EntityId,
    entity2: &EntityId,
) -> Result<f64> {
    // Stays the same (uses SQL similarity function or could use strsim if names fetched)
    let row_opt = conn
        .query_opt(
            "SELECT similarity(LOWER(o1.name), LOWER(o2.name))::DOUBLE PRECISION as name_similarity
         FROM entity e1 JOIN organization o1 ON e1.organization_id = o1.id,
              entity e2 JOIN organization o2 ON e2.organization_id = o2.id
         WHERE e1.id = $1 AND e2.id = $2 AND o1.name IS NOT NULL AND o2.name IS NOT NULL",
            &[&entity1.0, &entity2.0],
        )
        .await
        .context("DB query failed for calculate_name_similarity")?;
    Ok(row_opt.map_or(0.0, |row| {
        get_guaranteed_f64_from_row(
            &row,
            0,
            "name_similarity",
            &format!("{},{}", entity1.0, entity2.0),
        )
    }))
}

async fn calculate_embedding_similarity(
    conn: &(impl GenericClient + Sync),
    entity1: &EntityId,
    entity2: &EntityId,
    candle_device: Device,
) -> Result<f64> {
    debug!(
        "Calculating embedding similarity for pair ({}, {}) using Candle",
        entity1.0, entity2.0
    );
    let row_opt = conn
        .query_opt(
            "SELECT o1.embedding as emb1, o2.embedding as emb2
         FROM entity e1 JOIN organization o1 ON e1.organization_id = o1.id,
              entity e2 JOIN organization o2 ON e2.organization_id = o2.id
         WHERE e1.id = $1 AND e2.id = $2",
            &[&entity1.0, &entity2.0],
        )
        .await
        .context("DB query for embeddings failed in calculate_embedding_similarity")?;

    if let Some(row) = row_opt {
        let emb1_pg: Option<PgVector> = row.try_get("emb1").ok();
        let emb2_pg: Option<PgVector> = row.try_get("emb2").ok();

        if let (Some(e1), Some(e2)) = (emb1_pg, emb2_pg) {
            let v1_f32 = e1.to_vec();
            let v2_f32 = e2.to_vec();
            if v1_f32.is_empty() || v2_f32.is_empty() {
                warn!(
                    "Empty embedding vector(s) for pair ({}, {}). Returning 0.0 similarity.",
                    entity1.0, entity2.0
                );
                return Ok(0.0);
            }
            if v1_f32.len() != v2_f32.len() {
                warn!("Embeddings for pair ({}, {}) have different dimensions. V1 dim: {}, V2 dim: {}. Returning 0.0 similarity.", 
                       entity1.0, entity2.0, v1_f32.len(), v2_f32.len());
                return Ok(0.0);
            }
            let sim = cosine_similarity_candle(&v1_f32, &v2_f32)
                .map_err(|e| anyhow::anyhow!("Candle similarity calculation failed: {}", e))?;
            Ok(sim as f64)
        } else {
            Ok(0.0)
        } // Embeddings not found or null
    } else {
        Ok(0.0)
    } // Entities not found
}

async fn calculate_max_service_similarity(
    conn: &(impl GenericClient + Sync),
    entity1: &EntityId,
    entity2: &EntityId,
    candle_device: Device,
) -> Result<f64> {
    debug!(
        "Calculating max service similarity for pair ({}, {}) using Candle",
        entity1.0, entity2.0
    );
    let services1_rows = conn.query(
        "SELECT s.embedding_v2 FROM service s JOIN entity_feature ef ON ef.table_id = s.id WHERE ef.entity_id = $1 AND ef.table_name = 'service' AND s.embedding_v2 IS NOT NULL",
        &[&entity1.0]
    ).await.context(format!("DB query for services of entity1 ({}) failed", entity1.0))?;

let services2_rows = conn.query(
        "SELECT s.embedding_v2 FROM service s JOIN entity_feature ef ON ef.table_id = s.id WHERE ef.entity_id = $1 AND ef.table_name = 'service' AND s.embedding_v2 IS NOT NULL", // Changed $2 to $1
        &[&entity2.0]
    ).await.context(format!("DB query for services of entity2 ({}) failed", entity2.0))?;

    let s1_embeddings: Vec<Vec<f32>> = services1_rows
        .into_iter()
        .filter_map(|r| r.get::<_, Option<PgVector>>(0).map(|pg_v| pg_v.to_vec()))
        .filter(|v| !v.is_empty())
        .collect();
    let s2_embeddings: Vec<Vec<f32>> = services2_rows
        .into_iter()
        .filter_map(|r| r.get::<_, Option<PgVector>>(0).map(|pg_v| pg_v.to_vec()))
        .filter(|v| !v.is_empty())
        .collect();

    if s1_embeddings.is_empty() || s2_embeddings.is_empty() {
        return Ok(0.0);
    }

    // Validate all embeddings from s1 have the same dimension, and same for s2, and that this dimension is consistent.
    // For simplicity, assuming they are consistent if not empty. Proper validation would be more robust.
    if !s1_embeddings.is_empty()
        && !s2_embeddings.is_empty()
        && (s1_embeddings[0].len() != s2_embeddings[0].len())
    {
        warn!("Max service similarity: Service embeddings for pair ({}, {}) have different dimensions. S1 dim: {}, S2 dim: {}. Returning 0.0.",
                entity1.0, entity2.0, s1_embeddings[0].len(), s2_embeddings[0].len());
        return Ok(0.0);
    }

    let mut max_similarity = -1.0f32; // Smallest possible cosine similarity

    for emb1 in &s1_embeddings {
        for emb2 in &s2_embeddings {
            if emb1.len() != emb2.len() {
                // Check individual pairs too
                warn!(
                    "Skipping service pair due to dimension mismatch. Emb1 dim: {}, Emb2 dim: {}",
                    emb1.len(),
                    emb2.len()
                );
                continue;
            }

            let current_sim_f64 = cosine_similarity_candle(emb1, emb2).map_err(|e| {
                anyhow::anyhow!("Candle similarity for max service sim failed: {}", e)
            })?;

            let current_sim_f32 = current_sim_f64 as f32; // Explicitly cast f64 to f32

            if current_sim_f32 > max_similarity {
                max_similarity = current_sim_f32;
            }
        }
    }
    Ok(max_similarity as f64)
}

// Geographic distance remains SQL-based.
async fn calculate_geographic_distance(
    conn: &(impl GenericClient + Sync),
    entity1: &EntityId,
    entity2: &EntityId,
) -> Result<f64> {
    let row = conn
        .query_one(
            "WITH loc_distances AS (
            SELECT ST_Distance(l1.geom, l2.geom) as distance
            FROM location l1 JOIN entity_feature ef1 ON ef1.table_id = l1.id,
                 location l2 JOIN entity_feature ef2 ON ef2.table_id = l2.id
            WHERE ef1.entity_id = $1 AND ef1.table_name = 'location'
              AND ef2.entity_id = $2 AND ef2.table_name = 'location'
              AND l1.geom IS NOT NULL AND l2.geom IS NOT NULL
        )
        SELECT CASE WHEN COUNT(distance) = 0 THEN 0.0 
                    WHEN MIN(distance) > 10000 THEN 0.0
                    ELSE 1.0 - (MIN(distance) / 10000.0)
               END::DOUBLE PRECISION
        FROM loc_distances",
            &[&entity1.0, &entity2.0],
        )
        .await
        .context("DB query failed for calculate_geographic_distance")?;
    Ok(get_guaranteed_f64_from_row(
        &row,
        0,
        "geographic_distance",
        &format!("{},{}", entity1.0, entity2.0),
    ))
}

// Shared domain remains SQL-based.
async fn check_shared_domain(
    conn: &(impl GenericClient + Sync),
    entity1: &EntityId,
    entity2: &EntityId,
) -> Result<f64> {
    let row_opt = conn
        .query_opt(
            "WITH domain_extract AS (
            SELECT regexp_replace(LOWER(o1.url), '^https?://(www\\.)?|/.*$', '', 'g') as domain1,
                   regexp_replace(LOWER(o2.url), '^https?://(www\\.)?|/.*$', '', 'g') as domain2
            FROM entity e1 JOIN organization o1 ON e1.organization_id = o1.id,
                 entity e2 JOIN organization o2 ON e2.organization_id = o2.id
            WHERE e1.id = $1 AND e2.id = $2 AND o1.url IS NOT NULL AND o1.url <> ''
                                          AND o2.url IS NOT NULL AND o2.url <> ''
        )
        SELECT CASE WHEN domain1 = domain2 AND domain1 <> '' THEN 1.0 ELSE 0.0 END::DOUBLE PRECISION
        FROM domain_extract",
            &[&entity1.0, &entity2.0],
        )
        .await
        .context("DB query failed for check_shared_domain")?;
    Ok(row_opt.map_or(0.0, |row| {
        get_guaranteed_f64_from_row(
            &row,
            0,
            "shared_domain",
            &format!("{},{}", entity1.0, entity2.0),
        )
    }))
}

// Shared phone remains SQL-based.
async fn check_shared_phone(
    conn: &(impl GenericClient + Sync),
    entity1: &EntityId,
    entity2: &EntityId,
) -> Result<f64> {
    let row = conn.query_one(
        "SELECT CASE WHEN EXISTS (
            SELECT 1 FROM phone p1 JOIN entity_feature ef1 ON ef1.table_id = p1.id,
                         phone p2 JOIN entity_feature ef2 ON ef2.table_id = p2.id
            WHERE ef1.entity_id = $1 AND ef1.table_name = 'phone'
              AND ef2.entity_id = $2 AND ef2.table_name = 'phone'
              AND regexp_replace(p1.number, '[^0-9]', '', 'g') = regexp_replace(p2.number, '[^0-9]', '', 'g')
              AND regexp_replace(p1.number, '[^0-9]', '', 'g') <> ''
        ) THEN 1.0 ELSE 0.0 END::DOUBLE PRECISION",
        &[&entity1.0, &entity2.0]
    ).await.context("DB query failed for check_shared_phone")?;
    Ok(get_guaranteed_f64_from_row(
        &row,
        0,
        "shared_phone",
        &format!("{},{}", entity1.0, entity2.0),
    ))
}

async fn calculate_service_geo_semantic_score(
    conn: &(impl GenericClient + Sync),
    entity1: &EntityId,
    entity2: &EntityId,
    candle_device: Device,
) -> Result<f64> {
    debug!(
        "Calculating service_geo_semantic_score for pair ({}, {}) using Candle for semantics",
        entity1.0, entity2.0
    );
    // Fetch pairs of embeddings and their geo_distance
    let rows = conn.query(
        "SELECT s1.embedding_v2 as emb1, s2.embedding_v2 as emb2, ST_Distance(l1.geom, l2.geom) as geo_distance
         FROM service s1 JOIN entity_feature ef1 ON ef1.table_id = s1.id JOIN service_at_location sal1 ON sal1.service_id = s1.id JOIN location l1 ON l1.id = sal1.location_id,
              service s2 JOIN entity_feature ef2 ON ef2.table_id = s2.id JOIN service_at_location sal2 ON sal2.service_id = s2.id JOIN location l2 ON l2.id = sal2.location_id
         WHERE ef1.entity_id = $1 AND ef1.table_name = 'service' AND s1.embedding_v2 IS NOT NULL AND l1.geom IS NOT NULL
           AND ef2.entity_id = $2 AND ef2.table_name = 'service' AND s2.embedding_v2 IS NOT NULL AND l2.geom IS NOT NULL",
        &[&entity1.0, &entity2.0]
    ).await.context("DB query for service_geo_semantic_score failed")?;

    if rows.is_empty() {
        return Ok(0.0);
    }

    let mut total_weighted_score = 0.0;
    let mut count = 0;

    for row in rows {
        let emb1_pg: Option<PgVector> = row.try_get("emb1").ok();
        let emb2_pg: Option<PgVector> = row.try_get("emb2").ok();
        let geo_distance: Option<f64> = row.try_get("geo_distance").ok();

        if let (Some(e1), Some(e2), Some(dist)) = (emb1_pg, emb2_pg, geo_distance) {
            let v1_f32 = e1.to_vec();
            let v2_f32 = e2.to_vec();

            if v1_f32.is_empty() || v2_f32.is_empty() {
                continue;
            }
            if v1_f32.len() != v2_f32.len() {
                warn!("Service geo semantic: skipping pair due to dimension mismatch. V1 dim: {}, V2 dim: {}.", v1_f32.len(), v2_f32.len());
                continue;
            }

            let semantic_sim = cosine_similarity_candle(&v1_f32, &v2_f32).map_err(|e| {
                anyhow::anyhow!("Candle similarity for service_geo_semantic failed: {}", e)
            })? as f64;

            // Geo proximity score (0 to 1, higher is closer)
            // Max distance considered relevant is 10000 meters (10km)
            let geo_proximity = if dist > 10000.0 {
                0.0
            } else {
                1.0 - (dist / 10000.0)
            };

            total_weighted_score += semantic_sim * geo_proximity;
            count += 1;
        }
    }

    if count == 0 {
        Ok(0.0)
    } else {
        Ok((total_weighted_score / count as f64).max(0.0).min(1.0)) // Average and clamp
    }
}