// src/matching/geospatial/core.rs

use anyhow::{Context, Result};
use chrono::NaiveDateTime;
use log::{debug, error, info, trace, warn}; // Added error for explicit error logging
use std::collections::{HashMap, HashSet};
use tokio::sync::Mutex;
use tokio_postgres::Transaction; // Transaction is still used here as it's passed in
use uuid::Uuid;

// Local application/crate imports
use crate::config::{self as app_config, MIN_GEO_PAIR_CONFIDENCE_THRESHOLD}; // Assuming these are top-level config items
use crate::db::{self as project_db, PgPool}; // project_db for insert_suggestion, PgPool for RL context
use crate::models::{
    ActionType, EntityGroupId, EntityId, GeospatialMatchValue, MatchMethodType, MatchResult,
    MatchValues, NewSuggestedAction, SuggestionStatus,
};
use crate::reinforcement::{self, MatchingOrchestrator}; // reinforcement module

// Geospatial submodule specific imports
use super::config::THRESHOLDS; // Geospatial specific thresholds
use super::database::{self, DbStatements, ExistingGeoPairsMap}; // DbStatements now holds SQL strings
use super::postgis; // For PostGIS specific operations
use super::service_utils::{self, SERVICE_SIMILARITY_THRESHOLD}; // For service comparison logic
use super::utils::calculate_haversine_distance; // Fallback distance calculation

#[derive(Clone, Debug, Default)]
struct PairMlSnapshot {
    features: Option<Vec<f64>>,
    predicted_method: Option<MatchMethodType>,
    prediction_confidence: Option<f64>,
}

/// Orchestrates the entire geospatial pairwise matching process within a given transaction.
/// DbStatements now contains SQL query strings, not prepared statements.
pub async fn perform_geospatial_pairwise_matching_in_transaction(
    tx: &Transaction<'_>,    // The transaction for this batch of operations
    db_stmts: &DbStatements, // Contains SQL query strings
    all_unpaired_locations: &[(EntityId, f64, f64)],
    existing_geospatial_pairs: &ExistingGeoPairsMap,
    has_postgis: bool,
    now: &NaiveDateTime,
    pool: &PgPool, // For operations outside the transaction, like RL context fetching
    reinforcement_orchestrator: Option<&Mutex<MatchingOrchestrator>>,
    pipeline_run_id: &str,
) -> Result<MatchResult> {
    let mut overall_match_result = MatchResult {
        groups_created: 0,
        entities_matched: 0,
        entities_added: 0,
        entities_skipped: 0,
        processed_entities: HashSet::new(),
    };

    let mut current_unpaired_locations = all_unpaired_locations.to_vec();

    // Phase 1: Match entities from `current_unpaired_locations` against `existing_geospatial_pairs`
    if !existing_geospatial_pairs.is_empty() && !current_unpaired_locations.is_empty() {
        debug!(
            "Geo Core - Phase 1: Matching {} unpaired locations to {} existing geo pairs.",
            current_unpaired_locations.len(),
            existing_geospatial_pairs.len()
        );

        let mut still_unpaired_after_phase1 = Vec::new();
        for (new_entity_id, new_lat, new_lon) in current_unpaired_locations.iter() {
            if overall_match_result
                .processed_entities
                .contains(new_entity_id)
            {
                continue;
            }
            // Use consistency_check_sql from DbStatements
            let existing_db_pair_count_row = tx
                .query_one(db_stmts.consistency_check_sql, &[&new_entity_id.0])
                .await
                .with_context(|| {
                    format!("Consistency check failed for entity {}", new_entity_id.0)
                })?;
            let existing_db_pair_count: i64 = existing_db_pair_count_row.get(0);

            if existing_db_pair_count > 0 {
                overall_match_result.entities_skipped += 1;
                overall_match_result
                    .processed_entities
                    .insert(new_entity_id.clone());
                continue;
            }

            let (pair_created, _pair_details) = match_single_entity_to_existing_pairs(
                tx,
                db_stmts,
                new_entity_id,
                *new_lat,
                *new_lon,
                existing_geospatial_pairs,
                has_postgis,
                now,
                pool,
                reinforcement_orchestrator,
                pipeline_run_id,
                &THRESHOLDS,
                &mut overall_match_result.processed_entities,
            )
            .await?;

            if pair_created {
                overall_match_result.groups_created += 1;
            } else {
                still_unpaired_after_phase1.push((new_entity_id.clone(), *new_lat, *new_lon));
            }
        }
        current_unpaired_locations = still_unpaired_after_phase1;
        debug!(
            "Geo Core - Phase 1 done. {} locations remaining for new pair creation.",
            current_unpaired_locations.len()
        );
    }

    // Phase 2: Create new pairs from the remaining `current_unpaired_locations`.
    if !current_unpaired_locations.is_empty() {
        debug!(
            "Geo Core - Phase 2: Creating new pairs from {} remaining locations.",
            current_unpaired_locations.len()
        );

        let truly_unpaired_for_phase2: Vec<_> = current_unpaired_locations
            .into_iter()
            .filter(|(id, _, _)| !overall_match_result.processed_entities.contains(id))
            .collect();

        if !truly_unpaired_for_phase2.is_empty() {
            let phase2_result = create_new_pairs_from_candidates_list(
                tx,
                db_stmts,
                &truly_unpaired_for_phase2,
                has_postgis,
                now,
                pool,
                reinforcement_orchestrator,
                pipeline_run_id,
                &THRESHOLDS,
                &mut overall_match_result.processed_entities,
            )
            .await?;

            overall_match_result.groups_created += phase2_result.groups_created;
            overall_match_result.entities_skipped += phase2_result.entities_skipped;
        } else {
            debug!("Geo Core - Phase 2: No truly unpaired entities left after phase 1.");
        }
    }

    overall_match_result.entities_matched = overall_match_result.processed_entities.len();
    overall_match_result.entities_added = overall_match_result.entities_matched; // For pairwise, added is same as matched

    Ok(overall_match_result)
}

async fn match_single_entity_to_existing_pairs(
    tx: &Transaction<'_>,
    db_stmts: &DbStatements, // Contains SQL strings
    new_entity_id: &EntityId,
    new_lat: f64,
    new_lon: f64,
    existing_geospatial_pairs: &ExistingGeoPairsMap,
    has_postgis: bool,
    now: &NaiveDateTime,
    pool: &PgPool,
    reinforcement_orchestrator: Option<&Mutex<MatchingOrchestrator>>,
    pipeline_run_id: &str,
    thresholds_config: &[(f64, f64)],
    processed_entities_this_run: &mut HashSet<EntityId>,
) -> Result<(bool, Option<GeospatialMatchValue>)> {
    let mut best_rl_confidence = app_config::MIN_GEO_PAIR_CONFIDENCE_THRESHOLD - 0.01;
    let mut best_candidate_info: Option<(EntityId, f64, f64, f64, f64)> = None;
    let mut best_ml_snapshot: Option<PairMlSnapshot> = None;

    let max_dist_from_thresholds = thresholds_config
        .iter()
        .map(|(d, _)| *d)
        .fold(0.0, f64::max);

    for (_existing_pair_id, (e1_id_ref, e2_id_ref, geo_match_val, _current_pair_conf)) in
        existing_geospatial_pairs.iter()
    {
        let members_info_from_existing_pair = [
            (e1_id_ref, geo_match_val.latitude1, geo_match_val.longitude1),
            (e2_id_ref, geo_match_val.latitude2, geo_match_val.longitude2),
        ];

        for (candidate_id_ref, cand_lat, cand_lon) in members_info_from_existing_pair {
            if candidate_id_ref == new_entity_id
                || processed_entities_this_run.contains(candidate_id_ref)
            {
                continue;
            }

            let distance = calculate_distance_between_points(
                tx,
                db_stmts,
                new_lat,
                new_lon,
                cand_lat,
                cand_lon,
                has_postgis,
            )
            .await?;

            if distance <= max_dist_from_thresholds {
                match service_utils::compare_entity_services(tx, new_entity_id, candidate_id_ref)
                    .await
                {
                    Ok(service_similarity) => {
                        if service_similarity >= SERVICE_SIMILARITY_THRESHOLD {
                            let (pred_method, rl_conf_opt, features) = get_rl_prediction(
                                pool,
                                reinforcement_orchestrator,
                                new_entity_id,
                                candidate_id_ref,
                            )
                            .await?;

                            let base_conf_from_dist_thr = thresholds_config
                                .iter()
                                .find(|&&(thr_dist, _)| distance <= thr_dist)
                                .map_or(0.0, |&(_, base_c)| base_c);

                            let final_conf = calculate_final_confidence(
                                pred_method.as_ref(),
                                rl_conf_opt,
                                base_conf_from_dist_thr,
                            );

                            if final_conf > best_rl_confidence {
                                best_rl_confidence = final_conf;
                                best_candidate_info = Some((
                                    candidate_id_ref.clone(),
                                    cand_lat,
                                    cand_lon,
                                    distance,
                                    service_similarity,
                                ));
                                best_ml_snapshot = Some(PairMlSnapshot {
                                    features,
                                    predicted_method: pred_method,
                                    prediction_confidence: rl_conf_opt,
                                });
                            }
                        } else {
                            debug!("Service similarity check failed for new entity {} with candidate {} from existing pair. Similarity: {:.2}", new_entity_id.0, candidate_id_ref.0, service_similarity);
                        }
                    }
                    Err(e) => {
                        warn!("Service comparison API call failed between new entity {} and candidate {}: {}", new_entity_id.0, candidate_id_ref.0, e);
                    }
                }
            }
        }
    }

    if let Some((paired_entity_id, paired_lat, paired_lon, pair_distance, service_sim)) =
        best_candidate_info
    {
        if best_rl_confidence >= app_config::MIN_GEO_PAIR_CONFIDENCE_THRESHOLD {
            let (e1, e1_lat, e1_lon, e2, e2_lat, e2_lon) = canonical_order_pair(
                new_entity_id,
                new_lat,
                new_lon,
                &paired_entity_id,
                paired_lat,
                paired_lon,
            );

            let new_pair_geo_details = GeospatialMatchValue {
                latitude1: e1_lat,
                longitude1: e1_lon,
                latitude2: e2_lat,
                longitude2: e2_lon,
                distance: pair_distance,
            };
            let match_values_obj = MatchValues::Geospatial(new_pair_geo_details.clone());
            let new_pair_guid = Uuid::new_v4().to_string();

            // Use new_geospatial_pair_sql from DbStatements
            insert_new_pair_into_db(
                tx,
                db_stmts.new_geospatial_pair_sql,
                &new_pair_guid,
                &e1,
                &e2,
                &match_values_obj,
                best_rl_confidence,
                now,
            )
            .await?;

            info!("MATCH_TO_EXISTING: New geo pair {} ({}, {}) created. Dist: {:.1}m, ServSim: {:.2}, Conf: {:.4}",
                new_pair_guid, e1.0, e2.0, pair_distance, service_sim, best_rl_confidence);

            processed_entities_this_run.insert(e1.clone());
            processed_entities_this_run.insert(e2.clone());

            log_suggestion_if_needed(
                tx,
                pipeline_run_id,
                &e1,
                Some(&EntityGroupId(new_pair_guid)),
                best_rl_confidence,
                &match_values_obj,
                Some(service_sim),
            )
            .await;
            log_ml_feedback(
                pool,
                reinforcement_orchestrator,
                &e1,
                &e2,
                best_rl_confidence,
                best_ml_snapshot,
            )
            .await;

            return Ok((true, Some(new_pair_geo_details)));
        } else {
            debug!("Best candidate pair ({}, {}) did not meet MIN_GEO_PAIR_CONFIDENCE_THRESHOLD ({:.4} < {:.4})",
                new_entity_id.0, paired_entity_id.0, best_rl_confidence, app_config::MIN_GEO_PAIR_CONFIDENCE_THRESHOLD);
        }
    }

    Ok((false, None))
}

async fn create_new_pairs_from_candidates_list(
    tx: &Transaction<'_>,
    db_stmts: &DbStatements, // Contains SQL strings
    candidate_locations: &[(EntityId, f64, f64)],
    has_postgis: bool,
    now: &NaiveDateTime,
    pool: &PgPool,
    reinforcement_orchestrator: Option<&Mutex<MatchingOrchestrator>>,
    pipeline_run_id: &str,
    thresholds_config: &[(f64, f64)],
    globally_processed_entities: &mut HashSet<EntityId>,
) -> Result<MatchResult> {
    let mut local_match_result = MatchResult {
        groups_created: 0,
        entities_matched: 0,
        entities_added: 0,
        entities_skipped: 0,
        processed_entities: HashSet::<EntityId>::new(), // Initialize with an empty HashSet
    };

    for (threshold_dist, base_conf_for_thresh) in thresholds_config.iter() {
        let current_tier_candidates: Vec<_> = candidate_locations
            .iter()
            .filter(|(id, _, _)| !globally_processed_entities.contains(id))
            .cloned()
            .collect();
        if current_tier_candidates.len() < 2 {
            continue;
        }

        let raw_entity_id_clusters: Vec<Vec<EntityId>> = if has_postgis {
            postgis::create_temp_location_table(tx).await?;
            postgis::insert_location_data(tx, &current_tier_candidates).await?;
            let spatial_clusters = postgis::find_location_clusters(tx, *threshold_dist).await?;
            postgis::cleanup_temp_tables(tx).await?;
            spatial_clusters
                .into_iter()
                .map(|sc| sc.entity_ids)
                .collect()
        } else {
            build_haversine_proximity_clusters(&current_tier_candidates, *threshold_dist)
        };

        for entity_id_set in raw_entity_id_clusters {
            if entity_id_set.len() < 2 {
                continue;
            }

            for i in 0..entity_id_set.len() {
                for j in (i + 1)..entity_id_set.len() {
                    let e1_id = &entity_id_set[i];
                    let e2_id = &entity_id_set[j];

                    if globally_processed_entities.contains(e1_id)
                        || globally_processed_entities.contains(e2_id)
                    {
                        continue;
                    }
                    // Use consistency_check_sql from DbStatements
                    let e1_check_row = tx
                        .query_one(db_stmts.consistency_check_sql, &[&e1_id.0])
                        .await?;
                    let e1_check: i64 = e1_check_row.get(0);
                    let e2_check_row = tx
                        .query_one(db_stmts.consistency_check_sql, &[&e2_id.0])
                        .await?;
                    let e2_check: i64 = e2_check_row.get(0);

                    if e1_check > 0 || e2_check > 0 {
                        if e1_check > 0 {
                            globally_processed_entities.insert(e1_id.clone());
                        }
                        if e2_check > 0 {
                            globally_processed_entities.insert(e2_id.clone());
                        }
                        local_match_result.entities_skipped += 1;
                        continue;
                    }

                    let loc1_data = current_tier_candidates
                        .iter()
                        .find(|(id, _, _)| id == e1_id);
                    let loc2_data = current_tier_candidates
                        .iter()
                        .find(|(id, _, _)| id == e2_id);

                    if let (Some((_, lat1, lon1)), Some((_, lat2, lon2))) = (loc1_data, loc2_data) {
                        let distance = calculate_haversine_distance(*lat1, *lon1, *lat2, *lon2);

                        if distance <= *threshold_dist {
                            match service_utils::compare_entity_services(tx, e1_id, e2_id).await {
                                Ok(service_sim) if service_sim >= SERVICE_SIMILARITY_THRESHOLD => {
                                    let (pred_method, rl_conf, features) = get_rl_prediction(
                                        pool,
                                        reinforcement_orchestrator,
                                        e1_id,
                                        e2_id,
                                    )
                                    .await?;
                                    let final_conf = calculate_final_confidence(
                                        pred_method.as_ref(),
                                        rl_conf,
                                        *base_conf_for_thresh,
                                    );

                                    if final_conf >= MIN_GEO_PAIR_CONFIDENCE_THRESHOLD {
                                        let (id1_o, lat1_o, lon1_o, id2_o, lat2_o, lon2_o) =
                                            canonical_order_pair(
                                                e1_id, *lat1, *lon1, e2_id, *lat2, *lon2,
                                            );
                                        let geo_details = GeospatialMatchValue {
                                            latitude1: lat1_o,
                                            longitude1: lon1_o,
                                            latitude2: lat2_o,
                                            longitude2: lon2_o,
                                            distance,
                                        };
                                        let mv = MatchValues::Geospatial(geo_details);
                                        let new_pg_id = Uuid::new_v4().to_string();

                                        // Use new_geospatial_pair_sql from DbStatements
                                        insert_new_pair_into_db(
                                            tx,
                                            db_stmts.new_geospatial_pair_sql,
                                            &new_pg_id,
                                            &id1_o,
                                            &id2_o,
                                            &mv,
                                            final_conf,
                                            now,
                                        )
                                        .await?;

                                        info!("CREATE_NEW: Geo pair {} ({},{}) created. Dist:{:.1}m, ServSim:{:.2}, Conf:{:.4}", new_pg_id, id1_o.0, id2_o.0, distance, service_sim, final_conf);

                                        local_match_result.groups_created += 1;
                                        if globally_processed_entities.insert(id1_o.clone()) {
                                            local_match_result.entities_matched += 1;
                                        }
                                        if globally_processed_entities.insert(id2_o.clone()) {
                                            local_match_result.entities_matched += 1;
                                        }

                                        log_suggestion_if_needed(
                                            tx,
                                            pipeline_run_id,
                                            &id1_o,
                                            Some(&EntityGroupId(new_pg_id)),
                                            final_conf,
                                            &mv,
                                            Some(service_sim),
                                        )
                                        .await;
                                        log_ml_feedback(
                                            pool,
                                            reinforcement_orchestrator,
                                            &id1_o,
                                            &id2_o,
                                            final_conf,
                                            Some(PairMlSnapshot {
                                                features,
                                                predicted_method: pred_method,
                                                prediction_confidence: rl_conf,
                                            }),
                                        )
                                        .await;
                                    }
                                }
                                Ok(service_sim) => {
                                    // Service similarity below threshold
                                    debug!("Service similarity {:.2} for pair ({},{}) below threshold {:.2}. Skipping.", service_sim, e1_id.0, e2_id.0, SERVICE_SIMILARITY_THRESHOLD);
                                }
                                Err(e) => {
                                    // Error during service comparison
                                    warn!(
                                        "Service comparison failed for pair ({},{}): {}. Skipping.",
                                        e1_id.0, e2_id.0, e
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    local_match_result.entities_added = local_match_result.entities_matched;
    Ok(local_match_result)
}

// --- Helper Functions ---

fn calculate_final_confidence(
    predicted_method: Option<&MatchMethodType>,
    rl_confidence: Option<f64>,
    base_confidence_from_threshold: f64,
) -> f64 {
    match (predicted_method, rl_confidence) {
        (Some(MatchMethodType::Geospatial), Some(rl_conf)) => {
            (rl_conf * 0.7) + (base_confidence_from_threshold * 0.3)
        }
        (Some(_), Some(_rl_conf)) => base_confidence_from_threshold * 0.5, // RL predicted other method, penalize geo
        _ => base_confidence_from_threshold, // No RL or RL failed, rely on base
    }
}

async fn get_rl_prediction(
    pool: &PgPool, // Use PgPool for operations outside the main transaction
    orchestrator: Option<&Mutex<MatchingOrchestrator>>,
    e1_id: &EntityId,
    e2_id: &EntityId,
) -> Result<(Option<MatchMethodType>, Option<f64>, Option<Vec<f64>>)> {
    if let Some(orch_mutex) = orchestrator {
        // Context extraction might need its own connection if it's complex
        match MatchingOrchestrator::extract_pair_context(pool, e1_id, e2_id).await {
            Ok(features) => {
                let guard = orch_mutex.lock().await;
                match guard.predict_method_with_context(&features) {
                    Ok((method, confidence)) => {
                        Ok((Some(method), Some(confidence), Some(features)))
                    }
                    Err(e) => {
                        warn!(
                            "RL prediction for pair ({},{}) failed: {}. Features were extracted.",
                            e1_id.0, e2_id.0, e
                        );
                        Ok((None, None, Some(features))) // Return features even if prediction fails
                    }
                }
            }
            Err(e) => {
                warn!(
                    "RL context extraction for pair ({},{}) failed: {}",
                    e1_id.0, e2_id.0, e
                );
                Ok((None, None, None))
            }
        }
    } else {
        Ok((None, None, None))
    }
}

async fn calculate_distance_between_points(
    tx: &Transaction<'_>,
    db_stmts: &DbStatements, // db_stmts contains SQL string
    lat1: f64,
    lon1: f64,
    lat2: f64,
    lon2: f64,
    has_postgis: bool,
) -> Result<f64> {
    if has_postgis {
        // Use distance_sql from DbStatements
        let row = tx
            .query_one(db_stmts.distance_sql, &[&lon1, &lat1, &lon2, &lat2])
            .await
            .with_context(|| {
                format!(
                    "PostGIS distance calculation failed for points ({},{}) and ({},{})",
                    lat1, lon1, lat2, lon2
                )
            })?;
        Ok(row.get("distance"))
    } else {
        Ok(calculate_haversine_distance(lat1, lon1, lat2, lon2))
    }
}

fn canonical_order_pair<'a>(
    id1: &'a EntityId,
    lat1: f64,
    lon1: f64,
    id2: &'a EntityId,
    lat2: f64,
    lon2: f64,
) -> (EntityId, f64, f64, EntityId, f64, f64) {
    // Return owned EntityId
    if id1.0 < id2.0 {
        (id1.clone(), lat1, lon1, id2.clone(), lat2, lon2)
    } else {
        (id2.clone(), lat2, lon2, id1.clone(), lat1, lon1)
    }
}

async fn insert_new_pair_into_db(
    tx: &Transaction<'_>,
    new_geospatial_pair_sql: &str, // Pass the SQL string directly
    pair_guid: &str,
    e1_id: &EntityId,
    e2_id: &EntityId,
    match_values: &MatchValues,
    confidence: f64,
    now: &NaiveDateTime,
) -> Result<()> {
    let mv_json = serde_json::to_value(match_values).with_context(|| {
        format!(
            "Failed to serialize MatchValues for DB insert of pair {}",
            pair_guid
        )
    })?;
    tx.execute(
        new_geospatial_pair_sql, // Use the passed SQL string
        &[
            &pair_guid,
            &e1_id.0,
            &e2_id.0,
            &MatchMethodType::Geospatial.as_str(),
            &mv_json,
            &confidence,
            now,
            now,
        ],
    )
    .await
    .with_context(|| {
        format!(
            "DB insert failed for new_geospatial_pair_sql, pair_guid: {}",
            pair_guid
        )
    })?;
    Ok(())
}

fn build_haversine_proximity_clusters(
    locations: &[(EntityId, f64, f64)],
    threshold_dist: f64,
) -> Vec<Vec<EntityId>> {
    if locations.len() < 2 {
        return Vec::new();
    }
    let mut clusters: Vec<Vec<EntityId>> = Vec::new();
    let mut unclustered_indices: HashSet<usize> = (0..locations.len()).collect();

    while let Some(start_idx) = unclustered_indices.iter().next().copied() {
        // Copied to avoid borrow issue
        unclustered_indices.remove(&start_idx);
        let mut current_cluster_indices = vec![start_idx];
        let mut current_cluster_ids = vec![locations[start_idx].0.clone()];
        let mut i = 0;
        while i < current_cluster_indices.len() {
            let current_member_idx = current_cluster_indices[i];
            let (_, lat1, lon1) = locations[current_member_idx];

            let mut neighbors_found_this_iter = Vec::new();
            for &potential_neighbor_idx in unclustered_indices.iter() {
                // Iterate over a copy or by index
                let (_, lat2, lon2) = locations[potential_neighbor_idx];
                if calculate_haversine_distance(lat1, lon1, lat2, lon2) <= threshold_dist {
                    neighbors_found_this_iter.push(potential_neighbor_idx);
                }
            }
            for neighbor_idx in neighbors_found_this_iter {
                if unclustered_indices.remove(&neighbor_idx) {
                    current_cluster_indices.push(neighbor_idx);
                    current_cluster_ids.push(locations[neighbor_idx].0.clone());
                }
            }
            i += 1;
        }
        if current_cluster_ids.len() >= 2 {
            // Only add if it's a valid cluster (2 or more entities)
            clusters.push(current_cluster_ids);
        }
    }
    clusters
}

async fn log_suggestion_if_needed(
    tx: &Transaction<'_>,
    pipeline_run_id: &str,
    entity_id_for_suggestion: &EntityId,
    pair_group_id: Option<&EntityGroupId>,
    confidence: f64,
    match_values: &MatchValues,
    service_similarity: Option<f64>,
) {
    if confidence < app_config::MODERATE_LOW_SUGGESTION_THRESHOLD {
        let priority = if confidence < app_config::CRITICALLY_LOW_SUGGESTION_THRESHOLD {
            2
        } else {
            1
        };
        let details_json = serde_json::json!({
            "method_type": MatchMethodType::Geospatial.as_str(),
            "match_details": match_values,
            "service_similarity_score": service_similarity,
            "final_confidence": confidence,
            "pair_group_id": pair_group_id.map(|id| id.0.clone()),
        });
        let reason_msg = format!(
            "Entity {} part of geospatial pair {} with low confidence {:.4}. Service sim: {:.2?}.",
            entity_id_for_suggestion.0,
            pair_group_id.map_or_else(|| "N/A".to_string(), |id| id.0.clone()),
            confidence,
            service_similarity
        );
        // Use the refactored project_db::insert_suggestion which takes &impl GenericClient
        let suggestion = NewSuggestedAction {
            pipeline_run_id: Some(pipeline_run_id.to_string()),
            action_type: ActionType::ReviewEntityInGroup.as_str().to_string(),
            entity_id: Some(entity_id_for_suggestion.0.clone()), // Suggestion for one of the entities in the pair
            group_id_1: pair_group_id.map(|id| id.0.clone()),
            group_id_2: None,
            cluster_id: None,
            triggering_confidence: Some(confidence),
            details: Some(details_json),
            reason_code: Some("LOW_GEO_PAIR_CONF".to_string()),
            reason_message: Some(reason_msg),
            priority,
            status: SuggestionStatus::PendingReview.as_str().to_string(),
            reviewer_id: None,
            reviewed_at: None,
            review_notes: None,
        };
        if let Err(e) = project_db::insert_suggestion(tx, &suggestion).await {
            warn!(
                "Failed to log suggestion for geo pair involving entity {}: {}",
                entity_id_for_suggestion.0, e
            );
        } else {
            info!(
                "Logged suggestion for entity {} in geo pair {:?}",
                entity_id_for_suggestion.0,
                pair_group_id.map(|id| &id.0)
            );
        }
    }
}

async fn log_ml_feedback(
    pool: &PgPool,
    orchestrator: Option<&Mutex<MatchingOrchestrator>>,
    e1_id: &EntityId,
    e2_id: &EntityId,
    final_confidence: f64,
    ml_snapshot: Option<PairMlSnapshot>,
) {
    if let Some(orch_mutex) = orchestrator {
        if let Some(snapshot) = ml_snapshot {
            let mut guard = orch_mutex.lock().await;
            // Ensure actual_method_type and actual_confidence reflect the geospatial method's outcome
            if let Err(e) = guard
                .log_match_result(
                    pool,
                    e1_id,
                    e2_id,
                    &snapshot
                        .predicted_method
                        .unwrap_or(MatchMethodType::Geospatial), // ML's prediction
                    snapshot.prediction_confidence.unwrap_or(final_confidence), // ML's confidence for its prediction
                    true,                                                       // is_match
                    snapshot.features.as_ref(),
                    Some(&MatchMethodType::Geospatial), // actual_method_type that formed the pair
                    Some(final_confidence),             // actual_confidence of the geospatial pair
                )
                .await
            {
                warn!(
                    "Failed to log ML feedback for geo pair ({},{}): {}",
                    e1_id.0, e2_id.0, e
                );
            }
        }
    }
}
