// src/reinforcement/feedback_processor.rs

use anyhow::{Context, Result};
// chrono::Utc is used for timestamps if needed, but FeedbackItem doesn't have one.
// NaiveDateTime might be used if interfacing with database timestamps directly.
use futures::future::{join_all, try_join_all}; // try_join_all might be more appropriate if individual futures can fail
use log::{debug, info, warn};
use pgvector::Vector;
use std::collections::{HashMap, HashSet};
use tokio_postgres::Client as PgClient;
use uuid::Uuid; // Only if Uuid is used for new IDs generated in this specific file.

// Using existing types from provided files
use crate::models::EntityId;
use crate::reinforcement::get_stored_entity_features;
// From src/models.rs
use crate::reinforcement::types::{FeedbackItem, TrainingExample};
use crate::utils::{cosine_similarity_candle, extract_domain}; // From src/reinforcement/types.rs

// Local cache structure for organization details needed for pairwise features
#[derive(Debug, Clone)]
struct CachedOrgDetails {
    // id: EntityId, // Not strictly needed if key of HashMap is EntityId as String
    name: Option<String>,
    embedding: Option<Vec<f32>>,
    url: Option<String>,
}

// --- Data Structures for Raw Data Caches (local to this module) ---
#[derive(Debug, Clone)]
struct RawServiceData {
    id: String, // ServiceId could be a newtype from models.rs if defined
    embedding_v2: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
struct RawLocationData {
    id: String, // LocationId could be a newtype
    latitude: Option<f64>,
    longitude: Option<f64>,
    // geom: Option<String>, // WKT string, if used
}

// Helper struct to organize pairwise features before they are flattened into the final feature vector.
// This is an intermediate representation.
struct PairwiseFeatureSet {
    name_similarity: f64,
    embedding_similarity: f64,
    max_service_similarity: f64,
    geographic_distance_score: f64, // Normalized score
    shared_domain: f64,
    shared_phone: f64,
    service_geo_semantic_score: f64,
}

impl PairwiseFeatureSet {
    // Order must be consistent with how features are consumed,
    // typically aligned with feature metadata (indices 12-18 from feature_extraction.rs)
    fn to_vec(&self) -> Vec<f64> {
        vec![
            self.name_similarity,
            self.embedding_similarity,
            self.max_service_similarity,
            self.geographic_distance_score,
            self.shared_domain,
            self.shared_phone,
            self.service_geo_semantic_score,
        ]
    }
}

// Earth radius in kilometers, used for Haversine distance
const EARTH_RADIUS_KM: f64 = 6371.0;
// MAX_RELEVANT_DISTANCE_KM for normalizing general geographic distance score.
// Value should be based on `feature_extraction.rs` or project defaults.
const MAX_RELEVANT_DISTANCE_KM: f64 = 100.0;
// MAX_SERVICE_GEO_RELEVANT_DISTANCE_KM for service_geo_semantic_score's geo component.
const MAX_SERVICE_GEO_RELEVANT_DISTANCE_KM: f64 = 50.0;

// Helper for pgvector::Vector to Vec<f32>
fn convert_pg_vector_to_vec_f32(pg_vec: Option<Vector>) -> Option<Vec<f32>> {
    pg_vec.map(|v| v.as_slice().to_vec())
}

// Helper function to calculate Haversine distance between two points
fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let d_lat = (lat2 - lat1).to_radians();
    let d_lon = (lon2 - lon1).to_radians();
    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();

    let a =
        (d_lat / 2.0).sin().powi(2) + lat1_rad.cos() * lat2_rad.cos() * (d_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();

    EARTH_RADIUS_KM * c
}

pub async fn prepare_pairwise_training_data_batched(
    client: &PgClient,
    feedback_batch: &[FeedbackItem], // Using FeedbackItem from reinforcement/types.rs
) -> Result<Vec<TrainingExample>> {
    // Using TrainingExample from reinforcement/types.rs
    info!(
        "Starting batched pairwise training data preparation for {} feedback items.",
        feedback_batch.len()
    );

    // Step 1: Collect Unique Entity IDs from the feedback batch
    let mut unique_entity_ids_set = HashSet::new();
    for item in feedback_batch {
        unique_entity_ids_set.insert(item.entity_id1.clone()); // entity_id1 is String
        unique_entity_ids_set.insert(item.entity_id2.clone()); // entity_id2 is String
    }
    let entity_ids_vec_str: Vec<String> = unique_entity_ids_set.into_iter().collect();

    if entity_ids_vec_str.is_empty() {
        info!("No unique entity IDs found in feedback batch. Returning empty training data.");
        return Ok(Vec::new());
    }
    info!(
        "Collected {} unique entity IDs for feature extraction.",
        entity_ids_vec_str.len()
    );

    // Step 2: Batch Fetch/Extract Individual Entity Features
    let mut entity_features_cache: HashMap<String, Vec<f64>> = HashMap::new();
    // This loop assumes feature_extraction::get_stored_entity_features can be called individually.
    // A batch version `get_batch_stored_entity_features` would be more efficient if available.
    for entity_id_str in &entity_ids_vec_str {
        // feature_extraction::get_stored_entity_features expects &EntityId
        match get_stored_entity_features(client, &EntityId(entity_id_str.clone())).await {
            Ok(features) => {
                // Ensure it's 12 features as expected by individual feature set size
                if features.len() == 12 {
                    entity_features_cache.insert(entity_id_str.clone(), features);
                } else {
                    warn!("Features for entity {} had unexpected length {}. Expected 12. Using default.", entity_id_str, features.len());
                    entity_features_cache.insert(entity_id_str.clone(), vec![0.0; 12]);
                }
            }
            Err(e) => {
                warn!(
                    "Failed to get/extract features for entity {}: {}. Using default.",
                    entity_id_str, e
                );
                entity_features_cache.insert(entity_id_str.clone(), vec![0.0; 12]);
            }
        }
    }
    info!(
        "Fetched/Extracted individual features for {} entities.",
        entity_features_cache.len()
    );

    // Step 3: Batch Fetch Raw Data for Pairwise Calculations
    let mut org_details_cache: HashMap<String, CachedOrgDetails> = HashMap::new();
    let mut entity_services_cache: HashMap<String, Vec<RawServiceData>> = HashMap::new();
    let mut entity_locations_cache: HashMap<String, Vec<RawLocationData>> = HashMap::new();
    let mut entity_phones_cache: HashMap<String, Vec<String>> = HashMap::new();
    let mut service_to_location_ids_map: HashMap<String, Vec<String>> = HashMap::new();
    let mut all_raw_locations_for_services_cache: HashMap<String, RawLocationData> = HashMap::new();

    // Fetch Organization Details (name, embedding, url)
    if !entity_ids_vec_str.is_empty() {
        let org_rows = client
            .query(
                "SELECT e.id as entity_id_str, o.name, o.url, o.embedding
                 FROM entity e
                 JOIN organization o ON e.organization_id = o.id
                 WHERE e.id = ANY($1)",
                &[&entity_ids_vec_str],
            )
            .await
            .context("Failed to fetch organization details for entities")?;

        for row in org_rows {
            let entity_id_str: String = row.get("entity_id_str");
            org_details_cache.insert(
                entity_id_str,
                CachedOrgDetails {
                    name: row.try_get("name").ok(),
                    embedding: convert_pg_vector_to_vec_f32(
                        row.try_get("embedding").ok().flatten(),
                    ),
                    url: row.try_get("url").ok(),
                },
            );
        }
        info!(
            "Fetched organization details for {} entities.",
            org_details_cache.len()
        );
    }

    // Fetch Service Data for entities
    if !entity_ids_vec_str.is_empty() {
        let service_rows = client
            .query(
                "SELECT ef.entity_id as entity_id_str, s.id as service_id, s.embedding_v2
                 FROM entity_feature ef
                 JOIN service s ON ef.table_id = s.id
                 WHERE ef.table_name = 'service' AND ef.entity_id = ANY($1)",
                &[&entity_ids_vec_str],
            )
            .await
            .context("Failed to fetch service data for entities")?;

        for row in service_rows {
            let entity_id_str: String = row.get("entity_id_str");
            entity_services_cache
                .entry(entity_id_str)
                .or_default()
                .push(RawServiceData {
                    id: row.get("service_id"),
                    embedding_v2: convert_pg_vector_to_vec_f32(
                        row.try_get("embedding_v2").ok().flatten(),
                    ),
                });
        }
        info!(
            "Fetched service data into entity_services_cache for {} entities.",
            entity_services_cache.keys().len()
        );
    }

    // Fetch Location Data for entities
    if !entity_ids_vec_str.is_empty() {
        let location_rows = client
            .query(
                "SELECT ef.entity_id as entity_id_str, l.id as location_id, l.latitude, l.longitude
                 FROM entity_feature ef
                 JOIN location l ON ef.table_id = l.id
                 WHERE ef.table_name = 'location' AND ef.entity_id = ANY($1)",
                &[&entity_ids_vec_str],
            )
            .await
            .context("Failed to fetch location data for entities")?;

        for row in location_rows {
            let entity_id_str: String = row.get("entity_id_str");
            entity_locations_cache
                .entry(entity_id_str)
                .or_default()
                .push(RawLocationData {
                    id: row.get("location_id"),
                    latitude: row.try_get("latitude").ok(),
                    longitude: row.try_get("longitude").ok(),
                });
        }
        info!(
            "Fetched location data into entity_locations_cache for {} entities.",
            entity_locations_cache.keys().len()
        );
    }

    // Fetch Phone Data for entities
    if !entity_ids_vec_str.is_empty() {
        let phone_rows = client
            .query(
                "SELECT ef.entity_id as entity_id_str, p.number -- Assuming p.number is already normalized
                 FROM entity_feature ef
                 JOIN phone p ON ef.table_id = p.id
                 WHERE ef.table_name = 'phone' AND ef.entity_id = ANY($1) AND p.number IS NOT NULL AND p.number <> ''",
                &[&entity_ids_vec_str],
            )
            .await
            .context("Failed to fetch phone data for entities")?;

        for row in phone_rows {
            let entity_id_str: String = row.get("entity_id_str");
            let phone_number: String = row.get("number");
            // If normalization is needed and not done in DB:
            // let normalized_phone = string_utils::normalize_phone_number(&phone_number);
            // entity_phones_cache.entry(entity_id_str).or_default().push(normalized_phone);
            entity_phones_cache
                .entry(entity_id_str)
                .or_default()
                .push(phone_number);
        }
        info!(
            "Fetched phone data into entity_phones_cache for {} entities.",
            entity_phones_cache.keys().len()
        );
    }

    // Fetch Service-at-Location Data and their RawLocationData for service_geo_semantic_score
    let mut all_service_ids_for_entities: Vec<String> = Vec::new();
    for services_vec in entity_services_cache.values() {
        for service_data in services_vec {
            all_service_ids_for_entities.push(service_data.id.clone());
        }
    }
    // Deduplicate service IDs
    let unique_service_ids: Vec<String> = all_service_ids_for_entities
        .into_iter()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    if !unique_service_ids.is_empty() {
        let sal_rows = client
            .query(
                "SELECT sal.service_id, sal.location_id, loc.latitude, loc.longitude
                 FROM service_at_location sal
                 JOIN location loc ON sal.location_id = loc.id
                 WHERE sal.service_id = ANY($1)",
                &[&unique_service_ids],
            )
            .await
            .context("Failed to fetch service_at_location and associated location data")?;

        for row in sal_rows {
            let service_id: String = row.get("service_id");
            let location_id: String = row.get("location_id");

            service_to_location_ids_map
                .entry(service_id.clone())
                .or_default()
                .push(location_id.clone());

            // Cache the RawLocationData for these specific locations if not already cached
            // This avoids re-fetching or complex joins if entity_locations_cache is not comprehensive for service locations.
            if !all_raw_locations_for_services_cache.contains_key(&location_id) {
                all_raw_locations_for_services_cache.insert(
                    location_id.clone(),
                    RawLocationData {
                        id: location_id, // Store the ID with the data
                        latitude: row.try_get("latitude").ok(),
                        longitude: row.try_get("longitude").ok(),
                    },
                );
            }
        }
        info!(
            "Fetched service_at_location data for {} services.",
            service_to_location_ids_map.len()
        );
        info!(
            "Cached {} unique raw locations relevant to services.",
            all_raw_locations_for_services_cache.len()
        );
    }

    // Step 4 & 5: Calculate Pairwise Features and Construct TrainingExamples
    let mut training_examples_output: Vec<TrainingExample> =
        Vec::with_capacity(feedback_batch.len());

    for feedback_item in feedback_batch {
        let entity1_id_str = &feedback_item.entity_id1;
        let entity2_id_str = &feedback_item.entity_id2;

        let entity1_individual_features = entity_features_cache
            .get(entity1_id_str)
            .cloned()
            .unwrap_or_else(|| vec![0.0; 12]); // Default if missing
        let entity2_individual_features = entity_features_cache
            .get(entity2_id_str)
            .cloned()
            .unwrap_or_else(|| vec![0.0; 12]); // Default if missing

        let org1_details = org_details_cache.get(entity1_id_str);
        let org2_details = org_details_cache.get(entity2_id_str);

        // Calculate 7 pairwise features
        let name_similarity = match (org1_details, org2_details) {
            (Some(d1), Some(d2)) => match (&d1.name, &d2.name) {
                (Some(n1), Some(n2)) => {
                    strsim::jaro_winkler(&n1.to_lowercase(), &n2.to_lowercase())
                }
                _ => 0.0,
            },
            _ => 0.0,
        };

        let embedding_similarity = match (org1_details, org2_details) {
            (Some(d1), Some(d2)) => match (&d1.embedding, &d2.embedding) {
                (Some(e1), Some(e2)) => cosine_similarity_candle(e1, e2).unwrap_or(0.0),
                _ => 0.0,
            },
            _ => 0.0,
        };

        let services1 = entity_services_cache
            .get(entity1_id_str)
            .map_or_else(Vec::new, |v| v.clone());
        let services2 = entity_services_cache
            .get(entity2_id_str)
            .map_or_else(Vec::new, |v| v.clone());
        let max_service_similarity =
            calculate_max_service_similarity_from_raw(&services1, &services2);

        let locations1 = entity_locations_cache
            .get(entity1_id_str)
            .map_or_else(Vec::new, |v| v.clone());
        let locations2 = entity_locations_cache
            .get(entity2_id_str)
            .map_or_else(Vec::new, |v| v.clone());
        let geographic_distance_score =
            calculate_geographic_distance_score_from_raw(&locations1, &locations2);

        let shared_domain = match (org1_details, org2_details) {
            (Some(d1), Some(d2)) => match (&d1.url, &d2.url) {
                (Some(u1), Some(u2)) => {
                    let domain1 = extract_domain(u1);
                    let domain2 = extract_domain(u2);
                    if domain1.is_some() && domain1 == domain2 {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            },
            _ => 0.0,
        };

        let phones1 = entity_phones_cache
            .get(entity1_id_str)
            .map_or_else(Vec::new, |v| v.clone());
        let phones2 = entity_phones_cache
            .get(entity2_id_str)
            .map_or_else(Vec::new, |v| v.clone());
        let shared_phone = calculate_shared_phone_from_raw(&phones1, &phones2);

        let service_geo_semantic_score = calculate_service_geo_semantic_score_from_raw(
            entity1_id_str,
            entity2_id_str,
            &entity_services_cache,
            &service_to_location_ids_map,
            &all_raw_locations_for_services_cache,
        );

        let pairwise_features_set = PairwiseFeatureSet {
            name_similarity,
            embedding_similarity,
            max_service_similarity,
            geographic_distance_score,
            shared_domain,
            shared_phone,
            service_geo_semantic_score,
        };
        let pairwise_features_vec = pairwise_features_set.to_vec();

        // Construct the final feature vector: [entity1_features, entity2_features, pairwise_features]
        let mut final_features = Vec::with_capacity(12 + 12 + 7);
        final_features.extend_from_slice(&entity1_individual_features);
        final_features.extend_from_slice(&entity2_individual_features);
        final_features.extend_from_slice(&pairwise_features_vec);

        training_examples_output.push(TrainingExample {
            features: final_features,
            best_method: feedback_item.method_type.clone(), // From FeedbackItem
            confidence: feedback_item.confidence,           // From FeedbackItem
                                                            // The label (FeedbackItem.was_correct) is not part of reinforcement.types.TrainingExample.
                                                            // It's assumed the training loop will use was_correct alongside this TrainingExample.
        });
    }

    info!(
        "Successfully prepared {} training examples.",
        training_examples_output.len()
    );
    Ok(training_examples_output)
}

// --- Synchronous Pairwise Feature Calculation Helper Functions (Implementations from previous response) ---

fn calculate_max_service_similarity_from_raw(
    services1: &[RawServiceData],
    services2: &[RawServiceData],
) -> f64 {
    let mut max_similarity = 0.0;
    if services1.is_empty() || services2.is_empty() {
        return 0.0;
    }

    for s1_data in services1 {
        if let Some(ref emb1) = s1_data.embedding_v2 {
            if emb1.is_empty() {
                continue;
            }
            for s2_data in services2 {
                if let Some(ref emb2) = s2_data.embedding_v2 {
                    if emb2.is_empty() {
                        continue;
                    }
                    match cosine_similarity_candle(emb1, emb2) {
                        Ok(sim) => {
                            if sim > max_similarity {
                                max_similarity = sim;
                            }
                        }
                        Err(e) => {
                            // Log the error. Service IDs might be useful context here.
                            log::warn!(
                                "Candle cosine similarity failed for service pair (IDs: {}, {}): {}. Skipping this pair for max_similarity.",
                                s1_data.id,
                                s2_data.id,
                                e
                            );
                        }
                    }
                }
            }
        }
    }
    max_similarity
}

fn calculate_geographic_distance_score_from_raw(
    locations1: &[RawLocationData],
    locations2: &[RawLocationData],
) -> f64 {
    if locations1.is_empty() || locations2.is_empty() {
        return 0.0;
    }
    let mut min_distance = f64::MAX;
    let mut found_valid_pair = false;

    for loc1 in locations1 {
        if let (Some(lat1), Some(lon1)) = (loc1.latitude, loc1.longitude) {
            for loc2 in locations2 {
                if let (Some(lat2), Some(lon2)) = (loc2.latitude, loc2.longitude) {
                    min_distance = min_distance.min(haversine_distance(lat1, lon1, lat2, lon2));
                    found_valid_pair = true;
                }
            }
        }
    }

    if !found_valid_pair {
        return 0.0;
    }
    if min_distance >= MAX_RELEVANT_DISTANCE_KM {
        return 0.0;
    }

    1.0 - (min_distance / MAX_RELEVANT_DISTANCE_KM)
}

fn calculate_shared_phone_from_raw(
    phones1_normalized: &[String],
    phones2_normalized: &[String],
) -> f64 {
    if phones1_normalized.is_empty() || phones2_normalized.is_empty() {
        return 0.0;
    }
    let phone_set1: HashSet<&String> = phones1_normalized
        .iter()
        .filter(|p| !p.is_empty())
        .collect();
    for phone2 in phones2_normalized.iter().filter(|p| !p.is_empty()) {
        if phone_set1.contains(phone2) {
            return 1.0;
        }
    }
    0.0
}

fn calculate_service_geo_semantic_score_from_raw(
    entity1_id_str: &str, // Now receiving String IDs
    entity2_id_str: &str,
    entity_services_cache: &HashMap<String, Vec<RawServiceData>>,
    service_to_location_ids_map: &HashMap<String, Vec<String>>,
    all_locations_cache: &HashMap<String, RawLocationData>,
) -> f64 {
    let services1 = match entity_services_cache.get(entity1_id_str) {
        Some(s) => s,
        None => return 0.0,
    };
    let services2 = match entity_services_cache.get(entity2_id_str) {
        Some(s) => s,
        None => return 0.0,
    };

    if services1.is_empty() || services2.is_empty() {
        return 0.0;
    }

    let mut total_score_sum = 0.0;
    let mut valid_service_pairs_count = 0;

    for s1_data in services1 {
        if let Some(ref emb1) = s1_data.embedding_v2 {
            if emb1.is_empty() {
                continue;
            }
            let s1_location_ids = match service_to_location_ids_map.get(&s1_data.id) {
                Some(loc_ids) => loc_ids,
                None => continue,
            };
            if s1_location_ids.is_empty() {
                continue;
            }

            for s2_data in services2 {
                if let Some(ref emb2) = s2_data.embedding_v2 {
                    if emb2.is_empty() {
                        continue;
                    }
                    let s2_location_ids = match service_to_location_ids_map.get(&s2_data.id) {
                        Some(loc_ids) => loc_ids,
                        None => continue,
                    };
                    if s2_location_ids.is_empty() {
                        continue;
                    }

                    let semantic_sim = cosine_similarity_candle(emb1, emb2).unwrap_or(0.0);
                    if semantic_sim == 0.0 {
                        continue;
                    }

                    let mut min_geo_distance_s1_s2 = f64::MAX;
                    let mut found_s1_s2_loc_pair = false;

                    for loc_id1 in s1_location_ids {
                        if let Some(raw_loc1) = all_locations_cache.get(loc_id1) {
                            if let (Some(lat1), Some(lon1)) =
                                (raw_loc1.latitude, raw_loc1.longitude)
                            {
                                for loc_id2 in s2_location_ids {
                                    if let Some(raw_loc2) = all_locations_cache.get(loc_id2) {
                                        if let (Some(lat2), Some(lon2)) =
                                            (raw_loc2.latitude, raw_loc2.longitude)
                                        {
                                            min_geo_distance_s1_s2 = min_geo_distance_s1_s2
                                                .min(haversine_distance(lat1, lon1, lat2, lon2));
                                            found_s1_s2_loc_pair = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if !found_s1_s2_loc_pair {
                        continue;
                    }

                    let geo_proximity_score =
                        if min_geo_distance_s1_s2 >= MAX_SERVICE_GEO_RELEVANT_DISTANCE_KM {
                            0.0
                        } else {
                            1.0 - (min_geo_distance_s1_s2 / MAX_SERVICE_GEO_RELEVANT_DISTANCE_KM)
                        };

                    if geo_proximity_score > 0.0 {
                        total_score_sum += semantic_sim * geo_proximity_score;
                        valid_service_pairs_count += 1;
                    }
                }
            }
        }
    }

    if valid_service_pairs_count == 0 {
        0.0
    } else {
        total_score_sum / (valid_service_pairs_count as f64)
    }
}
