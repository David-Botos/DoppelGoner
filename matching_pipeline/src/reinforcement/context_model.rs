use anyhow::{Context, Result};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use smartcore::tree::decision_tree_classifier::SplitCriterion;
use std::collections::{HashMap, HashSet}; // Added HashSet
use uuid::Uuid;

// SmartCore imports
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;

use super::feature_extraction::get_feature_metadata; // For metadata structure
use super::types::TrainingExample;
use crate::db::PgPool;
use crate::models::EntityId; // Assuming EntityId is a newtype struct around String

// Wrapper for SmartCore RandomForestClassifier (remains the same)
#[derive(Serialize, Deserialize, Debug)]
struct RandomForestClassifierWrapper {
    serialized_model: String,
}

impl RandomForestClassifierWrapper {
    fn from_forest(
        forest: &RandomForestClassifier<f64, usize, DenseMatrix<f64>, Vec<usize>>,
    ) -> Result<Self> {
        let serialized_model = serde_json::to_string(forest)
            .map_err(|e| anyhow::anyhow!("Failed to serialize forest: {}", e))?;
        Ok(Self { serialized_model })
    }

    fn to_forest(
        &self,
    ) -> Result<RandomForestClassifier<f64, usize, DenseMatrix<f64>, Vec<usize>>> {
        let forest: RandomForestClassifier<f64, usize, DenseMatrix<f64>, Vec<usize>> =
            serde_json::from_str(&self.serialized_model)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize forest: {}", e))?;
        Ok(forest)
    }
}

#[derive(Serialize, Deserialize)]
pub struct ContextModel {
    #[serde(skip)]
    pub live_forest: Option<RandomForestClassifier<f64, usize, DenseMatrix<f64>, Vec<usize>>>,
    pub forest_for_serde: Option<RandomForestClassifierWrapper>,
    pub feature_names: Vec<String>,
    pub method_labels: Vec<String>,
    pub feature_importance: HashMap<String, f64>,
    pub version: u32,
}

impl ContextModel {
    pub fn new() -> Self {
        let feature_metadata = get_feature_metadata();
        let mut feature_names = Vec::new();
        // Entity 1 features
        for i in 0..12 {
            if i < feature_metadata.len() {
                feature_names.push(format!("{}1", feature_metadata[i].name));
            }
        }
        // Entity 2 features
        for i in 0..12 {
            if i < feature_metadata.len() {
                feature_names.push(format!("{}2", feature_metadata[i].name));
            }
        }
        // Pair relationship features
        for i in 12..feature_metadata.len() {
            feature_names.push(feature_metadata[i].name.clone());
        }

        Self {
            live_forest: None,
            forest_for_serde: None,
            feature_names,
            method_labels: vec![
                "email".to_string(),
                "phone".to_string(),
                "url".to_string(),
                "address".to_string(),
                "name".to_string(),
                "geospatial".to_string(),
            ],
            feature_importance: HashMap::new(),
            version: 1,
        }
    }

    pub async fn train(&mut self, pool: &PgPool) -> Result<()> {
        let training_data = self
            .collect_training_data(pool)
            .await
            .context("Failed to collect training data for ContextModel")?; // Added context
        if training_data.is_empty() {
            warn!("No training data available for context model. Model not trained.");
            return Ok(());
        }
        info!(
            "Training context model with {} examples",
            training_data.len()
        );
        self.train_with_data(&training_data)
            .context("Failed to train ContextModel with data")?; // Added context
        Ok(())
    }

    pub fn train_with_data(&mut self, training_data: &[TrainingExample]) -> Result<()> {
        if training_data.is_empty() {
            return Err(anyhow::anyhow!(
                "No training data provided for ContextModel."
            ));
        }
        let unique_methods: std::collections::HashSet<&String> =
            training_data.iter().map(|ex| &ex.best_method).collect();
        if unique_methods.len() < 2 {
            return Err(anyhow::anyhow!(
                "Need at least 2 different methods in training data to train. Found {}.",
                unique_methods.len()
            ));
        }

        let n_samples = training_data.len();
        // Ensure all feature vectors have the same length, matching self.feature_names
        if n_samples > 0 && training_data[0].features.len() != self.feature_names.len() {
            return Err(anyhow::anyhow!(
                "Training data feature count mismatch. Expected {}, got {}.",
                self.feature_names.len(),
                training_data[0].features.len()
            ));
        }
        let n_features = self.feature_names.len(); // Use expected feature count

        debug!(
            "Building training matrices: {} samples, {} features",
            n_samples, n_features
        );

        let mut feature_2d_vec: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
        for sample in training_data {
            if sample.features.len() == n_features {
                // Validate each sample
                feature_2d_vec.push(sample.features.clone());
            } else {
                warn!(
                    "Skipping training sample with incorrect feature count. Expected {}, got {}.",
                    n_features,
                    sample.features.len()
                );
            }
        }
        // If all samples were skipped due to feature count mismatch
        if feature_2d_vec.is_empty() && n_samples > 0 {
            return Err(anyhow::anyhow!(
                "All training samples had incorrect feature counts. Model not trained."
            ));
        }

        let x = DenseMatrix::from_2d_vec(&feature_2d_vec);
        let y: Vec<usize> = training_data
            .iter()
            .filter(|s| s.features.len() == n_features) // Filter to match feature_2d_vec
            .map(|sample| self.method_index(&sample.best_method))
            .collect();

        let forest_params = RandomForestClassifierParameters {
            criterion: SplitCriterion::Gini,
            max_depth: Some(20),
            min_samples_leaf: 1,
            min_samples_split: 5,
            n_trees: 100,
            m: None,
            keep_samples: false,
            seed: 42,
        };

        info!("Training Random Forest model for ContextModel...");
        let trained_forest_model = RandomForestClassifier::fit(&x, &y, forest_params)
            .map_err(|e| anyhow::anyhow!("Failed to train random forest: {}", e))?;

        self.live_forest = Some(trained_forest_model);
        if let Some(live_model_ref) = self.live_forest.as_ref() {
            match RandomForestClassifierWrapper::from_forest(live_model_ref) {
                Ok(wrapper) => self.forest_for_serde = Some(wrapper),
                Err(e) => {
                    warn!("Failed to create RandomForestClassifierWrapper from live_forest post-training: {}.", e);
                    self.forest_for_serde = None;
                }
            }
        }

        self.feature_importance.clear();
        for name in &self.feature_names {
            self.feature_importance
                .insert(name.clone(), 1.0 / n_features as f64);
        }
        self.version += 1;
        info!(
            "Context model training complete. New version: {}",
            self.version
        );
        Ok(())
    }

    pub fn predict_best_method(&self, features: &[f64]) -> Option<(String, f64)> {
        // ... (Implementation remains the same as in previous optimized version)
        if features.len() != self.feature_names.len() {
            warn!("Feature vector length mismatch for ContextModel prediction: got {}, expected {}. Features: {:?}", features.len(), self.feature_names.len(), features);
            return None;
        }
        if let Some(forest_model) = &self.live_forest {
            let feature_vec = vec![features.to_vec()];
            let x = DenseMatrix::from_2d_vec(&feature_vec);
            let predictions = match forest_model.predict(&x) {
                Ok(p) => p,
                Err(e) => {
                    warn!("ContextModel failed to generate prediction: {}", e);
                    return None;
                }
            };
            let class_idx = predictions[0];
            let confidence = 0.8; // Placeholder
            if class_idx < self.method_labels.len() {
                Some((self.method_labels[class_idx].clone(), confidence))
            } else {
                warn!("ContextModel prediction index out of bounds: got {}, max allowed {}. Labels: {:?}", class_idx, self.method_labels.len() - 1, self.method_labels);
                None
            }
        } else {
            debug!("No trained ContextModel (live_forest is None) available for prediction.");
            None
        }
    }

    fn method_index(&self, method: &str) -> usize {
        // ... (Implementation remains the same)
        self.method_labels.iter().position(|m| m == method).unwrap_or_else(|| {
            warn!("Method '{}' not found in method_labels for ContextModel, defaulting to index 0. Labels: {:?}", method, self.method_labels);
            0
        })
    }

    pub fn feature_importance(&self) -> Vec<(String, f64)> {
        // ... (Implementation remains the same)
        let mut result: Vec<(String, f64)> = self
            .feature_importance
            .iter()
            .map(|(name, score)| (name.clone(), *score))
            .collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Collects training data by fetching human review feedback and then batch-fetching entity features.
    async fn collect_training_data(&self, pool: &PgPool) -> Result<Vec<TrainingExample>> {
        info!("Collecting training data for ContextModel (batched approach)...");
        let conn = pool
            .get()
            .await
            .context("Failed to get DB connection for collect_training_data")?;

        // 1. Fetch initial feedback: (entity_id_A, correct_cluster_id_for_A, method_type_suggested_by_human)
        // We are interested in cases where a human indicated a specific method was correct for a pair,
        // or implied a pair should exist by assigning to the same cluster.
        // The original query focused on `was_correct` and `correct_cluster_id`.
        // Let's refine to get pairs from human feedback.
        let feedback_rows = conn.query(
            "SELECT 
                hrd.entity_id AS entity_id_a, 
                hrd.correct_cluster_id, 
                hrf.method_type AS reviewed_method_type,
                hrf.was_correct,
                hrd.affected_entity_group_id -- If available, helps identify the original pair
            FROM clustering_metadata.human_review_method_feedback hrf
            JOIN clustering_metadata.human_review_decisions hrd ON hrf.decision_id = hrd.id
            WHERE hrd.reviewer_id <> 'ml_system'
              AND hrd.entity_id IS NOT NULL 
              AND (hrf.was_correct = TRUE OR hrd.correct_cluster_id IS NOT NULL) -- Positive examples or explicit cluster assignments
            ORDER BY hrd.created_at DESC
            LIMIT 500", // Limit for safety during development/testing
            &[],
        ).await.context("Failed to query human review feedback")?;

        debug!(
            "Fetched {} raw feedback rows for training data.",
            feedback_rows.len()
        );

        struct TrainingPairCandidate {
            entity_a_id: String,
            entity_b_id: String,
            best_method: String, // The method confirmed or implied by human review
        }
        let mut training_pair_candidates: Vec<TrainingPairCandidate> = Vec::new();
        let mut all_entity_ids_needed: HashSet<String> = HashSet::new();
        let mut cluster_to_entities_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut clusters_to_fetch: HashSet<String> = HashSet::new();

        // First pass: identify entities and clusters from feedback
        for row in &feedback_rows {
            let entity_a_id: String = row.get("entity_id_a");
            let correct_cluster_id: Option<String> = row.get("correct_cluster_id");
            let reviewed_method_type: String = row.get("reviewed_method_type");
            let was_correct: bool = row.get("was_correct");
            let affected_entity_group_id: Option<String> = row.get("affected_entity_group_id");

            if was_correct {
                // If the system's pairing (and method) was deemed correct
                if let Some(group_id) = affected_entity_group_id {
                    // Fetch the original pair from entity_group
                    let pair_row_opt = conn
                        .query_opt(
                            "SELECT entity_id_1, entity_id_2 FROM public.entity_group WHERE id = $1",
                            &[&group_id],
                        )
                        .await
                        .context(format!("Failed to fetch pair for group_id {}", group_id))?;

                    if let Some(pair_row) = pair_row_opt {
                        let e1: String = pair_row.get("entity_id_1");
                        let e2: String = pair_row.get("entity_id_2");
                        training_pair_candidates.push(TrainingPairCandidate {
                            entity_a_id: e1.clone(),
                            entity_b_id: e2.clone(),
                            best_method: reviewed_method_type.clone(),
                        });
                        all_entity_ids_needed.insert(e1);
                        all_entity_ids_needed.insert(e2);
                    }
                }
            } else if let Some(cluster_id) = correct_cluster_id {
                // If assigned to a specific cluster
                all_entity_ids_needed.insert(entity_a_id.clone());
                clusters_to_fetch.insert(cluster_id.clone());
                // We'll resolve other entities in this cluster later
            }
        }

        // 2. Batch fetch cluster members if any clusters were identified
        if !clusters_to_fetch.is_empty() {
            let cluster_ids_vec: Vec<String> = clusters_to_fetch.into_iter().collect();
            let cluster_member_rows = conn
                .query(
                    "SELECT eg.group_cluster_id, ge.entity_id
                 FROM public.group_entity ge
                 JOIN entity_group eg ON ge.entity_group_id = eg.id
                 WHERE eg.group_cluster_id = ANY($1::TEXT[])",
                    &[&cluster_ids_vec],
                )
                .await
                .context("Failed to batch fetch cluster members")?;

            for row in cluster_member_rows {
                let cluster_id: String = row.get(0);
                let member_entity_id: String = row.get(1);
                cluster_to_entities_map
                    .entry(cluster_id)
                    .or_default()
                    .push(member_entity_id.clone());
                all_entity_ids_needed.insert(member_entity_id);
            }
        }

        // Second pass: form pairs from clustered feedback
        for row in feedback_rows {
            // Iterate original feedback again
            let entity_a_id: String = row.get("entity_id_a");
            let correct_cluster_id: Option<String> = row.get("correct_cluster_id");
            let reviewed_method_type: String = row.get("reviewed_method_type"); // This is the method under review
            let was_correct: bool = row.get("was_correct");

            if !was_correct && correct_cluster_id.is_some() {
                // Human corrected by assigning to a cluster
                if let Some(members) =
                    cluster_to_entities_map.get(correct_cluster_id.as_ref().unwrap())
                {
                    for entity_b_id in members {
                        if &entity_a_id != entity_b_id {
                            // What's the "best_method" here? The human didn't specify a method, just a cluster.
                            // We might need to infer it or use a generic "ClusterConfirmed" type,
                            // or use the `reviewed_method_type` if the context implies it was about *that method* for this new pair.
                            // For now, let's assume the `reviewed_method_type` is a candidate for this new pair.
                            training_pair_candidates.push(TrainingPairCandidate {
                                entity_a_id: entity_a_id.clone(),
                                entity_b_id: entity_b_id.clone(),
                                best_method: reviewed_method_type.clone(), // Or a more generic label
                            });
                            // all_entity_ids_needed already populated
                            break; // Take first other entity in cluster to match original logic
                        }
                    }
                }
            }
        }

        if all_entity_ids_needed.is_empty() {
            info!("No entity IDs identified for feature fetching. No training data generated.");
            return Ok(Vec::new());
        }

        // 3. Batch fetch entity features
        let entity_ids_param: Vec<String> = all_entity_ids_needed.into_iter().collect();
        let feature_rows = conn
            .query(
                "SELECT entity_id, feature_name, feature_value 
             FROM clustering_metadata.entity_context_features 
             WHERE entity_id = ANY($1::TEXT[])",
                &[&entity_ids_param],
            )
            .await
            .context("Failed to batch fetch entity features")?;

        let mut entity_features_cache: HashMap<String, HashMap<String, f64>> = HashMap::new();
        for row in feature_rows {
            let entity_id: String = row.get(0);
            let feature_name: String = row.get(1);
            let feature_value: f64 = row.get(2);
            entity_features_cache
                .entry(entity_id)
                .or_default()
                .insert(feature_name, feature_value);
        }
        debug!(
            "Cached features for {} unique entities.",
            entity_features_cache.len()
        );

        // 4. Construct TrainingExamples
        let mut training_examples = Vec::new();
        let feature_metadata = get_feature_metadata(); // For ordering
        const INDIVIDUAL_FEATURE_COUNT: usize = 12;

        for candidate in training_pair_candidates {
            let e1_features_ordered = Self::get_ordered_features_from_cache(
                &candidate.entity_a_id,
                &entity_features_cache,
                &feature_metadata,
                INDIVIDUAL_FEATURE_COUNT,
            );
            let e2_features_ordered = Self::get_ordered_features_from_cache(
                &candidate.entity_b_id,
                &entity_features_cache,
                &feature_metadata,
                INDIVIDUAL_FEATURE_COUNT,
            );

            // If features for either entity are incomplete (e.g. not found in cache), skip.
            if e1_features_ordered.len() != INDIVIDUAL_FEATURE_COUNT
                || e2_features_ordered.len() != INDIVIDUAL_FEATURE_COUNT
            {
                warn!("Skipping training candidate for pair ({}, {}) due to missing/incomplete entity features.", candidate.entity_a_id, candidate.entity_b_id);
                continue;
            }

            // Pair features are currently placeholders in this model's context.
            let pair_features = vec![0.5; 7]; // 7 pair features as per get_feature_metadata

            let mut combined_features = Vec::new();
            combined_features.extend(e1_features_ordered);
            combined_features.extend(e2_features_ordered);
            combined_features.extend(pair_features);

            // Ensure the total number of features matches model expectation
            if combined_features.len() == self.feature_names.len() {
                training_examples.push(TrainingExample {
                    features: combined_features,
                    best_method: candidate.best_method,
                    confidence: 1.0, // Human feedback implies high confidence in this label
                });
            } else {
                warn!("Skipping training example for pair ({}, {}) due to final feature count mismatch. Expected {}, got {}.", 
                    candidate.entity_a_id, candidate.entity_b_id, self.feature_names.len(), combined_features.len());
            }
        }

        info!(
            "Collected {} training examples for ContextModel after batch processing.",
            training_examples.len()
        );
        Ok(training_examples)
    }

    /// Helper to reconstruct an ordered feature vector from a cache of feature maps.
    fn get_ordered_features_from_cache(
        entity_id: &str,
        cache: &HashMap<String, HashMap<String, f64>>,
        metadata: &[super::types::FeatureMetadata], // Use the type from super
        expected_count: usize,
    ) -> Vec<f64> {
        let mut ordered_features = vec![0.0; expected_count]; // Default to 0.0
        if let Some(feature_map) = cache.get(entity_id) {
            for i in 0..expected_count {
                if i < metadata.len() {
                    if let Some(value) = feature_map.get(&metadata[i].name) {
                        ordered_features[i] = *value;
                    } else {
                        // Log missing specific feature for an entity if needed
                        // warn!("Entity {} missing feature '{}', defaulting to 0.0", entity_id, metadata[i].name);
                    }
                }
            }
        } else {
            // Log if entity has no features at all in cache
            // warn!("Entity {} not found in feature cache.", entity_id);
        }
        ordered_features
    }

    // get_entity_features and get_pair_features are no longer directly used by the revised
    // collect_training_data. If they were used by other parts of ContextModel, they would need
    // to be kept or adapted. For this refactoring, we assume their logic is now incorporated
    // or handled by the batch fetching within collect_training_data.

    pub async fn save_to_db(&mut self, pool: &PgPool) -> Result<String> {
        // ... (Implementation remains the same as in previous optimized version)
        let conn = pool
            .get()
            .await
            .context("Failed to get DB conn for save_to_db")?;
        if let Some(live_model_ref) = &self.live_forest {
            match RandomForestClassifierWrapper::from_forest(live_model_ref) {
                Ok(wrapper) => self.forest_for_serde = Some(wrapper),
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "Failed to wrap live_forest for saving: {}",
                        e
                    ))
                }
            }
        } else {
            self.forest_for_serde = None;
        }
        let model_json = serde_json::to_value(&*self)
            .map_err(|e| anyhow::anyhow!("Failed to serialize ContextModel to JSON: {}", e))?;
        let id = Uuid::new_v4().to_string();
        let existing = conn.query_opt("SELECT id FROM clustering_metadata.ml_models WHERE model_type = 'context_model' ORDER BY version DESC LIMIT 1", &[]).await?;
        let model_id = existing.map_or(id, |row| row.get(0));
        let parameters = json!({ "feature_names": self.feature_names, "method_labels": self.method_labels, "num_classes": self.method_labels.len(), "version": self.version });
        let metrics = json!({ "feature_importance": self.feature_importance().into_iter().map(|(name, score)| json!({"feature": name, "importance": score})).collect::<Vec<_>>() });
        conn.execute("INSERT INTO clustering_metadata.ml_models (id, model_type, parameters, metrics, version, created_at, updated_at) VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) ON CONFLICT (id) DO UPDATE SET parameters = $3, metrics = $4, version = $5, updated_at = CURRENT_TIMESTAMP", &[&model_id, &"context_model", &parameters, &metrics, &(self.version as i32)]).await?;
        let binary_id = format!("{}_binary", model_id);
        conn.execute("INSERT INTO clustering_metadata.ml_models (id, model_type, parameters, version, created_at, updated_at) VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) ON CONFLICT (id) DO UPDATE SET parameters = $3, version = $4, updated_at = CURRENT_TIMESTAMP", &[&binary_id, &"context_model_binary", &model_json, &(self.version as i32)]).await?;
        info!(
            "Saved context model to database, ID: {}, version: {}",
            model_id, self.version
        );
        Ok(model_id)
    }

    pub async fn load_from_db(pool: &PgPool) -> Result<Self> {
        // ... (Implementation remains the same as in previous optimized version)
        let conn = pool
            .get()
            .await
            .context("Failed to get DB conn for load_from_db")?;
        let binary_row_opt = conn.query_opt("SELECT parameters FROM clustering_metadata.ml_models WHERE model_type = 'context_model_binary' ORDER BY version DESC LIMIT 1", &[]).await?;
        if let Some(binary_row) = binary_row_opt {
            let model_json: Value = binary_row.get(0);
            let mut model: ContextModel = serde_json::from_value(model_json).map_err(|e| {
                anyhow::anyhow!("Failed to deserialize ContextModel from DB JSON: {}", e)
            })?;
            if let Some(wrapper) = &model.forest_for_serde {
                match wrapper.to_forest() {
                    Ok(forest_model) => model.live_forest = Some(forest_model),
                    Err(e) => warn!("Failed to convert wrapper to live_forest during load for model v{}: {}. Model unusable for prediction.", model.version, e),
                }
            } else {
                model.live_forest = None;
            }
            info!(
                "Loaded context model from database, version: {}",
                model.version
            );
            Ok(model)
        } else {
            info!("No existing context model in DB, creating new one.");
            Ok(Self::new())
        }
    }
}
