use anyhow::Result;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use smartcore::tree::decision_tree_classifier::SplitCriterion;
use std::collections::HashMap;
use uuid::Uuid;

// SmartCore imports
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::ClassificationMetrics;

use super::feature_extraction::get_feature_metadata;
use super::types::TrainingExample;
use crate::db::PgPool;

#[derive(Serialize, Deserialize)]
pub struct ContextModel {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    forest: Option<RandomForestClassifierWrapper>,
    feature_names: Vec<String>,
    method_labels: Vec<String>,
    feature_importance: HashMap<String, f64>,
    version: u32,
}

// Wrapper for SmartCore RandomForestClassifier to help with serialization/deserialization
#[derive(Serialize, Deserialize)]
struct RandomForestClassifierWrapper {
    serialized_model: String,
}

impl RandomForestClassifierWrapper {
    fn from_forest(
        forest: &RandomForestClassifier<f64, usize, DenseMatrix<f64>, Vec<usize>>,
    ) -> Result<Self> {
        // Serialize the model to a string
        let serialized_model = serde_json::to_string(forest)
            .map_err(|e| anyhow::anyhow!("Failed to serialize forest: {}", e))?;

        Ok(Self { serialized_model })
    }

    fn to_forest(
        &self,
    ) -> Result<RandomForestClassifier<f64, usize, DenseMatrix<f64>, Vec<usize>>> {
        // Deserialize from string to RandomForestClassifier
        let forest: RandomForestClassifier<f64, usize, DenseMatrix<f64>, Vec<usize>> =
            serde_json::from_str(&self.serialized_model)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize forest: {}", e))?;

        Ok(forest)
    }
}

impl ContextModel {
    pub fn new() -> Self {
        // Extract feature names from metadata
        let feature_metadata = get_feature_metadata();
        let mut feature_names = Vec::new();

        // Entity 1 features - individual (first 12 features from metadata)
        for i in 0..12 {
            if i < feature_metadata.len() {
                feature_names.push(format!("{}1", feature_metadata[i].name));
            }
        }

        // Entity 2 features - individual (first 12 features from metadata)
        for i in 0..12 {
            if i < feature_metadata.len() {
                feature_names.push(format!("{}2", feature_metadata[i].name));
            }
        }

        // Pair relationship features (last 7 features from metadata)
        for i in 12..feature_metadata.len() {
            feature_names.push(feature_metadata[i].name.clone());
        }

        Self {
            forest: None,
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
        // Fetch training data from human review decisions
        let training_data = self.collect_training_data(pool).await?;

        if training_data.is_empty() {
            warn!("No training data available for context model");
            return Ok(());
        }

        info!(
            "Training context model with {} examples",
            training_data.len()
        );

        // Train with the collected data
        self.train_with_data(&training_data)?;

        Ok(())
    }

    pub fn train_with_data(&mut self, training_data: &[TrainingExample]) -> Result<()> {
        // Need at least some data to train
        if training_data.is_empty() {
            return Err(anyhow::anyhow!("No training data provided"));
        }

        // Need at least 2 different classes to train a classifier
        let unique_methods: std::collections::HashSet<&String> =
            training_data.iter().map(|ex| &ex.best_method).collect();

        if unique_methods.len() < 2 {
            return Err(anyhow::anyhow!(
                "Need at least 2 different methods to train a classifier"
            ));
        }

        // Prepare training matrices
        let n_samples = training_data.len();
        let n_features = training_data[0].features.len();

        debug!(
            "Building training matrices: {} samples, {} features",
            n_samples, n_features
        );

        // Prepare feature matrices for SmartCore
        // Convert the flat feature vector into a 2D vector format for from_2d_vec
        let mut feature_2d_vec: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
        for sample in training_data {
            feature_2d_vec.push(sample.features.clone());
        }

        // Convert to DenseMatrix using from_2d_vec instead of from_array
        let x = DenseMatrix::from_2d_vec(&feature_2d_vec);

        // Prepare labels
        let y: Vec<usize> = training_data
            .iter()
            .map(|sample| self.method_index(&sample.best_method))
            .collect();

        // Configure Random Forest
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

        // Train the model
        info!("Training Random Forest model...");
        let forest = RandomForestClassifier::fit(&x, &y, forest_params)
            .map_err(|e| anyhow::anyhow!("Failed to train random forest: {}", e))?;

        // Calculate feature importance (SmartCore doesn't provide this directly,
        // but we could implement a simple permutation importance in the future)
        self.feature_importance.clear();
        // For now, use uniform importance
        for (i, name) in self.feature_names.iter().enumerate() {
            self.feature_importance
                .insert(name.clone(), 1.0 / n_features as f64);
        }

        // Store the trained forest
        match RandomForestClassifierWrapper::from_forest(&forest) {
            Ok(wrapper) => {
                self.forest = Some(wrapper);
                self.version += 1;
                info!("Context model training complete, version {}", self.version);
            }
            Err(e) => {
                return Err(anyhow::anyhow!("Failed to serialize random forest: {}", e));
            }
        }

        Ok(())
    }

    pub fn predict_best_method(&self, features: &[f64]) -> Option<(String, f64)> {
        if features.len() != self.feature_names.len() {
            warn!(
                "Feature vector length mismatch: got {}, expected {}",
                features.len(),
                self.feature_names.len()
            );
            return None;
        }

        if let Some(forest_wrapper) = &self.forest {
            // Convert wrapper back to forest for prediction
            let forest = match forest_wrapper.to_forest() {
                Ok(f) => f,
                Err(e) => {
                    warn!("Failed to load forest from wrapper: {}", e);
                    return None;
                }
            };

            // Prepare feature data for prediction
            let feature_vec = vec![features.to_vec()];
            let x = DenseMatrix::from_2d_vec(&feature_vec);

            // Get predictions
            let predictions = match forest.predict(&x) {
                Ok(p) => p,
                Err(e) => {
                    warn!("Failed to generate prediction: {}", e);
                    return None;
                }
            };

            // Get probabilities if available (not directly supported in RandomForest,
            // but we can get an estimate based on tree votes)
            let class_idx = predictions[0];

            // For confidence, we'll use a placeholder value of 0.8 for now
            // In the future, we could implement proper probability calibration
            let confidence = 0.8;

            // Return the method name and probability
            if class_idx < self.method_labels.len() {
                return Some((self.method_labels[class_idx].clone(), confidence));
            } else {
                warn!(
                    "Prediction index out of bounds: got {}, max allowed {}",
                    class_idx,
                    self.method_labels.len() - 1
                );
            }
        } else {
            debug!("No trained model available for prediction");
        }

        None
    }

    fn method_index(&self, method: &str) -> usize {
        self.method_labels
            .iter()
            .position(|m| m == method)
            .unwrap_or(0)
    }

    pub fn feature_importance(&self) -> Vec<(String, f64)> {
        let mut result = Vec::new();

        for (name, score) in &self.feature_importance {
            result.push((name.clone(), *score));
        }

        // Sort by importance (descending)
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        result
    }

    // Collect training data from human feedback
    async fn collect_training_data(&self, pool: &PgPool) -> Result<Vec<TrainingExample>> {
        let conn = pool.get().await?;

        info!("Collecting training data from human review feedback");

        // Query for feedback that indicates the correct matching method
        let rows = conn
            .query(
                "SELECT hrf.method_type, hrd.entity_id, hrd.correct_cluster_id, 
                    hrf.was_correct, hrf.confidence_adjustment
             FROM clustering_metadata.human_review_method_feedback hrf
             JOIN clustering_metadata.human_review_decisions hrd 
                 ON hrf.decision_id = hrd.id
             WHERE hrd.entity_id IS NOT NULL
             ORDER BY hrd.created_at DESC
             LIMIT 10000", // Reasonable limit for training data
                &[],
            )
            .await?;

        debug!("Found {} human review feedback records", rows.len());

        // Extract examples
        let mut examples = Vec::new();

        for row in rows {
            let method_type: String = row.get(0);
            let entity_id: String = row.get(1);
            let correct_cluster_id: Option<String> = row.get(2);
            let was_correct: bool = row.get(3);

            // Skip examples where we don't know the correct cluster
            let cluster_id = match (was_correct, correct_cluster_id) {
                (true, _) => None,             // Current cluster was correct
                (false, Some(id)) => Some(id), // We have a correction
                (false, None) => continue,     // No correction provided, skip
            };

            // For each entity ID, find a "best match" entity in the correct cluster
            if let Some(cluster_id) = cluster_id {
                // Find another entity in the correct cluster
                let entity_rows = conn
                    .query(
                        "SELECT ge.entity_id
                     FROM group_entity ge
                     JOIN entity_group eg ON ge.entity_group_id = eg.id
                     WHERE eg.group_cluster_id = $1
                     AND ge.entity_id <> $2
                     LIMIT 1",
                        &[&cluster_id, &entity_id],
                    )
                    .await?;

                if let Some(entity_row) = entity_rows.get(0) {
                    let other_entity_id: String = entity_row.get(0);

                    // Get stored features for both entities
                    let entity1_features = self.get_entity_features(&conn, &entity_id).await?;
                    let entity2_features =
                        self.get_entity_features(&conn, &other_entity_id).await?;

                    // Get pair features
                    let pair_features = self
                        .get_pair_features(&conn, &entity_id, &other_entity_id)
                        .await?;

                    // Combine all features
                    let mut features = Vec::new();
                    features.extend(entity1_features);
                    features.extend(entity2_features);
                    features.extend(pair_features);

                    // Add to examples
                    examples.push(TrainingExample {
                        features,
                        best_method: method_type,
                        confidence: 1.0, // Maximum confidence for human feedback
                    });
                }
            }
        }

        info!("Collected {} training examples", examples.len());

        Ok(examples)
    }

    // Helper to get entity features
    async fn get_entity_features(
        &self,
        conn: &tokio_postgres::Client,
        entity_id: &str,
    ) -> Result<Vec<f64>> {
        let rows = conn
            .query(
                "SELECT feature_name, feature_value
             FROM clustering_metadata.entity_context_features
             WHERE entity_id = $1
             ORDER BY feature_name",
                &[&entity_id],
            )
            .await?;

        let mut features = Vec::new();
        for row in rows {
            let value: f64 = row.get(1);
            features.push(value);
        }

        // If no features found, use zeros
        if features.is_empty() {
            // 12 entity features (see feature_extraction.rs)
            features = vec![0.0; 12];
        }

        Ok(features)
    }

    // Helper to get pair features
    async fn get_pair_features(
        &self,
        conn: &tokio_postgres::Client,
        entity1_id: &str,
        entity2_id: &str,
    ) -> Result<Vec<f64>> {
        // This would ideally pull from a cache, but for simplicity we'll use placeholder values
        // In a real implementation, we would extract these using functions from feature_extraction.rs

        // 7 pair features (see feature_extraction.rs)
        let pair_features = vec![0.5; 7];

        Ok(pair_features)
    }

    // Save model to database
    pub async fn save_to_db(&self, pool: &PgPool) -> Result<String> {
        let conn = pool.get().await?;

        // Serialize model to JSON
        let model_json = serde_json::to_value(self)?;

        // Generate a unique ID if saving for the first time
        let id = Uuid::new_v4().to_string();

        // Check if model already exists
        let existing = conn
            .query_opt(
                "SELECT id FROM clustering_metadata.ml_models
             WHERE model_type = 'context_model'
             ORDER BY version DESC
             LIMIT 1",
                &[],
            )
            .await?;

        let model_id = if let Some(row) = existing {
            row.get(0)
        } else {
            id
        };

        // Prepare parameters JSON
        let parameters = json!({
            "feature_names": self.feature_names,
            "method_labels": self.method_labels,
            "num_classes": self.method_labels.len(),
            "version": self.version,
        });

        // Prepare metrics JSON
        let metrics = json!({
            "feature_importance": self.feature_importance()
                .into_iter()
                .map(|(name, score)| json!({"feature": name, "importance": score}))
                .collect::<Vec<_>>()
        });

        // Insert or update the model
        conn.execute(
            "INSERT INTO clustering_metadata.ml_models
             (id, model_type, parameters, metrics, version, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
             ON CONFLICT (id) DO UPDATE
             SET parameters = $3, metrics = $4, version = $5, updated_at = CURRENT_TIMESTAMP",
            &[
                &model_id,
                &"context_model",
                &parameters,
                &metrics,
                &(self.version as i32),
            ],
        )
        .await?;

        // Store the serialized model as a separate record
        let binary_id = format!("{}_binary", model_id);
        conn.execute(
            "INSERT INTO clustering_metadata.ml_models
             (id, model_type, parameters, version, created_at, updated_at)
             VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
             ON CONFLICT (id) DO UPDATE
             SET parameters = $3, version = $4, updated_at = CURRENT_TIMESTAMP",
            &[
                &binary_id,
                &"context_model_binary",
                &model_json,
                &(self.version as i32),
            ],
        )
        .await?;

        info!(
            "Saved context model to database, ID: {}, version: {}",
            model_id, self.version
        );

        Ok(model_id)
    }

    // Load model from database
    pub async fn load_from_db(pool: &PgPool) -> Result<Self> {
        let conn = pool.get().await?;

        // Get latest model
        let binary_row = conn
            .query_opt(
                "SELECT id, parameters
             FROM clustering_metadata.ml_models
             WHERE model_type = 'context_model_binary'
             ORDER BY version DESC
             LIMIT 1",
                &[],
            )
            .await?;

        if let Some(row) = binary_row {
            let params: Value = row.get(1);

            // Deserialize the model
            let model: ContextModel = serde_json::from_value(params)?;

            info!(
                "Loaded context model from database, version: {}",
                model.version
            );

            return Ok(model);
        }

        // If no model found, return a new one
        info!("No existing context model found, creating new one");
        Ok(Self::new())
    }
}
