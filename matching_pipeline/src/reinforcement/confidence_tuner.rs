use anyhow::Result;
use log::{debug, info, warn};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use uuid::Uuid;

use super::types::ConfidenceClass;
use crate::db::PgPool;
use crate::models::MatchMethodType;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ConfidenceArm {
    confidence: f64,
    reward_sum: f64,
    trials: usize,
}

impl ConfidenceArm {
    pub fn new(confidence: f64) -> Self {
        Self {
            confidence,
            reward_sum: 0.0,
            trials: 0,
        }
    }

    pub fn update(&mut self, reward: f64) {
        self.trials += 1;
        self.reward_sum += reward;
    }

    fn average_reward(&self) -> f64 {
        if self.trials == 0 {
            return 0.0;
        }
        self.reward_sum / self.trials as f64
    }

    fn ucb_score(&self, total_trials: usize) -> f64 {
        if self.trials == 0 {
            return f64::INFINITY; // Encourage exploration
        }

        let exploitation = self.average_reward();
        let exploration = (2.0 * (total_trials as f64).ln() / self.trials as f64).sqrt();

        exploitation + exploration
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConfidenceTuner {
    method_arms: HashMap<String, Vec<ConfidenceArm>>,
    context_thresholds: HashMap<String, Vec<f64>>,
    epsilon: f64,
    version: u32,
}

impl ConfidenceTuner {
    pub fn new() -> Self {
        let mut method_arms = HashMap::new();

        // Initialize arms for each method with different confidence levels
        for method in &["email", "phone", "url", "address", "name", "geospatial"] {
            let confidence_levels = match *method {
                "email" => vec![0.9, 0.95, 1.0],
                "phone" => vec![0.8, 0.85, 0.9, 0.95],
                "url" => vec![0.8, 0.85, 0.9, 0.95],
                "address" => vec![0.85, 0.9, 0.95],
                "name" => vec![0.7, 0.75, 0.8, 0.85, 0.9],
                "geospatial" => vec![0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                _ => vec![0.8, 0.85, 0.9, 0.95],
            };

            let arms = confidence_levels
                .into_iter()
                .map(|conf| ConfidenceArm::new(conf))
                .collect();

            method_arms.insert(method.to_string(), arms);
        }

        Self {
            method_arms,
            context_thresholds: HashMap::new(),
            epsilon: 0.1, // 10% exploration
            version: 1,
        }
    }

    pub fn select_confidence(&self, method: &str, context_class: Option<&str>) -> f64 {
        let arms = match self.method_arms.get(method) {
            Some(a) => a,
            None => {
                warn!("Method not found for confidence selection: {}", method);
                return 0.85; // Default confidence
            }
        };

        if arms.is_empty() {
            return 0.85; // Default confidence
        }

        // With probability epsilon, explore randomly
        let mut rng = rand::thread_rng();
        if rng.r#gen::<f64>() < self.epsilon {
            // Use the correct syntax for rand 0.8
            let idx = rng.gen_range(0..arms.len());

            debug!(
                "Exploration mode: trying confidence {} for method {}",
                arms[idx].confidence, method
            );
            return arms[idx].confidence;
        }

        // Otherwise, use UCB for exploitation
        let total_trials: usize = arms.iter().map(|arm| arm.trials).sum();

        let mut best_idx = 0;
        let mut best_score = arms[0].ucb_score(total_trials);

        for (i, arm) in arms.iter().enumerate().skip(1) {
            let score = arm.ucb_score(total_trials);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        debug!(
            "Exploitation mode: selected confidence {} for method {} (UCB score: {})",
            arms[best_idx].confidence, method, best_score
        );

        arms[best_idx].confidence
    }

    // Get stats about performance
    pub fn get_stats(&self) -> HashMap<String, Vec<(f64, f64, usize)>> {
        let mut stats = HashMap::new();

        for (method, arms) in &self.method_arms {
            let method_stats: Vec<(f64, f64, usize)> = arms
                .iter()
                .map(|arm| (arm.confidence, arm.average_reward(), arm.trials))
                .collect();

            stats.insert(method.clone(), method_stats);
        }

        stats
    }

    // Add context-sensitive confidence selection
    pub fn select_confidence_with_context(
        &self,
        method: &str,
        features: &[f64],
        context_confidence: f64,
    ) -> f64 {
        // Basic confidence selection first
        let base_confidence = self.select_confidence(method, None);

        // Adjust based on context confidence (prediction confidence from random forest)
        // - If context confidence is high (>0.8), use the method's suggested confidence
        // - If context confidence is low (<0.5), regress toward a safer middle value (0.85)
        // - In between, scale linearly

        let context_weight = (context_confidence - 0.5) * 2.0;
        let context_weight = context_weight.max(0.0).min(1.0);

        let middle_confidence = 0.85;
        let adjusted_confidence =
            base_confidence * context_weight + middle_confidence * (1.0 - context_weight);

        debug!(
            "Confidence adjustment: base={}, context={}, adjusted={}",
            base_confidence, context_confidence, adjusted_confidence
        );

        adjusted_confidence
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
             WHERE model_type = 'confidence_tuner'
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

        // Prepare parameters JSON - summary of configuration
        let parameters = json!({
            "epsilon": self.epsilon,
            "methods": self.method_arms.keys().collect::<Vec<_>>(),
            "version": self.version,
        });

        // Prepare metrics JSON - performance by method
        let mut metrics = json!({});

        for (method, arms) in &self.method_arms {
            let method_stats = arms
                .iter()
                .map(|arm| {
                    json!({
                        "confidence": arm.confidence,
                        "reward": arm.average_reward(),
                        "trials": arm.trials
                    })
                })
                .collect::<Vec<_>>();

            metrics[method] = json!(method_stats);
        }

        // Insert or update the model
        conn.execute(
            "INSERT INTO clustering_metadata.ml_models
             (id, model_type, parameters, metrics, version, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
             ON CONFLICT (id) DO UPDATE
             SET parameters = $3, metrics = $4, version = $5, updated_at = CURRENT_TIMESTAMP",
            &[
                &model_id,
                &"confidence_tuner",
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
                &"confidence_tuner_binary",
                &model_json,
                &(self.version as i32),
            ],
        )
        .await?;

        info!(
            "Saved confidence tuner to database, ID: {}, version: {}",
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
             WHERE model_type = 'confidence_tuner_binary'
             ORDER BY version DESC
             LIMIT 1",
                &[],
            )
            .await?;

        if let Some(row) = binary_row {
            let params: Value = row.get(1);

            // Deserialize the model
            let model: ConfidenceTuner = serde_json::from_value(params)?;

            info!(
                "Loaded confidence tuner from database, version: {}",
                model.version
            );

            return Ok(model);
        }

        // If no model found, return a new one
        info!("No existing confidence tuner found, creating new one");
        Ok(Self::new())
    }

    pub fn update(&mut self, method: &str, confidence: f64, reward: f64) -> Result<()> {
        let arms = match self.method_arms.get_mut(method) {
            Some(a) => a,
            None => {
                warn!("Method not found for confidence update: {}", method);
                return Ok(());
            }
        };

        // Find the closest arm (confidence level)
        let mut closest_idx = 0;
        let mut min_diff = (arms[0].confidence - confidence).abs();

        for (i, arm) in arms.iter().enumerate().skip(1) {
            let diff = (arm.confidence - confidence).abs();
            if diff < min_diff {
                min_diff = diff;
                closest_idx = i;
            }
        }

        // Update the closest arm
        arms[closest_idx].update(reward);

        debug!(
            "Updated confidence arm for method {}: confidence={}, reward={}",
            method, arms[closest_idx].confidence, reward
        );

        Ok(())
    }
}
