// src/reinforcement/orchestrator.rs
use anyhow::{Context, Result}; // Added Context
use chrono::Utc;
use log::{debug, info, warn};
use uuid::Uuid;

use super::confidence_tuner::ConfidenceTuner;
use super::context_model::ContextModel;
use super::FeedbackItem;
// feature_extraction is used via a call in extract_pair_context, no direct use here
// use super::feature_extraction::{extract_context_for_pair, extract_entity_features};
use crate::db::{fetch_recent_feedback_items, PgPool};
use crate::models::{EntityId, MatchMethodType};
use crate::reinforcement::feedback_processor;

pub struct MatchingOrchestrator {
    pub context_model: ContextModel,
    pub confidence_tuner: ConfidenceTuner,
}

// This private helper function will now take the pool and other necessary details.
async fn record_entity_feedback(
    pool: &PgPool, // Accept the pool
    entity1_id: &EntityId,
    entity2_id: &EntityId,
    method_used: &MatchMethodType, // Method ultimately applied
    final_confidence_score: f64,   // Confidence of the applied match
    was_correct: bool,
    context_features: Option<&Vec<f64>>,
    ml_predicted_method: Option<&MatchMethodType>, // Method suggested by ContextModel
    ml_prediction_confidence: Option<f64>,         // Confidence of ContextModel's suggestion
    context_model_version: u32,                    // Pass from orchestrator
    confidence_tuner_version: u32,                 // Pass from orchestrator
) -> Result<()> {
    let conn = pool
        .get()
        .await
        .context("Failed to get DB connection from pool in record_entity_feedback")?;
    let decision_id = Uuid::new_v4().to_string();
    let now = Utc::now().naive_utc();

    // Insert into human_review_decisions (system's perspective)
    conn.execute(
        "INSERT INTO clustering_metadata.human_review_decisions
        (id, reviewer_id, decision_type, entity_id, confidence, created_at)
        VALUES ($1, $2, $3, $4, $5, $6)",
        &[
            &decision_id,
            &"ml_system",
            &"match_assessment",
            &entity1_id.0,
            &final_confidence_score,
            &now,
        ],
    )
    .await
    .context("Failed to insert into human_review_decisions")?; // Added context

    // Insert into human_review_method_feedback
    let confidence_adjustment = if was_correct { 0.05 } else { -0.05 };
    conn.execute(
        "INSERT INTO clustering_metadata.human_review_method_feedback
        (id, decision_id, method_type, was_correct, confidence_adjustment, created_at)
        VALUES ($1, $2, $3, $4, $5, $6)",
        &[
            &Uuid::new_v4().to_string(),
            &decision_id,
            &method_used.as_str(),
            &was_correct,
            // Ensure the type is explicitly f64 for ToSql trait
            &(confidence_adjustment as f64) as &(dyn tokio_postgres::types::ToSql + Sync),
            &now,
        ],
    )
    .await
    .context("Failed to insert into human_review_method_feedback")?; // Added context

    let snapshotted_features_json = context_features.map_or(serde_json::Value::Null, |features| {
        serde_json::to_value(features).unwrap_or(serde_json::Value::Null)
    });

    let method_for_snapshot_str = ml_predicted_method.unwrap_or(method_used).as_str();

    conn.execute(
        "INSERT INTO clustering_metadata.entity_match_pairs
        (decision_id, entity1_id, entity2_id, method_type, prediction_confidence, 
         snapshotted_features, context_model_version, confidence_tuner_version, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (decision_id) DO UPDATE SET
            entity1_id = EXCLUDED.entity1_id,
            entity2_id = EXCLUDED.entity2_id,
            method_type = EXCLUDED.method_type,
            prediction_confidence = EXCLUDED.prediction_confidence,
            snapshotted_features = EXCLUDED.snapshotted_features,
            context_model_version = EXCLUDED.context_model_version,
            confidence_tuner_version = EXCLUDED.confidence_tuner_version,
            created_at = EXCLUDED.created_at",
        &[
            &decision_id,
            &entity1_id.0,
            &entity2_id.0,
            &method_for_snapshot_str,
            &ml_prediction_confidence,
            &snapshotted_features_json,
            &(context_model_version as i32), // Use passed-in version
            &(confidence_tuner_version as i32), // Use passed-in version
            &now,
        ],
    )
    .await
    .context("Failed to insert into entity_match_pairs")?; // Added context

    Ok(())
}

impl MatchingOrchestrator {
    pub async fn new(pool: &PgPool) -> Result<Self> {
        let context_model = match ContextModel::load_from_db(pool).await {
            Ok(model) => {
                info!(
                    "Loaded context model from database version {}",
                    model.version
                );
                model
            }
            Err(e) => {
                warn!("Could not load context model: {}. Creating new one.", e);
                ContextModel::new()
            }
        };

        let confidence_tuner = match ConfidenceTuner::load_from_db(pool).await {
            Ok(tuner) => {
                info!(
                    "Loaded confidence tuner from database version {}",
                    tuner.version
                );
                tuner
            }
            Err(e) => {
                warn!("Could not load confidence tuner: {}. Creating new one.", e);
                ConfidenceTuner::new()
            }
        };

        Ok(Self {
            context_model,
            confidence_tuner,
        })
    }

    pub async fn extract_pair_context(
        pool: &PgPool,
        entity1: &EntityId,
        entity2: &EntityId,
    ) -> Result<Vec<f64>> {
        info!(
            "Extracting context for entities {:?} and {:?}",
            entity1, entity2
        );
        super::feature_extraction::extract_context_for_pair(pool, entity1, entity2).await
    }

    pub fn predict_method_with_context(
        &self,
        context_features: &Vec<f64>,
    ) -> Result<(MatchMethodType, f64)> {
        info!("Predicting matching method with pre-extracted context features");

        let (method_name, context_confidence) =
            match self.context_model.predict_best_method(context_features) {
                Some((method, confidence)) => {
                    info!(
                        "Context model (v{}) predicts method '{}' with confidence {:.4}",
                        self.context_model.version, method, confidence
                    );
                    (method, confidence)
                }
                None => {
                    info!(
                        "Context model (v{}) could not predict. Using default method (name).",
                        self.context_model.version
                    );
                    ("name".to_string(), 0.5)
                }
            };

        let confidence = self.confidence_tuner.select_confidence_with_context(
            &method_name,
            context_features,
            context_confidence,
        );

        info!(
            "Confidence tuner (v{}) selected confidence {:.4} for method '{}'",
            self.confidence_tuner.version, confidence, method_name
        );

        let method_type = MatchMethodType::from_str(&method_name); // Use from_str for consistency

        Ok((method_type, confidence))
    }

    pub async fn log_match_result(
        &mut self,
        pool: &PgPool, // Added pool parameter
        entity1_id: &EntityId,
        entity2_id: &EntityId,
        method_used: &MatchMethodType,
        final_confidence_score: f64,
        was_correct: bool,
        context_features: Option<&Vec<f64>>,
        ml_predicted_method: Option<&MatchMethodType>,
        ml_prediction_confidence: Option<f64>,
    ) -> Result<()> {
        let reward = if was_correct { 1.0 } else { 0.0 }; // Or -1.0 for incorrect, depends on strategy

        // Update tuner before potential version increment if it saves itself
        self.confidence_tuner
            .update(method_used.as_str(), final_confidence_score, reward)?;
        // If ConfidenceTuner::update might save & increment version, and you want to log the version *before* this update for this specific call,
        // you might fetch version numbers before calling update. For now, assume update doesn't auto-save/increment.

        let context_model_ver = self.context_model.version;
        let confidence_tuner_ver = self.confidence_tuner.version;

        // Call the refactored record_entity_feedback, now a free function or static method
        match record_entity_feedback(
            pool, // Pass the pool
            entity1_id,
            entity2_id,
            method_used,
            final_confidence_score,
            was_correct,
            context_features,
            ml_predicted_method,
            ml_prediction_confidence,
            context_model_ver,    // Pass current version
            confidence_tuner_ver, // Pass current version
        )
        .await
        {
            Ok(_) => debug!("Recorded detailed entity feedback (with snapshot) for ML training"),
            Err(e) => warn!(
                "Failed to record detailed entity feedback (with snapshot): {}",
                e
            ),
        }

        info!(
            "Logged match result for method {}: entities {} and {}, correct={}, final_confidence={:.4}. RL Models: Context v{}, Tuner v{}",
            method_used.as_str(),
            entity1_id.0,
            entity2_id.0,
            was_correct,
            final_confidence_score,
            context_model_ver,
            confidence_tuner_ver
        );
        Ok(())
    }

    // pub async fn save_models(&self, pool: &PgPool) -> Result<()> {
    //     info!("Saving ML models to database. ContextModel v{}, ConfidenceTuner v{}", self.context_model.version, self.confidence_tuner.version);
    //     let context_model_id = self.context_model.save_to_db(pool).await?;
    //     let confidence_tuner_id = self.confidence_tuner.save_to_db(pool).await?;
    //     info!(
    //         "Saved models with IDs: context={}, confidence={}",
    //         context_model_id, confidence_tuner_id
    //     );
    //     Ok(())
    // }

    pub async fn train_context_model(&mut self, pool: &PgPool) -> Result<()> {
        info!(
            "Manually training context model (current version: {}).",
            self.context_model.version
        );
        self.context_model.train(pool).await?; // train() should increment version internally on success
        info!(
            "Context model training complete. New version: {}.",
            self.context_model.version
        );
        Ok(())
    }

    pub fn get_confidence_stats(&self) -> String {
        let stats = self.confidence_tuner.get_stats();
        let mut output = format!(
            "Confidence Tuner (v{}) Statistics:\n",
            self.confidence_tuner.version
        );
        for (method, values) in stats {
            output.push_str(&format!("\nMethod: {}\n", method));
            output.push_str("  Confidence | Avg Reward | Trials\n");
            output.push_str("  -----------|------------|-------\n");
            for (confidence, reward, trials) in values {
                output.push_str(&format!(
                    "    {:.2}     |   {:.4}    |  {}\n",
                    confidence, reward, trials
                ));
            }
        }
        output
    }

    pub async fn extract_entity_features(
        // This method seems less used by orchestrator directly for pair matching
        &self, // but might be a utility.
        pool: &PgPool,
        entity_id: &EntityId,
    ) -> Result<Vec<f64>> {
        let conn = pool.get().await?;
        super::feature_extraction::extract_entity_features(&conn, entity_id).await
    }
}
