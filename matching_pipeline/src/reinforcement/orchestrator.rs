use anyhow::Result;
use log::{debug, info, warn};
use uuid::Uuid;

use super::confidence_tuner::ConfidenceTuner;
use super::context_model::ContextModel;
use super::feature_extraction::{extract_context_for_pair, extract_entity_features};
use crate::db::PgPool;
use crate::models::{EntityId, MatchMethodType};

pub struct MatchingOrchestrator {
    context_model: ContextModel,
    confidence_tuner: ConfidenceTuner,
}

impl MatchingOrchestrator {
    pub async fn new(pool: &PgPool) -> Result<Self> {
        // Try to load models from database
        let context_model = match ContextModel::load_from_db(pool).await {
            Ok(model) => {
                info!("Loaded context model from database");
                model
            }
            Err(e) => {
                warn!("Could not load context model: {}", e);
                info!("Creating new context model");
                ContextModel::new()
            }
        };

        let confidence_tuner = match ConfidenceTuner::load_from_db(pool).await {
            Ok(tuner) => {
                info!("Loaded confidence tuner from database");
                tuner
            }
            Err(e) => {
                warn!("Could not load confidence tuner: {}", e);
                info!("Creating new confidence tuner");
                ConfidenceTuner::new()
            }
        };

        Ok(Self {
            context_model,
            confidence_tuner,
        })
    }

    pub async fn select_matching_method(
        &self,
        pool: &PgPool,
        entity1: &EntityId,
        entity2: &EntityId,
    ) -> Result<(MatchMethodType, f64)> {
        info!(
            "Selecting matching method for entities {:?} and {:?}",
            entity1, entity2
        );

        // Extract context features for this entity pair
        let context = extract_context_for_pair(pool, entity1, entity2).await?;

        // Use the context model to predict best method
        let (method_name, context_confidence) =
            match self.context_model.predict_best_method(&context) {
                Some((method, confidence)) => {
                    info!(
                        "Context model predicts method '{}' with confidence {:.4}",
                        method, confidence
                    );
                    (method, confidence)
                }
                None => {
                    // Fallback to default method if model is not trained
                    info!("Using default method (name) due to missing context model");
                    ("name".to_string(), 0.5)
                }
            };

        // Use confidence tuner to get the confidence score (with context awareness)
        let confidence = self.confidence_tuner.select_confidence_with_context(
            &method_name,
            &context,
            context_confidence,
        );

        info!(
            "Selected confidence {:.4} for method '{}'",
            confidence, method_name
        );

        let method_type = match method_name.as_str() {
            "email" => MatchMethodType::Email,
            "phone" => MatchMethodType::Phone,
            "url" => MatchMethodType::Url,
            "address" => MatchMethodType::Address,
            "geospatial" => MatchMethodType::Geospatial,
            _ => MatchMethodType::Name, // Default
        };

        Ok((method_type, confidence))
    }

    pub async fn log_match_result(
        &mut self,
        method: &MatchMethodType,
        confidence: f64,
        was_correct: bool,
        entity1_id: &EntityId,
        entity2_id: &EntityId,
    ) -> Result<()> {
        // Calculate reward (1.0 if correct, 0.0 if incorrect)
        let reward = if was_correct { 1.0 } else { 0.0 };

        // Update the confidence tuner
        self.confidence_tuner
            .update(method.as_str(), confidence, reward)?;

        // Record detailed feedback in the database
        match self
            .record_entity_feedback(method, confidence, was_correct, entity1_id, entity2_id)
            .await
        {
            Ok(_) => debug!("Recorded detailed entity feedback for ML training"),
            Err(e) => warn!("Failed to record detailed entity feedback: {}", e),
        }

        info!(
            "Logged match result for method {}: entities {} and {}, correct={}, confidence={:.4}",
            method.as_str(),
            entity1_id.0,
            entity2_id.0,
            was_correct,
            confidence
        );

        Ok(())
    }

    // New method to record detailed entity feedback
    async fn record_entity_feedback(
        &self,
        method: &MatchMethodType,
        confidence: f64,
        was_correct: bool,
        entity1_id: &EntityId,
        entity2_id: &EntityId,
    ) -> Result<()> {
        let pool = crate::db::connect().await?;
        let conn = pool.get().await?;
        let decision_id = Uuid::new_v4().to_string();

        // First, create a human_review_decisions record
        let insert_decision = "
            INSERT INTO clustering_metadata.human_review_decisions 
            (id, reviewer_id, decision_type, entity_id, confidence, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
        ";

        conn.execute(
            insert_decision,
            &[
                &decision_id,
                &"ml_system",      // Using a standard ID for ML-generated feedback
                &"match_feedback", // New decision type for ML feedback
                &entity1_id.0,     // We'll record entity1 as the primary entity
                &confidence,
            ],
        )
        .await?;

        // Then, create a human_review_method_feedback record
        let insert_feedback = "
            INSERT INTO clustering_metadata.human_review_method_feedback
            (id, decision_id, method_type, was_correct, confidence_adjustment, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
        ";

        // Calculate a confidence adjustment (positive or negative)
        let confidence_adjustment = if was_correct { 0.05 } else { -0.05 };

        conn.execute(
            insert_feedback,
            &[
                &Uuid::new_v4().to_string(),
                &decision_id,
                &method.as_str(),
                &was_correct,
                &confidence_adjustment.to_string(),
            ],
        )
        .await?;

        // Also store the second entity in a custom field or table
        // This is important for training the context model later
        let insert_entity_pair = "
            INSERT INTO clustering_metadata.entity_match_pairs
            (decision_id, entity1_id, entity2_id, method_type, created_at)
            VALUES ($1, $2, $3, $4, NOW())
            ON CONFLICT DO NOTHING
        ";

        // First check if the entity_match_pairs table exists, create if not
        let create_table = "
            CREATE TABLE IF NOT EXISTS clustering_metadata.entity_match_pairs (
                decision_id TEXT NOT NULL REFERENCES clustering_metadata.human_review_decisions(id),
                entity1_id TEXT NOT NULL,
                entity2_id TEXT NOT NULL, 
                method_type TEXT NOT NULL,
                created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (decision_id)
            )
        ";

        // Create table first
        match conn.execute(create_table, &[]).await {
            Ok(_) => debug!("Entity match pairs table exists or was created"),
            Err(e) => warn!("Failed to create entity match pairs table: {}", e),
        }

        // Then insert the pair
        conn.execute(
            insert_entity_pair,
            &[&decision_id, &entity1_id.0, &entity2_id.0, &method.as_str()],
        )
        .await?;

        Ok(())
    }

    pub async fn save_models(&self, pool: &PgPool) -> Result<()> {
        // Save both models
        info!("Saving ML models to database");

        let context_model_id = self.context_model.save_to_db(pool).await?;
        let confidence_tuner_id = self.confidence_tuner.save_to_db(pool).await?;

        info!(
            "Saved models with IDs: context={}, confidence={}",
            context_model_id, confidence_tuner_id
        );

        Ok(())
    }

    // Force a retrain of the context model
    pub async fn train_context_model(&mut self, pool: &PgPool) -> Result<()> {
        info!("Manually training context model");
        self.context_model.train(pool).await?;
        Ok(())
    }

    // Get stats about the confidence tuner
    pub fn get_confidence_stats(&self) -> String {
        let stats = self.confidence_tuner.get_stats();

        let mut output = String::from("Confidence Tuner Statistics:\n");

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

    // Extract and store features for an entity
    pub async fn extract_entity_features(
        &self,
        pool: &PgPool,
        entity_id: &EntityId,
    ) -> Result<Vec<f64>> {
        let conn = pool.get().await?;
        extract_entity_features(&conn, entity_id).await
    }
}
