// src/reinforcement/orchestrator.rs
use anyhow::{Context, Result};
use chrono::Utc;
use log::{debug, info, warn};
use uuid::Uuid;

use super::confidence_tuner::ConfidenceTuner;
// ContextModel is removed for V1
// use super::context_model::ContextModel;
use super::feature_extraction; // For extract_context_for_pair
use crate::db::{insert_match_decision_detail, PgPool}; // Assuming this is the main pool type
use crate::models::{EntityId, MatchMethodType};
use crate::reinforcement::feedback_processor;
// feedback_processor and FeedbackItem are not directly used by orchestrator in V1 for logging,
// but feedback_processor will use ConfidenceTuner.

pub struct MatchingOrchestrator {
    // context_model: ContextModel, // Removed for V1
    pub confidence_tuner: ConfidenceTuner,
    // Add a field to store the pool if orchestrator methods need to perform DB operations directly
    // For V1, log_decision_snapshot will take the pool as an argument.
}

impl MatchingOrchestrator {
    pub async fn new(pool: &PgPool) -> Result<Self> {
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
            // context_model, // Removed
            confidence_tuner,
        })
    }

    /// Extracts the 31-element feature vector for a given entity pair.
    pub async fn extract_pair_context_features(
        pool: &PgPool, // Static method, takes pool
        entity1_id: &EntityId,
        entity2_id: &EntityId,
    ) -> Result<Vec<f64>> {
        debug!(
            "Orchestrator: Extracting context features for pair ({}, {})",
            entity1_id.0, entity2_id.0
        );
        feature_extraction::extract_context_for_pair(pool, entity1_id, entity2_id)
            .await
            .context(format!(
                "Failed to extract context for pair ({},{})",
                entity1_id.0, entity2_id.0
            ))
    }

    /// Gets a tuned confidence score from the ConfidenceTuner.
    /// The `pre_rl_confidence_score` is passed for potential logging or future tuner enhancements
    /// but is not directly used by the V1 `ConfidenceTuner::select_confidence` method.
    pub fn get_tuned_confidence(
        &self,
        method_type: &MatchMethodType,
        pre_rl_confidence_score: f64,
        context_features: &[f64],
    ) -> Result<f64> {
        info!(
            "Orchestrator (ConfidenceTuner v{}): Getting tuned confidence for method '{}', pre-RL score: {:.3}",
            self.confidence_tuner.version, method_type.as_str(), pre_rl_confidence_score
        );

        if context_features.len() != 31 {
            warn!(
                "Orchestrator: Expected 31 context features for ConfidenceTuner, got {}. This might lead to suboptimal tuning.",
                context_features.len()
            );
            // Proceeding, but this is a potential issue. The tuner might expect a fixed size.
        }

        let tuned_confidence = self.confidence_tuner.select_confidence(
            method_type.as_str(),
            context_features,
            pre_rl_confidence_score, // Pass it along
        );

        info!(
            "Orchestrator: Method '{}', Pre-RL confidence: {:.3}, Tuned confidence: {:.3}",
            method_type.as_str(),
            pre_rl_confidence_score,
            tuned_confidence
        );
        Ok(tuned_confidence)
    }

    /// Logs the details of a match decision (after an entity_group is created).
    /// This function is responsible for creating a snapshot in `match_decision_details`.
    /// The ConfidenceTuner itself is updated separately by the feedback processing loop.
    pub async fn log_decision_snapshot(
        &self, // Takes &self to access tuner version
        pool: &PgPool,
        entity_group_id: &str,
        pipeline_run_id: &str,
        snapshotted_features: &Vec<f64>,
        method_type_at_decision: &MatchMethodType,
        pre_rl_confidence_at_decision: f64,
        tuned_confidence_at_decision: f64,
        // No `was_correct` here, as this is about snapshotting the decision, not immediate feedback.
    ) -> Result<()> {
        let snapshot_features_json =
            serde_json::to_value(snapshotted_features).unwrap_or(serde_json::Value::Null);

        let confidence_tuner_ver = self.confidence_tuner.version;

        // This uses a function from crate::db (which should now contain the merged DB logic)
        match insert_match_decision_detail(
            pool,
            entity_group_id,
            pipeline_run_id,
            snapshot_features_json,
            method_type_at_decision.as_str(),
            pre_rl_confidence_at_decision,
            tuned_confidence_at_decision,
            confidence_tuner_ver,
        )
        .await
        {
            Ok(_) => {
                debug!(
                "Orchestrator: Logged decision snapshot for entity_group_id: {}, method: {}, pre-RL conf: {:.3}, tuned conf: {:.3}, tuner_v: {}",
                entity_group_id, method_type_at_decision.as_str(), pre_rl_confidence_at_decision, tuned_confidence_at_decision, confidence_tuner_ver
            );
            }
            Err(e) => {
                warn!("Failed to insert match decision detail: {}. This is non-critical and the pipeline will continue.", e);
                // Return Ok() to prevent the pipeline from failing due to logging issues
            }
        }

        Ok(())
    }

    /// Saves the ConfidenceTuner model to the database.
    /// The ContextModel save is removed.
    pub async fn save_models(&mut self, pool: &PgPool) -> Result<()> {
        info!(
            "Orchestrator: Saving ConfidenceTuner (v{}) to database.",
            self.confidence_tuner.version
        );
        // The save_to_db method on ConfidenceTuner increments its version.
        let confidence_tuner_id = self.confidence_tuner.save_to_db(pool).await?;
        info!(
            "Orchestrator: Saved ConfidenceTuner with new version {}. DB Record ID base: {}",
            self.confidence_tuner.version, confidence_tuner_id
        );
        Ok(())
    }

    // Method for manually triggering ConfidenceTuner training/update cycle.
    // This will be called by the main pipeline after feedback might have been ingested.
    pub async fn process_feedback_and_update_tuner(&mut self, pool: &PgPool) -> Result<()> {
        info!(
            "Orchestrator: Starting process_feedback_and_update_tuner for ConfidenceTuner (v{}).",
            self.confidence_tuner.version
        );
        feedback_processor::process_human_feedback_for_tuner(pool, &mut self.confidence_tuner)
            .await
            .context("Failed during feedback processing and tuner update cycle")?;
        info!("Orchestrator: Completed feedback processing and tuner update. ConfidenceTuner is now at v{}.", self.confidence_tuner.version);
        // Optionally save the tuner immediately after processing feedback
        self.save_models(pool)
            .await
            .context("Failed to save ConfidenceTuner after feedback processing")?;
        Ok(())
    }

    // Utility to get tuner stats - useful for reporting
    pub fn get_confidence_tuner_stats(&self) -> String {
        self.confidence_tuner.get_stats_display() // Assuming ConfidenceTuner has a display method
    }
}

// Adding a display method to ConfidenceTuner for the orchestrator to use
// This should ideally be in confidence_tuner.rs, but adding here for context if not already present.
// If ConfidenceTuner has `get_stats() -> HashMap<String, Vec<(f64, f64, usize)>>`, we can format it.
impl ConfidenceTuner {
    pub fn get_stats_display(&self) -> String {
        let stats = self.get_stats(); // Assuming this method exists
        let mut output = format!("Confidence Tuner (v{}) Statistics:\n", self.version);
        for (method, values) in stats {
            output.push_str(&format!("\nMethod: {}\n", method));
            output.push_str("  Target Confidence | Avg Reward | Trials\n");
            output.push_str("  -----------------|------------|-------\n");
            for (confidence, reward, trials) in values {
                output.push_str(&format!(
                    "        {:.3}       |   {:.4}   |  {}\n",
                    confidence, reward, trials
                ));
            }
        }
        output
    }
}
