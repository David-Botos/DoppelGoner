use anyhow::Result;
use log::{debug, info, warn};
use uuid::Uuid;

use super::confidence_tuner::ConfidenceTuner;
use super::context_model::ContextModel;
use super::types::FeedbackItem;
use crate::db::PgPool;

pub async fn process_feedback(pool: &PgPool) -> Result<()> {
    info!("Processing human review feedback");

    // Load models
    let mut context_model = match ContextModel::load_from_db(pool).await {
        Ok(model) => model,
        Err(e) => {
            warn!("Failed to load context model, creating new one: {}", e);
            ContextModel::new()
        }
    };

    let mut confidence_tuner = match ConfidenceTuner::load_from_db(pool).await {
        Ok(tuner) => tuner,
        Err(e) => {
            warn!("Failed to load confidence tuner, creating new one: {}", e);
            ConfidenceTuner::new()
        }
    };

    // Get recent human review decisions
    let feedback = fetch_recent_feedback(pool).await?;

    if feedback.is_empty() {
        info!("No new feedback to process");
        return Ok(());
    }

    info!("Processing {} feedback items", feedback.len());

    // Collect training data for context model
    let training_data = prepare_training_data(pool, &feedback).await?;

    // Retrain context model if we have enough data
    if !training_data.is_empty() {
        info!(
            "Retraining context model with {} examples",
            training_data.len()
        );
        context_model.train_with_data(&training_data)?;
        context_model.save_to_db(pool).await?;
    }

    // Update confidence tuner
    for item in &feedback {
        let reward = if item.was_correct { 1.0 } else { 0.0 };

        if let Err(e) = confidence_tuner.update(&item.method_type, item.confidence, reward) {
            warn!("Failed to update confidence tuner: {}", e);
        }
    }

    // Save updated confidence tuner
    confidence_tuner.save_to_db(pool).await?;

    info!("Feedback processing complete");

    Ok(())
}

async fn fetch_recent_feedback(pool: &PgPool) -> Result<Vec<FeedbackItem>> {
    let conn = pool.get().await?;

    // Query for recent feedback that has not been processed
    // This assumes we're adding some kind of 'processed' flag to the feedback table
    // For simplicity, we're just pulling the most recent records
    let rows = conn
        .query(
            "SELECT hrf.method_type, hrf.was_correct, hrf.confidence_adjustment,
                hrd.entity_id, hrd.cluster_id
         FROM clustering_metadata.human_review_method_feedback hrf
         JOIN clustering_metadata.human_review_decisions hrd ON hrf.decision_id = hrd.id
         WHERE hrd.created_at > NOW() - INTERVAL '1 day'
         ORDER BY hrd.created_at DESC
         LIMIT 1000",
            &[],
        )
        .await?;

    debug!("Found {} recent feedback items", rows.len());

    let mut feedback = Vec::new();

    for row in rows {
        let method_type: String = row.get(0);
        let was_correct: bool = row.get(1);
        let confidence_adjustment: f64 = row.get(2);
        let entity_id: String = row.get(3);
        let cluster_id: Option<String> = row.get(4);

        // For each entity in the feedback, find a related entity to create a pair
        // This is a simplified approach - in practice, you'd want to identify the exact
        // entity pairs involved in the matching decision
        if let Some(cluster_id) = cluster_id {
            if let Some(other_entity_id) =
                get_other_entity_in_cluster(&conn, &cluster_id, &entity_id).await?
            {
                // Confidence is either the original + adjustment or the default confidence
                let confidence = 0.85 + confidence_adjustment;

                feedback.push(FeedbackItem {
                    entity_id1: entity_id.clone(),
                    entity_id2: other_entity_id,
                    method_type: method_type.clone(),
                    confidence,
                    was_correct,
                });
            }
        }
    }

    info!(
        "Processed {} valid feedback items with entity pairs",
        feedback.len()
    );

    Ok(feedback)
}

async fn get_other_entity_in_cluster(
    conn: &tokio_postgres::Client,
    cluster_id: &str,
    exclude_entity_id: &str,
) -> Result<Option<String>> {
    // Find any other entity in the same cluster
    let row = conn
        .query_opt(
            "SELECT ge.entity_id
         FROM group_entity ge
         JOIN entity_group eg ON ge.entity_group_id = eg.id
         WHERE eg.group_cluster_id = $1
         AND ge.entity_id <> $2
         LIMIT 1",
            &[&cluster_id, &exclude_entity_id],
        )
        .await?;

    Ok(row.map(|r| r.get(0)))
}

async fn prepare_training_data(
    pool: &PgPool,
    feedback: &[FeedbackItem],
) -> Result<Vec<super::types::TrainingExample>> {
    let conn = pool.get().await?;

    let mut training_examples = Vec::new();

    for item in feedback {
        // Get stored features for both entities in the pair
        let entity1_features = get_entity_features(&conn, &item.entity_id1).await?;
        let entity2_features = get_entity_features(&conn, &item.entity_id2).await?;

        // Get features for the pair
        let pair_features = get_pair_features(&conn, &item.entity_id1, &item.entity_id2).await?;

        // Combine features
        let mut features = Vec::new();
        features.extend(entity1_features);
        features.extend(entity2_features);
        features.extend(pair_features);

        // Create training example
        training_examples.push(super::types::TrainingExample {
            features,
            best_method: item.method_type.clone(),
            confidence: item.confidence,
        });
    }

    Ok(training_examples)
}

// Helper to get stored entity features
async fn get_entity_features(conn: &tokio_postgres::Client, entity_id: &str) -> Result<Vec<f64>> {
    let rows = conn
        .query(
            "SELECT feature_name, feature_value
         FROM clustering_metadata.entity_context_features
         WHERE entity_id = $1
         ORDER BY feature_name",
            &[&entity_id],
        )
        .await?;

    // In a real implementation, we would ensure features are returned in a consistent order
    // For simplicity, we'll just collect values and ensure we have the expected number
    let mut features = Vec::new();
    for row in rows {
        let value: f64 = row.get(1);
        features.push(value);
    }

    // If no features found, use zeros
    if features.is_empty() {
        // 12 individual entity features (see feature_extraction.rs)
        features = vec![0.0; 12];
    } else if features.len() != 12 {
        warn!(
            "Unexpected number of features for entity {}: got {}, expected 12",
            entity_id,
            features.len()
        );

        // Pad or truncate to expected length
        if features.len() < 12 {
            features.resize(12, 0.0);
        } else {
            features.truncate(12);
        }
    }

    Ok(features)
}

// Helper to get pair features
async fn get_pair_features(
    conn: &tokio_postgres::Client,
    entity1_id: &str,
    entity2_id: &str,
) -> Result<Vec<f64>> {
    // In a real implementation, we would extract these features from the database
    // For simplicity, we'll return placeholder values

    // 7 pair features (see feature_extraction.rs)
    // In production, you'd calculate these properly using the actual feature extraction functions
    let pair_features = vec![0.5; 7];

    Ok(pair_features)
}

// Add a function to record human feedback
pub async fn record_human_feedback(
    pool: &PgPool,
    reviewer_id: &str,
    decision_type: &str,
    cluster_id: &str,
    entity_id: Option<&str>,
    correct_cluster_id: Option<&str>,
    confidence: f64,
    method_feedbacks: &[(&str, bool, f64)],
) -> Result<String> {
    let conn = pool.get().await?;

    // Generate IDs
    let decision_id = Uuid::new_v4().to_string();

    // Insert the decision
    conn.execute(
        "INSERT INTO clustering_metadata.human_review_decisions
         (id, reviewer_id, decision_type, cluster_id, entity_id, correct_cluster_id, confidence, created_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)",
        &[
            &decision_id,
            &reviewer_id,
            &decision_type,
            &cluster_id,
            &entity_id,
            &correct_cluster_id,
            &confidence,
        ],
    ).await?;

    // Insert method-specific feedback
    for (method_type, was_correct, confidence_adjustment) in method_feedbacks {
        let feedback_id = Uuid::new_v4().to_string();

        conn.execute(
            "INSERT INTO clustering_metadata.human_review_method_feedback
             (id, decision_id, method_type, was_correct, confidence_adjustment, created_at)
             VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)",
            &[
                &feedback_id,
                &decision_id,
                &method_type,
                &was_correct,
                &confidence_adjustment,
            ],
        )
        .await?;
    }

    Ok(decision_id)
}
