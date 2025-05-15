// src/reinforcement/types.rs
use serde::{Deserialize, Serialize};

// TrainingExample might be obsolete for V1 if ConfidenceTuner is updated directly via rewards.
// Kept for now, but review its necessity based on the final ConfidenceTuner update mechanism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub features: Vec<f64>,     // The 31-element feature vector
    pub method_type: String,    // The method that generated the pre_rl_score
    pub pre_rl_confidence: f64, // The confidence score from the heuristic method
    pub tuned_confidence: f64,  // The confidence score output by the tuner for this instance
    pub reward: f64,            // Typically 1.0 for correct, 0.0 or -1.0 for incorrect
}

// Represents a feedback item fetched from the database for tuner updates.
// This structure aligns with the new `clustering_metadata.human_feedback` table
// and the necessary details from `clustering_metadata.match_decision_details`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanFeedbackDataForTuner {
    pub feedback_id: uuid::Uuid, // From human_feedback.id
    pub entity_group_id: String, // From human_feedback.entity_group_id
    pub is_match_correct: bool,  // From human_feedback.is_match_correct

    // Details from the original decision, fetched from match_decision_details
    pub method_type_at_decision: String,
    pub snapshotted_features: Vec<f64>,
    pub tuned_confidence_at_decision: f64, // The confidence score the tuner originally outputted
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMetadata {
    pub name: String,
    pub description: String,
    pub min_value: f64,
    pub max_value: f64,
}

// ConfidenceClass can still be useful for categorizing or analyzing confidence scores.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConfidenceClass {
    VeryHigh, // 0.95-1.0
    High,     // 0.85-0.95
    Medium,   // 0.75-0.85
    Low,      // 0.65-0.75
    VeryLow,  // <0.65
}

impl ConfidenceClass {
    pub fn from_confidence(confidence: f64) -> Self {
        match confidence {
            c if c >= 0.95 => ConfidenceClass::VeryHigh,
            c if c >= 0.85 => ConfidenceClass::High,
            c if c >= 0.75 => ConfidenceClass::Medium,
            c if c >= 0.65 => ConfidenceClass::Low,
            _ => ConfidenceClass::VeryLow,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ConfidenceClass::VeryHigh => "very_high",
            ConfidenceClass::High => "high",
            ConfidenceClass::Medium => "medium",
            ConfidenceClass::Low => "low",
            ConfidenceClass::VeryLow => "very_low",
        }
    }
}

// Struct to hold model performance metrics (can be used for ConfidenceTuner evaluation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: Option<f64>, // Overall accuracy based on rewards
    pub average_reward: Option<f64>,
    pub trials_per_method_arm: Option<serde_json::Value>, // JSON representation of tuner stats
    pub sample_count: usize,                              // Number of feedback items processed
}

// Utility struct for tracking ML experiment results (can be used for ConfidenceTuner versions)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    pub experiment_id: String,
    pub model_name: String, // e.g., "ConfidenceTuner"
    pub model_version: u32,
    pub parameters: serde_json::Value, // e.g., tuner's epsilon, arm definitions
    pub metrics: ModelMetrics,
    pub created_at: String, // ISO 8601 timestamp
}
