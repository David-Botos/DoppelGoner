use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub features: Vec<f64>,
    pub best_method: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackItem {
    pub entity_id1: String,
    pub entity_id2: String,
    pub method_type: String,
    pub confidence: f64,
    pub was_correct: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMetadata {
    pub name: String,
    pub description: String,
    pub min_value: f64,
    pub max_value: f64,
}

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

// Add a struct to hold model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub sample_count: usize,
}

// Add a utility struct for tracking ML experiment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    pub experiment_id: String,
    pub model_version: u32,
    pub parameters: serde_json::Value,
    pub metrics: ModelMetrics,
    pub created_at: String,
}
