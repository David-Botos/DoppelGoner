// src/reinforcement/mod.rs

// Module declarations
mod confidence_tuner;
// context_model.rs is deleted for V1
// mod context_model;
mod feature_extraction;
pub mod feedback_processor; // Public for main pipeline to trigger feedback processing
mod orchestrator;
mod types;

// Public re-exports - V1 API
pub use confidence_tuner::ConfidenceTuner;
// ContextModel is removed
// pub use context_model::ContextModel;

pub use feature_extraction::{
    extract_context_for_pair, // Extracts the 31 features for a pair
    get_feature_metadata,
    get_stored_entity_features, // Gets 12 individual features, ensures extraction if not stored
                                // extract_entity_features is now a private helper within feature_extraction.rs
};

pub use orchestrator::MatchingOrchestrator;

// feedback_processor::prepare_pairwise_training_data_batched is removed as ContextModel is removed.
// process_human_feedback_for_tuner is called by the orchestrator or main pipeline.

// Core types used across the reinforcement learning system
pub use types::{
    // TrainingExample, // Review if needed for V1; ConfidenceTuner updates directly
    ConfidenceClass,
    ExperimentResult,
    FeatureMetadata,
    HumanFeedbackDataForTuner, // Renamed from FeedbackItem for clarity in its new role
    ModelMetrics,
};
