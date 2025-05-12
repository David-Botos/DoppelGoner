// src/reinforcement/mod.rs

// Module declarations
mod confidence_tuner;
mod context_model;
mod feature_extraction; // Contains individual and pair feature extraction logic
pub mod feedback_processor; // Contains batch data preparation, and potentially feedback fetching
mod orchestrator; // Contains MatchingOrchestrator and the main RL cycle logic
mod types; // Contains FeedbackItem, TrainingExample, etc.

// Public re-exports - carefully chosen for the library's public API
pub use confidence_tuner::ConfidenceTuner; // For direct interaction if needed, or internal use by orchestrator
pub use context_model::ContextModel; // For direct interaction if needed, or internal use by orchestrator

// Feature extraction functions are often useful standalone or for diagnostics
pub use feature_extraction::{
    extract_context_for_pair, // Extracts the 31 features for a pair
    extract_entity_features,  // Extracts 12 features for a single entity
    get_feature_metadata,
    get_stored_entity_features, // Gets 12 features, ensures extraction if not stored
};

// MatchingOrchestrator itself, if its methods for prediction are part of the public API
pub use orchestrator::MatchingOrchestrator;

// Feedback processing utilities if they need to be called directly (e.g., for specific data prep tasks)
pub use feedback_processor::prepare_pairwise_training_data_batched;
// Potentially a new function for fetching feedback if it's in feedback_processor
// pub use feedback_processor::fetch_recent_feedback_items;

// Core types used across the reinforcement learning system
pub use types::{ConfidenceClass, FeatureMetadata, FeedbackItem, ModelMetrics, TrainingExample};
// Note: `process_feedback` (the old top-level function) is removed as its role is taken over by `run_reinforcement_learning_cycle`.
