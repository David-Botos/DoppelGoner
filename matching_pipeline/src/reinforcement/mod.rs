// src/reinforcement/mod.rs
mod confidence_tuner;
mod context_model;
mod feature_extraction;
mod feedback_processor;
mod orchestrator;
mod types;

// Public re-exports
pub use confidence_tuner::ConfidenceTuner;
pub use context_model::ContextModel;
pub use feature_extraction::{extract_context_for_pair, extract_entity_features};
pub use feedback_processor::{process_feedback, record_human_feedback};
pub use orchestrator::MatchingOrchestrator;
pub use types::*;