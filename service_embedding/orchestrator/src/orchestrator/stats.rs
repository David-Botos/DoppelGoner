// Pipeline statistics with serialization traits
// orchestrator/src/orchestrator/stats.rs

use serde::{Deserialize, Serialize};

/// Pipeline statistics for monitoring
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PipelineStats {
    pub jobs_queued: usize,
    pub jobs_processed: usize,
    pub services_fetched: usize,
    pub documents_tokenized: usize,
    pub embeddings_generated: usize,
    pub errors: usize,
    pub fetch_time_ms: u64,
    pub tokenize_time_ms: u64,
    pub inference_time_ms: u64,
    pub storage_time_ms: u64,
    pub total_time_ms: u64,
}

impl PipelineStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, other: &Self) {
        self.jobs_queued += other.jobs_queued;
        self.jobs_processed += other.jobs_processed;
        self.services_fetched += other.services_fetched;
        self.documents_tokenized += other.documents_tokenized;
        self.embeddings_generated += other.embeddings_generated;
        self.errors += other.errors;
        self.fetch_time_ms += other.fetch_time_ms;
        self.tokenize_time_ms += other.tokenize_time_ms;
        self.inference_time_ms += other.inference_time_ms;
        self.storage_time_ms += other.storage_time_ms;
        self.total_time_ms += other.total_time_ms;
    }

    pub fn success_rate(&self) -> f64 {
        if self.jobs_processed == 0 {
            return 0.0;
        }

        let successful = self.jobs_processed.saturating_sub(self.errors);
        (successful as f64) / (self.jobs_processed as f64)
    }

    pub fn average_processing_time_ms(&self) -> u64 {
        if self.embeddings_generated == 0 {
            return 0;
        }

        self.total_time_ms / (self.embeddings_generated as u64)
    }
}
