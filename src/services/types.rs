// src/services/types.rs
use std::time::Duration;

#[derive(Clone)]
pub struct ServiceData {
    pub id: String,
    pub name: String,
    pub description: String,
    pub taxonomy_info: Vec<TaxonomyInfo>,
}

#[derive(Clone)]
pub struct TaxonomyInfo {
    pub term: String,
    pub description: Option<String>,
}

pub struct BatchMetrics {
    pub batch_id: String,
    pub fetch_time: Duration,
    pub tokenize_time: Duration,
    pub inference_time: Duration,
    pub db_time: Duration,
    pub total_time: Duration,
    pub num_services: usize,
    pub services_per_batch: usize,
    pub batches_processed: usize,
}
pub struct ProcessingBatch {
    pub service_data: Vec<ServiceData>,
    pub embeddings: Option<Vec<Vec<f32>>>,
    pub batch_id: String,
}

pub struct TokenizedBatch {
    pub service_ids: Vec<String>,
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u32>>,
    pub batch_id: String,
}
