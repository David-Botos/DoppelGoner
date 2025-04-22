// src/services/config.rs
pub const MODEL_PATH: &str = "./models/bge-small-en-v1.5/model.safetensors";
pub const TOKENIZER_PATH: &str = "./models/bge-small-en-v1.5/tokenizer.json";
pub const CONFIG_PATH: &str = "./models/bge-small-en-v1.5/config.json";
pub const BATCH_SIZE: usize = 64;
pub const CONCURRENT_BATCHES: usize = 4;
pub const MAX_TOKEN_LENGTH: usize = 512;
// pub const PIPELINE_DEPTH: usize = 8; // Number of batches to prefetch
