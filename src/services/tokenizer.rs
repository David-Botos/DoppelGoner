// src/services/tokenizer.rs

use anyhow::Result;
use log::{debug, warn};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

use crate::services::config::MAX_TOKEN_LENGTH;
use crate::services::types::{ServiceData, TokenizedBatch};

/// Loads a tokenizer from the specified path
///
/// # Arguments
/// * `tokenizer_path` - Path to the tokenizer JSON file
///
/// # Returns
/// * `Result<Arc<Tokenizer>>` - The loaded tokenizer wrapped in Arc for thread-safety
pub fn load_tokenizer(tokenizer_path: &str) -> Result<Arc<Tokenizer>> {
    debug!("Loading tokenizer from {}", tokenizer_path);

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {}: {}", tokenizer_path, e))?;

    debug!("Tokenizer loaded successfully");
    Ok(Arc::new(tokenizer))
}

/// Builds a text representation for embedding that combines service information with taxonomy data
///
/// # Arguments
/// * `service` - The service data containing name and description
/// * `tokenizer` - Reference to the tokenizer for length checking
///
/// # Returns
/// * `String` - The combined text representation for the service
pub fn build_text_for_embedding(service: &ServiceData, tokenizer: &Tokenizer) -> String {
    // Extract taxonomy terms and descriptions
    let taxonomy_terms: Vec<String> = service
        .taxonomy_info
        .iter()
        .map(|t| t.term.clone())
        .collect();

    let taxonomy_descriptions: Vec<String> = service
        .taxonomy_info
        .iter()
        .filter_map(|t| t.description.clone())
        .collect();

    // Use the templated approach with section markers
    let text = format!(
        "[SERVICE] {} [DESCRIPTION] {} [CATEGORIES] {} [CATEGORY_DETAILS] {}",
        service.name.trim(),
        service.description.trim(),
        taxonomy_terms.join(", "),
        taxonomy_descriptions.join(" ")
    );

    // Check if the text is too long for the model and truncate if necessary
    match tokenizer.encode(text.clone(), true) {
        Ok(encoding) => {
            if encoding.get_ids().len() > MAX_TOKEN_LENGTH {
                debug!(
                    "Text too long ({} tokens), truncating to {} tokens",
                    encoding.get_ids().len(),
                    MAX_TOKEN_LENGTH
                );

                // Start with the most important parts - service name and taxonomy terms
                let essential_text = format!(
                    "[SERVICE] {} [CATEGORIES] {}",
                    service.name.trim(),
                    taxonomy_terms.join(", ")
                );

                // Calculate remaining space
                match tokenizer.encode(essential_text.clone(), true) {
                    Ok(essential_encoding) => {
                        let remaining_tokens =
                            MAX_TOKEN_LENGTH - essential_encoding.get_ids().len();

                        if remaining_tokens <= 0 {
                            // Just return the essential parts
                            return essential_text;
                        }

                        // Now add description and taxonomy details with even split
                        let desc_tokens = remaining_tokens / 2;
                        let desc_truncated =
                            match tokenizer.encode(service.description.trim(), true) {
                                Ok(desc_encoding) => {
                                    if desc_encoding.get_ids().len() > desc_tokens {
                                        // Truncate description text
                                        let desc_token_ids = desc_encoding.get_ids();
                                        let truncated_ids = &desc_token_ids
                                            [0..desc_tokens.min(desc_token_ids.len())];
                                        tokenizer.decode(truncated_ids, false).unwrap_or_else(
                                            |_| service.description.trim().to_string(),
                                        )
                                    } else {
                                        service.description.trim().to_string()
                                    }
                                }
                                Err(_) => service.description.trim().to_string(),
                            };

                        let used_desc_tokens = match tokenizer.encode(desc_truncated.trim(), true) {
                            Ok(encoding) => encoding.get_ids().len(),
                            Err(_) => desc_tokens,
                        };

                        let tax_desc_tokens = remaining_tokens - used_desc_tokens;
                        let tax_desc_text = taxonomy_descriptions.join(" ");
                        let tax_desc_truncated = match tokenizer.encode(tax_desc_text.clone(), true)
                        {
                            Ok(tax_desc_encoding) => {
                                if tax_desc_encoding.get_ids().len() > tax_desc_tokens {
                                    let tax_desc_token_ids = tax_desc_encoding.get_ids();
                                    let truncated_ids = &tax_desc_token_ids
                                        [0..tax_desc_tokens.min(tax_desc_token_ids.len())];
                                    tokenizer
                                        .decode(truncated_ids, false)
                                        .unwrap_or_else(|_| tax_desc_text)
                                } else {
                                    tax_desc_text
                                }
                            }
                            Err(_) => tax_desc_text,
                        };

                        format!(
                            "[SERVICE] {} [DESCRIPTION] {} [CATEGORIES] {} [CATEGORY_DETAILS] {}",
                            service.name.trim(),
                            desc_truncated.trim(),
                            taxonomy_terms.join(", "),
                            tax_desc_truncated.trim()
                        )
                    }
                    Err(_) => essential_text,
                }
            } else {
                text
            }
        }
        Err(_) => {
            // If encoding fails, just return a simplified version
            warn!(
                "Failed to encode text for service {}, using simplified version",
                service.id
            );
            format!(
                "[SERVICE] {} [CATEGORIES] {}",
                service.name.trim(),
                taxonomy_terms.join(", ")
            )
        }
    }
}

/// Tokenizes a batch of service data for model inference
///
/// # Arguments
/// * `services` - Vec of ServiceData to tokenize
/// * `tokenizer` - Reference to the tokenizer
/// * `batch_id` - Identifier for the current batch (for logging)
///
/// # Returns
/// * `Result<TokenizedBatch>` - Tokenized representations ready for the model
pub fn tokenize_batch(
    services: &[ServiceData],
    tokenizer: &Arc<Tokenizer>,
    batch_id: &str,
) -> Result<TokenizedBatch> {
    debug!("{}: Tokenizing {} services", batch_id, services.len());

    // Prepare text for each service
    let mut texts = Vec::with_capacity(services.len());
    let mut service_ids = Vec::with_capacity(services.len());

    for service in services {
        let text = build_text_for_embedding(service, tokenizer);

        if !text.trim().is_empty() {
            service_ids.push(service.id.clone());
            texts.push(text);
        }
    }

    if texts.is_empty() {
        debug!("{}: No valid texts to process in batch", batch_id);
        return Ok(TokenizedBatch {
            service_ids: Vec::new(),
            input_ids: Vec::new(),
            attention_mask: Vec::new(),
            batch_id: batch_id.to_string(),
        });
    }

    // Encode all texts in batch
    let encodings = tokenizer
        .encode_batch(texts, true)
        .map_err(|e| anyhow::anyhow!("Failed to encode batch of texts: {}", e))?;

    // Extract input IDs and attention masks
    let input_ids: Vec<Vec<u32>> = encodings
        .iter()
        .map(|encoding| encoding.get_ids().to_vec())
        .collect();

    let attention_mask: Vec<Vec<u32>> = encodings
        .iter()
        .map(|encoding| encoding.get_attention_mask().to_vec())
        .collect();

    debug!(
        "{}: Successfully tokenized {} texts",
        batch_id,
        input_ids.len()
    );

    Ok(TokenizedBatch {
        service_ids,
        input_ids,
        attention_mask,
        batch_id: batch_id.to_string(),
    })
}

/// Manages tokenization of batches with a shared tokenizer
pub struct TokenizationManager {
    tokenizer: Arc<Tokenizer>,
    batch_counter: Arc<Mutex<usize>>,
}

impl TokenizationManager {
    /// Creates a new TokenizationManager with the provided tokenizer
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self {
            tokenizer,
            batch_counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Tokenizes a batch of services asynchronously
    ///
    /// # Arguments
    /// * `services` - Vec of ServiceData to process
    ///
    /// # Returns
    /// * `Result<TokenizedBatch>` - The tokenized batch
    pub async fn tokenize_services(&self, services: Vec<ServiceData>) -> Result<TokenizedBatch> {
        // Generate a unique batch ID
        let batch_number = {
            let mut counter = self.batch_counter.lock().await;
            *counter += 1;
            *counter
        };

        let batch_id = format!("batch-{}", batch_number);

        // Tokenize the batch
        let tokenized = tokenize_batch(&services, &self.tokenizer, &batch_id)?;

        Ok(tokenized)
    }
}

#[cfg(test)]
mod tests {

    use crate::services::types::TaxonomyInfo;

    use super::*;

    #[test]
    fn test_build_text_for_embedding() {
        // Mock tokenizer that pretends everything fits
        let tokenizer = Tokenizer::from_pretrained("bert-base-uncased", None).unwrap();

        // Test with complete data
        let service = ServiceData {
            id: "test123".to_string(),
            name: "Test Service".to_string(),
            description: "This is a test service description".to_string(),
            taxonomy_info: vec![
                TaxonomyInfo {
                    term: "Category1".to_string(),
                    description: Some("Description of category 1".to_string()),
                },
                TaxonomyInfo {
                    term: "Category2".to_string(),
                    description: None,
                },
            ],
        };

        let result = build_text_for_embedding(&service, &tokenizer);

        // Check that all sections are present
        assert!(result.contains("[SERVICE] Test Service"));
        assert!(result.contains("[DESCRIPTION] This is a test service description"));
        assert!(result.contains("[CATEGORIES] Category1, Category2"));
        assert!(result.contains("[CATEGORY_DETAILS] Description of category 1"));
    }
}
