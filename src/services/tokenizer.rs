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

/// Helper function to split text into sentences
fn split_into_sentences(text: &str) -> Vec<String> {
    // Basic sentence splitting - handles periods, question marks, exclamation points
    // followed by space or end of string
    let mut sentences = Vec::new();
    let mut current_sentence = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        current_sentence.push(c);

        // Check for end of sentence
        if (c == '.' || c == '?' || c == '!')
            && (chars.peek().map_or(true, |next| next.is_whitespace()))
        {
            // Add current sentence if not empty
            if !current_sentence.trim().is_empty() {
                sentences.push(current_sentence.trim().to_string());
                current_sentence.clear();
            }
        }
    }

    // Add any remaining text as a sentence
    if !current_sentence.trim().is_empty() {
        sentences.push(current_sentence.trim().to_string());
    }

    // If no sentences were found (no punctuation), just return the whole text as one sentence
    if sentences.is_empty() && !text.trim().is_empty() {
        sentences.push(text.trim().to_string());
    }

    sentences
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

                // 1. Start with service name (highest priority)
                let service_name_text = format!("[SERVICE] {}", service.name.trim());
                let service_name_tokens = match tokenizer.encode(service_name_text.clone(), true) {
                    Ok(enc) => enc.get_ids().len(),
                    Err(_) => 10, // Safe fallback if encoding fails
                };

                let mut remaining_tokens = MAX_TOKEN_LENGTH - service_name_tokens;

                // 2. Smart sentence-by-sentence allocation for description
                let desc_allocation = (remaining_tokens as f32 * 0.6).floor() as usize;
                let description_truncated = if desc_allocation > 0 {
                    // Break description into sentences for smart selection
                    let sentences = split_into_sentences(&service.description);
                    let mut selected_sentences = Vec::new();
                    let mut token_count = 0;

                    // First pass: add sentences until we approach the limit
                    for sentence in sentences.clone() {
                        let sentence_tokens = match tokenizer.encode(sentence.clone(), true) {
                            Ok(enc) => enc.get_ids().len(),
                            Err(_) => sentence.split_whitespace().count(), // Rough estimation
                        };

                        if token_count + sentence_tokens <= desc_allocation {
                            selected_sentences.push(sentence.clone());
                            token_count += sentence_tokens;
                        } else {
                            // If we can't add a full sentence, stop
                            break;
                        }
                    }

                    // If we didn't select any sentences but have allocation,
                    // take at least the first sentence and truncate it
                    if selected_sentences.is_empty() && !sentences.is_empty() {
                        let first_sentence = sentences[0].clone();
                        match tokenizer.encode(first_sentence.clone(), true) {
                            Ok(enc) => {
                                if enc.get_ids().len() > desc_allocation {
                                    // Truncate this single sentence
                                    let ids = enc.get_ids();
                                    let truncated_ids = &ids[0..desc_allocation.min(ids.len())];
                                    match tokenizer.decode(truncated_ids, false) {
                                        Ok(text) => selected_sentences.push(text),
                                        Err(_) => selected_sentences.push(first_sentence),
                                    }
                                } else {
                                    selected_sentences.push(first_sentence);
                                }
                            }
                            Err(_) => selected_sentences.push(first_sentence),
                        }
                    }

                    selected_sentences.join(" ")
                } else {
                    "".to_string()
                };

                // Calculate actual tokens used by description
                let desc_actual_tokens = match tokenizer.encode(description_truncated.clone(), true)
                {
                    Ok(enc) => enc.get_ids().len(),
                    Err(_) => desc_allocation,
                };

                remaining_tokens -= desc_actual_tokens;

                // 3. Allocate tokens to taxonomy terms
                let taxonomy_terms_text = taxonomy_terms.join(", ");
                let taxonomy_allocation = (remaining_tokens as f32 * 0.6).floor() as usize;

                let taxonomy_truncated = if taxonomy_allocation > 0 {
                    match tokenizer.encode(taxonomy_terms_text.clone(), true) {
                        Ok(tax_encoding) => {
                            if tax_encoding.get_ids().len() > taxonomy_allocation {
                                // Try to keep whole terms where possible
                                let terms_list = taxonomy_terms.clone();
                                let mut selected_terms = Vec::new();
                                let mut token_count = 0;

                                for term in terms_list {
                                    let term_tokens = match tokenizer.encode(term.clone(), true) {
                                        Ok(enc) => enc.get_ids().len(),
                                        Err(_) => term.split_whitespace().count(),
                                    };

                                    // Add separator tokens (comma, space)
                                    let separator_tokens =
                                        if selected_terms.is_empty() { 0 } else { 2 };

                                    if token_count + term_tokens + separator_tokens
                                        <= taxonomy_allocation
                                    {
                                        selected_terms.push(term);
                                        token_count += term_tokens + separator_tokens;
                                    } else {
                                        break;
                                    }
                                }

                                selected_terms.join(", ")
                            } else {
                                taxonomy_terms_text
                            }
                        }
                        Err(_) => taxonomy_terms_text,
                    }
                } else {
                    "".to_string()
                };

                // Calculate actual tokens used by taxonomy terms
                let tax_actual_tokens = match tokenizer.encode(taxonomy_truncated.clone(), true) {
                    Ok(enc) => enc.get_ids().len(),
                    Err(_) => taxonomy_allocation,
                };

                remaining_tokens -= tax_actual_tokens;

                // 4. Use any remaining tokens for taxonomy descriptions
                // Split taxonomy descriptions into sentences too
                let all_tax_desc_sentences = taxonomy_descriptions
                    .iter()
                    .flat_map(|desc| split_into_sentences(desc))
                    .collect::<Vec<String>>();

                let tax_desc_truncated =
                    if remaining_tokens > 0 && !all_tax_desc_sentences.is_empty() {
                        let mut selected_sentences = Vec::new();
                        let mut token_count = 0;

                        for sentence in all_tax_desc_sentences {
                            let sentence_tokens = match tokenizer.encode(sentence.clone(), true) {
                                Ok(enc) => enc.get_ids().len(),
                                Err(_) => sentence.split_whitespace().count(),
                            };

                            if token_count + sentence_tokens <= remaining_tokens {
                                selected_sentences.push(sentence);
                                token_count += sentence_tokens;
                            } else {
                                break;
                            }
                        }

                        selected_sentences.join(" ")
                    } else {
                        "".to_string()
                    };

                // Combine all parts with proper section markers
                format!(
                    "[SERVICE] {} [DESCRIPTION] {} [CATEGORIES] {} [CATEGORY_DETAILS] {}",
                    service.name.trim(),
                    description_truncated.trim(),
                    taxonomy_truncated.trim(),
                    tax_desc_truncated.trim()
                )
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
                "[SERVICE] {} [DESCRIPTION] {}",
                service.name.trim(),
                service.description.trim()
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
