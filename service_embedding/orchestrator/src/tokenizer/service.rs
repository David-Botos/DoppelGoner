// orchestrator/src/tokenizer/service.rs
// Implementation of the service tokenizer

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use regex::Regex;
use lazy_static::lazy_static;
use tracing::{debug, info};

use crate::types::types::{EmbeddingDocument, TaxonomyDocument};

/// Tokenization strategy to use
pub enum TruncationStrategy {
    /// Truncate document to max tokens, cutting from the end
    Simple,
    
    /// Maintain important information by selectively removing less important parts
    Intelligent,
    
    /// Keep only the beginning of the text
    StartOnly,
    
    /// Keep only the end of the text
    EndOnly,
    
    /// Use sliding window to create multiple chunks
    SlidingWindow,
}

impl From<&str> for TruncationStrategy {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "intelligent" => TruncationStrategy::Intelligent,
            "start_only" => TruncationStrategy::StartOnly,
            "end_only" => TruncationStrategy::EndOnly,
            "sliding_window" => TruncationStrategy::SlidingWindow,
            _ => TruncationStrategy::Simple,
        }
    }
}

/// Configuration for the tokenizer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Maximum number of tokens to include
    pub max_tokens: usize,
    
    /// Model ID to use for tokenization
    pub model_id: String,
    
    /// Truncation strategy to use when text exceeds max tokens
    pub truncation_strategy: String,
    
    /// Weight of service name (for intelligent truncation)
    pub name_weight: f32,
    
    /// Weight of service description (for intelligent truncation)
    pub description_weight: f32,
    
    /// Weight of taxonomies (for intelligent truncation)
    pub taxonomy_weight: f32,
    
    /// Include URL in tokenized text
    pub include_url: bool,
    
    /// Include email in tokenized text
    pub include_email: bool,
    
    /// Default language for tokenization
    pub language: String,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            max_tokens: 384, // Default for BGE-small-en-v1.5
            model_id: "bge-small-en-v1.5".to_string(),
            truncation_strategy: "intelligent".to_string(),
            name_weight: 1.5,
            description_weight: 1.0,
            taxonomy_weight: 0.8,
            include_url: true,
            include_email: false,
            language: "en".to_string(),
        }
    }
}

/// Service Tokenizer for preparing documents for embedding
pub struct ServiceTokenizer {
    config: TokenizerConfig,
}

impl ServiceTokenizer {
    /// Create a new tokenizer with default configuration
    pub fn new() -> Self {
        Self {
            config: TokenizerConfig::default(),
        }
    }
    
    /// Create a new tokenizer with custom configuration
    pub fn with_config(config: TokenizerConfig) -> Self {
        Self { config }
    }
    
    /// Tokenize a service document
    pub fn tokenize(&self, doc: &EmbeddingDocument) -> Result<(String, usize)> {
        // Create the base document text
        let mut content = self.create_base_document(doc)?;
        
        // Apply truncation if needed
        let strategy = TruncationStrategy::from(self.config.truncation_strategy.as_str());
        let tokenized = self.apply_truncation(content, &strategy)?;
        
        // Count tokens (approximate - in a real implementation we would use a proper tokenizer)
        let token_count = self.count_tokens(&tokenized);
        
        debug!(
            "Tokenized service {} with {} tokens (strategy: {:?})",
            doc.service_id, token_count, strategy
        );
        
        Ok((tokenized, token_count))
    }
    
    /// Create the base document text
    fn create_base_document(&self, doc: &EmbeddingDocument) -> Result<String> {
        let mut content = String::new();
        
        // Add service name
        content.push_str(&format!("Service: {}\n\n", doc.service_name));
        
        // Add description if available
        if !doc.service_desc.is_empty() {
            content.push_str(&format!("Description: {}\n\n", doc.service_desc));
        }
        
        // Add taxonomies if available
        if !doc.taxonomies.is_empty() {
            content.push_str("Categories: ");
            let terms: Vec<String> = doc.taxonomies.iter()
                .map(|t| t.taxonomy_name.clone())
                .collect();
            content.push_str(&terms.join(", "));
            content.push_str("\n\n");
            
            // Add taxonomy descriptions if available
            for taxonomy in &doc.taxonomies {
                if let Some(desc) = &taxonomy.taxonomy_desc {
                    if !desc.is_empty() {
                        content.push_str(&format!("{}: {}\n", taxonomy.taxonomy_name, desc));
                    }
                }
            }
            if doc.taxonomies.iter().any(|t| t.taxonomy_desc.is_some()) {
                content.push_str("\n");
            }
        }
        
        Ok(content)
    }
    
    /// Apply truncation strategy to text
    fn apply_truncation(&self, text: String, strategy: &TruncationStrategy) -> Result<String> {
        let approx_token_count = self.count_tokens(&text);
        
        // If under max tokens, no truncation needed
        if approx_token_count <= self.config.max_tokens {
            return Ok(text);
        }
        
        // Apply truncation strategies
        match strategy {
            TruncationStrategy::Simple => {
                // Simple truncation just cuts off at max tokens
                let words: Vec<&str> = text.split_whitespace().collect();
                let truncated = words.into_iter()
                    .take(self.config.max_tokens)
                    .collect::<Vec<&str>>()
                    .join(" ");
                
                Ok(truncated)
            },
            TruncationStrategy::Intelligent => {
                // Intelligent truncation preserves important parts
                self.intelligent_truncation(text)
            },
            TruncationStrategy::StartOnly => {
                // Keep only the beginning
                let words: Vec<&str> = text.split_whitespace().collect();
                let truncated = words.into_iter()
                    .take(self.config.max_tokens)
                    .collect::<Vec<&str>>()
                    .join(" ");
                
                Ok(truncated)
            },
            TruncationStrategy::EndOnly => {
                // Keep only the end
                let words: Vec<&str> = text.split_whitespace().collect();
                let start_idx = if words.len() > self.config.max_tokens {
                    words.len() - self.config.max_tokens
                } else {
                    0
                };
                
                let truncated = words.into_iter()
                    .skip(start_idx)
                    .collect::<Vec<&str>>()
                    .join(" ");
                
                Ok(truncated)
            },
            TruncationStrategy::SlidingWindow => {
                // Sliding window not implemented in this version
                // In a real implementation, this would create multiple chunks
                // Since we only return one chunk here, we'll use simple truncation
                let words: Vec<&str> = text.split_whitespace().collect();
                let truncated = words.into_iter()
                    .take(self.config.max_tokens)
                    .collect::<Vec<&str>>()
                    .join(" ");
                
                Ok(truncated)
            },
        }
    }
    
    /// Intelligent truncation that preserves important parts of the document
    fn intelligent_truncation(&self, text: String) -> Result<String> {
        // Split content into sections
        lazy_static! {
            static ref SERVICE_RE: Regex = Regex::new(r"Service:\s*(.+?)\n\n").unwrap();
            static ref DESC_RE: Regex = Regex::new(r"Description:\s*(.+?)\n\n").unwrap();
            static ref CATEGORIES_RE: Regex = Regex::new(r"Categories:\s*(.+?)\n\n").unwrap();
            static ref TAXONOMY_RE: Regex = Regex::new(r"([^:]+):\s*(.+?)\n").unwrap();
        }
        
        // Extract sections
        let service_name = SERVICE_RE.captures(&text)
            .map(|c| c[1].to_string())
            .unwrap_or_default();
        
        let description = DESC_RE.captures(&text)
            .map(|c| c[1].to_string())
            .unwrap_or_default();
        
        let categories = CATEGORIES_RE.captures(&text)
            .map(|c| c[1].to_string())
            .unwrap_or_default();
        
        // Calculate tokens for each section
        let name_tokens = self.count_tokens(&service_name);
        let desc_tokens = self.count_tokens(&description);
        let categories_tokens = self.count_tokens(&categories);
        
        // Total available tokens
        let available_tokens = self.config.max_tokens;
        
        // Always include the service name
        let mut result = format!("Service: {}\n\n", service_name);
        let mut tokens_used = name_tokens + 3; // +3 for "Service: " and "\n\n"
        
        // Add categories if they fit
        if tokens_used + categories_tokens + 12 <= available_tokens {
            result.push_str(&format!("Categories: {}\n\n", categories));
            tokens_used += categories_tokens + 12; // +12 for "Categories: " and "\n\n"
        }
        
        // Calculate tokens available for description
        let desc_tokens_available = available_tokens.saturating_sub(tokens_used);
        
        // Add truncated description if needed
        if !description.is_empty() {
            let desc_words: Vec<&str> = description.split_whitespace().collect();
            
            if desc_tokens <= desc_tokens_available {
                // Entire description fits
                result.push_str(&format!("Description: {}\n\n", description));
            } else {
                // Truncate description
                let words_to_take = (desc_tokens_available as f32 * 0.9) as usize; // 90% of available
                let truncated_desc = desc_words.into_iter()
                    .take(words_to_take)
                    .collect::<Vec<&str>>()
                    .join(" ");
                
                result.push_str(&format!("Description: {}...\n\n", truncated_desc));
            }
        }
        
        Ok(result)
    }
    
    /// Count tokens in text (approximate)
    fn count_tokens(&self, text: &str) -> usize {
        // In a real implementation, we would use a proper tokenizer from the model
        // For now, we'll use a simple whitespace tokenizer with some adjustments
        
        // Split on whitespace
        let word_count = text.split_whitespace().count();
        
        // Adjust for punctuation and special characters (approximation)
        // For BGE-small-en-v1.5, most words are 1 token, but some special chars are separate tokens
        let punct_count = text.chars()
            .filter(|c| ",.!?;:()[]{}\"'".contains(*c))
            .count();
        
        word_count + (punct_count / 2) // Approximate adjustment
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tokenize_simple() {
        let tokenizer = ServiceTokenizer::new();
        
        let doc = EmbeddingDocument {
            service_id: "test-123".to_string(),
            service_name: "Test Service".to_string(),
            service_desc: "This is a test service description.".to_string(),
            taxonomies: vec![
                TaxonomyDocument {
                    taxonomy_name: "Category 1".to_string(),
                    taxonomy_desc: Some("Description of category 1".to_string()),
                },
                TaxonomyDocument {
                    taxonomy_name: "Category 2".to_string(),
                    taxonomy_desc: None,
                },
            ],
        };
        
        let result = tokenizer.tokenize(&doc).unwrap();
        assert!(result.1 > 0);
        assert!(result.0.contains("Service: Test Service"));
        assert!(result.0.contains("Categories: Category 1, Category 2"));
        assert!(result.0.contains("Description: This is a test service description"));
    }
}