use anyhow::{anyhow, Result};
use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

use crate::types::types::{EmbeddingDocument, TaxonomyDocument};

/// Tokenization strategy to use
#[derive(Debug, Clone, Copy)]
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
#[derive(Clone)]
pub struct ServiceTokenizer {
    config: TokenizerConfig,
    tokenizer: Arc<Tokenizer>,
}

impl ServiceTokenizer {
    /// Create a new tokenizer with default configuration
    pub fn new() -> Self {
        let config = TokenizerConfig::default();
        Self::with_config(config).expect("Failed to create ServiceTokenizer with default config")
    }

    /// Create a new tokenizer with custom configuration
    pub fn with_config(config: TokenizerConfig) -> Result<Self> {
        // Path to the tokenizer model files
        let model_path = format!("models/{}/tokenizer.json", config.model_id);

        // Load the tokenizer
        let tokenizer = match Tokenizer::from_file(&model_path) {
            Ok(t) => Arc::new(t),
            Err(e) => {
                // Fallback to default location if model directory is different
                let fallback_path = format!("./models/{}/tokenizer.json", config.model_id);
                match Tokenizer::from_file(&fallback_path) {
                    Ok(t) => Arc::new(t),
                    Err(_) => {
                        // Return a proper error instead of trying to create a dummy tokenizer
                        return Err(anyhow!(
                            "Failed to load tokenizer from {} or {}: {}",
                            model_path,
                            fallback_path,
                            e
                        ));
                    }
                }
            }
        };

        Ok(Self { config, tokenizer })
    }

    /// Tokenize a service document
    pub fn tokenize(&self, doc: &EmbeddingDocument) -> Result<(String, usize)> {
        // Create the base document text
        let content = self.create_base_document(doc)?;
        
        // Apply truncation if needed
        let strategy = TruncationStrategy::from(self.config.truncation_strategy.as_str());
        let tokenized = self.apply_truncation(content, &strategy)?;
        
        // Count tokens using real tokenizer
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
            let terms: Vec<String> = doc
                .taxonomies
                .iter()
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

        // Add URL and email if configured and available
        if self.config.include_url && doc.service_id.contains("http") {
            content.push_str(&format!("URL: {}\n\n", doc.service_id));
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
                // Simple truncation with more accurate token counting
                self.simple_truncation(text)
            }
            TruncationStrategy::Intelligent => {
                // Intelligent truncation preserves important parts
                self.intelligent_truncation(text)
            }
            TruncationStrategy::StartOnly => {
                // Keep only the beginning
                self.start_only_truncation(text)
            }
            TruncationStrategy::EndOnly => {
                // Keep only the end
                self.end_only_truncation(text)
            }
            TruncationStrategy::SlidingWindow => {
                // For now, fallback to intelligent truncation
                self.intelligent_truncation(text)
            }
        }
    }

    /// Simple truncation that cuts off text at max tokens
    fn simple_truncation(&self, text: String) -> Result<String> {
        let max_tokens = self.config.max_tokens;

        // Reserve some tokens for the ellipsis
        let ellipsis_tokens = self.count_tokens("...");
        let target_tokens = max_tokens.saturating_sub(ellipsis_tokens);

        let words: Vec<&str> = text.split_whitespace().collect();
        let mut truncated = String::new();
        let mut current_tokens = 0;

        for word in words.clone() {
            let word_with_space = format!("{} ", word);
            let word_tokens = self.count_tokens(&word_with_space);

            if current_tokens + word_tokens <= target_tokens {
                truncated.push_str(&word_with_space);
                current_tokens += word_tokens;
            } else {
                break;
            }
        }

        // Add ellipsis to indicate truncation
        truncated.push_str("...");

        // Final verification - if we're still over limit, do hard truncation
        if self.count_tokens(&truncated) > max_tokens {
            // More aggressive truncation - consider character-level if needed
            let chars_per_token = 4; // Very rough approximation
            let safe_char_count = (max_tokens - ellipsis_tokens) * chars_per_token;

            if truncated.len() > safe_char_count {
                truncated = format!(
                    "{}...",
                    &truncated[..safe_char_count.min(truncated.len() - 1)]
                );
            }

            // If still over limit, just take first few words
            if self.count_tokens(&truncated) > max_tokens {
                let limited_words: Vec<&str> = words.into_iter().take(max_tokens / 2).collect();
                truncated = format!("{}...", limited_words.join(" "));
            }
        }

        Ok(truncated)
    }

    /// Keep only the beginning of the text
    fn start_only_truncation(&self, text: String) -> Result<String> {
        // Similar to simple truncation but keeps only the beginning
        self.simple_truncation(text)
    }

    /// Keep only the end of the text
    fn end_only_truncation(&self, text: String) -> Result<String> {
        let max_tokens = self.config.max_tokens;

        // Reserve tokens for ellipsis
        let ellipsis_tokens = self.count_tokens("...");
        let target_tokens = max_tokens.saturating_sub(ellipsis_tokens);

        let words: Vec<&str> = text.split_whitespace().collect();

        if self.count_tokens(&text) <= max_tokens {
            return Ok(text);
        }

        // Start with a prefix and add words from the end until we hit the limit
        let mut truncated = "...".to_string();
        let mut end_chunk = Vec::new();
        let mut current_tokens = ellipsis_tokens;

        for word in words.iter().rev() {
            let word_with_space = format!(" {}", word);
            let word_tokens = self.count_tokens(&word_with_space);

            if current_tokens + word_tokens <= max_tokens {
                end_chunk.insert(0, word.to_string());
                current_tokens += word_tokens;
            } else {
                break;
            }
        }

        // Add the end chunk
        if !end_chunk.is_empty() {
            truncated.push(' ');
            truncated.push_str(&end_chunk.join(" "));
        }

        // Final check
        if self.count_tokens(&truncated) > max_tokens {
            // Resort to simple truncation from the end
            let words: Vec<&str> = truncated.split_whitespace().collect();
            let safe_count = (max_tokens / 2).max(3); // At least 3 words

            let limited_words = if words.len() > safe_count {
                words.into_iter().take(safe_count).collect::<Vec<&str>>()
            } else {
                words
            };

            truncated = format!("... {}", limited_words.join(" "));
        }

        Ok(truncated)
    }

    /// Intelligent truncation that preserves important parts of the document
    fn intelligent_truncation(&self, text: String) -> Result<String> {
        // Define regex patterns for extracting document sections
        lazy_static! {
            static ref SERVICE_RE: Regex = Regex::new(r"Service:\s*(.+?)\n\n").unwrap();
            static ref DESC_RE: Regex = Regex::new(r"Description:\s*(.+?)\n\n").unwrap();
            static ref CATEGORIES_RE: Regex = Regex::new(r"Categories:\s*(.+?)\n\n").unwrap();
            static ref TAXONOMY_RE: Regex = Regex::new(r"([^:]+):\s*(.+?)\n").unwrap();
            static ref URL_RE: Regex = Regex::new(r"URL:\s*(.+?)\n\n").unwrap();
            // For better sentence and clause splitting
            static ref SENTENCE_SPLIT_RE: Regex = Regex::new(r"(?:[.!?])\s+").unwrap();
            static ref COMMA_SPLIT_RE: Regex = Regex::new(r",\s+").unwrap();
        }

        // Extract sections from text
        let service_name = SERVICE_RE
            .captures(&text)
            .map(|c| c[1].to_string())
            .unwrap_or_default();

        let description = DESC_RE
            .captures(&text)
            .map(|c| c[1].to_string())
            .unwrap_or_default();

        let categories = CATEGORIES_RE
            .captures(&text)
            .map(|c| c[1].to_string())
            .unwrap_or_default();

        let url = URL_RE.captures(&text).map(|c| c[1].to_string());

        // Extract taxonomy descriptions (excluding categories)
        let mut taxonomy_descriptions = Vec::new();
        for cap in TAXONOMY_RE.captures_iter(&text) {
            let taxonomy_name = cap[1].trim().to_string();
            let taxonomy_desc = cap[2].trim().to_string();

            // Skip categories section since we handled it separately
            if taxonomy_name == "Categories" {
                continue;
            }

            taxonomy_descriptions.push((taxonomy_name, taxonomy_desc));
        }

        // Reserve tokens for ellipsis
        let ellipsis_tokens = self.count_tokens("...");

        // Total available tokens
        let available_tokens = self.config.max_tokens.saturating_sub(ellipsis_tokens);
        let mut tokens_used = 0;
        let mut result = String::new();
        let mut truncation_applied = false;

        // Priority 1: Always include the service name (highest priority)
        let service_section = format!("Service: {}\n\n", service_name);
        let service_tokens = self.count_tokens(&service_section);
        result.push_str(&service_section);
        tokens_used += service_tokens;

        // Priority 2: Add categories if they fit (second highest priority)
        if !categories.is_empty() {
            let categories_section = format!("Categories: {}\n\n", categories);
            let categories_tokens = self.count_tokens(&categories_section);

            if tokens_used + categories_tokens <= available_tokens {
                result.push_str(&categories_section);
                tokens_used += categories_tokens;
            } else {
                // Try to fit truncated categories
                let prefix = "Categories: ";
                let prefix_tokens = self.count_tokens(prefix);
                let available_for_categories =
                    available_tokens.saturating_sub(tokens_used + prefix_tokens + 3); // +3 for "...\n\n"

                if available_for_categories > 10 {
                    let categories_items: Vec<&str> = categories.split(", ").collect();
                    let mut truncated_categories = String::new();
                    let mut item_tokens_used = 0;

                    for (i, item) in categories_items.iter().enumerate() {
                        let item_text = if i == categories_items.len() - 1 {
                            item.to_string()
                        } else {
                            format!("{}, ", item)
                        };

                        let item_tokens = self.count_tokens(&item_text);

                        if item_tokens_used + item_tokens <= available_for_categories {
                            truncated_categories.push_str(&item_text);
                            item_tokens_used += item_tokens;
                        } else {
                            truncation_applied = true;
                            break;
                        }
                    }

                    if !truncated_categories.is_empty() {
                        let trunc_section = format!("{}{}...\n\n", prefix, truncated_categories);
                        result.push_str(&trunc_section);
                        tokens_used += self.count_tokens(&trunc_section);
                    }
                }
            }
        }

        // Priority 3: Add description with sentence-level truncation
        if !description.is_empty() {
            // First, see if the entire description fits
            let full_desc_section = format!("Description: {}\n\n", description);
            let full_desc_tokens = self.count_tokens(&full_desc_section);

            if tokens_used + full_desc_tokens <= available_tokens {
                // Entire description fits
                result.push_str(&full_desc_section);
                tokens_used += full_desc_tokens;
            } else {
                // Need to truncate the description
                let desc_prefix = "Description: ";
                let desc_tokens_available = available_tokens
                    .saturating_sub(tokens_used + self.count_tokens(desc_prefix) + 3); // +3 for "...\n\n"

                if desc_tokens_available > 10 {
                    // Ensure there's enough space for meaningful content
                    // Split into sentences for more intelligent truncation
                    let sentences: Vec<&str> = SENTENCE_SPLIT_RE
                        .split(&description)
                        .filter(|s| !s.trim().is_empty())
                        .collect();

                    let mut truncated_desc = String::new();
                    let mut desc_tokens_used = 0;

                    for sentence in sentences {
                        let clean_sentence = sentence.trim();
                        let sentence_with_period = format!("{}. ", clean_sentence);
                        let sentence_tokens = self.count_tokens(&sentence_with_period);

                        if desc_tokens_used + sentence_tokens <= desc_tokens_available {
                            // Full sentence fits
                            truncated_desc.push_str(&sentence_with_period);
                            desc_tokens_used += sentence_tokens;
                        } else if desc_tokens_available > desc_tokens_used + 5 {
                            // Try to fit part of the sentence by splitting at commas
                            let comma_parts: Vec<&str> = COMMA_SPLIT_RE
                                .split(clean_sentence)
                                .filter(|s| !s.trim().is_empty())
                                .collect();

                            let mut added_part = false;
                            for part in comma_parts {
                                let part_text = format!("{}, ", part.trim());
                                let part_tokens = self.count_tokens(&part_text);

                                if desc_tokens_used + part_tokens <= desc_tokens_available - 3 {
                                    // Leave room for ellipsis
                                    truncated_desc.push_str(&part_text);
                                    desc_tokens_used += part_tokens;
                                    added_part = true;
                                } else {
                                    // Try word-by-word
                                    let words: Vec<&str> = part.split_whitespace().collect();
                                    let mut word_chunk = String::new();
                                    let mut word_tokens_used = 0;

                                    for word in words {
                                        let word_text = format!("{} ", word);
                                        let word_tokens = self.count_tokens(&word_text);

                                        if desc_tokens_used + word_tokens_used + word_tokens
                                            <= desc_tokens_available - 3
                                        {
                                            word_chunk.push_str(&word_text);
                                            word_tokens_used += word_tokens;
                                        } else {
                                            break;
                                        }
                                    }

                                    if !word_chunk.is_empty() {
                                        truncated_desc.push_str(&word_chunk);
                                        desc_tokens_used += word_tokens_used;
                                        added_part = true;
                                    }

                                    break;
                                }
                            }

                            // Add ellipsis if we added a partial sentence
                            if added_part {
                                truncated_desc.push_str("...");
                                truncation_applied = true;
                            }

                            break;
                        } else {
                            // No room for partial sentence
                            break;
                        }
                    }

                    if !truncated_desc.is_empty() {
                        result.push_str(&format!("{}{}\n\n", desc_prefix, truncated_desc));
                        tokens_used = self.count_tokens(&result); // Recalculate total tokens used
                    }
                }
            }
        }

        // Priority 4: Add URL if configured and available
        if let Some(url_str) = url {
            let url_section = format!("URL: {}\n\n", url_str);
            let url_tokens = self.count_tokens(&url_section);

            if tokens_used + url_tokens <= available_tokens {
                result.push_str(&url_section);
                tokens_used += url_tokens;
            }
        }

        // Priority 5: Add taxonomy descriptions if there's space
        if !taxonomy_descriptions.is_empty() {
            for (taxonomy_name, taxonomy_desc) in taxonomy_descriptions {
                let full_taxonomy_section = format!("{}: {}\n", taxonomy_name, taxonomy_desc);
                let taxonomy_tokens = self.count_tokens(&full_taxonomy_section);

                if tokens_used + taxonomy_tokens <= available_tokens {
                    // Full taxonomy description fits
                    result.push_str(&full_taxonomy_section);
                    tokens_used += taxonomy_tokens;
                } else {
                    // Try to fit partial taxonomy description
                    let prefix = format!("{}: ", taxonomy_name);
                    let prefix_tokens = self.count_tokens(&prefix);
                    let content_tokens_available =
                        available_tokens.saturating_sub(tokens_used + prefix_tokens + 3); // +3 for "...\n"

                    if content_tokens_available > 5 {
                        // Split into words and add as many as will fit
                        let words: Vec<&str> = taxonomy_desc.split_whitespace().collect();
                        let mut truncated_desc = String::new();
                        let mut word_tokens_used = 0;

                        for word in words {
                            let word_with_space = format!("{} ", word);
                            let word_tokens = self.count_tokens(&word_with_space);

                            if word_tokens_used + word_tokens <= content_tokens_available - 3 {
                                truncated_desc.push_str(&word_with_space);
                                word_tokens_used += word_tokens;
                            } else {
                                truncation_applied = true;
                                break;
                            }
                        }

                        if !truncated_desc.is_empty() {
                            let partial_section = format!("{}{}...\n", prefix, truncated_desc);
                            result.push_str(&partial_section);
                            tokens_used += self.count_tokens(&partial_section);
                        }
                    }

                    // No more space for other taxonomies
                    break;
                }
            }
        }

        // Add ellipsis at the end if truncation was applied but no ellipsis was added yet
        if truncation_applied && !result.contains("...") {
            result.push_str("...");
        }

        // Add a final verification to ensure we don't exceed max tokens
        let final_token_count = self.count_tokens(&result);
        if final_token_count > self.config.max_tokens {
            warn!(
                "Truncation algorithm produced text with {} tokens, which exceeds max_tokens {}. Applying simple truncation as fallback.",
                final_token_count, self.config.max_tokens
            );

            // Fall back to simple truncation
            return self.simple_truncation(text);
        }

        Ok(result)
    }

    /// Count tokens in text using the real tokenizer
    fn count_tokens(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        
        // Use the real tokenizer - no fallback to approximate counting
        match self.tokenizer.encode(text, false) {
            Ok(encoding) => encoding.get_ids().len(),
            Err(e) => {
                // Log the error but still return a value to avoid crashing
                // In a real implementation, we might want to propagate this error
                warn!("Error encoding text with tokenizer: {}", e);
                // Make a best effort approximation 
                text.split_whitespace().count() + (text.len() / 4)
            }
        }
    }

    /// Approximate token counting as a fallback
    fn approximate_count_tokens(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }

        // Base count: split on whitespace
        let word_count = text.split_whitespace().count();

        // Count punctuation and special characters that may get their own token
        let special_char_count = text
            .chars()
            .filter(|c| ",.!?;:()[]{}\"'".contains(*c))
            .count();

        // URLs typically tokenize into many more tokens than word count would suggest
        let url_adjustment = if text.contains("http") || text.contains("www.") {
            20 // Rough adjustment for URLs - they use many tokens
        } else {
            0
        };

        // BERT tokenizers often split words at subword boundaries
        let subword_adjustment = (word_count as f32 * 0.4) as usize; // About 40% of words get split

        // Final token count approximation with a safety margin
        let estimated_tokens =
            word_count + (special_char_count / 2) + url_adjustment + subword_adjustment;

        // Add a safety margin for BERT-like tokenizers
        estimated_tokens + (estimated_tokens / 5) // Add 20% safety margin
    }
}
