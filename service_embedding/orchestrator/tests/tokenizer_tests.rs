// orchestrator/tests/tokenizer_tests.rs

use orchestrator::tokenizer::{ServiceTokenizer, TokenizerConfig};
use orchestrator::types::types::{EmbeddingDocument, TaxonomyDocument};

#[test]
fn test_tokenizer_no_truncation_needed() {
    // Create a tokenizer with a token limit matching BGE-small-en-v1.5
    let config = TokenizerConfig {
        max_tokens: 384,
        model_id: "bge-small-en-v1.5".to_string(),
        truncation_strategy: "intelligent".to_string(),
        name_weight: 1.5,
        description_weight: 1.0,
        taxonomy_weight: 0.8,
        include_url: true,
        include_email: false,
        language: "en".to_string(),
    };

    // Unwrap the Result to get the tokenizer
    let tokenizer =
        ServiceTokenizer::with_config(config.clone()).expect("Failed to create tokenizer");

    // Create a document that shouldn't need truncation
    let doc = EmbeddingDocument {
        service_id: "service-123".to_string(),
        service_name: "Sample Service".to_string(),
        service_desc: "This is a sample service description that provides functionality."
            .to_string(),
        taxonomies: vec![
            TaxonomyDocument {
                taxonomy_name: "Category 1".to_string(),
                taxonomy_desc: Some("Description of category 1".to_string()),
            },
            TaxonomyDocument {
                taxonomy_name: "Category 2".to_string(),
                taxonomy_desc: Some("Description of category 2".to_string()),
            },
        ],
    };

    // Tokenize the document
    let (tokenized, token_count) = tokenizer.tokenize(&doc).unwrap();

    // Verify the content is preserved
    assert!(
        tokenized.contains("Service: Sample Service"),
        "Service name should be preserved"
    );
    assert!(
        tokenized.contains("Description: This is a sample service"),
        "Description should be preserved"
    );
    assert!(
        tokenized.contains("Categories: Category 1, Category 2"),
        "Categories should be preserved"
    );
    assert!(
        tokenized.contains("Category 1: Description of category 1"),
        "Category 1 description should be preserved"
    );
    assert!(
        tokenized.contains("Category 2: Description of category 2"),
        "Category 2 description should be preserved"
    );

    // Verify token count is reasonable
    assert!(
        token_count > 0 && token_count <= 384,
        "Token count should be within limits"
    );
}

#[test]
fn test_tokenizer_intelligent_truncation() {
    // Create a tokenizer with a small token limit to force truncation
    let config = TokenizerConfig {
        max_tokens: 50, // Very small limit to force intelligent truncation
        model_id: "bge-small-en-v1.5".to_string(),
        truncation_strategy: "intelligent".to_string(),
        name_weight: 1.5,
        description_weight: 1.0,
        taxonomy_weight: 0.8,
        include_url: true,
        include_email: false,
        language: "en".to_string(),
    };

    // Unwrap the Result to get the tokenizer
    let tokenizer = ServiceTokenizer::with_config(config).expect("Failed to create tokenizer");

    // Create a document with lots of content that will need truncation
    let doc = EmbeddingDocument {
        service_id: "service-456".to_string(),
        service_name: "Enterprise Resource Planning System".to_string(),
        service_desc: "This enterprise resource planning system offers comprehensive solutions for businesses of all sizes. It integrates various functions including inventory management, human resources, customer relationship management, and financial reporting into a unified platform. The system is designed to streamline operations, improve efficiency, and provide real-time insights for better decision making. It features a user-friendly interface, robust security measures, and customizable modules to meet specific business needs.".to_string(),
        taxonomies: vec![
            TaxonomyDocument {
                taxonomy_name: "Business Software".to_string(),
                taxonomy_desc: Some("Software designed for business applications and enterprise operations".to_string()),
            },
            TaxonomyDocument {
                taxonomy_name: "Cloud Services".to_string(),
                taxonomy_desc: Some("Services delivered over the internet rather than on-premises".to_string()),
            },
            TaxonomyDocument {
                taxonomy_name: "Premium Tier".to_string(),
                taxonomy_desc: Some("High-end service offerings with advanced features and support".to_string()),
            },
        ],
    };

    // Tokenize the document
    let (tokenized, token_count) = tokenizer.tokenize(&doc).unwrap();

    // Verify prioritization of content
    assert!(
        tokenized.contains("Service: Enterprise Resource Planning System"),
        "Service name should always be preserved"
    );

    assert!(
        tokenized.contains("Categories: Business Software, Cloud Services, Premium Tier")
            || tokenized.contains("Categories: Business Software, Cloud Services")
            || tokenized.contains("Categories: Business Software"),
        "At least some categories should be preserved"
    );

    // Verify token count is within limits
    assert!(
        token_count <= 50,
        "Token count should be within the configured limit"
    );

    // Check for ellipsis indicating truncation
    assert!(
        tokenized.contains("..."),
        "Should have ellipsis indicating truncation"
    );

    // Check that lower priority items are truncated before higher priority ones
    let has_taxonomy_descriptions =
        tokenized.contains("Business Software: Software designed for business");
    let has_full_description = tokenized.contains("This enterprise resource planning system offers comprehensive solutions for businesses of all sizes");

    // If we have taxonomy descriptions, we should definitely have the description
    if has_taxonomy_descriptions {
        assert!(
            has_full_description,
            "If taxonomy descriptions are included, the main description should also be included"
        );
    }
}

#[test]
fn test_tokenizer_prioritization() {
    // Create a tokenizer with a medium token limit to test prioritization
    let config = TokenizerConfig {
        max_tokens: 100,
        model_id: "bge-small-en-v1.5".to_string(),
        truncation_strategy: "intelligent".to_string(),
        name_weight: 1.5,
        description_weight: 1.0,
        taxonomy_weight: 0.8,
        include_url: true,
        include_email: false,
        language: "en".to_string(),
    };

    // Unwrap the Result to get the tokenizer
    let tokenizer = ServiceTokenizer::with_config(config).expect("Failed to create tokenizer");

    // Create a document with more content than the token limit
    let doc = EmbeddingDocument {
        service_id: "service-789".to_string(),
        service_name: "Data Analytics Platform".to_string(),
        service_desc: "Advanced analytics platform for processing large datasets. Features include data visualization, machine learning capabilities, and integration with various data sources. The platform supports real-time analytics and batch processing.".to_string(),
        taxonomies: vec![
            TaxonomyDocument {
                taxonomy_name: "Analytics".to_string(),
                taxonomy_desc: Some("Tools for analyzing and visualizing data".to_string()),
            },
            TaxonomyDocument {
                taxonomy_name: "Big Data".to_string(),
                taxonomy_desc: Some("Solutions for handling large volumes of data efficiently".to_string()),
            },
            TaxonomyDocument {
                taxonomy_name: "Machine Learning".to_string(),
                taxonomy_desc: Some("Systems that can learn from and make predictions based on data".to_string()),
            },
            TaxonomyDocument {
                taxonomy_name: "Enterprise".to_string(),
                taxonomy_desc: Some("Solutions designed for large organizations with complex needs".to_string()),
            },
        ],
    };

    // Tokenize the document
    let (tokenized, token_count) = tokenizer.tokenize(&doc).unwrap();

    // Verify token count is within limits
    assert!(
        token_count <= 100,
        "Token count should be within the configured limit"
    );

    // Check prioritization of elements
    // These items should be included (highest priority)
    assert!(
        tokenized.contains("Service: Data Analytics Platform"),
        "Service name must be included"
    );
    assert!(
        tokenized.contains("Categories:"),
        "Categories section must be included"
    );

    // The main description should be included but might be truncated
    assert!(
        tokenized.contains("Description:"),
        "Description section must be included"
    );

    // Not all taxonomy descriptions may be included (lower priority)
    let taxonomy_descriptions_count = tokenized.matches(": ").count() - 2; // Subtract 2 for "Service:" and "Description:"
    assert!(
        taxonomy_descriptions_count <= 4,
        "Not all taxonomy descriptions may fit"
    );

    // Check for proper truncation indicators
    if !tokenized.contains("The platform supports real-time analytics and batch processing") {
        assert!(
            tokenized.contains("..."),
            "Should indicate truncation with ellipsis"
        );
    }
}

#[test]
fn test_tokenizer_empty_fields() {
    // Create a tokenizer with default config
    let tokenizer = ServiceTokenizer::new();

    // Create a document with some empty fields
    let doc = EmbeddingDocument {
        service_id: "service-empty".to_string(),
        service_name: "Empty Service".to_string(),
        service_desc: "".to_string(), // Empty description
        taxonomies: vec![],           // No taxonomies
    };

    // Tokenize the document
    let (tokenized, token_count) = tokenizer.tokenize(&doc).unwrap();

    // Verify the document was tokenized correctly despite empty fields
    assert!(
        tokenized.contains("Service: Empty Service"),
        "Service name should be included"
    );
    assert!(
        !tokenized.contains("Description:"),
        "Empty description should be omitted"
    );
    assert!(
        !tokenized.contains("Categories:"),
        "Empty categories should be omitted"
    );

    // Verify token count is reasonable
    assert!(token_count > 0, "Token count should be greater than zero");
    assert!(
        token_count < 20,
        "Token count should be low for this minimal document"
    );
}

#[test]
fn test_tokenizer_very_long_content() {
    // Create a tokenizer with default config
    let tokenizer = ServiceTokenizer::new();

    // Create a document with extremely long content
    let long_description = "This is an extremely long service description that goes into great detail about all the features and capabilities. ".repeat(100);

    let doc = EmbeddingDocument {
        service_id: "service-long".to_string(),
        service_name: "Long Content Service".to_string(),
        service_desc: long_description.clone(),
        taxonomies: vec![
            TaxonomyDocument {
                taxonomy_name: "Category X".to_string(),
                taxonomy_desc: Some("This is a very detailed category description that elaborates extensively on what this category encompasses. ".repeat(20)),
            },
        ],
    };

    // Tokenize the document
    let (tokenized, token_count) = tokenizer.tokenize(&doc).unwrap();

    // Verify token count is within the default limit
    assert!(
        token_count <= 384,
        "Token count should be within the default limit of 384"
    );

    // Verify that the document contains the high-priority content
    assert!(
        tokenized.contains("Service: Long Content Service"),
        "Service name should be included"
    );
    assert!(
        tokenized.contains("Categories: Category X"),
        "Categories should be included"
    );

    // Verify truncation of long content
    assert!(
        tokenized.contains("..."),
        "Should indicate truncation with ellipsis"
    );

    // The tokenized content should be significantly shorter than the input
    assert!(
        tokenized.len() < long_description.len() / 10,
        "Tokenized content should be much shorter than input"
    );
}

#[test]
fn test_tokenizer_sentence_boundary_truncation() {
    // Create a tokenizer with a medium token limit
    let config = TokenizerConfig {
        max_tokens: 150,
        model_id: "bge-small-en-v1.5".to_string(),
        truncation_strategy: "intelligent".to_string(),
        name_weight: 1.5,
        description_weight: 1.0,
        taxonomy_weight: 0.8,
        include_url: true,
        include_email: false,
        language: "en".to_string(),
    };

    // Unwrap the Result to get the tokenizer
    let tokenizer = ServiceTokenizer::with_config(config).expect("Failed to create tokenizer");

    // Create a document with multiple sentences
    let doc = EmbeddingDocument {
        service_id: "service-sentences".to_string(),
        service_name: "Multi-Sentence Service".to_string(),
        service_desc: "This is the first sentence of the description. This is the second sentence with more details. This is the third sentence explaining additional features. This is the fourth sentence covering edge cases. This is the fifth sentence about implementation. This is the final sentence with conclusions.".to_string(),
        taxonomies: vec![
            TaxonomyDocument {
                taxonomy_name: "Category A".to_string(),
                taxonomy_desc: Some("First sentence of taxonomy description. Second sentence with details.".to_string()),
            },
        ],
    };

    // Tokenize the document
    let (tokenized, token_count) = tokenizer.tokenize(&doc).unwrap();

    // Verify token count is within limits
    assert!(
        token_count <= 150,
        "Token count should be within the configured limit"
    );

    // Check that truncation happens at sentence boundaries where possible
    let description_excerpt = tokenized
        .split("Description: ")
        .nth(1)
        .unwrap_or("")
        .split("\n\n")
        .next()
        .unwrap_or("");

    // Count complete sentences in the truncated description
    let sentence_count = description_excerpt.matches(". ").count();

    // We can't predict exactly how many sentences will fit, but we should have at least one complete sentence
    assert!(
        sentence_count >= 1,
        "Should have at least one complete sentence in description"
    );

    // If the description was truncated (not all 6 sentences fit), it should end with an ellipsis
    if sentence_count < 5 {
        // Less than 5 periods means at least one sentence was cut
        assert!(
            description_excerpt.contains("..."),
            "Truncated description should end with ellipsis"
        );
    }
}

#[test]
fn test_tokenizer_url_inclusion() {
    // Test with URL inclusion enabled
    let config = TokenizerConfig {
        max_tokens: 100,
        model_id: "bge-small-en-v1.5".to_string(),
        truncation_strategy: "intelligent".to_string(),
        name_weight: 1.5,
        description_weight: 1.0,
        taxonomy_weight: 0.8,
        include_url: true,
        include_email: false,
        language: "en".to_string(),
    };

    // Unwrap the Result to get the tokenizer
    let tokenizer =
        ServiceTokenizer::with_config(config.clone()).expect("Failed to create tokenizer");

    // Create a document with a URL in the service_id
    let doc = EmbeddingDocument {
        service_id: "https://example.com/service".to_string(),
        service_name: "URL Service".to_string(),
        service_desc: "This service has a URL.".to_string(),
        taxonomies: vec![],
    };

    // Tokenize the document
    let (tokenized, _) = tokenizer.tokenize(&doc).unwrap();

    // Verify URL is included
    assert!(
        tokenized.contains("URL: https://example.com/service"),
        "URL should be included"
    );

    // Test with URL inclusion disabled
    let config = TokenizerConfig {
        include_url: false,
        ..config
    };

    // Unwrap the Result to get the tokenizer
    let tokenizer = ServiceTokenizer::with_config(config).expect("Failed to create tokenizer");

    // Tokenize again
    let (tokenized, _) = tokenizer.tokenize(&doc).unwrap();

    // Verify URL is not included
    assert!(
        !tokenized.contains("URL:"),
        "URL should not be included when disabled"
    );
}

#[test]
fn test_tokenizer_different_strategies() {
    // Create documents to test with
    let doc = EmbeddingDocument {
        service_id: "service-strategies".to_string(),
        service_name: "Strategy Test Service".to_string(),
        service_desc: "This description is lengthy enough to demonstrate different truncation strategies. It contains multiple sentences that will be handled differently depending on the strategy used. We need to ensure that each strategy behaves as expected.".to_string(),
        taxonomies: vec![
            TaxonomyDocument {
                taxonomy_name: "Test Category".to_string(),
                taxonomy_desc: Some("This is a test category description.".to_string()),
            },
        ],
    };

    // Test simple truncation
    let simple_config = TokenizerConfig {
        max_tokens: 50,
        truncation_strategy: "simple".to_string(),
        ..TokenizerConfig::default()
    };

    // Unwrap the Result to get the tokenizer
    let simple_tokenizer =
        ServiceTokenizer::with_config(simple_config).expect("Failed to create simple tokenizer");

    let (simple_result, simple_count) = simple_tokenizer.tokenize(&doc).unwrap();

    assert!(
        simple_count <= 50,
        "Simple truncation should respect token limit"
    );

    // Test start-only truncation
    let start_config = TokenizerConfig {
        max_tokens: 50,
        truncation_strategy: "start_only".to_string(),
        ..TokenizerConfig::default()
    };

    // Unwrap the Result to get the tokenizer
    let start_tokenizer =
        ServiceTokenizer::with_config(start_config).expect("Failed to create start_only tokenizer");

    let (start_result, start_count) = start_tokenizer.tokenize(&doc).unwrap();

    assert!(
        start_count <= 50,
        "Start-only truncation should respect token limit"
    );
    assert!(
        start_result.contains("Service: Strategy Test Service"),
        "Start-only should include the beginning"
    );

    // Test end-only truncation
    let end_config = TokenizerConfig {
        max_tokens: 50,
        truncation_strategy: "end_only".to_string(),
        ..TokenizerConfig::default()
    };

    // Unwrap the Result to get the tokenizer
    let end_tokenizer =
        ServiceTokenizer::with_config(end_config).expect("Failed to create end_only tokenizer");

    let (end_result, end_count) = end_tokenizer.tokenize(&doc).unwrap();

    assert!(
        end_count <= 50,
        "End-only truncation should respect token limit"
    );

    // Compare the strategies to verify they produce different results
    assert_ne!(
        simple_result, end_result,
        "Simple and end-only truncation should produce different results"
    );
    assert_ne!(
        start_result, end_result,
        "Start-only and end-only truncation should produce different results"
    );
}

#[test]
fn test_token_counter() {
    // Create a tokenizer with default config
    let tokenizer = ServiceTokenizer::new();

    // Test token counting for various inputs
    let text1 = "This is a simple text.";
    let text2 = "This text, has; some: punctuation! And various - symbols? To test the counter.";
    let text3 = "https://www.example.com/this/is/a/url/that/should/count/as/more/tokens";
    let text4 = ""; // Empty text

    // Use the private count_tokens method through our tokenize method
    let doc1 = EmbeddingDocument {
        service_id: "service-count-1".to_string(),
        service_name: "Counter Test 1".to_string(),
        service_desc: text1.to_string(),
        taxonomies: vec![],
    };

    let doc2 = EmbeddingDocument {
        service_id: "service-count-2".to_string(),
        service_name: "Counter Test 2".to_string(),
        service_desc: text2.to_string(),
        taxonomies: vec![],
    };

    let doc3 = EmbeddingDocument {
        service_id: "service-count-3".to_string(),
        service_name: "Counter Test 3".to_string(),
        service_desc: text3.to_string(),
        taxonomies: vec![],
    };

    let doc4 = EmbeddingDocument {
        service_id: "service-count-4".to_string(),
        service_name: "Counter Test 4".to_string(),
        service_desc: text4.to_string(),
        taxonomies: vec![],
    };

    // Get token counts through tokenization
    let (_, count1) = tokenizer.tokenize(&doc1).unwrap();
    let (_, count2) = tokenizer.tokenize(&doc2).unwrap();
    let (_, count3) = tokenizer.tokenize(&doc3).unwrap();
    let (_, count4) = tokenizer.tokenize(&doc4).unwrap();

    // Verify token counts are reasonable
    assert!(count1 > 0, "Simple text should have non-zero token count");
    assert!(
        count2 > count1,
        "Text with punctuation should have more tokens"
    );
    assert!(
        count3 > count1,
        "URL should have more tokens than simple text"
    );
    assert!(
        count4 < count1,
        "Empty text should have fewer tokens than simple text"
    );

    // Verify URL counting is working as expected
    let url_overhead = count3 - text3.split_whitespace().count();
    assert!(
        url_overhead > 5,
        "URLs should have significant token overhead"
    );
}
