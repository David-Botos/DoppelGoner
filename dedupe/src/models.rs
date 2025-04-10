// src/models.rs

use serde::{Deserialize, Serialize};
// use std::collections::HashMap;

/// Represents a group of similar entities (organizations/services)
/// that have been matched based on one or more criteria.
///
/// Used throughout the pipeline to hold temporary or reviewed matches.
/// Output of: `match_emails`, `match_phones`, `match_urls`, `match_addresses`,
/// `match_semantic`, `match_by_taxonomy_and_region`, `run_llm_review`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchGroup {
    /// Unique identifiers of matched records (e.g., UUIDs from `organization` or `service` tables)
    pub record_ids: Vec<String>,

    /// Strategy or combination of strategies used to match these records
    pub method: MatchingMethod,

    /// Confidence score [0.0 - 1.0], used to prioritize human review or auto-confirmation
    pub confidence: f32,

    /// Optional justification or evidence for the match (e.g., "emails identical")
    pub notes: Option<String>,

    /// Indicates if a human has reviewed this match group
    pub is_reviewed: bool,
}

/// Enum describing the method or strategy used to match records.
/// This is attached to each `MatchGroup` and helps explain to reviewers
/// why a group was formed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MatchingMethod {
    /// Matched by identical or near-identical email
    Email,

    /// Matched by identical or near-identical phone number
    Phone,

    /// Matched by comparing URLs or normalized domains
    Url,

    /// Matched by geocoded or parsed addresses
    Address,

    /// Matched using NLP on names and descriptions
    Semantic,

    /// Matched based on HSIS taxonomy and geographic region
    TaxonomyRegion,

    /// Matched by LLM during Phase 4
    LLMReview,

    /// Confirmed or rejected by a human reviewer
    HumanVerified,

    /// Used when merging two or more MatchGroups that used different strategies
    Merged(Vec<MatchingMethod>),
}

/// Output format from LLM review step or other human-in-the-loop system.
/// This helps capture the intent behind matches (e.g., reason provided)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMReviewResult {
    /// The reviewed group of entity IDs
    pub group: MatchGroup,

    /// LLM justification (optional)
    pub reasoning: Option<String>,

    /// Whether LLM recommends keeping the group
    pub is_valid: bool,
}

/// Struct representing an HSIS Taxonomy assignment.
/// This can be used to populate classifications during semantic processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxonomyLabel {
    /// E.g., "BD-1800.2000-300"
    pub code: String,

    /// E.g., "Gluten Free Food"
    pub label: String,

    /// Optional: hierarchical depth or score
    pub confidence: Option<f32>,
}

/// Represents a simplified organization or service entity
/// used during feature extraction or similarity analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordFingerprint {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub email: Option<String>,
    pub phone: Option<String>,
    pub url: Option<String>,
    pub address: Option<String>,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub taxonomy_labels: Vec<TaxonomyLabel>,
}

/// Stores a sparse similarity matrix between records.
/// You can use this during LLM review or cluster merging to avoid recomputing.
// pub type SimilarityMatrix = HashMap<(String, String), f32>;

/// Represents a feature of an entity (organization, service, location, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityFeature {
    /// The table this feature comes from
    pub table_name: String,
    
    /// The ID of the record in that table
    pub table_id: String,
    
    /// Optional feature type (e.g., "primary", "secondary")
    pub feature_type: Option<String>,
    
    /// Optional importance weight
    pub weight: Option<f32>,
}

/// Represents a complete entity within a cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterEntity {
    /// Generated ID for this entity
    pub id: Option<String>,
    
    /// Whether this is the primary entity in the cluster (usually the organization)
    pub is_primary: bool,
    
    /// All features that make up this entity
    pub features: Vec<EntityFeature>,
}

/// Extended version of MatchGroup that supports hierarchical entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalMatchGroup {
    /// Original flat match group
    pub base_group: MatchGroup,
    
    /// Organized hierarchical entities
    pub entities: Vec<ClusterEntity>,
}