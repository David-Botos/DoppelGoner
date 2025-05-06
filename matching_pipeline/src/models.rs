// src/models.rs

use std::collections::{HashMap, HashSet};

use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

//------------------------------------------------------------------------------
// IDENTIFIER TYPES
//------------------------------------------------------------------------------
// Using newtype pattern for type safety to prevent mixing different ID types

/// Strongly typed identifier for Entity records
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityId(pub String);

/// Strongly typed identifier for Organization records from HSDS
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrganizationId(pub String);

/// Strongly typed identifier for EntityGroup records
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityGroupId(pub String);

/// Strongly typed identifier for GroupCluster records
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GroupClusterId(pub String);

/// Strongly typed identifier for Service records from HSDS
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ServiceId(pub String);

//------------------------------------------------------------------------------
// CORE DOMAIN MODELS
//------------------------------------------------------------------------------

/// Represents an organization entity in the federation
///
/// This is the foundation of the grouping system, representing a single
/// organization from any data source in the federation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier for this entity
    pub id: EntityId,

    /// Reference to the original organization in HSDS
    pub organization_id: OrganizationId,

    /// Human-readable name of the entity (typically from organization.name)
    pub name: Option<String>,

    /// When this entity was first created
    pub created_at: NaiveDateTime,

    /// When this entity was last updated
    pub updated_at: NaiveDateTime,

    /// Identifier for the source system in the federation
    pub source_system: Option<String>,

    /// Original ID from the source system
    pub source_id: Option<String>,
}

/// Links an entity to related records in other HSDS tables
///
/// This enables tracking which services, phones, locations, etc.
/// are associated with each entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityFeature {
    /// Unique identifier for this feature link
    pub id: String,

    /// The entity this feature belongs to
    pub entity_id: EntityId,

    /// Name of the HSDS table this feature comes from (e.g., "service", "phone")
    pub table_name: String,

    /// ID of the specific record in that table
    pub table_id: String,

    /// When this feature link was created
    pub created_at: NaiveDateTime,
}

/// Represents a group of entities that have been matched
///
/// Groups are formed when entities are found to match based on
/// one or more criteria (phone numbers, emails, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityGroup {
    /// Unique identifier for this group
    pub id: EntityGroupId,

    /// Optional descriptive name
    pub name: Option<String>,

    /// The cluster this group belongs to (null until clustering is performed)
    pub group_cluster_id: Option<GroupClusterId>,

    /// When this group was first created
    pub created_at: NaiveDateTime,

    /// When this group was last updated
    pub updated_at: NaiveDateTime,

    /// Aggregate confidence score for the matches in this group (0.0-1.0)
    pub confidence_score: f64,

    /// Number of entities in this group
    pub entity_count: i32,
}

/// Maps which entities belong to which groups
///
/// This is a many-to-many relationship table since entities
/// can belong to multiple groups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupEntity {
    /// Unique identifier for this mapping
    pub id: String,

    /// The group that contains the entity
    pub entity_group_id: EntityGroupId,

    /// The entity that belongs to the group
    pub entity_id: EntityId,

    /// When this mapping was created
    pub created_at: NaiveDateTime,
}

/// Enum for supported matching method types
///
/// Defines the standardized method types used for entity matching
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MatchMethodType {
    /// Matching based on normalized website domains
    Url,

    /// Matching based on normalized phone numbers
    Phone,

    /// Matching based on normalized email addresses
    Email,

    /// Matching based on physical address similarity
    Address,

    /// Matching based on geographic proximity
    Geospatial,
    
    /// Matching based on organization name similarity
    Name,

    /// Custom matcher type (for extensibility)
    Custom(String),
}

impl MatchMethodType {
    /// Converts the enum to a string representation
    pub fn as_str(&self) -> &str {
        match self {
            Self::Url => "url",
            Self::Phone => "phone",
            Self::Email => "email",
            Self::Address => "address",
            Self::Geospatial => "geospatial",
            Self::Name => "name",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Creates the enum from a string representation
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "url" => Self::Url,
            "phone" => Self::Phone,
            "email" => Self::Email,
            "address" => Self::Address,
            "geospatial" => Self::Geospatial,
            "name" => Self::Name,
            _ => Self::Custom(s.to_string()),
        }
    }
}

/// Records details about how entities in a group were matched
///
/// This stores the evidence for why entities were grouped together
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupMethod {
    /// Unique identifier for this matching method record
    pub id: String,

    /// The group these matches apply to
    pub entity_group_id: EntityGroupId,

    /// The type of matching used (e.g., phone, email)
    pub method_type: MatchMethodType,

    /// Human-readable explanation of the match
    pub description: Option<String>,

    /// The actual values that matched (structured by match type)
    pub match_values: MatchValues,

    /// Confidence level for this specific matching method (0.0-1.0)
    pub confidence_score: Option<f32>,

    /// When this matching method was recorded
    pub created_at: NaiveDateTime,
}

/// Represents a consolidated set of overlapping groups
///
/// Clusters are formed when groups are found to share entities,
/// indicating they represent the same real-world organization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupCluster {
    /// Unique identifier for this cluster
    pub id: GroupClusterId,

    /// Optional descriptive name
    pub name: Option<String>,

    /// Details about what this cluster represents
    pub description: Option<String>,

    /// When this cluster was first created
    pub created_at: NaiveDateTime,

    /// When this cluster was last updated
    pub updated_at: NaiveDateTime,

    /// Total unique entities across all groups in this cluster
    pub entity_count: i32,

    /// Number of groups in this cluster
    pub group_count: i32,
}

/// Enum for service match status
///
/// Tracks the lifecycle of a potential service match
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ServiceMatchStatus {
    /// Match has been identified but not reviewed
    Potential,

    /// Match has been reviewed and confirmed as correct
    Confirmed,

    /// Match has been reviewed and rejected as incorrect
    Rejected,
}

impl ServiceMatchStatus {
    /// Converts the enum to a string representation
    pub fn as_str(&self) -> &str {
        match self {
            Self::Potential => "potential",
            Self::Confirmed => "confirmed",
            Self::Rejected => "rejected",
        }
    }

    /// Creates the enum from a string representation
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "confirmed" => Self::Confirmed,
            "rejected" => Self::Rejected,
            _ => Self::Potential,
        }
    }
}

/// Records potential matches between services within a cluster
///
/// Used to identify services that may represent the same real-world service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMatch {
    /// Unique identifier for this service match
    pub id: String,

    /// The cluster containing the potentially matching services
    pub group_cluster_id: GroupClusterId,

    /// First service in the potential match
    pub service_id_1: ServiceId,

    /// Second service in the potential match
    pub service_id_2: ServiceId,

    /// Measure of service similarity (0.0-1.0)
    pub similarity_score: f32,

    /// Explanation of why services matched
    pub match_reasons: Option<String>,

    /// When this potential match was created
    pub created_at: NaiveDateTime,

    /// Current status of the match
    pub status: ServiceMatchStatus,

    /// User who reviewed the match (if applicable)
    pub reviewed_by: Option<String>,

    /// When the match was reviewed (if applicable)
    pub reviewed_at: Option<NaiveDateTime>,
}

//------------------------------------------------------------------------------
// MATCH VALUE TYPES
//------------------------------------------------------------------------------

/// Union type for different kinds of match values
///
/// This is a strongly-typed representation of the JSONB data
/// stored in the group_method.match_values column
/// Union type for different kinds of match values
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "values")]
pub enum MatchValues {
    /// URL matching values
    Url(Vec<UrlMatchValue>),

    /// Phone matching values
    Phone(Vec<PhoneMatchValue>),

    /// Email matching values
    Email(Vec<EmailMatchValue>),

    /// Address matching values
    Address(Vec<AddressMatchValue>),

    /// Geospatial matching values
    Geospatial(Vec<GeospatialMatchValue>),
    
    /// Name matching values
    Name(Vec<NameMatchValue>),

    /// Generic string matching values (for extensibility)
    Generic(Vec<String>),
}

/// Represents a matched URL value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrlMatchValue {
    /// Original URL as found in the data
    pub original: String,

    /// Normalized domain (the basis for matching)
    pub domain: String,

    /// Source entity ID this URL belongs to
    pub entity_id: EntityId,
}

/// Represents a matched phone number
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhoneMatchValue {
    /// Original phone number as found in the data
    pub original: String,

    /// Normalized phone number (the basis for matching)
    pub normalized: String,

    /// Optional extension (not used for primary matching)
    pub extension: Option<String>,

    /// Source entity ID this phone belongs to
    pub entity_id: EntityId,
}

/// Represents a matched email address
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailMatchValue {
    /// Original email as found in the data
    pub original: String,

    /// Normalized email (the basis for matching)
    pub normalized: String,

    /// Email domain (may be used for secondary matching)
    pub domain: String,

    /// Source entity ID this email belongs to
    pub entity_id: EntityId,
}

/// Represents a matched physical address
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddressMatchValue {
    /// Original address as found in the data
    pub original: String,

    /// Normalized address (the basis for matching)
    pub normalized: String,

    /// Match score for fuzzy matching (0.0-1.0)
    pub match_score: Option<f32>,

    /// Source entity ID this address belongs to
    pub entity_id: EntityId,
}

/// Represents a matched geospatial location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeospatialMatchValue {
    /// Latitude coordinate
    pub latitude: f64,

    /// Longitude coordinate
    pub longitude: f64,

    /// Distance in meters to the centroid of the match
    pub distance_to_center: Option<f64>,

    /// Source entity ID this location belongs to
    pub entity_id: EntityId,
}

/// Represents a matched organization name
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NameMatchValue {
    /// Original name as found in the data
    pub original: String,
    
    /// Normalized name (the basis for matching)
    pub normalized: String,
    
    /// Match score (fuzzy or semantic similarity)
    pub similarity_score: f32,
    
    /// Type of matching used (fuzzy, semantic, or combined)
    pub match_type: String,
    
    /// Source entity ID this name belongs to
    pub entity_id: EntityId,
    
    /// The entity ID that this entity matched with
    pub matched_entity_id: Option<EntityId>,
}

/// Struct to hold location information for unprocessed entities
pub struct LocationResults {
    pub locations: Vec<(EntityId, f64, f64)>,
    pub has_postgis: bool,
}

/// Struct to hold information about existing groups
pub struct GroupResults {
    /// Map of group_id to (method_id, centroid_lat, centroid_lon, version)
    pub groups: HashMap<String, (String, f64, f64, i32)>,
    /// Map of group_id to vec of (entity_id, lat, lon)
    pub group_entities: HashMap<String, Vec<(EntityId, f64, f64)>>,
}

/// Represents a centroid of a spatial cluster
pub struct Centroid {
    pub latitude: f64,
    pub longitude: f64,
}

/// Represents a spatial cluster found through PostgreSQL
pub struct SpatialCluster {
    pub cluster_id: i32,
    pub entity_ids: Vec<String>,
    pub centroid: Centroid,
    pub entity_count: i64,
}

/// Represents the result of a matching operation
pub struct MatchResult {
    pub entities_added: usize,
    pub entities_skipped: usize,
    pub groups_created: usize,
    pub processed_entities: HashSet<EntityId>,
}
