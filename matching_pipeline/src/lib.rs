// src/lib.rs
pub mod consolidate_clusters;
pub mod db;
pub mod entity_organizations;
pub mod matching;
pub mod models;
pub mod reinforcement;
pub mod results;
pub mod service_matching;

// Re-export common types for easier access
pub use models::{
    EntityId, EntityGroupId, GroupClusterId, MatchMethodType,
    Entity, EntityGroup, GroupCluster, ServiceMatchStatus,
};

// Re-export important functionality
pub use reinforcement::MatchingOrchestrator;
pub use db::PgPool;