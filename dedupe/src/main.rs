// src/main.rs

mod db;
mod models;
mod matching;
mod results;
mod entity_organization;

use crate::db::PgPool;
use crate::models::{LLMReviewResult, MatchGroup, HierarchicalMatchGroup};
use crate::matching::{
    address::match_addresses,
    email::match_emails,
    phone::match_phones,
    url::match_urls,
};
// use crate::matching::semantic::{assign_taxonomies, match_by_taxonomy_and_region, match_semantic};
use crate::results::{merge_match_groups, save_clusters};
use crate::entity_organization::organize_entities;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load environment variables
    dotenv::dotenv().ok();
    
    // Connect to database
    let pool: PgPool = db::connect()?;
    println!("📊 Connected to database successfully");
    println!("📋 Database schema: HSDS (Human Services Data Specification)");
    
    // === Phase 1: Deterministic Matching ===
    println!("🔄 Starting Phase 1: Deterministic Matching");
    
    let mut groups: Vec<MatchGroup> = Vec::new();
    
    // Email matching
    println!("📧 Running email matching across organization, service, and contact tables...");
    let email_matches: Vec<MatchGroup> = match_emails(&pool).await?;
    println!("   Found {} email matches", email_matches.len());
    groups.extend(email_matches);
    
    // Phone matching
    println!("📞 Running phone matching from phone table with links to all entities...");
    let phone_matches: Vec<MatchGroup> = match_phones(&pool).await?;
    println!("   Found {} phone matches", phone_matches.len());
    groups.extend(phone_matches);
    
    // URL matching
    println!("🔗 Running URL matching from organization and service tables...");
    let url_matches: Vec<MatchGroup> = match_urls(&pool).await?;
    println!("   Found {} URL matches", url_matches.len());
    groups.extend(url_matches);
    
    // Address matching
    println!("🏢 Running address matching using location coordinates and address tables...");
    let address_matches: Vec<MatchGroup> = match_addresses(&pool).await?;
    println!("   Found {} address matches", address_matches.len());
    groups.extend(address_matches);
    
    // === Merge Overlapping Groups ===
    println!("🔀 Merging overlapping match groups...");
    let unified_groups: Vec<MatchGroup> = merge_match_groups(groups);
    println!("   Merged into {} unified groups", unified_groups.len());
    
    // === Organize Entities into Hierarchical Structure ===
    println!("🏗️ Organizing entities into hierarchical structure...");
    let hierarchical_groups: Vec<HierarchicalMatchGroup> = organize_entities(&pool, &unified_groups).await?;
    println!("   Organized {} groups with hierarchical structure", hierarchical_groups.len());
    
    // === Simplified Implementation: Skip Phase 2, 3, 4 for now ===
    // Comment out the more sophisticated matching strategies to focus on deterministic methods
    /*
    // === Phase 2: Semantic Matching (Name + Description) ===
    println!("🔄 Starting Phase 2: Semantic Matching");
    let semantic_groups = match_semantic(&pool, &unified_groups).await?;
    println!("   Found {} semantic matches", semantic_groups.len());
    unified_groups = merge_match_groups([unified_groups, semantic_groups].concat());
    println!("   Merged into {} unified groups", unified_groups.len());
    
    // === Phase 3: Taxonomy-Based Regional Matching ===
    println!("🔄 Starting Phase 3: Taxonomy-Based Regional Matching");
    assign_taxonomies(&pool).await?;
    println!("   Assigned taxonomy codes to records");
    let taxonomy_groups = match_by_taxonomy_and_region(&pool).await?;
    println!("   Found {} taxonomy-region matches", taxonomy_groups.len());
    unified_groups = merge_match_groups([unified_groups, taxonomy_groups].concat());
    println!("   Merged into {} unified groups", unified_groups.len());
    
    // === Phase 4: LLM Review for Ambiguous Cases ===
    println!("🔄 Starting Phase 4: LLM Review");
    let reviewed_groups = run_llm_review(&pool, &unified_groups).await?;
    println!("   Completed LLM review of {} groups", reviewed_groups.len());
    */
    
    // Create a simple version of reviewed_groups without LLM processing
    let reviewed_groups: Vec<LLMReviewResult> = unified_groups
        .iter()
        .map(|group| LLMReviewResult {
            group: group.clone(),
            reasoning: Some("Deterministic matching only (Phase 1)".to_string()),
            is_valid: true, // All deterministic matches are considered valid
        })
        .collect();
    
    // === Persist Results for Human Review ===
    println!("💾 Saving deterministic match clusters to database...");
    save_clusters(&pool, &reviewed_groups, &hierarchical_groups).await?;
    
    println!("✅ Phase 1 deduplication completed successfully!");
    println!("   Total groups identified: {}", reviewed_groups.len());
    
    Ok(())
}