// src/matching/url.rs

use crate::models::{MatchGroup, MatchingMethod};
use crate::db::PgPool;
use anyhow::{Result, Context};
use std::collections::HashMap;
use tokio_postgres::Row;
use url::Url;

/// match_urls performs deterministic matching on URL fields.
/// It queries records with non-null URLs, normalizes them by extracting the domain,
/// and groups record IDs by the normalized domain.
pub async fn match_urls(pool: &PgPool) -> Result<Vec<MatchGroup>> {
    let mut url_map: HashMap<String, Vec<String>> = HashMap::new();
    
    // Connect to the database
    let client = pool.get().await.context("Failed to get DB connection for URL matching")?;
    
    // Query organizations with non-null URLs
    let org_rows = client
        .query(
            "SELECT id, url FROM organization WHERE url IS NOT NULL AND url != ''",
            &[]
        )
        .await
        .context("Failed to query organizations with URLs")?;
    
    // Add organization URLs to the map
    for row in org_rows {
        process_url_row(&mut url_map, &row, "organization")?;
    }
    
    // Query services with non-null URLs
    let service_rows = client
        .query(
            "SELECT id, url FROM service WHERE url IS NOT NULL AND url != ''", 
            &[]
        )
        .await
        .context("Failed to query services with URLs")?;
    
    // Add service URLs to the map
    for row in service_rows {
        process_url_row(&mut url_map, &row, "service")?;
    }
    
    // Create match groups from the URL map
    let groups = create_match_groups_from_map(url_map);
    
    Ok(groups)
}

/// Helper function to process a row containing a URL field
fn process_url_row(url_map: &mut HashMap<String, Vec<String>>, row: &Row, entity_type: &str) -> Result<()> {
    let id: String = row.get("id");
    let url_str: String = row.get("url");
    
    // Normalize the URL
    if let Some(normalized) = normalize_url(&url_str) {
        // Store the ID with the normalized domain as the key
        url_map
            .entry(normalized)
            .or_default()
            .push(format!("{}:{}", entity_type, id)); // Prefix ID with entity type
    }
    
    Ok(())
}

/// Normalize a URL by:
/// - Extracting the domain name
/// - Removing "www." prefix
/// - Converting to lowercase
fn normalize_url(url_str: &str) -> Option<String> {
    // Ensure URL has a scheme
    let url_with_scheme = if !url_str.starts_with("http://") && !url_str.starts_with("https://") {
        format!("https://{}", url_str)
    } else {
        url_str.to_string()
    };
    
    // Parse the URL
    match Url::parse(&url_with_scheme) {
        Ok(url) => {
            // Extract the host/domain
            if let Some(host) = url.host_str() {
                // Normalize domain: lowercase and remove www prefix
                let normalized = host.to_lowercase()
                    .trim_start_matches("www.")
                    .to_string();
                
                // Skip empty domains after normalization
                if !normalized.is_empty() {
                    Some(normalized)
                } else {
                    None
                }
            } else {
                None
            }
        },
        Err(_) => {
            // If parsing fails, try a basic domain extraction
            basic_domain_extraction(url_str)
        }
    }
}

/// Fallback domain extraction for URLs that the URL crate can't parse
fn basic_domain_extraction(url_str: &str) -> Option<String> {
    let lower = url_str.to_lowercase();
    
    // Strip common schemes
    let without_scheme = lower
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_start_matches("ftp://");
    
    // Extract everything before first slash or querystring
    let domain_part = without_scheme
        .split('/')
        .next()?
        .split('?')
        .next()?;
    
    // Remove www prefix
    let normalized = domain_part.trim_start_matches("www.").to_string();
    
    if !normalized.is_empty() && normalized.contains('.') {
        Some(normalized)
    } else {
        None
    }
}

/// Convert the URL mapping into formal MatchGroup structs
fn create_match_groups_from_map(url_map: HashMap<String, Vec<String>>) -> Vec<MatchGroup> {
    let mut groups = Vec::new();
    
    for (normalized_domain, record_ids) in url_map {
        // Only consider groups with more than one record as duplicates
        if record_ids.len() > 1 {
            groups.push(MatchGroup {
                record_ids,
                method: MatchingMethod::Url,
                confidence: 1.0, // Exact domain matches have high confidence
                notes: Some(format!("Matched on normalized domain: {}", normalized_domain)),
                is_reviewed: false, // Not yet reviewed by a human
            });
        }
    }
    
    groups
}