// src/matching/email.rs

use crate::models::{MatchGroup, MatchingMethod};
use crate::db::PgPool;
use anyhow::{Result, Context};
use std::collections::HashMap;
use tokio_postgres::Row;

/// match_emails performs deterministic matching on email fields.
/// It queries the database for records with non-null email addresses,
/// normalizes them (e.g., lowercases and trims whitespace),
/// and groups record IDs by their normalized email.
/// Only groups with more than one record are considered duplicates.
pub async fn match_emails(pool: &PgPool) -> Result<Vec<MatchGroup>> {
    let mut email_map: HashMap<String, Vec<String>> = HashMap::new();
    
    // Connect to the database
    let client = pool.get().await.context("Failed to get DB connection for email matching")?;
    
    // Query organizations with non-null emails
    let org_rows = client
        .query(
            "SELECT id, email FROM organization WHERE email IS NOT NULL AND email != ''",
            &[]
        )
        .await
        .context("Failed to query organizations with emails")?;
    
    // Add organization emails to the map
    for row in org_rows {
        process_email_row(&mut email_map, &row, "organization");
    }
    
    // Query services with non-null emails
    let service_rows = client
        .query(
            "SELECT id, email FROM service WHERE email IS NOT NULL AND email != ''", 
            &[]
        )
        .await
        .context("Failed to query services with emails")?;
    
    // Add service emails to the map
    for row in service_rows {
        process_email_row(&mut email_map, &row, "service");
    }
    
    // Query contacts with non-null emails
    // let contact_rows = client
    //     .query(
    //         "SELECT id, email FROM contact WHERE email IS NOT NULL AND email != ''", 
    //         &[]
    //     )
    //     .await
    //     .context("Failed to query contacts with emails")?;
    
    // Add contact emails to the map
    // for row in contact_rows {
    //     process_email_row(&mut email_map, &row, "contact");
    // }
    
    // Create match groups from the email map
    let groups = create_match_groups_from_map(email_map);
    
    Ok(groups)
}

/// Helper function to process a row containing an email field
fn process_email_row(email_map: &mut HashMap<String, Vec<String>>, row: &Row, entity_type: &str) {
    let id: String = row.get("id");
    let email: String = row.get("email");
    
    // Normalize the email
    let normalized = normalize_email(&email);
    
    // Skip empty emails after normalization
    if normalized.is_empty() {
        return;
    }
    
    // Store the ID with the normalized email as the key
    email_map
        .entry(normalized)
        .or_default()
        .push(format!("{}:{}", entity_type, id)); // Prefix ID with entity type for clarity
}

/// Normalize an email by:
/// - Converting to lowercase
/// - Trimming whitespace
/// - Removing dots before @ in Gmail addresses
/// - Removing plus addressing (everything between + and @ is removed)
/// - Normalizing common domains (e.g., googlemail.com -> gmail.com)
fn normalize_email(email: &str) -> String {
    // Basic normalization
    let email = email.trim().to_lowercase();
    
    // If it doesn't contain an @ symbol, just return the trimmed lowercase version
    if !email.contains('@') {
        return email;
    }
    
    // Split the email into local part and domain
    let parts: Vec<&str> = email.split('@').collect();
    if parts.len() != 2 {
        return email; // Return original if multiple @ symbols
    }
    
    let (local_part, domain) = (parts[0], parts[1]);
    
    // Normalize the local part based on domain-specific rules
    let normalized_local = if domain == "gmail.com" || domain == "googlemail.com" {
        // For Gmail: remove dots and anything after a plus sign
        let without_dots = local_part.replace(".", "");
        match without_dots.split('+').next() {
            Some(username) => username.to_string(),
            None => local_part.to_string()
        }
    } else {
        // For other domains: just remove anything after a plus sign
        match local_part.split('+').next() {
            Some(username) => username.to_string(),
            None => local_part.to_string()
        }
    };
    
    // Normalize domain
    let normalized_domain = match domain {
        "googlemail.com" => "gmail.com",
        "hotmail.com" | "live.com" | "outlook.com" => "outlook.com",
        "ymail.com" | "rocketmail.com" => "yahoo.com",
        _ => domain,
    };
    
    format!("{}@{}", normalized_local, normalized_domain)
}

/// Convert the email mapping into formal MatchGroup structs
fn create_match_groups_from_map(email_map: HashMap<String, Vec<String>>) -> Vec<MatchGroup> {
    let mut groups = Vec::new();
    
    for (normalized_email, record_ids) in email_map {
        // Only consider groups with more than one record as duplicates
        if record_ids.len() > 1 {
            groups.push(MatchGroup {
                record_ids,
                method: MatchingMethod::Email,
                confidence: 1.0, // Exact email matches have high confidence
                notes: Some(format!("Matched on normalized email: {}", normalized_email)),
                is_reviewed: false, // Not yet reviewed by a human
            });
        }
    }
    
    groups
}