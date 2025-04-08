// src/matching/phone.rs

use crate::models::{MatchGroup, MatchingMethod};
use crate::db::PgPool;
use anyhow::{Result, Context};
use std::collections::HashMap;

/// match_phones performs deterministic matching on phone numbers.
/// It queries the database for records with non-null phone numbers,
/// normalizes them to a standard format (E.164 where possible),
/// and groups records sharing the same number.
pub async fn match_phones(pool: &PgPool) -> Result<Vec<MatchGroup>> {
    let mut phone_map: HashMap<String, Vec<String>> = HashMap::new();
    
    // Connect to the database
    let client = pool.get().await.context("Failed to get DB connection for phone matching")?;
    
    // Query phones table for all phone numbers
    let phone_rows = client
        .query(
            "SELECT id, organization_id, service_id, location_id, contact_id, service_at_location_id, number 
             FROM phone 
             WHERE number IS NOT NULL AND number != ''",
            &[]
        )
        .await
        .context("Failed to query phone records")?;
    
    // Process each phone row
    for row in phone_rows {
        let phone_id: String = row.get("id");
        let phone_number: String = row.get("number");
        
        // Check all possible foreign keys to determine the parent entity
        if let Ok(org_id) = row.try_get::<_, String>("organization_id") {
            if !org_id.is_empty() {
                add_phone_to_map(&mut phone_map, &phone_number, format!("organization:{}", org_id));
            }
        }
        
        if let Ok(service_id) = row.try_get::<_, String>("service_id") {
            if !service_id.is_empty() {
                add_phone_to_map(&mut phone_map, &phone_number, format!("service:{}", service_id));
            }
        }
        
        if let Ok(location_id) = row.try_get::<_, String>("location_id") {
            if !location_id.is_empty() {
                add_phone_to_map(&mut phone_map, &phone_number, format!("location:{}", location_id));
            }
        }
        
        if let Ok(contact_id) = row.try_get::<_, String>("contact_id") {
            if !contact_id.is_empty() {
                add_phone_to_map(&mut phone_map, &phone_number, format!("contact:{}", contact_id));
            }
        }
        
        if let Ok(sal_id) = row.try_get::<_, String>("service_at_location_id") {
            if !sal_id.is_empty() {
                add_phone_to_map(&mut phone_map, &phone_number, format!("service_at_location:{}", sal_id));
            }
        }
    }
    
    // Create match groups from the phone map
    let groups = create_match_groups_from_map(phone_map);
    
    Ok(groups)
}

/// Helper function to add a phone to the map
fn add_phone_to_map(phone_map: &mut HashMap<String, Vec<String>>, phone: &str, entity_id: String) {
    // Normalize the phone number
    let normalized = normalize_phone(phone);
    
    // Skip empty phone numbers after normalization
    if normalized.is_empty() {
        return;
    }
    
    // Store the ID with the normalized phone as the key
    phone_map
        .entry(normalized)
        .or_default()
        .push(entity_id);
}

/// Normalize a phone number by:
/// - Removing all non-numeric characters
/// - Handling country codes
/// - Standardizing format
fn normalize_phone(phone: &str) -> String {
    // 1. Remove all non-numeric characters (spaces, dashes, parentheses, etc.)
    let digits_only: String = phone
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect();
    
    // Skip if we don't have enough digits
    if digits_only.len() < 7 {
        return String::new();
    }
    
    // 2. Handle different phone number formats
    // For US numbers:
    // - If 10 digits, assume US number
    // - If 11 digits and starts with 1, assume US number with country code
    
    // This is a simplified approach - for production, use a proper phone library
    // like phonenumber or similar
    
    if digits_only.len() == 10 {
        // Standard US number, keep as is
        return digits_only;
    } else if digits_only.len() == 11 && digits_only.starts_with('1') {
        // US number with country code, strip the leading 1
        return digits_only[1..].to_string();
    }
    
    // Return whatever we have for other cases
    // In a production system, you'd use a library to properly normalize international numbers
    digits_only
}

/// Convert the phone mapping into formal MatchGroup structs
fn create_match_groups_from_map(phone_map: HashMap<String, Vec<String>>) -> Vec<MatchGroup> {
    let mut groups = Vec::new();
    
    for (normalized_phone, record_ids) in phone_map {
        // Only consider groups with more than one record as duplicates
        if record_ids.len() > 1 {
            groups.push(MatchGroup {
                record_ids,
                method: MatchingMethod::Phone,
                confidence: 1.0, // Exact phone matches have high confidence
                notes: Some(format!("Matched on normalized phone: {}", normalized_phone)),
                is_reviewed: false, // Not yet reviewed by a human
            });
        }
    }
    
    groups
}