// src/matching/address.rs

use crate::models::{MatchGroup, MatchingMethod};
use crate::db::PgPool;
use anyhow::{Result, Context};
use std::collections::{HashMap, HashSet};

/// Struct to represent a location with coordinates
#[derive(Debug, Clone)]
struct Location {
    id: String,
    entity_type: String,
    organization_id: Option<String>,
    address: String,
    latitude: Option<f64>,
    longitude: Option<f64>,
}

/// match_addresses performs deterministic matching on addresses.
/// It fetches location records, uses address normalization techniques,
/// and groups addresses based on either exact matches or geospatial proximity.
pub async fn match_addresses(pool: &PgPool) -> Result<Vec<MatchGroup>> {
    let mut groups = Vec::new();
    
    // Connect to the database
    let client = pool.get().await.context("Failed to get DB connection for address matching")?;
    
    // 1. Collect all location data from both organizations and locations tables
    let mut locations = Vec::new();
    
    // Query locations with coordinates
    let location_rows = client
        .query(
            "SELECT id, organization_id, latitude, longitude FROM location 
             WHERE (latitude IS NOT NULL AND longitude IS NOT NULL)",
            &[]
        )
        .await
        .context("Failed to query locations with coordinates")?;
    
    // Process location coordinates
    for row in location_rows {
        let id: String = row.get("id");
        let org_id: String = row.get("organization_id");
        let latitude: Option<f64> = row.try_get("latitude").ok().map(|v: f64| v.into());
        let longitude: Option<f64> = row.try_get("longitude").ok().map(|v: f64| v.into());
        
        // Skip if we don't have both coordinates
        if latitude.is_none() || longitude.is_none() {
            continue;
        }
        
        locations.push(Location {
            id: format!("location:{}", id),
            entity_type: "location".to_string(),
            organization_id: Some(org_id),
            address: String::new(), // We'll get the address separately
            latitude,
            longitude,
        });
    }
    
    // Query addresses linked to locations
    let address_rows = client
        .query(
            "SELECT id, location_id, address_1, address_2, city, state_province, postal_code, country 
             FROM address 
             WHERE location_id IS NOT NULL",
            &[]
        )
        .await
        .context("Failed to query address table")?;
    
    // Process address rows and link to locations
    for row in address_rows {
        let id: String = row.get("id");
        let location_id: String = row.get("location_id");
        let address_1: String = row.get("address_1");
        let address_2: Option<String> = row.try_get("address_2").ok();
        let city: String = row.get("city");
        let state_province: String = row.get("state_province");
        let postal_code: String = row.get("postal_code");
        let country: String = row.get("country");
        
        // Assemble full address
        let full_address = format!(
            "{}{}, {}, {} {}, {}",
            address_1,
            address_2.map_or("".to_string(), |a| format!(", {}", a)),
            city,
            state_province,
            postal_code,
            country
        );
        
        // Either update an existing location or add a new one
        let existing_location = locations.iter_mut().find(|loc| loc.id == format!("location:{}", location_id));
        
        if let Some(location) = existing_location {
            // Update existing location with address
            location.address = full_address;
        } else {
            // Add a new location with just the address
            locations.push(Location {
                id: format!("location:{}", location_id),
                entity_type: "location".to_string(),
                organization_id: None, // We don't have this info here
                address: full_address,
                latitude: None,
                longitude: None,
            });
        }
    }
    
    // 2. Match by exact normalized address
    let exact_matches = match_by_normalized_address(&locations);
    groups.extend(exact_matches);
    
    // 3. Match by geospatial proximity (only for records with coordinates)
    let geo_matches = match_by_geospatial_proximity(&locations);
    groups.extend(geo_matches);
    
    Ok(groups)
}

// These functions are replaced by inline processing in the match_addresses function
// since we're now handling the actual database schema directly

/// Match records based on normalized address strings
fn match_by_normalized_address(locations: &[Location]) -> Vec<MatchGroup> {
    let mut address_map: HashMap<String, Vec<String>> = HashMap::new();
    
    // Group records by normalized addresses
    for location in locations {
        if !location.address.is_empty() {
            let normalized = normalize_address(&location.address);
            
            // Skip empty addresses after normalization
            if !normalized.is_empty() {
                address_map
                    .entry(normalized)
                    .or_default()
                    .push(location.id.clone());
            }
        }
    }
    
    // Create match groups for identical normalized addresses
    let mut groups = Vec::new();
    for (normalized_address, record_ids) in address_map {
        if record_ids.len() > 1 {
            groups.push(MatchGroup {
                record_ids,
                method: MatchingMethod::Address,
                confidence: 0.95, // Slightly less than 1.0 due to address normalization variations
                notes: Some(format!("Matched on normalized address: {}", normalized_address)),
                is_reviewed: false,
            });
        }
    }
    
    groups
}

/// Match records based on geospatial proximity
fn match_by_geospatial_proximity(locations: &[Location]) -> Vec<MatchGroup> {
    let mut groups = Vec::new();
    let mut processed_ids = HashSet::new();
    
    // Only consider locations with valid coordinates
    let valid_locations: Vec<&Location> = locations
        .iter()
        .filter(|loc| loc.latitude.is_some() && loc.longitude.is_some())
        .collect();
    
    // Threshold distance in meters
    const PROXIMITY_THRESHOLD: f64 = 100.0; // 100 meters
    
    // For each location, find nearby locations
    for (i, loc1) in valid_locations.iter().enumerate() {
        // Skip if already in a group
        if processed_ids.contains(&loc1.id) {
            continue;
        }
        
        let mut nearby_ids = vec![loc1.id.clone()];
        processed_ids.insert(loc1.id.clone());
        
        // Compare with all other locations
        for loc2 in valid_locations.iter().skip(i + 1) {
            // Skip if already in a group
            if processed_ids.contains(&loc2.id) {
                continue;
            }
            
            // Calculate distance between the two points
            let distance = calculate_haversine_distance(
                loc1.latitude.unwrap(), 
                loc1.longitude.unwrap(),
                loc2.latitude.unwrap(), 
                loc2.longitude.unwrap()
            );
            
            // If within threshold, add to nearby locations
            if distance <= PROXIMITY_THRESHOLD {
                nearby_ids.push(loc2.id.clone());
                processed_ids.insert(loc2.id.clone());
            }
        }
        
        // Create a match group if we found nearby locations
        if nearby_ids.len() > 1 {
            // Calculate confidence based on proximity
            let confidence = 0.9; // Base confidence for geospatial matches
            
            groups.push(MatchGroup {
                record_ids: nearby_ids,
                method: MatchingMethod::Address,
                confidence,
                notes: Some(format!("Matched on geospatial proximity within {} meters", PROXIMITY_THRESHOLD)),
                is_reviewed: false,
            });
        }
    }
    
    groups
}

/// Normalize an address by:
/// - Converting to lowercase
/// - Removing punctuation
/// - Standardizing common abbreviations
/// - Removing apartment/suite numbers
/// 
/// This is a simple implementation. For production, use a library like libpostal.
fn normalize_address(address: &str) -> String {
    // Convert to lowercase
    let lower = address.to_lowercase();
    
    // Remove punctuation and extra whitespace
    let mut normalized = lower
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>();
    
    normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
    
    // Replace common abbreviations (this is a very simplified approach)
    normalized = normalized
        .replace(" st ", " street ")
        .replace(" rd ", " road ")
        .replace(" ave ", " avenue ")
        .replace(" blvd ", " boulevard ")
        .replace(" dr ", " drive ")
        .replace(" ln ", " lane ")
        .replace(" apt ", " ")
        .replace(" suite ", " ")
        .replace(" unit ", " ")
        .replace(" #", " ");
    
    // Remove trailing commas and normalize state codes
    normalized = normalized
        .trim_end_matches(',')
        .to_string();
    
    normalized
}

/// Calculate the Haversine distance between two points in meters
fn calculate_haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS: f64 = 6371000.0; // Earth radius in meters
    
    // Convert degrees to radians
    let lat1_rad = lat1.to_radians();
    let lon1_rad = lon1.to_radians();
    let lat2_rad = lat2.to_radians();
    let lon2_rad = lon2.to_radians();
    
    // Calculate differences
    let dlat = lat2_rad - lat1_rad;
    let dlon = lon2_rad - lon1_rad;
    
    // Haversine formula
    let a = (dlat / 2.0).sin().powi(2) + 
            lat1_rad.cos() * lat2_rad.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    
    EARTH_RADIUS * c // Distance in meters
}