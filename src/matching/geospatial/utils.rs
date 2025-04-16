// src/matching/geospatial/utils.rs
//
// Utility functions for geospatial calculations

use crate::models::EntityId;

/// Calculate the Haversine distance between two points in meters
/// Used as fallback when PostGIS is not available
pub fn calculate_haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
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
    let a =
        (dlat / 2.0).sin().powi(2) + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS * c // Distance in meters
}

/// Calculate the centroid (average position) of multiple points
pub fn calculate_centroid(points: &[(EntityId, f64, f64)]) -> (f64, f64) {
    if points.is_empty() {
        return (0.0, 0.0);
    }

    let mut lat_sum = 0.0;
    let mut lon_sum = 0.0;

    for (_, lat, lon) in points {
        lat_sum += lat;
        lon_sum += lon;
    }

    // Round to 10 decimal places to ensure consistent values
    let lat_avg = (lat_sum / points.len() as f64 * 10_000_000_000.0).round() / 10_000_000_000.0;
    let lon_avg = (lon_sum / points.len() as f64 * 10_000_000_000.0).round() / 10_000_000_000.0;

    (lat_avg, lon_avg)
}

/// Calculate the centroid from points with distance included
pub fn calculate_centroid_from_full(points: &[(EntityId, f64, f64, f64)]) -> (f64, f64) {
    if points.is_empty() {
        return (0.0, 0.0);
    }

    let mut lat_sum = 0.0;
    let mut lon_sum = 0.0;

    for (_, lat, lon, _) in points {
        lat_sum += lat;
        lon_sum += lon;
    }

    // Round to 10 decimal places to ensure consistent values
    let lat_avg = (lat_sum / points.len() as f64 * 10_000_000_000.0).round() / 10_000_000_000.0;
    let lon_avg = (lon_sum / points.len() as f64 * 10_000_000_000.0).round() / 10_000_000_000.0;

    (lat_avg, lon_avg)
}

/// Calculate distance from a point to the centroid
pub fn distance_to_centroid(lat: f64, lon: f64, centroid_lat: f64, centroid_lon: f64) -> f64 {
    calculate_haversine_distance(lat, lon, centroid_lat, centroid_lon)
}
