// src/matching/geospatial/config.rs
//
// Contains configuration constants for the geospatial matching module

/// Distance thresholds and confidence levels for geospatial matching
/// Format: (distance_in_meters, confidence_score)
pub const THRESHOLDS: [(f64, f64); 3] = [
    (100.0, 0.95),  // Very close: within 100m, high confidence
    (500.0, 0.85),  // Close: within 500m, medium-high confidence
    (2000.0, 0.75), // Nearby: within 2km, medium confidence
];
