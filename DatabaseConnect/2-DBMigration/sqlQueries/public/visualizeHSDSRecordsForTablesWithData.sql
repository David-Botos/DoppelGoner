-- Script to visualize random records from each table with simple JSON format
-- Each table will have its own section with a clear header

-- ADDRESS TABLE
SELECT '==== ADDRESS TABLE ====' AS table_header;
SELECT 
    jsonb_build_object(
        'id', id,
        'location_id', location_id,
        'attention', attention,
        'address_1', address_1,
        'address_2', address_2,
        'city', city,
        'region', region,
        'state_province', state_province,
        'postal_code', postal_code,
        'country', country
    ) AS record
FROM 
    address
TABLESAMPLE SYSTEM (50)
LIMIT 5;

-- ORGANIZATION TABLE
SELECT '==== ORGANIZATION TABLE ====' AS table_header;
SELECT 
    jsonb_build_object(
        'id', id,
        'name', name,
        'alternate_name', alternate_name,
        'description', description,
        'email', email,
        'url', url,
        'tax_status', tax_status,
        'tax_id', tax_id,
        'legal_status', legal_status
    ) AS record
FROM 
    organization
TABLESAMPLE SYSTEM (50)
LIMIT 5;

-- LOCATION TABLE
SELECT '==== LOCATION TABLE ====' AS table_header;
SELECT 
    jsonb_build_object(
        'id', id,
        'organization_id', organization_id,
        'name', name,
        'alternate_name', alternate_name,
        'description', description,
        'latitude', latitude,
        'longitude', longitude,
        'location_type', location_type
    ) AS record
FROM 
    location
TABLESAMPLE SYSTEM (50)
LIMIT 5;

-- SERVICE TABLE
SELECT '==== SERVICE TABLE ====' AS table_header;
SELECT 
    jsonb_build_object(
        'id', id,
        'organization_id', organization_id,
        'name', name,
        'alternate_name', alternate_name,
        'description', description,
        'url', url,
        'email', email,
        'status', status
    ) AS record
FROM 
    service
TABLESAMPLE SYSTEM (50)
LIMIT 5;

-- PHONE TABLE
SELECT '==== PHONE TABLE ====' AS table_header;
SELECT 
    jsonb_build_object(
        'id', id,
        'location_id', location_id,
        'service_id', service_id,
        'organization_id', organization_id,
        'service_at_location_id', service_at_location_id,
        'number', number,
        'extension', extension,
        'type', type,
        'language', language
    ) AS record
FROM 
    phone
TABLESAMPLE SYSTEM (50)
LIMIT 5;

-- SERVICE_AT_LOCATION TABLE
SELECT '==== SERVICE_AT_LOCATION TABLE ====' AS table_header;
SELECT 
    jsonb_build_object(
        'id', id,
        'service_id', service_id,
        'location_id', location_id,
        'description', description
    ) AS record
FROM 
    service_at_location
TABLESAMPLE SYSTEM (50)
LIMIT 5;

-- MATCH_CLUSTERS TABLE
SELECT '==== MATCH_CLUSTERS TABLE ====' AS table_header;
SELECT 
    jsonb_build_object(
        'id', id,
        'confidence', confidence,
        'notes', notes,
        'reasoning', reasoning,
        'is_reviewed', is_reviewed,
        'review_result', review_result,
        'reviewed_by', reviewed_by,
        'reviewed_at', reviewed_at
    ) AS record
FROM 
    match_clusters
TABLESAMPLE SYSTEM (50)
LIMIT 5;

-- CLUSTER_ENTITIES TABLE
SELECT '==== CLUSTER_ENTITIES TABLE ====' AS table_header;
SELECT 
    jsonb_build_object(
        'id', id,
        'cluster_id', cluster_id,
        'entity_type', entity_type,
        'entity_id', entity_id,
        'created_at', created_at
    ) AS record
FROM 
    cluster_entities
TABLESAMPLE SYSTEM (50)
LIMIT 5;

-- MATCHING_METHODS TABLE
SELECT '==== MATCHING_METHODS TABLE ====' AS table_header;
SELECT 
    jsonb_build_object(
        'id', id,
        'cluster_id', cluster_id,
        'method_name', method_name,
        'confidence', confidence,
        'details', details
    ) AS record
FROM 
    matching_methods
TABLESAMPLE SYSTEM (50)
LIMIT 5;