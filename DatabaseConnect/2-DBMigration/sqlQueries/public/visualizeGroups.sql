-- Enhanced Entity Cluster Visualization with more diverse samples and human-readable data
-- Modified to combine all results into a single query for better compatibility with the PostgreSQL client

-- First, create temporary tables to store our intermediate results
-- This allows us to run multiple queries but return a single result set

-- Create a temporary table for the diverse clusters we want to analyze
CREATE TEMP TABLE temp_diverse_clusters AS
(
    (SELECT id FROM match_clusters WHERE confidence = 1 ORDER BY random() LIMIT 1)
    UNION ALL
    (SELECT id FROM match_clusters WHERE confidence < 1 AND confidence >= 0.7 ORDER BY random() LIMIT 1)
    UNION ALL
    (SELECT id FROM match_clusters ORDER BY created_at ASC LIMIT 1)
);

-- Create a temporary table for cluster information
CREATE TEMP TABLE temp_cluster_info AS
SELECT 
    mc.id AS cluster_id,
    mc.confidence AS confidence_score,
    mc.notes AS matching_evidence,
    mc.reasoning,
    mc.is_reviewed,
    mc.created_at,
    STRING_AGG(DISTINCT mm.method_name, ', ') AS matching_methods
FROM match_clusters mc
JOIN temp_diverse_clusters dc ON mc.id = dc.id
LEFT JOIN matching_methods mm ON mc.id = mm.cluster_id
GROUP BY mc.id, mc.confidence, mc.notes, mc.reasoning, mc.is_reviewed, mc.created_at;

-- Create a temporary table for entity details
CREATE TEMP TABLE temp_entity_details AS
SELECT
    mc.id AS cluster_id,
    ce.entity_type,
    ce.entity_id,
    CASE 
        WHEN ce.entity_type = 'organization' THEN o.name
        WHEN ce.entity_type = 'service' THEN s.name
        WHEN ce.entity_type = 'location' THEN l.name
        WHEN ce.entity_type = 'service_at_location' THEN CONCAT(s2.name, ' at ', l2.name)
        ELSE 'Unknown'
    END AS name,
    CASE 
        WHEN ce.entity_type = 'organization' THEN COALESCE(o.description, 'No description')
        WHEN ce.entity_type = 'service' THEN COALESCE(s.description, 'No description')
        WHEN ce.entity_type = 'location' THEN COALESCE(l.description, 'No description')
        WHEN ce.entity_type = 'service_at_location' THEN COALESCE(sal.description, 'No description')
        ELSE 'No description'
    END AS description,
    CASE 
        WHEN ce.entity_type = 'organization' THEN o.email
        WHEN ce.entity_type = 'service' THEN s.email
        ELSE NULL
    END AS email,
    CASE 
        WHEN ce.entity_type = 'organization' THEN o.url
        WHEN ce.entity_type = 'service' THEN s.url
        ELSE NULL
    END AS url
FROM temp_diverse_clusters dc
JOIN match_clusters mc ON dc.id = mc.id
JOIN cluster_entities ce ON mc.id = ce.cluster_id
LEFT JOIN organization o ON ce.entity_type = 'organization' AND ce.entity_id = o.id
LEFT JOIN service s ON ce.entity_type = 'service' AND ce.entity_id = s.id
LEFT JOIN location l ON ce.entity_type = 'location' AND ce.entity_id = l.id
LEFT JOIN service_at_location sal ON ce.entity_type = 'service_at_location' AND ce.entity_id = sal.id
LEFT JOIN service s2 ON sal.service_id = s2.id
LEFT JOIN location l2 ON sal.location_id = l2.id;

-- Create a temporary table for contact details
CREATE TEMP TABLE temp_contact_details AS
WITH cluster_org_entities AS (
    SELECT
        mc.id AS cluster_id,
        ce.entity_type,
        ce.entity_id,
        CASE 
            WHEN ce.entity_type = 'organization' THEN ce.entity_id
            WHEN ce.entity_type = 'service' THEN s.organization_id
            WHEN ce.entity_type = 'location' THEN l.organization_id
            WHEN ce.entity_type = 'service_at_location' THEN s2.organization_id
            ELSE NULL
        END AS organization_id
    FROM temp_diverse_clusters dc
    JOIN match_clusters mc ON dc.id = mc.id
    JOIN cluster_entities ce ON mc.id = ce.cluster_id
    LEFT JOIN service s ON ce.entity_type = 'service' AND ce.entity_id = s.id
    LEFT JOIN location l ON ce.entity_type = 'location' AND ce.entity_id = l.id
    LEFT JOIN service_at_location sal ON ce.entity_type = 'service_at_location' AND ce.entity_id = sal.id
    LEFT JOIN service s2 ON sal.service_id = s2.id
    WHERE ce.entity_type IN ('organization', 'service', 'location', 'service_at_location')
)
SELECT
    coe.cluster_id,
    o.name AS organization_name,
    CONCAT(a.address_1, 
           CASE WHEN a.address_2 IS NOT NULL THEN CONCAT(', ', a.address_2) ELSE '' END,
           ', ', a.city, ', ', a.state_province, ' ', a.postal_code) AS address,
    p.number AS phone_number,
    p.type AS phone_type
FROM cluster_org_entities coe
JOIN organization o ON coe.organization_id = o.id
LEFT JOIN location l ON l.organization_id = o.id
LEFT JOIN address a ON a.location_id = l.id
LEFT JOIN phone p ON p.organization_id = o.id
WHERE coe.organization_id IS NOT NULL;

-- Now return a single combined result set with section headers
SELECT '1. CLUSTER INFORMATION' AS section, 
       NULL AS entity_type, 
       NULL AS entity_id, 
       cluster_id::text, 
       confidence_score::text, 
       matching_evidence AS detail1, 
       reasoning AS detail2, 
       is_reviewed::text AS detail3,
       created_at::text AS detail4,
       matching_methods AS detail5
FROM temp_cluster_info

UNION ALL

SELECT '2. ENTITY DETAILS' AS section,
       entity_type,
       entity_id::text,
       cluster_id::text,
       name AS detail_name,
       description AS detail1,
       email AS detail2,
       url AS detail3,
       NULL AS detail4,
       NULL AS detail5
FROM temp_entity_details

UNION ALL

SELECT '3. CONTACT INFORMATION' AS section,
       NULL AS entity_type,
       NULL AS entity_id,
       cluster_id::text,
       organization_name AS detail_name,
       address AS detail1,
       phone_number AS detail2,
       phone_type AS detail3,
       NULL AS detail4,
       NULL AS detail5
FROM temp_contact_details

ORDER BY section, cluster_id;

-- Clean up temporary tables
DROP TABLE temp_diverse_clusters;
DROP TABLE temp_cluster_info;
DROP TABLE temp_entity_details;
DROP TABLE temp_contact_details;