-- List all schemas in the database
SELECT 
    nspname AS schema_name,
    pg_catalog.pg_get_userbyid(nspowner) AS schema_owner,
    pg_catalog.obj_description(oid, 'pg_namespace') AS description
FROM 
    pg_catalog.pg_namespace
WHERE 
    nspname NOT LIKE 'pg_%'     -- Exclude system schemas that start with pg_
    AND nspname != 'information_schema'  -- Exclude information_schema
ORDER BY 
    schema_name;