-- Check if test schema exists
SELECT schema_name 
FROM information_schema.schemata 
WHERE schema_name = 'test';

-- Attempt to create it if it doesn't exist
CREATE SCHEMA IF NOT EXISTS test;

-- Grant permissions (if needed)
GRANT ALL ON SCHEMA test TO current_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA test GRANT ALL ON TABLES TO current_user;

-- Check for errors in the schema creation process
SELECT 
    pid, 
    usename, 
    application_name,
    state, 
    query
FROM 
    pg_stat_activity 
WHERE 
    query LIKE '%test%' 
    AND state != 'idle';
    
-- List schemas and their owners
SELECT 
    n.nspname AS schema_name,
    pg_catalog.pg_get_userbyid(n.nspowner) AS owner
FROM pg_catalog.pg_namespace n
WHERE n.nspname !~ '^pg_' AND n.nspname != 'information_schema'
ORDER BY 1;