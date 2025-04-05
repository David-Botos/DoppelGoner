-- Start a transaction so we can roll back if something goes wrong
BEGIN;

-- Create the new postgis schema
CREATE SCHEMA IF NOT EXISTS postgis;

-- Check if PostGIS is installed in public schema
DO $$
DECLARE
    postgis_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 
        FROM pg_extension 
        WHERE extname = 'postgis' AND extnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
    ) INTO postgis_exists;

    IF postgis_exists THEN
        -- Update search_path to include both schemas
        EXECUTE 'ALTER DATABASE ' || current_database() || ' SET search_path TO "$user", public, postgis';
        
        -- Create a temporary extension first because we can't directly move an extension
        CREATE EXTENSION IF NOT EXISTS postgis_sfcgal WITH SCHEMA postgis;
        CREATE EXTENSION IF NOT EXISTS postgis_topology WITH SCHEMA postgis;
        CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder WITH SCHEMA postgis;
        DROP EXTENSION postgis CASCADE;
        
        -- Recreate PostGIS extension in the postgis schema
        CREATE EXTENSION postgis WITH SCHEMA postgis;
        
        RAISE NOTICE 'PostGIS has been successfully moved from public to postgis schema';
    ELSE
        -- If PostGIS is not in public, just create it in postgis schema
        CREATE EXTENSION postgis WITH SCHEMA postgis;
        RAISE NOTICE 'PostGIS extension has been created in the postgis schema';
    END IF;
END
$$;

-- Set search_path for current session too
SET search_path TO "$user", public, postgis;

-- Verify the PostGIS is in the correct schema
SELECT n.nspname as schema, e.extname as extension
FROM pg_extension e 
JOIN pg_namespace n ON e.extnamespace = n.oid
WHERE e.extname LIKE 'postgis%';

-- If everything looks good, commit the transaction
COMMIT;