DO $$ 
DECLARE
    r RECORD;
BEGIN
    -- Disable triggers temporarily
    SET session_replication_role = 'replica';
    
    -- Loop through all tables in the current schema
    FOR r IN (
        SELECT tablename FROM pg_tables 
        WHERE schemaname = 'public' -- Change if you use a different schema
        ORDER BY tablename
    )
    LOOP
        EXECUTE 'TRUNCATE TABLE "' || r.tablename || '" CASCADE';
    END LOOP;
    
    -- Re-enable triggers
    SET session_replication_role = 'origin';
END $$;