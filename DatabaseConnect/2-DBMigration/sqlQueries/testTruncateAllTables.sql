-- Script to safely truncate all tables in the TEST schema
DO $$ 
DECLARE
    current_user_name text;
BEGIN
    -- Get current user
    SELECT current_user INTO current_user_name;
    
    -- Use RAISE to output an informational notice
    RAISE NOTICE '
    ******************************************************************************
    *                                                                            *
    *                    TRUNCATING ALL TABLES IN TEST SCHEMA                    *
    *                                                                            *
    * Current user: %                                                          *
    *                                                                            *
    ******************************************************************************
    ', current_user_name;
    
    -- Disable foreign key constraints
    SET session_replication_role = 'replica';
    
    -- Truncate all tables in the test schema in one command
    EXECUTE 'TRUNCATE TABLE 
        test.accessibility_for_disabilities,
        test.address,
        test.contact,
        test.failed_migration_records,
        test.funding,
        test.language,
        test.other_attribute,
        test.phone,
        test.required_document,
        test.schedule,
        test.service_area,
        test.service_at_location,
        test.service_taxonomy,
        test.service,
        test.program,
        test.location,
        test.organization,
        test.metadata,
        test.migration_log,
        test.taxonomy_term
        CASCADE';
    
    -- Re-enable foreign key constraints
    SET session_replication_role = 'origin';
    
    RAISE NOTICE 'SUCCESS: All test schema tables have been truncated by user %', current_user_name;
END $$;