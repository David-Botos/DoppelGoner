-- Script to safely truncate all tables in the PUBLIC schema with confirmation prompt
DO $$ 
DECLARE
    confirmation boolean := false;
    current_user_name text;
BEGIN
    -- Get current user
    SELECT current_user INTO current_user_name;
    
    -- Use RAISE to output a very visible warning
    RAISE WARNING '
    ******************************************************************************
    *                                                                            *
    *                               DANGER ZONE                                  *
    *                                                                            *
    *             YOU ARE ABOUT TO DELETE ALL DATA FROM PUBLIC SCHEMA            *
    *                                                                            *
    *                       THIS CANNOT BE UNDONE!                               *
    *                                                                            *
    ******************************************************************************
    *                                                                            *
    * Current user: %                                                          *
    *                                                                            *
    * To continue, change "confirmation := false" to "confirmation := true"      *
    * in the SQL script and run it again.                                        *
    *                                                                            *
    ******************************************************************************
    ', current_user_name;
    
    -- Safety check - User must explicitly set this to true in the script
    IF confirmation THEN
        -- Disable foreign key constraints
        SET session_replication_role = 'replica';
        
        -- Truncate all tables in the public schema in a specific order
        -- that respects foreign key relationships
        EXECUTE 'TRUNCATE TABLE 
            public.accessibility_for_disabilities,
            public.address,
            public.contact,
            public.failed_migration_records,
            public.funding,
            public.language,
            public.other_attribute,
            public.phone,
            public.required_document,
            public.schedule,
            public.service_area,
            public.service_at_location,
            public.service_taxonomy,
            public.service,
            public.program,
            public.location,
            public.organization,
            public.metadata,
            public.migration_log,
            public.taxonomy_term
            CASCADE';
        
        -- Re-enable foreign key constraints
        SET session_replication_role = 'origin';
        
        RAISE NOTICE 'SUCCESS: All public schema tables have been truncated by user %', current_user_name;
    ELSE
        RAISE NOTICE 'Operation canceled. No tables were truncated.';
    END IF;
END $$;