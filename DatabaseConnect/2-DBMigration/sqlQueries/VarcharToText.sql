-- Step 1: Identify PostGIS-related tables and views to exclude
DO $$
DECLARE
    postgis_objects TEXT[];
BEGIN
    -- Create a temporary table to store PostGIS objects
    CREATE TEMP TABLE postgis_objects_list (
        object_name TEXT
    );
    
    -- Add known PostGIS system views
    INSERT INTO postgis_objects_list VALUES 
        ('spatial_ref_sys'),
        ('geometry_columns'),
        ('geography_columns'),
        ('raster_columns'),
        ('raster_overviews');
    
    -- Add any tables that have geometry or geography columns
    INSERT INTO postgis_objects_list
    SELECT DISTINCT table_name 
    FROM information_schema.columns 
    WHERE udt_name IN ('geometry', 'geography');
    
    RAISE NOTICE 'Identified PostGIS-related objects to exclude from modification';
END $$;

-- Step 2: Save non-PostGIS view definitions before dropping them
DO $$
DECLARE
    view_record RECORD;
BEGIN
    -- Create temporary table to store view definitions
    CREATE TEMP TABLE view_defs (
        view_name TEXT,
        view_definition TEXT
    );
    
    -- Get all non-PostGIS views
    FOR view_record IN 
        SELECT 
            viewname AS view_name,
            definition AS view_definition
        FROM 
            pg_catalog.pg_views
        WHERE 
            schemaname NOT IN ('pg_catalog', 'information_schema') AND
            viewname NOT IN (SELECT object_name FROM postgis_objects_list)
        ORDER BY 
            viewname
    LOOP
        -- Store the view definition
        INSERT INTO view_defs (view_name, view_definition)
        VALUES (view_record.view_name, view_record.view_definition);
        
        -- Drop the view
        EXECUTE 'DROP VIEW IF EXISTS ' || view_record.view_name || ' CASCADE';
        RAISE NOTICE 'Dropped view %', view_record.view_name;
    END LOOP;
    
    RAISE NOTICE 'All non-PostGIS views have been dropped and their definitions saved';
END $$;

-- Step 3: Change VARCHAR to TEXT except in PostGIS-related tables
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN
        SELECT 
            table_name, 
            column_name
        FROM 
            information_schema.columns
        WHERE 
            table_schema = 'public' AND 
            data_type = 'character varying' AND
            table_name NOT IN (SELECT object_name FROM postgis_objects_list)
        ORDER BY
            table_name, column_name
    LOOP
        EXECUTE format('ALTER TABLE %I ALTER COLUMN %I TYPE TEXT', 
                      r.table_name, r.column_name);
        
        RAISE NOTICE 'Altered column % in table % to TEXT type', 
                    r.column_name, r.table_name;
    END LOOP;
    
    RAISE NOTICE 'All applicable VARCHAR columns converted to TEXT';
END $$;

-- Step 4: Recreate all non-PostGIS views
DO $$
DECLARE
    view_record RECORD;
BEGIN
    -- Recreate views from saved definitions
    FOR view_record IN 
        SELECT * FROM view_defs
        ORDER BY view_name -- Recreate in alphabetical order
    LOOP
        BEGIN
            EXECUTE view_record.view_definition;
            RAISE NOTICE 'Recreated view %', view_record.view_name;
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'Error recreating view %: %', view_record.view_name, SQLERRM;
        END;
    END LOOP;
    
    -- Clean up
    DROP TABLE IF EXISTS view_defs;
    DROP TABLE IF EXISTS postgis_objects_list;
    RAISE NOTICE 'All non-PostGIS views have been recreated';
END $$;