-- Get all views in the current database
SELECT 
    schemaname AS schema_name,
    viewname AS view_name,
    definition AS view_definition
FROM 
    pg_catalog.pg_views
WHERE 
    schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY 
    schemaname, viewname;

-- For more details, including dependent objects:
SELECT 
    n.nspname AS schema_name,
    c.relname AS view_name,
    r.rolname AS owner,
    pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size,
    c.reltuples::bigint AS row_estimate,
    pg_catalog.obj_description(c.oid, 'pg_class') AS description,
    pg_get_viewdef(c.oid) AS view_definition
FROM 
    pg_catalog.pg_class c
    LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
    LEFT JOIN pg_catalog.pg_roles r ON r.oid = c.relowner
WHERE 
    c.relkind = 'v'
    AND n.nspname NOT IN ('pg_catalog', 'information_schema')
ORDER BY 
    n.nspname, c.relname;

-- To find dependencies (what tables/columns a view depends on):
SELECT 
    dependent_view.relname AS view_name,
    source_table.relname AS referenced_table,
    source_column.attname AS referenced_column
FROM 
    pg_depend 
    JOIN pg_rewrite ON pg_depend.objid = pg_rewrite.oid 
    JOIN pg_class as dependent_view ON pg_rewrite.ev_class = dependent_view.oid 
    JOIN pg_class as source_table ON pg_depend.refobjid = source_table.oid 
    JOIN pg_attribute as source_column ON pg_depend.refobjid = source_column.attrelid 
                                       AND pg_depend.refobjsubid = source_column.attnum
WHERE 
    dependent_view.relkind = 'v'
    AND source_table.relkind = 'r'
    AND dependent_view.relnamespace NOT IN (SELECT oid FROM pg_namespace WHERE nspname IN ('pg_catalog', 'information_schema'))
ORDER BY 
    view_name, referenced_table, referenced_column;