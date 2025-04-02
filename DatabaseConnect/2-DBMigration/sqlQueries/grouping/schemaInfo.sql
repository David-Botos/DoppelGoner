SELECT 
    n.nspname as schema,
    c.relname as table_name,
    a.attname as column_name,
    format_type(a.atttypid, a.atttypmod) as data_type,
    CASE WHEN a.attnotnull THEN 'NOT NULL' ELSE 'NULL' END as nullable,
    CASE 
        WHEN co.contype = 'p' THEN 'PRIMARY KEY'
        WHEN co.contype = 'u' THEN 'UNIQUE'
        WHEN co.contype = 'f' THEN 'FOREIGN KEY'
        ELSE NULL
    END as constraint_type,
    CASE WHEN a.atthasdef THEN pg_get_expr(d.adbin, d.adrelid) ELSE NULL END as default_value
FROM 
    pg_attribute a
JOIN 
    pg_class c ON a.attrelid = c.oid
JOIN 
    pg_namespace n ON c.relnamespace = n.oid
LEFT JOIN 
    pg_constraint co ON (co.conrelid = c.oid AND a.attnum = ANY(co.conkey))
LEFT JOIN 
    pg_attrdef d ON (d.adrelid = a.attrelid AND d.adnum = a.attnum)
WHERE 
    n.nspname = 'grouping'
    AND a.attnum > 0
    AND NOT a.attisdropped
    AND c.relkind = 'r'
ORDER BY 
    schema, table_name, a.attnum;