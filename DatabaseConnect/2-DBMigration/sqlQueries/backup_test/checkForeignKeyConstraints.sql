SELECT
    c.conname AS constraint_name,
    ns1.nspname AS source_schema,
    t1.relname AS source_table,
    ns2.nspname AS referenced_schema,
    t2.relname AS referenced_table
FROM
    pg_constraint c
    JOIN pg_class t1 ON c.conrelid = t1.oid
    JOIN pg_namespace ns1 ON t1.relnamespace = ns1.oid
    JOIN pg_class t2 ON c.confrelid = t2.oid
    JOIN pg_namespace ns2 ON t2.relnamespace = ns2.oid
WHERE
    c.contype = 'f'
    AND (
        -- Tables in backup_test referencing other schemas
        (ns1.nspname = 'backup_test' AND ns2.nspname != 'backup_test')
        OR
        -- Tables in other schemas referencing backup_test
        (ns1.nspname != 'backup_test' AND ns2.nspname = 'backup_test')
    )
ORDER BY
    source_schema, source_table;