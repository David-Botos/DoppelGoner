SELECT
    relname AS table_name,
    n_live_tup AS record_count
FROM
    pg_stat_user_tables
WHERE
    schemaname = 'grouping'
ORDER BY
    relname;