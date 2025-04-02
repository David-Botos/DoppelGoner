SELECT 
    error_message,
    COUNT(*) as error_count
FROM 
    public.failed_migration_records
GROUP BY 
    error_message
ORDER BY 
    error_count DESC;