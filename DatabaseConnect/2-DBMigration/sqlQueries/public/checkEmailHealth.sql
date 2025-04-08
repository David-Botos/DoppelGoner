-- 1. Case-insensitive duplicates
SELECT LOWER(email) AS normalized_email, COUNT(*) AS occurrences
FROM organization
WHERE email IS NOT NULL AND email != ''
GROUP BY LOWER(email)
HAVING COUNT(*) > 1
ORDER BY occurrences DESC;

-- 2. Emails with dots in the local part (before the @)
SELECT email
FROM organization
WHERE email ~ '^[^@]+(\.[^@]+)+@'
ORDER BY email;

-- 3. Malformed or invalid emails
SELECT email
FROM organization
WHERE email !~* '^[^@\s]+@[^@\s]+\.[^@\s]+$'
ORDER BY email;

-- 4. Group by domain
SELECT SPLIT_PART(email, '@', 2) AS domain, COUNT(*) AS count
FROM organization
WHERE email IS NOT NULL AND email LIKE '%@%'
GROUP BY domain
ORDER BY count DESC;

-- 5. Optional fuzzy comparison if pg_trgm is enabled
-- CREATE EXTENSION IF NOT EXISTS pg_trgm;
-- SELECT a.id AS org1_id, a.email AS email1, b.id AS org2_id, b.email AS email2
-- FROM organization a
-- JOIN organization b ON a.id < b.id
-- WHERE levenshtein(a.email, b.email) <= 2;
