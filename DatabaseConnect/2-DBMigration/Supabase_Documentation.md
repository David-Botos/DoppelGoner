# HSDS Database Documentation for Supabase

This document provides a comprehensive guide to the Human Services Data Specification (HSDS) database implemented in Supabase. It covers the structure of each table, explains each field, and provides example records.

## Table of Contents

1. [Organization](#organization)
2. [Service](#service)
3. [Location](#location)
4. [Service at Location](#service-at-location)
5. [Address](#address)
6. [Phone](#phone)
7. [Program](#program)
8. [Contact](#contact)
9. [Service Area](#service-area)
10. [Required Document](#required-document)
11. [Schedule](#schedule)
12. [Taxonomy Term](#taxonomy-term)
13. [Service Taxonomy](#service-taxonomy)
14. [Other Attribute](#other-attribute)
15. [Accessibility For Disabilities](#accessibility-for-disabilities)
16. [Language](#language)
17. [Funding](#funding)
18. [Migration Log](#migration-log)
19. [Views and Functions](#views-and-functions)

---

## Organization

The organization table stores information about service providers.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | CHAR(36) | Primary key | Yes | "550e8400-e29b-41d4-a716-446655440000" |
| name | VARCHAR(255) | Organization name | Yes | "Community Food Bank" |
| alternate_name | VARCHAR(255) | Other name | No | "CFB" |
| description | TEXT | Organization description | No | "A non-profit organization providing food assistance to the community." |
| email | VARCHAR(255) | Contact email | No | "info@communityfoodbank.org" |
| url | VARCHAR(255) | Website | No | "https://www.communityfoodbank.org" |
| tax_status | VARCHAR(255) | Tax status | No | "501(c)(3)" |
| tax_id | VARCHAR(255) | Tax ID | No | "12-3456789" |
| year_incorporated | CHAR(4) | Year founded | No | "1995" |
| legal_status | VARCHAR(255) | Legal designation | No | "Non-profit corporation" |
| parent_organization_id | CHAR(36) | Foreign key to parent org | No | "440e8400-e29b-41d4-a716-446655550000" |
| last_modified | TIMESTAMP | Last update time | No | "2023-06-15 14:30:45" |
| created | TIMESTAMP | Creation time | No | "2022-11-10 09:15:22" |
| original_id | VARCHAR(100) | Original Snowflake ID | No | "SRVTAX123456" |

### Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440013",
  "service_id": "550e8400-e29b-41d4-a716-446655440001",
  "taxonomy_term_id": "550e8400-e29b-41d4-a716-446655440010",
  "last_modified": "2023-07-15 09:25:18",
  "created": "2022-11-28 14:22:35",
  "original_id": "SRVTAX123456"
}
```

---

## Other Attribute

The other_attribute table links entities other than services to taxonomy terms.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | CHAR(36) | Primary key | Yes | "550e8400-e29b-41d4-a716-446655440014" |
| link_id | CHAR(36) | ID of the linked entity | Yes | "550e8400-e29b-41d4-a716-446655440002" |
| link_type | VARCHAR(50) | Type of linked entity | Yes | "location" |
| taxonomy_term_id | CHAR(36) | Foreign key to taxonomy_term | No | "550e8400-e29b-41d4-a716-446655440015" |
| last_modified | TIMESTAMP | Last update time | No | "2023-07-16 11:35:28" |
| created | TIMESTAMP | Creation time | No | "2022-12-02 13:45:22" |
| original_id | VARCHAR(100) | Original Snowflake ID | No | "OTHATR123456" |

### Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440014",
  "link_id": "550e8400-e29b-41d4-a716-446655440002",
  "link_type": "location",
  "taxonomy_term_id": "550e8400-e29b-41d4-a716-446655440015",
  "last_modified": "2023-07-16 11:35:28",
  "created": "2022-12-02 13:45:22",
  "original_id": "OTHATR123456"
}
```

---

## Accessibility For Disabilities

The accessibility_for_disabilities table describes accommodations at locations.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | CHAR(36) | Primary key | Yes | "550e8400-e29b-41d4-a716-446655440016" |
| location_id | CHAR(36) | Foreign key to location | Yes | "550e8400-e29b-41d4-a716-446655440002" |
| accessibility | VARCHAR(255) | Accommodation type | Yes | "wheelchair" |
| details | TEXT | Additional information | No | "Ramp at main entrance, elevator to all floors, accessible restrooms" |
| last_modified | TIMESTAMP | Last update time | No | "2023-07-18 10:20:45" |
| created | TIMESTAMP | Creation time | No | "2022-12-05 09:15:38" |
| original_id | VARCHAR(100) | Original Snowflake ID | No | "ACCFOR123456" |

### Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440016",
  "location_id": "550e8400-e29b-41d4-a716-446655440002",
  "accessibility": "wheelchair",
  "details": "Ramp at main entrance, elevator to all floors, accessible restrooms. Wide doorways and hallways throughout the facility.",
  "last_modified": "2023-07-18 10:20:45",
  "created": "2022-12-05 09:15:38",
  "original_id": "ACCFOR123456"
}
```

---

## Language

The language table lists languages supported at services and locations.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | CHAR(36) | Primary key | Yes | "550e8400-e29b-41d4-a716-446655440017" |
| service_id | CHAR(36) | Foreign key to service | No | "550e8400-e29b-41d4-a716-446655440001" |
| location_id | CHAR(36) | Foreign key to location | No | null |
| language | VARCHAR(10) | ISO language code | Yes | "es" |
| last_modified | TIMESTAMP | Last update time | No | "2023-07-14 14:18:32" |
| created | TIMESTAMP | Creation time | No | "2022-12-03 15:55:28" |
| original_id | VARCHAR(100) | Original Snowflake ID | No | "LANG123456" |

### Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440017",
  "service_id": "550e8400-e29b-41d4-a716-446655440001",
  "location_id": null,
  "language": "es",
  "last_modified": "2023-07-14 14:18:32",
  "created": "2022-12-03 15:55:28",
  "original_id": "LANG123456"
}
```

---

## Funding

The funding table records funding sources for organizations and services.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | CHAR(36) | Primary key | Yes | "550e8400-e29b-41d4-a716-446655440018" |
| organization_id | CHAR(36) | Foreign key to organization | No | "550e8400-e29b-41d4-a716-446655440000" |
| service_id | CHAR(36) | Foreign key to service | No | null |
| source | VARCHAR(255) | Funding source | No | "USDA Emergency Food Assistance Program" |
| last_modified | TIMESTAMP | Last update time | No | "2023-07-19 13:28:42" |
| created | TIMESTAMP | Creation time | No | "2022-12-08 11:45:20" |
| original_id | VARCHAR(100) | Original Snowflake ID | No | "FUND123456" |

### Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440018",
  "organization_id": "550e8400-e29b-41d4-a716-446655440000",
  "service_id": null,
  "source": "USDA Emergency Food Assistance Program",
  "last_modified": "2023-07-19 13:28:42",
  "created": "2022-12-08 11:45:20",
  "original_id": "FUND123456"
}
```

---

## Migration Log

The migration_log table tracks the migration process from Snowflake.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | SERIAL | Primary key (auto-incremented) | Yes | 1 |
| source_table | VARCHAR(100) | Original Snowflake table | Yes | "ORGANIZATION" |
| target_table | VARCHAR(100) | Target PostgreSQL table | Yes | "organization" |
| records_migrated | INT | Number of records processed | Yes | 100 |
| success_count | INT | Successfully migrated records | Yes | 98 |
| failure_count | INT | Failed migrations | Yes | 2 |
| error_messages | TEXT | Details of errors | No | "Failed to process records with IDs: ORG789, ORG456. Missing required fields." |
| started_at | TIMESTAMP | Migration start time | Yes | "2023-08-01 08:00:00" |
| completed_at | TIMESTAMP | Migration end time | Yes | "2023-08-01 08:05:32" |
| execution_time_seconds | NUMERIC(10,2) | Duration in seconds | Yes | 332.45 |

### Example Record

```json
{
  "id": 1,
  "source_table": "ORGANIZATION",
  "target_table": "organization",
  "records_migrated": 100,
  "success_count": 98,
  "failure_count": 2,
  "error_messages": "Failed to process records with IDs: ORG789, ORG456. Missing required fields.",
  "started_at": "2023-08-01 08:00:00",
  "completed_at": "2023-08-01 08:05:32",
  "execution_time_seconds": 332.45
}
```

---

## Views and Functions

### Service Location View

This view combines data from service, organization, location, and address tables for easier querying.

#### Usage Example

```sql
-- Find all active services with their locations in Austin
SELECT 
    service_id, 
    service_name, 
    organization_name, 
    location_name, 
    address_1, 
    city
FROM 
    service_location_view
WHERE 
    service_status = 'active' 
    AND city = 'Austin';
```

### Find Nearby Services Function

This function uses the Haversine formula to find services within a specified radius.

#### Usage Example

```sql
-- Find services within 5 miles of a specific location
SELECT 
    service_id, 
    service_name, 
    organization_name, 
    distance_miles, 
    address_1, 
    city, 
    postal_code
FROM 
    find_nearby_services(30.267153, -97.743057, 5.0)
ORDER BY 
    distance_miles;
```

## Common Query Patterns

### 1. Find all services by an organization

```sql
SELECT 
    s.id, 
    s.name, 
    s.description, 
    s.status
FROM 
    service s
WHERE 
    s.organization_id = '550e8400-e29b-41d4-a716-446655440000'
    AND s.status = 'active';
```

### 2. Find all services at a location

```sql
SELECT 
    s.id, 
    s.name, 
    s.description, 
    sal.description as service_at_location_description
FROM 
    service s
JOIN 
    service_at_location sal ON s.id = sal.service_id
WHERE 
    sal.location_id = '550e8400-e29b-41d4-a716-446655440002'
    AND s.status = 'active';
```

### 3. Find service contact information

```sql
SELECT 
    s.name as service_name,
    c.name as contact_name,
    c.title,
    c.email,
    p.number as phone_number,
    p.extension
FROM 
    service s
LEFT JOIN 
    contact c ON s.id = c.service_id
LEFT JOIN 
    phone p ON s.id = p.service_id
WHERE 
    s.id = '550e8400-e29b-41d4-a716-446655440001';
```

### 4. Find services by taxonomy term

```sql
SELECT 
    s.id, 
    s.name, 
    s.description
FROM 
    service s
JOIN 
    service_taxonomy st ON s.id = st.service_id
JOIN 
    taxonomy_term tt ON st.taxonomy_term_id = tt.id
WHERE 
    tt.term = 'Food Assistance'
    AND s.status = 'active';
```

### 5. Find locations with accessibility features

```sql
SELECT 
    l.id,
    l.name,
    a.address_1,
    a.city,
    a.state_province,
    afd.accessibility,
    afd.details
FROM 
    location l
JOIN 
    address a ON l.id = a.location_id
JOIN 
    accessibility_for_disabilities afd ON l.id = afd.location_id
WHERE 
    afd.accessibility = 'wheelchair';
```

## Working with Supabase

### Authentication

When working with Supabase, use the client library to authenticate:

```javascript
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = 'https://your-project-url.supabase.co'
const supabaseKey = 'your-anon-key'
const supabase = createClient(supabaseUrl, supabaseKey)
```

### Basic Queries

Use Supabase's query builder for most operations:

```javascript
// Get all active services
const { data, error } = await supabase
  .from('service')
  .select('id, name, description, organization_id')
  .eq('status', 'active')
```

### Using the Service Location View

```javascript
// Find services in Austin
const { data, error } = await supabase
  .from('service_location_view')
  .select('*')
  .eq('city', 'Austin')
```

### Using PostGIS Functions

```javascript
// Run a raw query to find nearby services
const { data, error } = await supabase
  .rpc('find_nearby_services', { 
    lat: 30.267153, 
    lon: -97.743057, 
    radius_miles: 5.0 
  })
```

## Database Maintenance

### Data Migration

Track migration progress using the migration_log table:

```sql
SELECT 
    source_table, 
    target_table, 
    records_migrated, 
    success_count, 
    failure_count, 
    started_at, 
    completed_at 
FROM 
    migration_log
ORDER BY 
    started_at DESC;
```

### Index Maintenance

If query performance degrades over time, consider reindexing:

```sql
REINDEX TABLE organization;
REINDEX TABLE service;
REINDEX TABLE location;
```

### Monitoring Performance

Use Supabase's dashboard to monitor query performance and identify slow queries for optimization.
ORG123456" |
| original_translations_id | VARCHAR(100) | Original translation ID | No | "ORGTRANS123456" |

### Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Community Food Bank",
  "alternate_name": "CFB",
  "description": "A non-profit organization providing food assistance to the community through multiple programs including a food pantry, mobile distribution, and nutrition education.",
  "email": "info@communityfoodbank.org",
  "url": "https://www.communityfoodbank.org",
  "tax_status": "501(c)(3)",
  "tax_id": "12-3456789",
  "year_incorporated": "1995",
  "legal_status": "Non-profit corporation",
  "parent_organization_id": null,
  "last_modified": "2023-06-15 14:30:45",
  "created": "2022-11-10 09:15:22",
  "original_id": "ORG123456",
  "original_translations_id": "ORGTRANS123456"
}
```

---

## Service

The service table contains services provided by organizations.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | CHAR(36) | Primary key | Yes | "550e8400-e29b-41d4-a716-446655440001" |
| organization_id | CHAR(36) | Foreign key to organization | Yes | "550e8400-e29b-41d4-a716-446655440000" |
| program_id | CHAR(36) | Foreign key to program | No | "550e8400-e29b-41d4-a716-446655440011" |
| name | VARCHAR(255) | Service name | Yes | "Emergency Food Pantry" |
| alternate_name | VARCHAR(255) | Other name | No | "Food Assistance" |
| description | TEXT | Service description | No | "Provides emergency food boxes to eligible individuals and families." |
| short_description | TEXT | Brief description | No | "Emergency food assistance" |
| url | VARCHAR(255) | Service-specific website | No | "https://www.communityfoodbank.org/services/pantry" |
| email | VARCHAR(255) | Service-specific email | No | "pantry@communityfoodbank.org" |
| status | VARCHAR(50) | Current status | Yes | "active" |
| interpretation_services | TEXT | Language support info | No | "Spanish and Vietnamese interpretation available" |
| application_process | TEXT | How to apply | No | "Walk-in during hours of operation. Bring ID and proof of address." |
| wait_time | VARCHAR(255) | Expected wait | No | "15-30 minutes" |
| fees_description | TEXT | Cost information | No | "No cost to eligible clients" |
| accreditations | TEXT | Certifications | No | "Feeding America partner" |
| licenses | VARCHAR(255) | Legal licenses | No | "County Food Handler Certification" |
| minimum_age | INT | Minimum age to receive service | No | 0 |
| maximum_age | INT | Maximum age to receive service | No | null |
| eligibility_description | TEXT | Who can receive service | No | "Residents of Travis County with income below 200% of federal poverty level" |
| alert | TEXT | Temporary notices | No | "Holiday hours in effect Dec 24-Jan 2" |
| last_modified | TIMESTAMP | Last update time | No | "2023-07-20 11:45:32" |
| created | TIMESTAMP | Creation time | No | "2022-11-15 10:20:30" |
| original_id | VARCHAR(100) | Original Snowflake ID | No | "SRV123456" |
| original_translations_id | VARCHAR(100) | Original translation ID | No | "SRVTRANS123456" |

### Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440001",
  "organization_id": "550e8400-e29b-41d4-a716-446655440000",
  "program_id": "550e8400-e29b-41d4-a716-446655440011",
  "name": "Emergency Food Pantry",
  "alternate_name": "Food Assistance",
  "description": "Provides emergency food boxes to eligible individuals and families. Each box contains a 3-day supply of nutritionally balanced, non-perishable food items. Fresh produce, bread, and dairy products are available when in stock.",
  "short_description": "Emergency food assistance",
  "url": "https://www.communityfoodbank.org/services/pantry",
  "email": "pantry@communityfoodbank.org",
  "status": "active",
  "interpretation_services": "Spanish and Vietnamese interpretation available during regular hours",
  "application_process": "Walk-in during hours of operation. Bring ID and proof of address. First-time clients need to complete an intake form.",
  "wait_time": "15-30 minutes",
  "fees_description": "No cost to eligible clients",
  "accreditations": "Feeding America partner",
  "licenses": "County Food Handler Certification",
  "minimum_age": 0,
  "maximum_age": null,
  "eligibility_description": "Residents of Travis County with income below 200% of federal poverty level",
  "alert": "Holiday hours in effect Dec 24-Jan 2",
  "last_modified": "2023-07-20 11:45:32",
  "created": "2022-11-15 10:20:30",
  "original_id": "SRV123456",
  "original_translations_id": "SRVTRANS123456"
}
```

---

## Location

The location table stores physical places where services are provided.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | CHAR(36) | Primary key | Yes | "550e8400-e29b-41d4-a716-446655440002" |
| organization_id | CHAR(36) | Foreign key to organization | Yes | "550e8400-e29b-41d4-a716-446655440000" |
| name | VARCHAR(255) | Location name | No | "Main Distribution Center" |
| alternate_name | VARCHAR(255) | Other name | No | "Downtown Location" |
| description | TEXT | Location description | No | "Main facility with warehouse and client service area." |
| short_description | TEXT | Brief description | No | "Main food distribution center" |
| transportation | TEXT | How to get there | No | "Bus routes 7 and 20 stop in front of the building." |
| latitude | DECIMAL(10,6) | Y coordinate | No | 30.267153 |
| longitude | DECIMAL(10,6) | X coordinate | No | -97.743057 |
| location_type | VARCHAR(50) | Type of location | No | "physical" |
| last_modified | TIMESTAMP | Last update time | No | "2023-06-30 09:22:18" |
| created | TIMESTAMP | Creation time | No | "2022-11-12 13:40:55" |
| original_id | VARCHAR(100) | Original Snowflake ID | No | "LOC123456" |
| original_translations_id | VARCHAR(100) | Original translation ID | No | "LOCTRANS123456" |

### Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440002",
  "organization_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Main Distribution Center",
  "alternate_name": "Downtown Location",
  "description": "Main facility with warehouse and client service area. Includes a waiting room, private intake offices, and a large food distribution area.",
  "short_description": "Main food distribution center",
  "transportation": "Bus routes 7 and 20 stop in front of the building. Limited client parking available in the lot behind the building.",
  "latitude": 30.267153,
  "longitude": -97.743057,
  "location_type": "physical",
  "last_modified": "2023-06-30 09:22:18",
  "created": "2022-11-12 13:40:55",
  "original_id": "LOC123456",
  "original_translations_id": "LOCTRANS123456"
}
```

---

## Service at Location

The service_at_location table links services to the locations where they're offered.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | CHAR(36) | Primary key | Yes | "550e8400-e29b-41d4-a716-446655440003" |
| service_id | CHAR(36) | Foreign key to service | Yes | "550e8400-e29b-41d4-a716-446655440001" |
| location_id | CHAR(36) | Foreign key to location | Yes | "550e8400-e29b-41d4-a716-446655440002" |
| description | TEXT | Description of service at this location | No | "Full-service food pantry with all programs available." |
| last_modified | TIMESTAMP | Last update time | No | "2023-07-01 15:10:40" |
| created | TIMESTAMP | Creation time | No | "2022-11-18 14:25:33" |
| original_id | VARCHAR(100) | Original Snowflake ID | No | "SAL123456" |
| original_translations_id | VARCHAR(100) | Original translation ID | No | "SALTRANS123456" |

### Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440003",
  "service_id": "550e8400-e29b-41d4-a716-446655440001",
  "location_id": "550e8400-e29b-41d4-a716-446655440002",
  "description": "Full-service food pantry with all programs available. This location offers the complete range of food assistance services including emergency food boxes, SNAP application assistance, and nutrition education.",
  "last_modified": "2023-07-01 15:10:40",
  "created": "2022-11-18 14:25:33",
  "original_id": "SAL123456",
  "original_translations_id": "SALTRANS123456"
}
```

---

## Address

The address table contains physical and mailing addresses for locations.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | CHAR(36) | Primary key | Yes | "550e8400-e29b-41d4-a716-446655440004" |
| location_id | CHAR(36) | Foreign key to location | Yes | "550e8400-e29b-41d4-a716-446655440002" |
| attention | VARCHAR(255) | Person or dept. attention | No | "Client Services Department" |
| address_1 | VARCHAR(255) | Primary address line | Yes | "1234 Main Street" |
| address_2 | VARCHAR(255) | Secondary address line | No | "Suite 500" |
| city | VARCHAR(255) | City | Yes | "Austin" |
| region | VARCHAR(255) | Region | No | "Central" |
| state_province | VARCHAR(100) | State/province | Yes | "TX" |
| postal_code | VARCHAR(20) | ZIP/postal code | Yes | "78701" |
| country | CHAR(2) | Country code | Yes | "US" |
| address_type | VARCHAR(20) | Type of address | Yes | "physical" |
| last_modified | TIMESTAMP | Last update time | No | "2023-06-25 17:05:12" |
| created | TIMESTAMP | Creation time | No | "2022-11-14 11:30:27" |
| original_id | VARCHAR(100) | Original Snowflake ID | No | "ADDR123456" |

### Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440004",
  "location_id": "550e8400-e29b-41d4-a716-446655440002",
  "attention": "Client Services Department",
  "address_1": "1234 Main Street",
  "address_2": "Suite 500",
  "city": "Austin",
  "region": "Central",
  "state_province": "TX",
  "postal_code": "78701",
  "country": "US",
  "address_type": "physical",
  "last_modified": "2023-06-25 17:05:12",
  "created": "2022-11-14 11:30:27",
  "original_id": "ADDR123456"
}
```

---

## Phone

The phone table stores phone numbers for various entities.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | CHAR(36) | Primary key | Yes | "550e8400-e29b-41d4-a716-446655440005" |
| location_id | CHAR(36) | Foreign key to location | No | "550e8400-e29b-41d4-a716-446655440002" |
| service_id | CHAR(36) | Foreign key to service | No | null |
| organization_id | CHAR(36) | Foreign key to organization | No | null |
| contact_id | CHAR(36) | Foreign key to contact | No | null |
| service_at_location_id | CHAR(36) | Foreign key to service_at_location | No | null |
| number | VARCHAR(50) | Phone number | Yes | "512-555-1234" |
| extension | VARCHAR(20) | Extension | No | "123" |
| type | VARCHAR(20) | Type of phone | No | "voice" |
| language | VARCHAR(10) | Language for this number | No | "en" |
| description | TEXT | Additional information | No | "Main reception line, staffed 9am-5pm weekdays" |
| priority | INT | Priority order | No | 1 |
| last_modified | TIMESTAMP | Last update time | No | "2023-06-28 14:15:50" |
| created | TIMESTAMP | Creation time | No | "2022-11-16 10:45:38" |
| original_id | VARCHAR(100) | Original Snowflake ID | No | "PHN123456" |
| original_translations_id | VARCHAR(100) | Original translation ID | No | "PHNTRANS123456" |

### Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440005",
  "location_id": "550e8400-e29b-41d4-a716-446655440002",
  "service_id": null,
  "organization_id": null,
  "contact_id": null,
  "service_at_location_id": null,
  "number": "512-555-1234",
  "extension": "123",
  "type": "voice",
  "language": "en",
  "description": "Main reception line, staffed 9am-5pm weekdays",
  "priority": 1,
  "last_modified": "2023-06-28 14:15:50",
  "created": "2022-11-16 10:45:38",
  "original_id": "PHN123456",
  "original_translations_id": "PHNTRANS123456"
}
```

---

## Program

The program table organizes services into programs.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | CHAR(36) | Primary key | Yes | "550e8400-e29b-41d4-a716-446655440011" |
| organization_id | CHAR(36) | Foreign key to organization | Yes | "550e8400-e29b-41d4-a716-446655440000" |
| name | VARCHAR(255) | Program name | Yes | "Food Assistance Program" |
| alternate_name | VARCHAR(255) | Other name | No | "FAP" |
| last_modified | TIMESTAMP | Last update time | No | "2023-06-22 13:25:42" |
| created | TIMESTAMP | Creation time | No | "2022-11-11 08:50:15" |
| original_id | VARCHAR(100) | Original Snowflake ID | No | "PRG123456" |
| original_translations_id | VARCHAR(100) | Original translation ID | No | "PRGTRANS123456" |

### Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440011",
  "organization_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Food Assistance Program",
  "alternate_name": "FAP",
  "last_modified": "2023-06-22 13:25:42",
  "created": "2022-11-11 08:50:15",
  "original_id": "PRG123456",
  "original_translations_id": "PRGTRANS123456"
}
```

---

## Contact

The contact table contains information about specific people at organizations and services.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | CHAR(36) | Primary key | Yes | "550e8400-e29b-41d4-a716-446655440006" |
| organization_id | CHAR(36) | Foreign key to organization | No | "550e8400-e29b-41d4-a716-446655440000" |
| service_id | CHAR(36) | Foreign key to service | No | null |
| service_at_location_id | CHAR(36) | Foreign key to service_at_location | No | null |
| name | VARCHAR(255) | Contact name | No | "Jane Smith" |
| title | VARCHAR(255) | Job title | No | "Program Director" |
| department | VARCHAR(255) | Department | No | "Client Services" |
| email | VARCHAR(255) | Email address | No | "jane.smith@communityfoodbank.org" |
| last_modified | TIMESTAMP | Last update time | No | "2023-07-05 11:20:30" |
| created | TIMESTAMP | Creation time | No | "2022-11-20 09:35:48" |
| original_id | VARCHAR(100) | Original Snowflake ID | No | "CNT123456" |
| original_translations_id | VARCHAR(100) | Original translation ID | No | "CNTTRANS123456" |

### Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440006",
  "organization_id": "550e8400-e29b-41d4-a716-446655440000",
  "service_id": null,
  "service_at_location_id": null,
  "name": "Jane Smith",
  "title": "Program Director",
  "department": "Client Services",
  "email": "jane.smith@communityfoodbank.org",
  "last_modified": "2023-07-05 11:20:30",
  "created": "2022-11-20 09:35:48",
  "original_id": "CNT123456",
  "original_translations_id": "CNTTRANS123456"
}
```

---

## Service Area

The service_area table defines geographic areas where services are available.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | CHAR(36) | Primary key | Yes | "550e8400-e29b-41d4-a716-446655440007" |
| service_id | CHAR(36) | Foreign key to service | Yes | "550e8400-e29b-41d4-a716-446655440001" |
| service_area | VARCHAR(255) | Area name | No | "Travis County" |
| description | TEXT | Area description | No | "All of Travis County including Austin and surrounding municipalities." |
| extent | GEOMETRY | Geographic boundary | No | [POLYGON data] |
| extent_type | VARCHAR(50) | Format of extent | No | "geojson" |
| last_modified | TIMESTAMP | Last update time | No | "2023-07-10 16:12:35" |
| created | TIMESTAMP | Creation time | No | "2022-11-22 14:58:20" |
| original_id | VARCHAR(100) | Original Snowflake ID | No | "SAREA123456" |
| original_translations_id | VARCHAR(100) | Original translation ID | No | "SAREATRANS123456" |

### Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440007",
  "service_id": "550e8400-e29b-41d4-a716-446655440001",
  "service_area": "Travis County",
  "description": "All of Travis County including Austin and surrounding municipalities.",
  "extent": "0103000000010000000500000000000000008041C000000000008051400000000000804140000000000080514000000000008041400000000000804B4000000000008041C00000000000804B4000000000008041C000000000008051400",
  "extent_type": "geojson",
  "last_modified": "2023-07-10 16:12:35",
  "created": "2022-11-22 14:58:20",
  "original_id": "SAREA123456",
  "original_translations_id": "SAREATRANS123456"
}
```

---

## Required Document

The required_document table lists documents clients need to access services.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | CHAR(36) | Primary key | Yes | "550e8400-e29b-41d4-a716-446655440008" |
| service_id | CHAR(36) | Foreign key to service | Yes | "550e8400-e29b-41d4-a716-446655440001" |
| document | VARCHAR(255) | Document name | Yes | "Photo ID" |
| last_modified | TIMESTAMP | Last update time | No | "2023-07-08 10:30:25" |
| created | TIMESTAMP | Creation time | No | "2022-11-25 13:22:18" |
| original_id | VARCHAR(100) | Original Snowflake ID | No | "RDOC123456" |

### Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440008",
  "service_id": "550e8400-e29b-41d4-a716-446655440001",
  "document": "Photo ID",
  "last_modified": "2023-07-08 10:30:25",
  "created": "2022-11-25 13:22:18",
  "original_id": "RDOC123456"
}
```

---

## Schedule

The schedule table defines when services are available.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | CHAR(36) | Primary key | Yes | "550e8400-e29b-41d4-a716-446655440009" |
| service_id | CHAR(36) | Foreign key to service | No | "550e8400-e29b-41d4-a716-446655440001" |
| location_id | CHAR(36) | Foreign key to location | No | null |
| service_at_location_id | CHAR(36) | Foreign key to service_at_location | No | null |
| valid_from | DATE | Start date | No | "2023-01-01" |
| valid_to | DATE | End date | No | "2023-12-31" |
| dtstart | DATE | First event date | No | "2023-01-02" |
| freq | VARCHAR(20) | Frequency | No | "WEEKLY" |
| interval | INT | Interval | No | 1 |
| byday | VARCHAR(50) | Days of week | No | "MO,TU,WE,TH,FR" |
| description | TEXT | Text description | No | "Open Monday-Friday, 9am-5pm" |
| last_modified | TIMESTAMP | Last update time | No | "2023-07-12 12:40:15" |
| created | TIMESTAMP | Creation time | No | "2022-12-01 10:15:42" |
| original_id | VARCHAR(100) | Original Snowflake ID | No | "SCH123456" |

### Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440009",
  "service_id": "550e8400-e29b-41d4-a716-446655440001",
  "location_id": null,
  "service_at_location_id": null,
  "valid_from": "2023-01-01",
  "valid_to": "2023-12-31",
  "dtstart": "2023-01-02",
  "freq": "WEEKLY",
  "interval": 1,
  "byday": "MO,TU,WE,TH,FR",
  "description": "Open Monday-Friday, 9am-5pm. Closed on federal holidays.",
  "last_modified": "2023-07-12 12:40:15",
  "created": "2022-12-01 10:15:42",
  "original_id": "SCH123456"
}
```

---

## Taxonomy Term

The taxonomy_term table contains classification terms for services.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | CHAR(36) | Primary key | Yes | "550e8400-e29b-41d4-a716-446655440010" |
| term | VARCHAR(255) | Classification term | Yes | "Food Assistance" |
| description | TEXT | Term description | No | "Programs that provide food to individuals and families." |
| parent_id | CHAR(36) | Foreign key to parent term | No | "550e8400-e29b-41d4-a716-446655440012" |
| taxonomy | VARCHAR(255) | Taxonomy system | No | "AIRS" |
| language | VARCHAR(10) | Language code | No | "en" |
| last_modified | TIMESTAMP | Last update time | No | "2023-06-20 15:48:22" |
| created | TIMESTAMP | Creation time | No | "2022-11-05 11:30:15" |
| original_id | VARCHAR(100) | Original Snowflake ID | No | "TAXTRM123456" |

### Example Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440010",
  "term": "Food Assistance",
  "description": "Programs that provide food to individuals and families who are in need. Includes food pantries, meal programs, and food banks.",
  "parent_id": "550e8400-e29b-41d4-a716-446655440012",
  "taxonomy": "AIRS",
  "language": "en",
  "last_modified": "2023-06-20 15:48:22",
  "created": "2022-11-05 11:30:15",
  "original_id": "TAXTRM123456"
}
```

---

## Service Taxonomy

The service_taxonomy table links services to taxonomy terms.

### Columns

| Column Name | Data Type | Description | Required | Example |
|-------------|-----------|-------------|----------|---------|
| id | CHAR(36) | Primary key | Yes | "550e8400-e29b-41d4-a716-446655440013" |
| service_id | CHAR(36) | Foreign key to service | Yes | "550e8400-e29b-41d4-a716-446655440001" |
| taxonomy_term_id | CHAR(36) | Foreign key to taxonomy_term | Yes | "550e8400-e29b-41d4-a716-446655440010" |
| last_modified | TIMESTAMP | Last update time | No | "2023-07-15 09:25:18" |
| created | TIMESTAMP | Creation time | No | "2022-11-28 14:22:35" |
| original_id | VARCHAR(100) | Original Snowflake ID | No | "