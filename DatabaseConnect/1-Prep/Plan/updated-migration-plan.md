# Updated Migration Plan: Snowflake to PostgreSQL on GCP Implementation

## Revised Core Principles

1. **Focus on Critical Tables**: Prioritize migration of the most critical tables first: Organization, Service, Location, Service_at_Location, Address, and Phone
2. **English-Only Initially**: Support only English translations in the initial implementation
3. **Direct Mapping with Traceability**: Maintain traceability to original Snowflake records through original ID fields
4. **PostGIS Support**: Utilize PostgreSQL's PostGIS extension for geographic data (particularly for SERVICE_AREA)
5. **Referential Integrity**: Implement proper foreign key constraints for the core tables
6. **Small Test Migration**: Start with 100 records migration to validate the approach
7. **Cost Optimization**: Utilize PostgreSQL on GCP for cost efficiency
8. **Simplified Address Model**: Use a single address table with an address_type field instead of separate physical and postal address tables

## Table Prioritization

### High Priority Tables (Immediate Migration)
- organization
- service
- location
- service_at_location
- address (consolidated from physical_address & postal_address)
- phone

### Deferred Tables (Create Schema Only)
- accessibility_for_disabilities
- contact
- funding
- language
- program
- required_document
- schedule
- service_area
- taxonomy_term
- service_taxonomy
- other_attribute

## Updated Table Mapping for Priority Tables

| HSDS Table | Snowflake Source Table(s) | Action | ID Tracking Approach |
|------------|---------------------------|--------|---------------------|
| organization | ORGANIZATION + ORGANIZATION_TRANSLATIONS (where LOCALE='en') | Merge | Add original_id, original_translations_id |
| service | SERVICE + SERVICE_TRANSLATIONS (where LOCALE='en') | Merge | Add original_id, original_translations_id |
| location | LOCATION + LOCATION_TRANSLATIONS (where LOCALE='en') | Merge | Add original_id, original_translations_id |
| service_at_location | SERVICE_AT_LOCATION + SERVICE_AT_LOCATION_TRANSLATIONS (where LOCALE='en') | Merge | Add original_id, original_translations_id |
| address | ADDRESS | Transform | Add original_id, retain address_type field |
| phone | PHONE + PHONE_TRANSLATIONS (where LOCALE='en') | Merge | Add original_id, original_translations_id |

## Updated Field Mapping Details for Priority Tables

### 1. Organization Table

**GCP PostgreSQL Table Name**: `organization`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic | Traceability Fields |
|------------|---------------|-----------------|---------------------|-------------------|
| id | CHAR(36) | ORGANIZATION.ID | Direct mapping | |
| name | VARCHAR(255) | ORGANIZATION.NAME | Direct mapping | |
| alternate_name | VARCHAR(255) | ORGANIZATION.ALTERNATE_NAME | Direct mapping | |
| description | TEXT | ORGANIZATION_TRANSLATIONS.DESCRIPTION (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English description | |
| email | VARCHAR(255) | ORGANIZATION.EMAIL | Direct mapping | |
| url | VARCHAR(255) | ORGANIZATION.WEBSITE | Direct mapping | |
| tax_status | VARCHAR(255) | NULL | New field per HSDS | |
| tax_id | VARCHAR(255) | From ORGANIZATION_IDENTIFIER.IDENTIFIER where IDENTIFIER_TYPE='US-EIN' | Extract from identifiers | |
| year_incorporated | CHAR(4) | ORGANIZATION.YEAR_INCORPORATED | Direct mapping | |
| legal_status | VARCHAR(255) | ORGANIZATION.LEGAL_STATUS | Direct mapping | |
| parent_organization_id | CHAR(36) | ORGANIZATION.PARENT_ORGANIZATION_ID | Direct mapping | |
| last_modified | TIMESTAMP | ORGANIZATION.LAST_MODIFIED | Convert to proper timestamp | |
| created | TIMESTAMP | ORGANIZATION.CREATED | Convert to proper timestamp | |
| original_id | VARCHAR(100) | ORGANIZATION.ID | Store original Snowflake ID | New tracking field |
| original_translations_id | VARCHAR(100) | ORGANIZATION_TRANSLATIONS.ID | Store original translation ID | New tracking field |

### 2. Service Table

**GCP PostgreSQL Table Name**: `service`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic | Traceability Fields |
|------------|---------------|-----------------|---------------------|-------------------|
| id | CHAR(36) | SERVICE.ID | Direct mapping | |
| organization_id | CHAR(36) | SERVICE.ORGANIZATION_ID | Direct mapping | |
| program_id | CHAR(36) | SERVICE.PROGRAM_ID | Direct mapping | |
| name | VARCHAR(255) | SERVICE_TRANSLATIONS.NAME (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English name | |
| alternate_name | VARCHAR(255) | SERVICE_TRANSLATIONS.ALTERNATE_NAME (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English alternate name | |
| description | TEXT | SERVICE_TRANSLATIONS.DESCRIPTION (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English description | |
| short_description | TEXT | SERVICE_TRANSLATIONS.SHORT_DESCRIPTION (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English short description | |
| url | VARCHAR(255) | SERVICE.URL | Direct mapping | |
| email | VARCHAR(255) | SERVICE.EMAIL | Direct mapping | |
| status | VARCHAR(50) | SERVICE.STATUS | Direct mapping | |
| interpretation_services | TEXT | SERVICE_TRANSLATIONS.INTERPRETATION_SERVICES (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English interpretation services | |
| application_process | TEXT | SERVICE_TRANSLATIONS.APPLICATION_PROCESS (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English application process | |
| wait_time | VARCHAR(255) | NULL | New field per HSDS | |
| fees_description | TEXT | SERVICE_TRANSLATIONS.FEES_DESCRIPTION (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English fees description | |
| accreditations | TEXT | SERVICE_TRANSLATIONS.ACCREDITATIONS (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English accreditations | |
| licenses | VARCHAR(255) | NULL | New field per HSDS | |
| minimum_age | INT | SERVICE.MINIMUM_AGE | Direct mapping | |
| maximum_age | INT | SERVICE.MAXIMUM_AGE | Direct mapping | |
| eligibility_description | TEXT | SERVICE_TRANSLATIONS.ELIGIBILITY_DESCRIPTION (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English eligibility description | |
| alert | TEXT | SERVICE_TRANSLATIONS.ALERT (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English alert | |
| last_modified | TIMESTAMP | SERVICE.LAST_MODIFIED | Convert to proper timestamp | |
| created | TIMESTAMP | SERVICE.CREATED | Convert to proper timestamp | |
| original_id | VARCHAR(100) | SERVICE.ID | Store original Snowflake ID | New tracking field |
| original_translations_id | VARCHAR(100) | SERVICE_TRANSLATIONS.ID | Store original translation ID | New tracking field |

### 3. Location Table

**GCP PostgreSQL Table Name**: `location`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic | Traceability Fields |
|------------|---------------|-----------------|---------------------|-------------------|
| id | CHAR(36) | LOCATION.ID | Direct mapping | |
| organization_id | CHAR(36) | LOCATION.ORGANIZATION_ID | Direct mapping | |
| name | VARCHAR(255) | LOCATION.NAME | Direct mapping | |
| alternate_name | VARCHAR(255) | LOCATION.ALTERNATE_NAME | Direct mapping | |
| description | TEXT | LOCATION_TRANSLATIONS.DESCRIPTION (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English description | |
| short_description | TEXT | LOCATION_TRANSLATIONS.SHORT_DESCRIPTION (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English short description | |
| transportation | TEXT | LOCATION_TRANSLATIONS.TRANSPORTATION (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English transportation | |
| latitude | DECIMAL(10,6) | LOCATION.LATITUDE | Direct mapping | |
| longitude | DECIMAL(10,6) | LOCATION.LONGITUDE | Direct mapping | |
| location_type | VARCHAR(50) | LOCATION.LOCATION_TYPE | Direct mapping | |
| last_modified | TIMESTAMP | LOCATION.LAST_MODIFIED | Convert to proper timestamp | |
| created | TIMESTAMP | LOCATION.CREATED | Convert to proper timestamp | |
| original_id | VARCHAR(100) | LOCATION.ID | Store original Snowflake ID | New tracking field |
| original_translations_id | VARCHAR(100) | LOCATION_TRANSLATIONS.ID | Store original translation ID | New tracking field |

### 4. Service At Location Table

**GCP PostgreSQL Table Name**: `service_at_location`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic | Traceability Fields |
|------------|---------------|-----------------|---------------------|-------------------|
| id | CHAR(36) | SERVICE_AT_LOCATION.ID | Direct mapping | |
| service_id | CHAR(36) | SERVICE_AT_LOCATION.SERVICE_ID | Direct mapping | |
| location_id | CHAR(36) | SERVICE_AT_LOCATION.LOCATION_ID | Direct mapping | |
| description | TEXT | SERVICE_AT_LOCATION_TRANSLATIONS.DESCRIPTION (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English description | |
| last_modified | TIMESTAMP | SERVICE_AT_LOCATION.LAST_MODIFIED | Convert to proper timestamp | |
| created | TIMESTAMP | SERVICE_AT_LOCATION.CREATED | Convert to proper timestamp | |
| original_id | VARCHAR(100) | SERVICE_AT_LOCATION.ID | Store original Snowflake ID | New tracking field |
| original_translations_id | VARCHAR(100) | SERVICE_AT_LOCATION_TRANSLATIONS.ID | Store original translation ID | New tracking field |

### 5. Consolidated Address Table

**GCP PostgreSQL Table Name**: `address`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic | Traceability Fields |
|------------|---------------|-----------------|---------------------|-------------------|
| id | CHAR(36) | ADDRESS.ID | Direct mapping | |
| location_id | CHAR(36) | ADDRESS.LOCATION_ID | Direct mapping | |
| attention | VARCHAR(255) | ADDRESS.ATTENTION | Direct mapping | |
| address_1 | VARCHAR(255) | ADDRESS.ADDRESS_1 | Direct mapping | |
| address_2 | VARCHAR(255) | ADDRESS.ADDRESS_2 | Direct mapping | |
| city | VARCHAR(255) | ADDRESS.CITY | Direct mapping | |
| region | VARCHAR(255) | ADDRESS.REGION | Direct mapping | |
| state_province | VARCHAR(100) | ADDRESS.STATE_PROVINCE | Direct mapping | |
| postal_code | VARCHAR(20) | ADDRESS.POSTAL_CODE | Direct mapping | |
| country | CHAR(2) | ADDRESS.COUNTRY | Ensure 2-letter ISO code | |
| address_type | VARCHAR(20) | ADDRESS.ADDRESS_TYPE | Direct mapping | Retained original field |
| last_modified | TIMESTAMP | ADDRESS.LAST_MODIFIED | Convert to proper timestamp | |
| created | TIMESTAMP | ADDRESS.CREATED | Convert to proper timestamp | |
| original_id | VARCHAR(100) | ADDRESS.ID | Store original Snowflake ID | New tracking field |

### 6. Phone Table

**GCP PostgreSQL Table Name**: `phone`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic | Traceability Fields |
|------------|---------------|-----------------|---------------------|-------------------|
| id | CHAR(36) | PHONE.ID | Direct mapping | |
| location_id | CHAR(36) | PHONE.LOCATION_ID | Direct mapping | |
| service_id | CHAR(36) | PHONE.SERVICE_ID | Direct mapping | |
| organization_id | CHAR(36) | PHONE.ORGANIZATION_ID | Direct mapping | |
| contact_id | CHAR(36) | PHONE.CONTACT_ID | Direct mapping | |
| service_at_location_id | CHAR(36) | PHONE.SERVICE_AT_LOCATION_ID | Direct mapping | |
| number | VARCHAR(50) | PHONE.NUMBER | Direct mapping | |
| extension | VARCHAR(20) | PHONE.EXTENSION | Direct mapping | |
| type | VARCHAR(20) | PHONE.TYPE | Direct mapping | |
| language | VARCHAR(10) | 'en' | Default to English | |
| description | TEXT | PHONE_TRANSLATIONS.DESCRIPTION (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English description | |
| priority | INT | PHONE.PRIORITY | Direct mapping | |
| last_modified | TIMESTAMP | PHONE.LAST_MODIFIED | Convert to proper timestamp | |
| created | TIMESTAMP | PHONE.CREATED | Convert to proper timestamp | |
| original_id | VARCHAR(100) | PHONE.ID | Store original Snowflake ID | New tracking field |
| original_translations_id | VARCHAR(100) | PHONE_TRANSLATIONS.ID | Store original translation ID | New tracking field |

## PostgreSQL Schema on GCP with PostGIS Support

```sql
-- Enable PostGIS
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create organization table
CREATE TABLE organization (
  id CHAR(36) PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  alternate_name VARCHAR(255),
  description TEXT,
  email VARCHAR(255),
  url VARCHAR(255),
  tax_status VARCHAR(255),
  tax_id VARCHAR(255),
  year_incorporated CHAR(4),
  legal_status VARCHAR(255),
  parent_organization_id CHAR(36),
  last_modified TIMESTAMP,
  created TIMESTAMP,
  original_id VARCHAR(100),
  original_translations_id VARCHAR(100),
  FOREIGN KEY (parent_organization_id) REFERENCES organization(id)
);

-- Create service table
CREATE TABLE service (
  id CHAR(36) PRIMARY KEY,
  organization_id CHAR(36) NOT NULL,
  program_id CHAR(36),
  name VARCHAR(255) NOT NULL,
  alternate_name VARCHAR(255),
  description TEXT,
  short_description TEXT,
  url VARCHAR(255),
  email VARCHAR(255),
  status VARCHAR(50) NOT NULL CHECK (status IN ('active', 'inactive', 'defunct', 'temporarily closed')),
  interpretation_services TEXT,
  application_process TEXT,
  wait_time VARCHAR(255),
  fees_description TEXT,
  accreditations TEXT,
  licenses VARCHAR(255),
  minimum_age INT,
  maximum_age INT,
  eligibility_description TEXT,
  alert TEXT,
  last_modified TIMESTAMP,
  created TIMESTAMP,
  original_id VARCHAR(100),
  original_translations_id VARCHAR(100),
  FOREIGN KEY (organization_id) REFERENCES organization(id)
);

-- Create location table
CREATE TABLE location (
  id CHAR(36) PRIMARY KEY,
  organization_id CHAR(36) NOT NULL,
  name VARCHAR(255),
  alternate_name VARCHAR(255),
  description TEXT,
  short_description TEXT,
  transportation TEXT,
  latitude DECIMAL(10,6),
  longitude DECIMAL(10,6),
  location_type VARCHAR(50),
  last_modified TIMESTAMP,
  created TIMESTAMP,
  original_id VARCHAR(100),
  original_translations_id VARCHAR(100),
  FOREIGN KEY (organization_id) REFERENCES organization(id)
);

-- Create service_at_location table
CREATE TABLE service_at_location (
  id CHAR(36) PRIMARY KEY,
  service_id CHAR(36) NOT NULL,
  location_id CHAR(36) NOT NULL,
  description TEXT,
  last_modified TIMESTAMP,
  created TIMESTAMP,
  original_id VARCHAR(100),
  original_translations_id VARCHAR(100),
  FOREIGN KEY (service_id) REFERENCES service(id),
  FOREIGN KEY (location_id) REFERENCES location(id)
);

-- Create consolidated address table
CREATE TABLE address (
  id CHAR(36) PRIMARY KEY,
  location_id CHAR(36) NOT NULL,
  attention VARCHAR(255),
  address_1 VARCHAR(255) NOT NULL,
  address_2 VARCHAR(255),
  city VARCHAR(255) NOT NULL,
  region VARCHAR(255),
  state_province VARCHAR(100) NOT NULL,
  postal_code VARCHAR(20) NOT NULL,
  country CHAR(2) NOT NULL,
  address_type VARCHAR(20) NOT NULL,
  last_modified TIMESTAMP,
  created TIMESTAMP,
  original_id VARCHAR(100),
  FOREIGN KEY (location_id) REFERENCES location(id)
);

-- Create phone table
CREATE TABLE phone (
  id CHAR(36) PRIMARY KEY,
  location_id CHAR(36),
  service_id CHAR(36),
  organization_id CHAR(36),
  contact_id CHAR(36),
  service_at_location_id CHAR(36),
  number VARCHAR(50) NOT NULL,
  extension VARCHAR(20),
  type VARCHAR(20),
  language VARCHAR(10) DEFAULT 'en',
  description TEXT,
  priority INT,
  last_modified TIMESTAMP,
  created TIMESTAMP,
  original_id VARCHAR(100),
  original_translations_id VARCHAR(100),
  FOREIGN KEY (location_id) REFERENCES location(id),
  FOREIGN KEY (service_id) REFERENCES service(id),
  FOREIGN KEY (organization_id) REFERENCES organization(id),
  FOREIGN KEY (service_at_location_id) REFERENCES service_at_location(id)
);

-- Create migration_log table for tracking migration progress
CREATE TABLE migration_log (
  id SERIAL PRIMARY KEY,
  source_table VARCHAR(100) NOT NULL,
  target_table VARCHAR(100) NOT NULL,
  records_migrated INT NOT NULL,
  success_count INT NOT NULL,
  failure_count INT NOT NULL,
  error_messages TEXT,
  started_at TIMESTAMP NOT NULL,
  completed_at TIMESTAMP NOT NULL,
  execution_time_seconds NUMERIC(10,2) NOT NULL
);
```

## GCP PostgreSQL Setup

### 1. Provisioning PostgreSQL on GCP
```bash
# Example command to provision a Cloud SQL PostgreSQL instance
gcloud sql instances create hsds-postgres \
  --database-version=POSTGRES_14 \
  --cpu=2 \
  --memory=4GB \
  --region=us-central1 \
  --storage-type=SSD \
  --storage-size=10GB \
  --availability-type=ZONAL \
  --backup-start-time=23:00 \
  --enable-point-in-time-recovery

# Create database
gcloud sql databases create hsds \
  --instance=hsds-postgres

# Create user
gcloud sql users create hsds-admin \
  --instance=hsds-postgres \
  --password=[PASSWORD]
```

### 2. Installing PostGIS Extension
```bash
# Connect to PostgreSQL instance
gcloud sql connect hsds-postgres --user=hsds-admin

# Create extension
CREATE EXTENSION IF NOT EXISTS postgis;
```

## Migration Implementation Strategy

### 1. Migration Order
To maintain referential integrity, tables must be migrated in this order:
1. organization
2. location
3. service
4. service_at_location  
5. address (consolidated)
6. phone

### 2. Python ETL Processing Flow
```python
# Simplified pseudocode for migration process
def migrate_table(source_table, target_table, transform_function, batch_size=100):
    # 1. Extract data from Snowflake
    source_data = extract_from_snowflake(source_table, batch_size)
    
    # 2. For translation tables, join with main table
    if needs_translation_data(target_table):
        translation_data = extract_from_snowflake(source_table + '_TRANSLATIONS', 
                                               filter="LOCALE='en' AND IS_CANONICAL=True",
                                               batch_size)
        combined_data = join_data(source_data, translation_data)
    else:
        combined_data = source_data
        
    # 3. Transform data according to mapping rules
    transformed_data = transform_function(combined_data)
    
    # 4. Load data to GCP PostgreSQL
    load_to_postgres(target_table, transformed_data)
    
    # 5. Log migration results
    log_migration_results(source_table, target_table, len(transformed_data))
```

### 3. Handling Missing Data
When merging translation tables, there may be cases where no canonical English translation exists. The migration process should:

1. Always prefer canonical English translations (IS_CANONICAL=True)
2. Fall back to any English translation if no canonical one exists
3. For fields without translations, leave as NULL but maintain the relationship structure
4. Log records with missing translations for potential manual follow-up

### 4. Foreign Key Resolution
The migration order described above ensures that referenced records exist before the referencing records are created:

1. Migrate organization first (self-referencing organization.parent_organization_id)
2. Migrate location next (references organization)
3. Migrate service (references organization)
4. Migrate service_at_location (references both service and location)
5. Migrate addresses (reference location)
6. Migrate phone (references multiple entities)

### 5. PostGIS for SERVICE_AREA
Although not in the initial migration, SERVICE_AREA will need special handling:

```sql
-- Example when implementing service_area with PostGIS
CREATE TABLE service_area (
  id CHAR(36) PRIMARY KEY,
  service_id CHAR(36) NOT NULL,
  service_area VARCHAR(255),
  description TEXT,
  extent GEOMETRY, -- PostGIS geometry type
  extent_type VARCHAR(50),
  last_modified TIMESTAMP,
  created TIMESTAMP,
  original_id VARCHAR(100),
  original_translations_id VARCHAR(100),
  FOREIGN KEY (service_id) REFERENCES service(id)
);

-- When loading data, transform JSON to PostGIS geometry:
INSERT INTO service_area(id, service_id, extent, extent_type)
VALUES ('...', '...', ST_GeomFromGeoJSON('{"type":"Polygon","coordinates":[...]}'), 'geojson');
```

## Enhanced Validation Strategy

1. **Pre-migration validation**: Check source data integrity and required fields  
2. **During migration validation**: Ensure foreign key constraints are satisfied
3. **Post-migration validation**: 
   - Verify record counts match expected values
   - Spot-check sample records for data fidelity
   - Test referential queries across the migrated tables

## Cost Optimization Strategies for GCP

1. **Right-sizing the instance**: Start with a smaller instance and scale up as needed
2. **Automated backup scheduling**: Configure automated backups during off-peak hours
3. **Connection pooling**: Implement connection pooling to reduce database connection overhead
4. **Read replicas**: Consider read replicas for reporting workloads if needed
5. **Resource monitoring**: Set up Cloud Monitoring to track resource usage and costs

## Next Steps

1. Provision PostgreSQL instance on GCP with PostGIS extension
2. Implement the database schema in PostgreSQL
3. Create ETL scripts for the 6 priority tables
4. Migrate 100 test records following the defined order
5. Validate the test migration
6. Iterate on any issues identified during testing
7. Implement schema for deferred tables without migration
8. Plan for complete migration once the test is successful
