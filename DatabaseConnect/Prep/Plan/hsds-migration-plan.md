# Snowflake to RDS Migration Plan: HSDS Schema Implementation

## Overview

This migration plan outlines how to transform the current Snowflake database (NORSE_STAGING.WA211) into an Amazon RDS implementation that closely follows the Human Services Data Specification (HSDS) schema. The plan prioritizes maintaining data integrity while simplifying the database structure where appropriate.

## Core Principles

1. **Direct Mapping**: Map existing Snowflake tables directly to HSDS core tables where possible
2. **Consolidation**: Merge translation tables into main tables with language-specific fields
3. **Standardization**: Normalize field types and sizes based on HSDS specifications
4. **Simplification**: Eliminate unused or sparse tables with minimal data
5. **Constraint Implementation**: Add proper constraints based on Dagster quality checks

## Table Mapping Overview

| HSDS Table | Snowflake Source Table(s) | Action |
|------------|---------------------------|--------|
| organization | ORGANIZATION + ORGANIZATION_TRANSLATIONS | Merge |
| program | PROGRAM + PROGRAM_TRANSLATIONS | Merge |
| service | SERVICE + SERVICE_TRANSLATIONS | Merge |
| location | LOCATION + LOCATION_TRANSLATIONS | Merge |
| service_at_location | SERVICE_AT_LOCATION + SERVICE_AT_LOCATION_TRANSLATIONS | Merge |
| taxonomy_term | TAXONOMY_TERM + TAXONOMY_TERM_TRANSLATIONS | Merge |
| service_taxonomy | ATTRIBUTE (where link_entity='service') | Transform |
| other_attribute | ATTRIBUTE (where link_entity≠'service') | Transform |
| contact | CONTACT + CONTACT_TRANSLATIONS | Merge |
| phone | PHONE + PHONE_TRANSLATIONS | Merge |
| physical_address | ADDRESS (where address_type='physical') | Transform |
| postal_address | ADDRESS (where address_type≠'physical') | Transform |
| schedule | SCHEDULE + SCHEDULE_TRANSLATIONS | Merge |
| funding | FUNDING + FUNDING_TRANSLATIONS | Merge |
| service_area | SERVICE_AREA + SERVICE_AREA_TRANSLATIONS | Merge |
| required_document | REQUIRED_DOCUMENT + REQUIRED_DOCUMENT_TRANSLATIONS | Merge |
| language | LANGUAGE + LANGUAGE_TRANSLATIONS | Merge |
| accessibility_for_disabilities | ACCESSIBILITY + ACCESSIBILITY_TRANSLATIONS | Merge |

## Detailed Field Mapping by Table

### 1. Organization Table

**RDS Table Name**: `organization`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | ORGANIZATION.ID | Direct mapping |
| name | VARCHAR(255) | ORGANIZATION.NAME | Direct mapping |
| alternate_name | VARCHAR(255) | ORGANIZATION.ALTERNATE_NAME | Direct mapping |
| description | TEXT | ORGANIZATION_TRANSLATIONS.DESCRIPTION (where IS_CANONICAL=True) | Extract canonical description |
| description_<lang> | TEXT | ORGANIZATION_TRANSLATIONS.DESCRIPTION (for each LOCALE) | Create language-specific columns |
| short_description | TEXT | ORGANIZATION_TRANSLATIONS.SHORT_DESCRIPTION (where IS_CANONICAL=True) | Extract canonical short description |
| short_description_<lang> | TEXT | ORGANIZATION_TRANSLATIONS.SHORT_DESCRIPTION (for each LOCALE) | Create language-specific columns |
| email | VARCHAR(255) | ORGANIZATION.EMAIL | Direct mapping |
| url | VARCHAR(255) | ORGANIZATION.WEBSITE | Direct mapping |
| tax_status | VARCHAR(255) | NULL | New field per HSDS |
| tax_id | VARCHAR(255) | From ORGANIZATION_IDENTIFIER.IDENTIFIER where IDENTIFIER_TYPE matches tax ID | Extract from identifiers |
| year_incorporated | CHAR(4) | ORGANIZATION.YEAR_INCORPORATED | Direct mapping |
| legal_status | VARCHAR(255) | ORGANIZATION.LEGAL_STATUS | Direct mapping |
| parent_organization_id | CHAR(36) | ORGANIZATION.PARENT_ORGANIZATION_ID | Direct mapping |
| last_modified | TIMESTAMP | ORGANIZATION.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | ORGANIZATION.CREATED | Convert to proper timestamp |

### 2. Program Table

**RDS Table Name**: `program`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | PROGRAM.ID | Direct mapping |
| organization_id | CHAR(36) | PROGRAM.ORGANIZATION_ID | Direct mapping |
| name | VARCHAR(255) | PROGRAM_TRANSLATIONS.NAME (where IS_CANONICAL=True) | Extract canonical name |
| name_<lang> | VARCHAR(255) | PROGRAM_TRANSLATIONS.NAME (for each LOCALE) | Create language-specific columns |
| alternate_name | VARCHAR(255) | PROGRAM_TRANSLATIONS.ALTERNATE_NAME (where IS_CANONICAL=True) | Extract canonical alternate name |
| alternate_name_<lang> | VARCHAR(255) | PROGRAM_TRANSLATIONS.ALTERNATE_NAME (for each LOCALE) | Create language-specific columns |
| description | TEXT | PROGRAM_TRANSLATIONS.DESCRIPTION (where IS_CANONICAL=True) | Extract canonical description |
| description_<lang> | TEXT | PROGRAM_TRANSLATIONS.DESCRIPTION (for each LOCALE) | Create language-specific columns |
| last_modified | TIMESTAMP | PROGRAM.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | PROGRAM.CREATED | Convert to proper timestamp |

### 3. Service Table

**RDS Table Name**: `service`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | SERVICE.ID | Direct mapping |
| organization_id | CHAR(36) | SERVICE.ORGANIZATION_ID | Direct mapping |
| program_id | CHAR(36) | SERVICE.PROGRAM_ID | Direct mapping |
| name | VARCHAR(255) | SERVICE_TRANSLATIONS.NAME (where IS_CANONICAL=True) | Extract canonical name |
| name_<lang> | VARCHAR(255) | SERVICE_TRANSLATIONS.NAME (for each LOCALE) | Create language-specific columns |
| alternate_name | VARCHAR(255) | SERVICE_TRANSLATIONS.ALTERNATE_NAME (where IS_CANONICAL=True) | Extract canonical alternate name |
| alternate_name_<lang> | VARCHAR(255) | SERVICE_TRANSLATIONS.ALTERNATE_NAME (for each LOCALE) | Create language-specific columns |
| description | TEXT | SERVICE_TRANSLATIONS.DESCRIPTION (where IS_CANONICAL=True) | Extract canonical description |
| description_<lang> | TEXT | SERVICE_TRANSLATIONS.DESCRIPTION (for each LOCALE) | Create language-specific columns |
| short_description | TEXT | SERVICE_TRANSLATIONS.SHORT_DESCRIPTION (where IS_CANONICAL=True) | Extract canonical short description |
| short_description_<lang> | TEXT | SERVICE_TRANSLATIONS.SHORT_DESCRIPTION (for each LOCALE) | Create language-specific columns |
| url | VARCHAR(255) | SERVICE.URL | Direct mapping |
| email | VARCHAR(255) | SERVICE.EMAIL | Direct mapping |
| status | VARCHAR(50) | SERVICE.STATUS | Direct mapping |
| interpretation_services | TEXT | SERVICE_TRANSLATIONS.INTERPRETATION_SERVICES (where IS_CANONICAL=True) | Extract canonical interpretation services |
| interpretation_services_<lang> | TEXT | SERVICE_TRANSLATIONS.INTERPRETATION_SERVICES (for each LOCALE) | Create language-specific columns |
| application_process | TEXT | SERVICE_TRANSLATIONS.APPLICATION_PROCESS (where IS_CANONICAL=True) | Extract canonical application process |
| application_process_<lang> | TEXT | SERVICE_TRANSLATIONS.APPLICATION_PROCESS (for each LOCALE) | Create language-specific columns |
| wait_time | VARCHAR(255) | NULL | New field per HSDS |
| fees_description | TEXT | SERVICE_TRANSLATIONS.FEES_DESCRIPTION (where IS_CANONICAL=True) | Extract canonical fees description |
| fees_description_<lang> | TEXT | SERVICE_TRANSLATIONS.FEES_DESCRIPTION (for each LOCALE) | Create language-specific columns |
| accreditations | TEXT | SERVICE_TRANSLATIONS.ACCREDITATIONS (where IS_CANONICAL=True) | Extract canonical accreditations |
| accreditations_<lang> | TEXT | SERVICE_TRANSLATIONS.ACCREDITATIONS (for each LOCALE) | Create language-specific columns |
| licenses | VARCHAR(255) | NULL | New field per HSDS |
| minimum_age | INT | SERVICE.MINIMUM_AGE | Direct mapping |
| maximum_age | INT | SERVICE.MAXIMUM_AGE | Direct mapping |
| eligibility_description | TEXT | SERVICE_TRANSLATIONS.ELIGIBILITY_DESCRIPTION (where IS_CANONICAL=True) | Extract canonical eligibility description |
| eligibility_description_<lang> | TEXT | SERVICE_TRANSLATIONS.ELIGIBILITY_DESCRIPTION (for each LOCALE) | Create language-specific columns |
| alert | TEXT | SERVICE_TRANSLATIONS.ALERT (where IS_CANONICAL=True) | Extract canonical alert |
| alert_<lang> | TEXT | SERVICE_TRANSLATIONS.ALERT (for each LOCALE) | Create language-specific columns |
| last_modified | TIMESTAMP | SERVICE.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | SERVICE.CREATED | Convert to proper timestamp |

### 4. Location Table

**RDS Table Name**: `location`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | LOCATION.ID | Direct mapping |
| organization_id | CHAR(36) | LOCATION.ORGANIZATION_ID | Direct mapping |
| name | VARCHAR(255) | LOCATION.NAME | Direct mapping |
| alternate_name | VARCHAR(255) | LOCATION.ALTERNATE_NAME | Direct mapping |
| description | TEXT | LOCATION_TRANSLATIONS.DESCRIPTION (where IS_CANONICAL=True) | Extract canonical description |
| description_<lang> | TEXT | LOCATION_TRANSLATIONS.DESCRIPTION (for each LOCALE) | Create language-specific columns |
| short_description | TEXT | LOCATION_TRANSLATIONS.SHORT_DESCRIPTION (where IS_CANONICAL=True) | Extract canonical short description |
| short_description_<lang> | TEXT | LOCATION_TRANSLATIONS.SHORT_DESCRIPTION (for each LOCALE) | Create language-specific columns |
| transportation | TEXT | LOCATION_TRANSLATIONS.TRANSPORTATION (where IS_CANONICAL=True) | Extract canonical transportation |
| transportation_<lang> | TEXT | LOCATION_TRANSLATIONS.TRANSPORTATION (for each LOCALE) | Create language-specific columns |
| latitude | DECIMAL(10,6) | LOCATION.LATITUDE | Direct mapping |
| longitude | DECIMAL(10,6) | LOCATION.LONGITUDE | Direct mapping |
| location_type | VARCHAR(50) | LOCATION.LOCATION_TYPE | Direct mapping |
| last_modified | TIMESTAMP | LOCATION.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | LOCATION.CREATED | Convert to proper timestamp |

### 5. Service At Location Table

**RDS Table Name**: `service_at_location`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | SERVICE_AT_LOCATION.ID | Direct mapping |
| service_id | CHAR(36) | SERVICE_AT_LOCATION.SERVICE_ID | Direct mapping |
| location_id | CHAR(36) | SERVICE_AT_LOCATION.LOCATION_ID | Direct mapping |
| description | TEXT | SERVICE_AT_LOCATION_TRANSLATIONS.DESCRIPTION (where IS_CANONICAL=True) | Extract canonical description |
| description_<lang> | TEXT | SERVICE_AT_LOCATION_TRANSLATIONS.DESCRIPTION (for each LOCALE) | Create language-specific columns |
| last_modified | TIMESTAMP | SERVICE_AT_LOCATION.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | SERVICE_AT_LOCATION.CREATED | Convert to proper timestamp |

### 6. Taxonomy Term Table

**RDS Table Name**: `taxonomy_term`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | TAXONOMY_TERM.ID | Direct mapping |
| term | VARCHAR(255) | TAXONOMY_TERM_TRANSLATIONS.NAME (where IS_CANONICAL=True) | Extract canonical name |
| term_<lang> | VARCHAR(255) | TAXONOMY_TERM_TRANSLATIONS.NAME (for each LOCALE) | Create language-specific columns |
| description | TEXT | TAXONOMY_TERM_TRANSLATIONS.DESCRIPTION (where IS_CANONICAL=True) | Extract canonical description |
| description_<lang> | TEXT | TAXONOMY_TERM_TRANSLATIONS.DESCRIPTION (for each LOCALE) | Create language-specific columns |
| parent_id | CHAR(36) | TAXONOMY_TERM.PARENT_ID | Direct mapping |
| taxonomy | VARCHAR(255) | Join with TAXONOMY table on TAXONOMY_ID | Include TAXONOMY.NAME |
| language | VARCHAR(10) | TAXONOMY_TERM_TRANSLATIONS.LANGUAGE (where IS_CANONICAL=True) | Extract canonical language |
| code | VARCHAR(50) | TAXONOMY_TERM.CODE | Direct mapping |
| last_modified | TIMESTAMP | TAXONOMY_TERM.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | TAXONOMY_TERM.CREATED | Convert to proper timestamp |

### 7. Service Taxonomy Table

**RDS Table Name**: `service_taxonomy`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | ATTRIBUTE.ID (where LINK_ENTITY='service') | Direct mapping |
| service_id | CHAR(36) | ATTRIBUTE.LINK_ID (where LINK_ENTITY='service') | Direct mapping |
| taxonomy_term_id | CHAR(36) | ATTRIBUTE.TAXONOMY_TERM_ID | Direct mapping |
| last_modified | TIMESTAMP | ATTRIBUTE.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | ATTRIBUTE.CREATED | Convert to proper timestamp |

### 8. Other Attribute Table

**RDS Table Name**: `other_attribute`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | ATTRIBUTE.ID (where LINK_ENTITY≠'service') | Direct mapping |
| link_id | CHAR(36) | ATTRIBUTE.LINK_ID | Direct mapping |
| link_type | VARCHAR(50) | ATTRIBUTE.LINK_ENTITY | Direct mapping |
| taxonomy_term_id | CHAR(36) | ATTRIBUTE.TAXONOMY_TERM_ID | Direct mapping |
| last_modified | TIMESTAMP | ATTRIBUTE.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | ATTRIBUTE.CREATED | Convert to proper timestamp |

### 9. Contact Table

**RDS Table Name**: `contact`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | CONTACT.ID | Direct mapping |
| organization_id | CHAR(36) | CONTACT.ORGANIZATION_ID | Direct mapping |
| service_id | CHAR(36) | CONTACT.SERVICE_ID | Direct mapping |
| service_at_location_id | CHAR(36) | CONTACT.SERVICE_AT_LOCATION_ID | Direct mapping |
| name | VARCHAR(255) | CONTACT.NAME | Direct mapping |
| title | VARCHAR(255) | CONTACT_TRANSLATIONS.TITLE (where IS_CANONICAL=True) | Extract canonical title |
| title_<lang> | VARCHAR(255) | CONTACT_TRANSLATIONS.TITLE (for each LOCALE) | Create language-specific columns |
| department | VARCHAR(255) | CONTACT_TRANSLATIONS.DEPARTMENT (where IS_CANONICAL=True) | Extract canonical department |
| department_<lang> | VARCHAR(255) | CONTACT_TRANSLATIONS.DEPARTMENT (for each LOCALE) | Create language-specific columns |
| email | VARCHAR(255) | CONTACT.EMAIL | Direct mapping |
| last_modified | TIMESTAMP | CONTACT.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | CONTACT.CREATED | Convert to proper timestamp |

### 10. Phone Table

**RDS Table Name**: `phone`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | PHONE.ID | Direct mapping |
| location_id | CHAR(36) | PHONE.LOCATION_ID | Direct mapping |
| service_id | CHAR(36) | PHONE.SERVICE_ID | Direct mapping |
| organization_id | CHAR(36) | PHONE.ORGANIZATION_ID | Direct mapping |
| contact_id | CHAR(36) | PHONE.CONTACT_ID | Direct mapping |
| service_at_location_id | CHAR(36) | PHONE.SERVICE_AT_LOCATION_ID | Direct mapping |
| number | VARCHAR(50) | PHONE.NUMBER | Direct mapping |
| extension | VARCHAR(20) | PHONE.EXTENSION | Direct mapping |
| type | VARCHAR(20) | PHONE.TYPE | Direct mapping |
| language | VARCHAR(10) | Join with LANGUAGE table on PHONE_ID | Extract language codes |
| description | TEXT | PHONE_TRANSLATIONS.DESCRIPTION (where IS_CANONICAL=True) | Extract canonical description |
| description_<lang> | TEXT | PHONE_TRANSLATIONS.DESCRIPTION (for each LOCALE) | Create language-specific columns |
| priority | INT | PHONE.PRIORITY | Direct mapping |
| last_modified | TIMESTAMP | PHONE.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | PHONE.CREATED | Convert to proper timestamp |

### 11. Physical Address Table

**RDS Table Name**: `physical_address`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | ADDRESS.ID (where ADDRESS_TYPE='physical') | Direct mapping |
| location_id | CHAR(36) | ADDRESS.LOCATION_ID | Direct mapping |
| attention | VARCHAR(255) | ADDRESS.ATTENTION | Direct mapping |
| address_1 | VARCHAR(255) | ADDRESS.ADDRESS_1 | Direct mapping |
| address_2 | VARCHAR(255) | ADDRESS.ADDRESS_2 | Direct mapping |
| city | VARCHAR(255) | ADDRESS.CITY | Direct mapping |
| region | VARCHAR(255) | ADDRESS.REGION | Direct mapping |
| state_province | VARCHAR(100) | ADDRESS.STATE_PROVINCE | Direct mapping |
| postal_code | VARCHAR(20) | ADDRESS.POSTAL_CODE | Direct mapping |
| country | CHAR(2) | ADDRESS.COUNTRY | Ensure 2-letter ISO code |
| last_modified | TIMESTAMP | ADDRESS.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | ADDRESS.CREATED | Convert to proper timestamp |

### 12. Postal Address Table

**RDS Table Name**: `postal_address`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | ADDRESS.ID (where ADDRESS_TYPE≠'physical') | Direct mapping |
| location_id | CHAR(36) | ADDRESS.LOCATION_ID | Direct mapping |
| attention | VARCHAR(255) | ADDRESS.ATTENTION | Direct mapping |
| address_1 | VARCHAR(255) | ADDRESS.ADDRESS_1 | Direct mapping |
| address_2 | VARCHAR(255) | ADDRESS.ADDRESS_2 | Direct mapping |
| city | VARCHAR(255) | ADDRESS.CITY | Direct mapping |
| region | VARCHAR(255) | ADDRESS.REGION | Direct mapping |
| state_province | VARCHAR(100) | ADDRESS.STATE_PROVINCE | Direct mapping |
| postal_code | VARCHAR(20) | ADDRESS.POSTAL_CODE | Direct mapping |
| country | CHAR(2) | ADDRESS.COUNTRY | Ensure 2-letter ISO code |
| last_modified | TIMESTAMP | ADDRESS.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | ADDRESS.CREATED | Convert to proper timestamp |

### 13. Schedule Table

**RDS Table Name**: `schedule`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | SCHEDULE.ID | Direct mapping |
| service_id | CHAR(36) | SCHEDULE.SERVICE_ID | Direct mapping |
| location_id | CHAR(36) | SCHEDULE.LOCATION_ID | Direct mapping |
| service_at_location_id | CHAR(36) | SCHEDULE.SERVICE_AT_LOCATION_ID | Direct mapping |
| valid_from | DATE | SCHEDULE.VALID_FROM | Convert to proper date |
| valid_to | DATE | SCHEDULE.VALID_TO | Convert to proper date |
| dtstart | DATE | SCHEDULE.DTSTART | Convert to proper date |
| freq | VARCHAR(20) | SCHEDULE.FREQ | Direct mapping |
| interval | INT | SCHEDULE.INTERVAL | Direct mapping |
| byday | VARCHAR(50) | SCHEDULE.BYDAY | Direct mapping |
| byweekno | VARCHAR(50) | SCHEDULE.BYWEEKNO | Direct mapping |
| bymonthday | VARCHAR(50) | SCHEDULE.BYMONTHDAY | Direct mapping |
| byyearday | VARCHAR(50) | SCHEDULE.BYYEARDAY | Direct mapping |
| description | TEXT | SCHEDULE_TRANSLATIONS.DESCRIPTION (where IS_CANONICAL=True) | Extract canonical description |
| description_<lang> | TEXT | SCHEDULE_TRANSLATIONS.DESCRIPTION (for each LOCALE) | Create language-specific columns |
| opens_at | TIME | SCHEDULE.OPENS_AT | Convert to proper time |
| closes_at | TIME | SCHEDULE.CLOSES_AT | Convert to proper time |
| timezone | VARCHAR(50) | SCHEDULE.TIMEZONE | Convert to proper timezone format |
| attending_type | VARCHAR(50) | SCHEDULE.ATTENDING_TYPE | Direct mapping |
| priority | INT | SCHEDULE.PRIORITY | Direct mapping |
| last_modified | TIMESTAMP | SCHEDULE.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | SCHEDULE.CREATED | Convert to proper timestamp |

### 14. Funding Table

**RDS Table Name**: `funding`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | FUNDING.ID | Direct mapping |
| organization_id | CHAR(36) | FUNDING.ORGANIZATION_ID | Direct mapping |
| service_id | CHAR(36) | FUNDING.SERVICE_ID | Direct mapping |
| source | TEXT | FUNDING_TRANSLATIONS.SOURCE (where IS_CANONICAL=True) | Extract canonical source |
| source_<lang> | TEXT | FUNDING_TRANSLATIONS.SOURCE (for each LOCALE) | Create language-specific columns |
| last_modified | TIMESTAMP | FUNDING.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | FUNDING.CREATED | Convert to proper timestamp |

### 15. Service Area Table

**RDS Table Name**: `service_area`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | SERVICE_AREA.ID | Direct mapping |
| service_id | CHAR(36) | SERVICE_AREA.SERVICE_ID | Direct mapping |
| service_area | VARCHAR(255) | SERVICE_AREA.NAME | Direct mapping |
| description | TEXT | SERVICE_AREA_TRANSLATIONS.DESCRIPTION (where IS_CANONICAL=True) | Extract canonical description |
| description_<lang> | TEXT | SERVICE_AREA_TRANSLATIONS.DESCRIPTION (for each LOCALE) | Create language-specific columns |
| extent | TEXT | SERVICE_AREA.EXTENT | Direct mapping, consider using PostGIS geometry type |
| extent_type | VARCHAR(50) | SERVICE_AREA.EXTENT_TYPE | Direct mapping |
| last_modified | TIMESTAMP | SERVICE_AREA.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | SERVICE_AREA.CREATED | Convert to proper timestamp |

### 16. Required Document Table

**RDS Table Name**: `required_document`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | REQUIRED_DOCUMENT.ID | Direct mapping |
| service_id | CHAR(36) | REQUIRED_DOCUMENT.SERVICE_ID | Direct mapping |
| document | TEXT | REQUIRED_DOCUMENT_TRANSLATIONS.DOCUMENT (where IS_CANONICAL=True) | Extract canonical document |
| document_<lang> | TEXT | REQUIRED_DOCUMENT_TRANSLATIONS.DOCUMENT (for each LOCALE) | Create language-specific columns |
| last_modified | TIMESTAMP | REQUIRED_DOCUMENT.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | REQUIRED_DOCUMENT.CREATED | Convert to proper timestamp |

### 17. Language Table

**RDS Table Name**: `language`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | LANGUAGE.ID | Direct mapping |
| service_id | CHAR(36) | LANGUAGE.SERVICE_ID | Direct mapping |
| location_id | CHAR(36) | LANGUAGE.LOCATION_ID | Direct mapping |
| phone_id | CHAR(36) | LANGUAGE.PHONE_ID | Direct mapping |
| language | VARCHAR(10) | LANGUAGE.CODE | Ensure proper ISO language code |
| name | VARCHAR(100) | LANGUAGE_TRANSLATIONS.NAME (where IS_CANONICAL=True) | Extract canonical name |
| name_<lang> | VARCHAR(100) | LANGUAGE_TRANSLATIONS.NAME (for each LOCALE) | Create language-specific columns |
| note | TEXT | LANGUAGE_TRANSLATIONS.NOTE (where IS_CANONICAL=True) | Extract canonical note |
| note_<lang> | TEXT | LANGUAGE_TRANSLATIONS.NOTE (for each LOCALE) | Create language-specific columns |
| last_modified | TIMESTAMP | LANGUAGE.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | LANGUAGE.CREATED | Convert to proper timestamp |

### 18. Accessibility For Disabilities Table

**RDS Table Name**: `accessibility_for_disabilities`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic |
|------------|---------------|-----------------|---------------------|
| id | CHAR(36) | ACCESSIBILITY.ID | Direct mapping |
| location_id | CHAR(36) | ACCESSIBILITY.LOCATION_ID | Direct mapping |
| accessibility | TEXT | ACCESSIBILITY_TRANSLATIONS.DESCRIPTION (where IS_CANONICAL=True) | Extract canonical description |
| accessibility_<lang> | TEXT | ACCESSIBILITY_TRANSLATIONS.DESCRIPTION (for each LOCALE) | Create language-specific columns |
| details | TEXT | ACCESSIBILITY_TRANSLATIONS.DETAILS (where IS_CANONICAL=True) | Extract canonical details |
| details_<lang> | TEXT | ACCESSIBILITY_TRANSLATIONS.DETAILS (for each LOCALE) | Create language-specific columns |
| url | VARCHAR(255) | ACCESSIBILITY.URL | Direct mapping |
| last_modified | TIMESTAMP | ACCESSIBILITY.LAST_MODIFIED | Convert to proper timestamp |
| created | TIMESTAMP | ACCESSIBILITY.CREATED | Convert to proper timestamp |

## Technical Implementation Considerations

### 1. Handling Translations

Rather than maintaining separate translation tables, we'll extend the core tables with language-specific fields following the pattern:
- `field_name` for the canonical (default language) content
- `field_name_<lang>` for translations, where `<lang>` is the language code from LOCALE field

This simplifies queries while maintaining the multilingual capabilities.

### 2. Data Type Sizing

- **ID fields**: Fixed at CHAR(36) for UUID values
- **Name fields**: VARCHAR(255) should be sufficient for most names
- **Description fields**: Use TEXT type to accommodate variable-length content
- **Email/URL fields**: VARCHAR(255) should be adequate
- **Code fields**: VARCHAR(50) for taxonomy codes
- **Numeric fields**: Follow source field precision where available

### 3. Constraint Implementation

Based on Dagster quality checks, add these constraints to RDS:

- **Primary Keys**: All ID fields
- **Foreign Keys**: All reference fields (e.g., organization_id, service_id)
- **Not Null**: Fields marked as required in HSDS or critical in Dagster checks
- **Check Constraints**: For fields with enumerated values (e.g., service.status)
- **Unique Constraints**: For fields or combinations marked as unique in HSDS

### 4. Geographic Data

For SERVICE_AREA.EXTENT which contains GeoJSON:
- If using PostgreSQL RDS, leverage PostGIS extension to store as proper geometry types
- If using another RDS engine, consider storing as TEXT with application-level parsing

### 5. Migration Implementation Strategy

1. **Extract and Transform**: 
   - Create ETL scripts to extract data from Snowflake
   - Transform according to mapping rules
   - Load into staging tables

2. **Validation**:
   - Apply Dagster validation rules to staging data
   - Correct data issues before final load

3. **Production Load**:
   - Create properly constrained schema in RDS
   - Load data from staging tables
   - Verify referential integrity

4. **Cut-over Plan**:
   - Freeze writes to Snowflake
   - Perform final data sync
   - Redirect application to new RDS database

## Sample SQL Implementations

### Create Tables SQL (PostgreSQL Sample)

```sql
-- Organization Table
CREATE TABLE organization (
  id CHAR(36) PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  alternate_name VARCHAR(255),
  description TEXT,
  description_es TEXT,  -- Example language column
  email VARCHAR(255),
  url VARCHAR(255),
  tax_status VARCHAR(255),
  tax_id VARCHAR(255),
  year_incorporated CHAR(4),
  legal_status VARCHAR(255),
  parent_organization_id CHAR(36),
  FOREIGN KEY (parent_organization_id) REFERENCES organization(id),
  last_modified TIMESTAMP,
  created TIMESTAMP
);

-- Service Table
CREATE TABLE service (
  id CHAR(36) PRIMARY KEY,
  organization_id CHAR(36) NOT NULL,
  program_id CHAR(36),
  name VARCHAR(255) NOT NULL,
  alternate_name VARCHAR(255),
  description TEXT,
  url VARCHAR(255),
  email VARCHAR(255),
  status VARCHAR(50) NOT NULL CHECK (status IN ('active', 'inactive', 'defunct', 'temporarily closed')),
  FOREIGN KEY (organization_id) REFERENCES organization(id),
  FOREIGN KEY (program_id) REFERENCES program(id),
  last_modified TIMESTAMP,
  created TIMESTAMP
);

-- Similar CREATE TABLE statements for all other tables...
```

### Example Data Migration SQL (PostgreSQL)

```sql
-- Migration for organization table
INSERT INTO organization (
  id, name, alternate_name, description, description_es, email, url, 
  tax_id, year_incorporated, legal_status, parent_organization_id,
  last_modified, created
)
SELECT 
  o.id, 
  o.name, 
  o.alternate_name,
  (SELECT description FROM organization_translations WHERE organization_id = o.id AND is_canonical = TRUE LIMIT 1),
  (SELECT description FROM organization_translations WHERE organization_id = o.id AND locale = 'es' LIMIT 1),
  o.email,
  o.website,
  (SELECT identifier FROM organization_identifier WHERE organization_id = o.id AND identifier_type = 'US-EIN' LIMIT 1),
  o.year_incorporated,
  o.legal_status,
  o.parent_organization_id,
  CAST(o.last_modified AS TIMESTAMP),
  CAST(o.created AS TIMESTAMP)
FROM organization o;

-- Additional migration statements for other tables...
```
