# Updated Migration Plan: Snowflake to Supabase Implementation

## Revised Core Principles

1. **Focus on Critical Tables**: Prioritize migration of the most critical tables first: Organization, Service, Location, Service_at_Location, Address, and Phone
2. **English-Only Initially**: Support only English translations in the initial implementation
3. **Direct Mapping with Traceability**: Maintain traceability to original Snowflake records through original ID fields
4. **PostGIS Support**: Utilize Supabase's built-in PostGIS extension for geographic data (particularly for SERVICE_AREA)
5. **Referential Integrity**: Implement proper foreign key constraints for the core tables
6. **Small Test Migration**: Start with 100 records migration to validate the approach
7. **Cost Optimization**: Utilize Supabase's pricing tier appropriate for the dataset size
8. **Simplified Address Model**: Use a single address table with an address_type field instead of separate physical and postal address tables
9. **Leverage Supabase Features**: Utilize Row-Level Security (RLS), built-in authentication, and Realtime features

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

**Supabase Table Name**: `organization`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic | Traceability Fields |
|------------|---------------|-----------------|---------------------|-------------------|
| id | UUID | ORGANIZATION.ID | Convert to UUID format | |
| name | VARCHAR(255) | ORGANIZATION.NAME | Direct mapping | |
| alternate_name | VARCHAR(255) | ORGANIZATION.ALTERNATE_NAME | Direct mapping | |
| description | TEXT | ORGANIZATION_TRANSLATIONS.DESCRIPTION (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English description | |
| email | VARCHAR(255) | ORGANIZATION.EMAIL | Direct mapping | |
| url | VARCHAR(255) | ORGANIZATION.WEBSITE | Direct mapping | |
| tax_status | VARCHAR(255) | NULL | New field per HSDS | |
| tax_id | VARCHAR(255) | From ORGANIZATION_IDENTIFIER.IDENTIFIER where IDENTIFIER_TYPE='US-EIN' | Extract from identifiers | |
| year_incorporated | CHAR(4) | ORGANIZATION.YEAR_INCORPORATED | Direct mapping | |
| legal_status | VARCHAR(255) | ORGANIZATION.LEGAL_STATUS | Direct mapping | |
| parent_organization_id | UUID | ORGANIZATION.PARENT_ORGANIZATION_ID | Convert to UUID format | |
| last_modified | TIMESTAMPTZ | ORGANIZATION.LAST_MODIFIED | Convert to proper timestamp with timezone | |
| created | TIMESTAMPTZ | ORGANIZATION.CREATED | Convert to proper timestamp with timezone | |
| original_id | VARCHAR(100) | ORGANIZATION.ID | Store original Snowflake ID | New tracking field |
| original_translations_id | VARCHAR(100) | ORGANIZATION_TRANSLATIONS.ID | Store original translation ID | New tracking field |

### 2. Service Table

**Supabase Table Name**: `service`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic | Traceability Fields |
|------------|---------------|-----------------|---------------------|-------------------|
| id | UUID | SERVICE.ID | Convert to UUID format | |
| organization_id | UUID | SERVICE.ORGANIZATION_ID | Convert to UUID format | |
| program_id | UUID | SERVICE.PROGRAM_ID | Convert to UUID format | |
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
| last_modified | TIMESTAMPTZ | SERVICE.LAST_MODIFIED | Convert to proper timestamp with timezone | |
| created | TIMESTAMPTZ | SERVICE.CREATED | Convert to proper timestamp with timezone | |
| original_id | VARCHAR(100) | SERVICE.ID | Store original Snowflake ID | New tracking field |
| original_translations_id | VARCHAR(100) | SERVICE_TRANSLATIONS.ID | Store original translation ID | New tracking field |

### 3. Location Table

**Supabase Table Name**: `location`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic | Traceability Fields |
|------------|---------------|-----------------|---------------------|-------------------|
| id | UUID | LOCATION.ID | Convert to UUID format | |
| organization_id | UUID | LOCATION.ORGANIZATION_ID | Convert to UUID format | |
| name | VARCHAR(255) | LOCATION.NAME | Direct mapping | |
| alternate_name | VARCHAR(255) | LOCATION.ALTERNATE_NAME | Direct mapping | |
| description | TEXT | LOCATION_TRANSLATIONS.DESCRIPTION (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English description | |
| short_description | TEXT | LOCATION_TRANSLATIONS.SHORT_DESCRIPTION (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English short description | |
| transportation | TEXT | LOCATION_TRANSLATIONS.TRANSPORTATION (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English transportation | |
| latitude | DECIMAL(10,6) | LOCATION.LATITUDE | Direct mapping | |
| longitude | DECIMAL(10,6) | LOCATION.LONGITUDE | Direct mapping | |
| location_type | VARCHAR(50) | LOCATION.LOCATION_TYPE | Direct mapping | |
| last_modified | TIMESTAMPTZ | LOCATION.LAST_MODIFIED | Convert to proper timestamp with timezone | |
| created | TIMESTAMPTZ | LOCATION.CREATED | Convert to proper timestamp with timezone | |
| original_id | VARCHAR(100) | LOCATION.ID | Store original Snowflake ID | New tracking field |
| original_translations_id | VARCHAR(100) | LOCATION_TRANSLATIONS.ID | Store original translation ID | New tracking field |

### 4. Service At Location Table

**Supabase Table Name**: `service_at_location`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic | Traceability Fields |
|------------|---------------|-----------------|---------------------|-------------------|
| id | UUID | SERVICE_AT_LOCATION.ID | Convert to UUID format | |
| service_id | UUID | SERVICE_AT_LOCATION.SERVICE_ID | Convert to UUID format | |
| location_id | UUID | SERVICE_AT_LOCATION.LOCATION_ID | Convert to UUID format | |
| description | TEXT | SERVICE_AT_LOCATION_TRANSLATIONS.DESCRIPTION (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English description | |
| last_modified | TIMESTAMPTZ | SERVICE_AT_LOCATION.LAST_MODIFIED | Convert to proper timestamp with timezone | |
| created | TIMESTAMPTZ | SERVICE_AT_LOCATION.CREATED | Convert to proper timestamp with timezone | |
| original_id | VARCHAR(100) | SERVICE_AT_LOCATION.ID | Store original Snowflake ID | New tracking field |
| original_translations_id | VARCHAR(100) | SERVICE_AT_LOCATION_TRANSLATIONS.ID | Store original translation ID | New tracking field |

### 5. Consolidated Address Table

**Supabase Table Name**: `address`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic | Traceability Fields |
|------------|---------------|-----------------|---------------------|-------------------|
| id | UUID | ADDRESS.ID | Convert to UUID format | |
| location_id | UUID | ADDRESS.LOCATION_ID | Convert to UUID format | |
| attention | VARCHAR(255) | ADDRESS.ATTENTION | Direct mapping | |
| address_1 | VARCHAR(255) | ADDRESS.ADDRESS_1 | Direct mapping | |
| address_2 | VARCHAR(255) | ADDRESS.ADDRESS_2 | Direct mapping | |
| city | VARCHAR(255) | ADDRESS.CITY | Direct mapping | |
| region | VARCHAR(255) | ADDRESS.REGION | Direct mapping | |
| state_province | VARCHAR(100) | ADDRESS.STATE_PROVINCE | Direct mapping | |
| postal_code | VARCHAR(20) | ADDRESS.POSTAL_CODE | Direct mapping | |
| country | CHAR(2) | ADDRESS.COUNTRY | Ensure 2-letter ISO code | |
| address_type | VARCHAR(20) | ADDRESS.ADDRESS_TYPE | Direct mapping | Retained original field |
| last_modified | TIMESTAMPTZ | ADDRESS.LAST_MODIFIED | Convert to proper timestamp with timezone | |
| created | TIMESTAMPTZ | ADDRESS.CREATED | Convert to proper timestamp with timezone | |
| original_id | VARCHAR(100) | ADDRESS.ID | Store original Snowflake ID | New tracking field |

### 6. Phone Table

**Supabase Table Name**: `phone`

| HSDS Field | Type (Format) | Snowflake Source | Transformation Logic | Traceability Fields |
|------------|---------------|-----------------|---------------------|-------------------|
| id | UUID | PHONE.ID | Convert to UUID format | |
| location_id | UUID | PHONE.LOCATION_ID | Convert to UUID format | |
| service_id | UUID | PHONE.SERVICE_ID | Convert to UUID format | |
| organization_id | UUID | PHONE.ORGANIZATION_ID | Convert to UUID format | |
| contact_id | UUID | PHONE.CONTACT_ID | Convert to UUID format | |
| service_at_location_id | UUID | PHONE.SERVICE_AT_LOCATION_ID | Convert to UUID format | |
| number | VARCHAR(50) | PHONE.NUMBER | Direct mapping | |
| extension | VARCHAR(20) | PHONE.EXTENSION | Direct mapping | |
| type | VARCHAR(20) | PHONE.TYPE | Direct mapping | |
| language | VARCHAR(10) | 'en' | Default to English | |
| description | TEXT | PHONE_TRANSLATIONS.DESCRIPTION (where LOCALE='en' and IS_CANONICAL=True) | Extract canonical English description | |
| priority | INT | PHONE.PRIORITY | Direct mapping | |
| last_modified | TIMESTAMPTZ | PHONE.LAST_MODIFIED | Convert to proper timestamp with timezone | |
| created | TIMESTAMPTZ | PHONE.CREATED | Convert to proper timestamp with timezone | |
| original_id | VARCHAR(100) | PHONE.ID | Store original Snowflake ID | New tracking field |
| original_translations_id | VARCHAR(100) | PHONE_TRANSLATIONS.ID | Store original translation ID | New tracking field |

## Supabase Schema with PostGIS Support

```sql
-- Create organization table
CREATE TABLE organization (
  id UUID PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  alternate_name VARCHAR(255),
  description TEXT,
  email VARCHAR(255),
  url VARCHAR(255),
  tax_status VARCHAR(255),
  tax_id VARCHAR(255),
  year_incorporated CHAR(4),
  legal_status VARCHAR(255),
  parent_organization_id UUID,
  last_modified TIMESTAMPTZ,
  created TIMESTAMPTZ,
  original_id VARCHAR(100),
  original_translations_id VARCHAR(100),
  FOREIGN KEY (parent_organization_id) REFERENCES organization(id)
);

-- Create service table
CREATE TABLE service (
  id UUID PRIMARY KEY,
  organization_id UUID NOT NULL,
  program_id UUID,
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
  last_modified TIMESTAMPTZ,
  created TIMESTAMPTZ,
  original_id VARCHAR(100),
  original_translations_id VARCHAR(100),
  FOREIGN KEY (organization_id) REFERENCES organization(id)
);

-- Create location table
CREATE TABLE location (
  id UUID PRIMARY KEY,
  organization_id UUID NOT NULL,
  name VARCHAR(255),
  alternate_name VARCHAR(255),
  description TEXT,
  short_description TEXT,
  transportation TEXT,
  latitude DECIMAL(10,6),
  longitude DECIMAL(10,6),
  location_type VARCHAR(50),
  last_modified TIMESTAMPTZ,
  created TIMESTAMPTZ,
  original_id VARCHAR(100),
  original_translations_id VARCHAR(100),
  FOREIGN KEY (organization_id) REFERENCES organization(id)
);

-- Create service_at_location table
CREATE TABLE service_at_location (
  id UUID PRIMARY KEY,
  service_id UUID NOT NULL,
  location_id UUID NOT NULL,
  description TEXT,
  last_modified TIMESTAMPTZ,
  created TIMESTAMPTZ,
  original_id VARCHAR(100),
  original_translations_id VARCHAR(100),
  FOREIGN KEY (service_id) REFERENCES service(id),
  FOREIGN KEY (location_id) REFERENCES location(id)
);

-- Create consolidated address table
CREATE TABLE address (
  id UUID PRIMARY KEY,
  location_id UUID NOT NULL,
  attention VARCHAR(255),
  address_1 VARCHAR(255) NOT NULL,
  address_2 VARCHAR(255),
  city VARCHAR(255) NOT NULL,
  region VARCHAR(255),
  state_province VARCHAR(100) NOT NULL,
  postal_code VARCHAR(20) NOT NULL,
  country CHAR(2) NOT NULL,
  address_type VARCHAR(20) NOT NULL,
  last_modified TIMESTAMPTZ,
  created TIMESTAMPTZ,
  original_id VARCHAR(100),
  FOREIGN KEY (location_id) REFERENCES location(id)
);

-- Create phone table
CREATE TABLE phone (
  id UUID PRIMARY KEY,
  location_id UUID,
  service_id UUID,
  organization_id UUID,
  contact_id UUID,
  service_at_location_id UUID,
  number VARCHAR(50) NOT NULL,
  extension VARCHAR(20),
  type VARCHAR(20),
  language VARCHAR(10) DEFAULT 'en',
  description TEXT,
  priority INT,
  last_modified TIMESTAMPTZ,
  created TIMESTAMPTZ,
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
  started_at TIMESTAMPTZ NOT NULL,
  completed_at TIMESTAMPTZ NOT NULL,
  execution_time_seconds NUMERIC(10,2) NOT NULL
);
```

## Supabase-Specific Setup

### 1. Setting up Supabase
```bash
# Install Supabase CLI if not already installed
npm install -g supabase

# Login to Supabase
supabase login

# Initialize new Supabase project
supabase init

# Start Supabase local development
supabase start

# Link to your Supabase project
supabase link --project-ref your-project-ref
```

### 2. Leveraging Supabase Features

#### Row-Level Security (RLS)
```sql
-- Example RLS policy for organization table
ALTER TABLE organization ENABLE ROW LEVEL SECURITY;

-- Create policy for authenticated users
CREATE POLICY "Authenticated users can view organizations" 
ON organization FOR SELECT 
TO authenticated 
USING (true);

-- Create policy for specific roles (admin)
CREATE POLICY "Only admins can update organizations" 
ON organization FOR UPDATE 
TO authenticated 
USING (auth.uid() IN (
  SELECT user_id FROM admin_users
));
```

#### TypeScript Type Generation
```bash
# Generate TypeScript types for your Supabase tables
supabase gen types typescript --local > src/types/supabase.ts
```

## Migration Implementation Strategy with TypeScript

### 1. Migration Order
To maintain referential integrity, tables must be migrated in this order:
1. organization
2. location
3. service
4. service_at_location  
5. address (consolidated)
6. phone

### 2. TypeScript ETL Processing Flow

```typescript
// src/migration/index.ts
import { createClient } from '@supabase/supabase-js';
import { SnowflakeService } from './snowflake-service';

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

const snowflake = new SnowflakeService({
  account: process.env.SNOWFLAKE_ACCOUNT,
  username: process.env.SNOWFLAKE_USERNAME,
  password: process.env.SNOWFLAKE_PASSWORD,
  database: process.env.SNOWFLAKE_DATABASE,
  schema: process.env.SNOWFLAKE_SCHEMA,
  warehouse: process.env.SNOWFLAKE_WAREHOUSE,
});

// Migration function for a table
async function migrateTable(
  sourceTable: string, 
  targetTable: string, 
  transformFunction: Function, 
  batchSize: number = 100
) {
  const startTime = new Date();
  let successCount = 0;
  let failureCount = 0;
  let errorMessages = [];
  
  try {
    // 1. Extract data from Snowflake
    const sourceData = await snowflake.query(`SELECT * FROM ${sourceTable} LIMIT ${batchSize}`);
    
    // 2. For translation tables, join with main table
    let combinedData;
    if (needsTranslationData(targetTable)) {
      const translationData = await snowflake.query(
        `SELECT * FROM ${sourceTable}_TRANSLATIONS WHERE LOCALE='en' AND IS_CANONICAL=True LIMIT ${batchSize}`
      );
      combinedData = joinData(sourceData, translationData);
    } else {
      combinedData = sourceData;
    }
    
    // 3. Transform data according to mapping rules
    const transformedData = transformFunction(combinedData);
    
    // 4. Load data to Supabase
    const { data, error } = await supabase
      .from(targetTable)
      .insert(transformedData);
      
    if (error) {
      throw error;
    }
    
    successCount = transformedData.length;
  } catch (error) {
    failureCount = batchSize;
    errorMessages.push(error.message);
    console.error(`Error migrating ${sourceTable} to ${targetTable}:`, error);
  }
  
  // 5. Log migration results
  const endTime = new Date();
  const executionTime = (endTime.getTime() - startTime.getTime()) / 1000;
  
  await supabase.from('migration_log').insert({
    source_table: sourceTable,
    target_table: targetTable,
    records_migrated: successCount + failureCount,
    success_count: successCount,
    failure_count: failureCount,
    error_messages: errorMessages.join('\n'),
    started_at: startTime.toISOString(),
    completed_at: endTime.toISOString(),
    execution_time_seconds: executionTime
  });
  
  return {
    success: successCount,
    failure: failureCount,
    errors: errorMessages
  };
}

// Helper functions
function needsTranslationData(targetTable: string): boolean {
  const tablesWithTranslations = [
    'organization', 'service', 'location', 
    'service_at_location', 'phone'
  ];
  return tablesWithTranslations.includes(targetTable);
}

function joinData(mainData: any[], translationData: any[]): any[] {
  // Join logic here, using Map for efficient lookups
  const translationMap = new Map();
  
  translationData.forEach(translation => {
    translationMap.set(translation.id, translation);
  });
  
  return mainData.map(item => {
    const translation = translationMap.get(item.id);
    return { ...item, translation };
  });
}

// Transform functions for each table
const transformOrganization = (data: any[]) => {
  return data.map(item => {
    const translation = item.translation || {};
    
    return {
      id: parseUUID(item.id),
      name: item.name,
      alternate_name: item.alternate_name,
      description: translation.description,
      email: item.email,
      url: item.website,
      tax_status: null,
      tax_id: null, // Extract from identifiers if available
      year_incorporated: item.year_incorporated,
      legal_status: item.legal_status,
      parent_organization_id: parseUUID(item.parent_organization_id),
      last_modified: new Date(item.last_modified).toISOString(),
      created: new Date(item.created).toISOString(),
      original_id: item.id,
      original_translations_id: translation.id
    };
  });
};

// Helper for UUID conversion
function parseUUID(id: string): string {
  // Convert string ID to proper UUID format if needed
  if (!id) return null;
  
  // If already in UUID format, return as is
  if (/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)) {
    return id;
  }
  
  // Otherwise, generate a UUID using a hash of the original ID for consistency
  // This ensures the same ID always maps to the same UUID
  return generateConsistentUUID(id);
}

function generateConsistentUUID(input: string): string {
  // Implementation of a hashing function to convert string to UUID
  // This is simplified - in production use a robust UUID generator
  // or hash function that ensures consistency
  const crypto = require('crypto');
  const hash = crypto.createHash('md5').update(input).digest('hex');
  
  // Format as UUID v5 (name-based)
  return [
    hash.substr(0, 8),
    hash.substr(8, 4),
    '5' + hash.substr(13, 3), // Version 5
    '8' + hash.substr(17, 3), // Variant 8
    hash.substr(20, 12)
  ].join('-');
}

// Main migration function
export async function migrateData() {
  // Migrate in order to maintain referential integrity
  await migrateTable('ORGANIZATION', 'organization', transformOrganization);
  // Implement other transform functions and migration calls
}
```

### 3. Handling Missing Data and Translation Tables
Looking at the Snowflake sample data, we see multiple translation tables with various languages. The migration process should:

1. Always prefer canonical English translations (IS_CANONICAL=True)
2. Fall back to any English translation if no canonical one exists
3. For fields without English translations, leave as NULL but maintain the relationship structure
4. Log records with missing translations for potential manual follow-up

```typescript
// Example of translation handling function
function findBestTranslation(translations: any[], preferredLocale = 'en'): any {
  if (!translations || translations.length === 0) return null;
  
  // First try to find a canonical translation in the preferred locale
  const canonicalPreferred = translations.find(t => 
    t.LOCALE?.toLowerCase() === preferredLocale.toLowerCase() && t.IS_CANONICAL === 'True'
  );
  if (canonicalPreferred) return canonicalPreferred;
  
  // Then any translation in the preferred locale
  const anyPreferred = translations.find(t => 
    t.LOCALE?.toLowerCase() === preferredLocale.toLowerCase()
  );
  if (anyPreferred) return anyPreferred;
  
  // Then any canonical translation
  const anyCanonical = translations.find(t => t.IS_CANONICAL === 'True');
  if (anyCanonical) return anyCanonical;
  
  // Finally, just take the first one
  return translations[0];
}

// Usage in transformation
const organizationWithTranslation = organizations.map(org => {
  const translations = organizationTranslations.filter(t => t.ORGANIZATION_ID === org.ID);
  const bestTranslation = findBestTranslation(translations);
  
  return {
    ...org,
    description: bestTranslation?.DESCRIPTION || null,
    // other translated fields
    original_translations_id: bestTranslation?.ID || null
  };
});
```

### 4. UUID Conversion
Since Supabase typically uses UUIDs but Snowflake might be using different ID formats:

1. Create a consistent algorithm to convert source IDs to UUIDs
2. Ensure the same source ID always converts to the same UUID
3. Store the original ID in the traceability field

### 5. PostGIS for SERVICE_AREA
Supabase supports PostGIS out of the box:

```typescript
// Example in TypeScript for transforming GeoJSON to PostGIS
const transformServiceArea = (data: any[]) => {
  return data.map(item => {
    return {
      id: parseUUID(item.id),
      service_id: parseUUID(item.service_id),
      service_area: item.service_area,
      description: item.translation?.description,
      // Convert GeoJSON to PostGIS geometry
      extent: item.extent, // Supabase handles the conversion automatically
      extent_type: 'geojson',
      last_modified: new Date(item.last_modified).toISOString(),
      created: new Date(item.created).toISOString(),
      original_id: item.id,
      original_translations_id: item.translation?.id
    };
  });
};
```

## Enhanced Validation Strategy

Based on the Snowflake sample data, we need a robust validation approach:

1. **Pre-migration validation**:
   - Check source data integrity and required fields
   - Identify tables with minimal data (some tables like ACCESSIBILITY have very few records)
   - Validate ID formats and consistency across related tables
   - Check for translations in preferred language (English)

2. **During migration validation**:
   - Ensure foreign key constraints are satisfied
   - Validate UUID conversion consistency
   - Handle GeoJSON conversion for SERVICE_AREA table
   - Log any data conversion issues for manual review

3. **Post-migration validation**: 
   - Verify record counts match expected values
   - Spot-check sample records for data fidelity
   - Test referential integrity across the migrated tables
   - Verify PostGIS functionality for spatial data
   - Test Supabase Row-Level Security (RLS) policies

```typescript
// Example validation function
async function validateMigration() {
  const validationLog = [];
  
  // Check record counts
  const tables = [
    'organization', 'service', 'location', 
    'service_at_location', 'address', 'phone'
  ];
  
  for (const table of tables) {
    // Get source count from Snowflake
    const sourceCount = await snowflake.query(
      `SELECT COUNT(*) as count FROM ${table.toUpperCase()}`
    );
    
    // Get target count from Supabase
    const { data: targetCount, error } = await supabase
      .from(table.toLowerCase())
      .select('count', { count: 'exact', head: true });
      
    if (error) {
      validationLog.push({
        table,
        status: 'ERROR',
        message: `Failed to get count: ${error.message}`
      });
      continue;
    }
    
    // Compare counts
    if (sourceCount[0].count !== targetCount) {
      validationLog.push({
        table,
        status: 'WARNING',
        message: `Record count mismatch: Snowflake=${sourceCount[0].count}, Supabase=${targetCount}`
      });
    } else {
      validationLog.push({
        table,
        status: 'SUCCESS',
        message: `Record count matches: ${sourceCount[0].count}`
      });
    }
  }
  
  // Save validation log
  await supabase.from('migration_log').insert({
    source_table: 'ALL',
    target_table: 'ALL',
    records_migrated: 0,
    success_count: 0,
    failure_count: 0,
    error_messages: JSON.stringify(validationLog),
    started_at: new Date().toISOString(),
    completed_at: new Date().toISOString(),
    execution_time_seconds: 0
  });
  
  return validationLog;
}

```typescript
// Validation function example
async function validateMigration() {
  // Check record counts
  const sourceCount = await snowflake.query(
    'SELECT COUNT(*) as count FROM ORGANIZATION'
  );
  
  const { data: targetCount, error } = await supabase
    .from('organization')
    .select('count', { count: 'exact', head: true });
    
  console.log(`Source: ${sourceCount[0].count}, Target: ${targetCount}`);
  
  // Check sample records
  const sourceSample = await snowflake.query(
    'SELECT * FROM ORGANIZATION LIMIT 5'
  );
  
  for (const record of sourceSample) {
    const { data, error } = await supabase
      .from('organization')
      .select('*')
      .eq('original_id', record.ID)
      .single();
      
    if (error || !data) {
      console.error(`Record not found or error: ${record.ID}`);
      continue;
    }
    
    // Validate fields
    if (record.NAME !== data.name) {
      console.error(`Name mismatch for ${record.ID}: ${record.NAME} vs ${data.name}`);
    }
    // Check other fields...
  }
}
```

## Supabase-Specific Optimizations

1. **Using Realtime Features**: Enable real-time notifications for critical tables if needed
   ```sql
   BEGIN;
     -- Enable realtime for specific tables
     ALTER PUBLICATION supabase_realtime ADD TABLE organization, service;
   COMMIT;
   ```

2. **Efficient Supabase Queries**: Use Supabase client efficiently
   ```typescript
   // Efficient filtering with Supabase
   const { data, error } = await supabase
     .from('organization')
     .select(`
       id, 
       name,
       service:service(id, name)
     `)
     .eq('status', 'active')
     .order('name');
   ```

3. **Batch Processing**: Use batch operations for better performance
   ```typescript
   // Process in manageable chunks
   const BATCH_SIZE = 100;
   for (let i = 0; i < totalRecords; i += BATCH_SIZE) {
     await migrateTable('ORGANIZATION', 'organization', transformOrganization, BATCH_SIZE);
   }
   ```

4. **Type Safety**: Leverage TypeScript with generated types
   ```typescript
   import { definitions } from '../types/supabase';
   
   type Organization = definitions['organization'];
   
   const transformOrganization = (data: any[]): Organization[] => {
     // Type-safe transformation
     return data.map(item => ({...}));
   };
   ```

## Handling Edge Cases from Sample Data

Based on the Snowflake sample data, we should address these edge cases:

1. **Empty Tables**: Some tables like ACCESSIBILITY, CONTACT, and FUNDING have very few records. We should still create these tables but mark them as low-priority.

2. **Non-English Translations**: Many translation tables contain Korean (ko), Chinese (zh, zh-Hans), and other languages. We'll store only English initially but design the schema to support re-importing other languages later.

3. **Complex JSON Data**: SERVICE_AREA contains nested JSON data in both the extent field and SERVICE_AREA field. We need to handle both properly.

4. **Missing Required Fields**: Some records may be missing required fields according to the schema. We should log these for review and decide on default values.

## Next Steps

1. Initialize a Supabase project (either locally or in the cloud)
2. Generate TypeScript type definitions from the Supabase schema
3. Create TypeScript migration scripts with proper types
4. Implement the database schema in Supabase with all necessary constraints
5. Set up Row-Level Security policies based on the data access patterns
6. Configure PostGIS extension for spatial data
7. Migrate 100 test records following the defined order (organization → location → service → service_at_location → address → phone)
8. Validate the test migration against the expected data model
9. Create views and functions similar to those in the Supabase documentation
10. Iterate on any issues identified during testing
11. Implement schema for deferred tables without migration
12. Plan for complete migration once the test is successful
13. Configure Supabase API endpoints and permissions

```typescript
// Example of migration orchestration
async function runMigration() {
  // Setup - ensure the order is correct due to foreign key constraints
  const migrationOrder = [
    { sourceTable: 'ORGANIZATION', targetTable: 'organization', transform: transformOrganization },
    { sourceTable: 'LOCATION', targetTable: 'location', transform: transformLocation },
    { sourceTable: 'SERVICE', targetTable: 'service', transform: transformService },
    { sourceTable: 'SERVICE_AT_LOCATION', targetTable: 'service_at_location', transform: transformServiceAtLocation },
    { sourceTable: 'ADDRESS', targetTable: 'address', transform: transformAddress },
    { sourceTable: 'PHONE', targetTable: 'phone', transform: transformPhone }
  ];
  
  // Run migrations in sequence
  for (const migration of migrationOrder) {
    console.log(`Starting migration: ${migration.sourceTable} → ${migration.targetTable}`);
    
    const result = await migrateTable(
      migration.sourceTable, 
      migration.targetTable, 
      migration.transform,
      100 // Limit to 100 records for test migration
    );
    
    console.log(`Migration completed: ${result.success} succeeded, ${result.failure} failed`);
    
    // Break if significant failures to prevent cascade issues
    if (result.failure > result.success * 0.1) {
      console.error(`Too many failures, stopping migration sequence`);
      break;
    }
  }
  
  // Run validation
  const validationResults = await validateMigration();
  console.log('Validation complete:', validationResults);
}
