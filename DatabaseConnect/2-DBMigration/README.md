# Snowflake to Supabase Migration Tool

A command-line tool for migrating HSDS (Human Services Data Specification) data from Snowflake to Supabase.

## Overview

This migration tool facilitates the Extract, Transform, Load (ETL) process for moving data from Snowflake to Supabase, following the migration plan that prioritizes critical tables and maintains data integrity. The tool focuses on English-only data initially and maintains traceability to original Snowflake records.

## Prerequisites

- Node.js (v16.0.0 or higher)
- TypeScript
- Snowflake account with access credentials
- Supabase project with appropriate permissions
- PostgreSQL client tools (for direct database operations if needed)

## Installation

1. Clone the repository
2. Install dependencies:

```bash
npm install
```

3. Build the TypeScript code:

```bash
npm run build
```

## Environment Setup

Create a `.env` file in the root directory with the following variables:

```
# Snowflake Configuration
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_ROLE=your_role

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
```

Additional database configuration may be required in the config files.

## Usage

### Running Migrations

To run a migration with default settings:

```bash
npm run migrate
```

With specific options:

```bash
npm run migrate -- -e organization -b 100 -l 500 -o 0 --locale en
```

Or using the Node executable directly:

```bash
node dist/index.js migrate -e organization -b 100 -l 500 -o 0 --locale en
```

### Command Line Options

- `-e, --entity <entity>` - Specify which entity to migrate (e.g., organization, service, location)
- `-b, --batch-size <size>` - Number of records to process in each batch (default defined in config)
- `-l, --limit <limit>` - Maximum number of records to migrate (default: 1000)
- `-o, --offset <offset>` - Starting offset for pagination (default: 0)
- `--locale <locale>` - Locale for translations (default: "en")

### Running Validation

To validate migrated data:

```bash
npm run validate
```

Or with specific entity:

```bash
npm run validate -- -e organization
```

## Migration Order

Based on referential integrity requirements, the entities are migrated in this order:

1. organization
2. location
3. service
4. service_at_location
5. address
6. phone

## Project Structure

- `src/services/` - Service classes for database connections (SnowflakeClient)
- `src/3-Load/` - Data loading modules (SupabaseLoader)
- `src/utils/` - Utility functions (IdConverter for UUID conversion)
- `src/managers/` - Business logic managers (MigrationManager)
- `src/config/` - Configuration files
- `scripts/` - Database setup scripts

## Data Transformation Details

### ID Tracking Approach

For all migrated tables, the original Snowflake IDs are preserved:

- `original_id`: Stores the original Snowflake record ID
- `original_translations_id`: Stores the original translation record ID (for tables with translations)

### Translation Handling

The migration tool focuses on English-only translations initially:

- Prioritizes canonical English translations (IS_CANONICAL=True)
- Falls back to any English translation if no canonical one exists
- For fields without English translations, values are left as NULL

### UUID Conversion

Snowflake IDs are converted to UUIDs for Supabase:

- Consistent algorithm ensures the same source ID always maps to the same UUID
- Original IDs are preserved in tracking fields for traceability

## Development

To modify the migration logic:

1. Update entity-specific transformation logic in the appropriate files
2. Adjust the configuration in `config/config.js` to modify global settings
3. Add validation rules for new entities as needed

### Adding Support for New Entities

To add support for additional entities beyond the prioritized tables:

1. Create a new transformation module for the entity
2. Define the mapping rules based on the migration plan
3. Add the entity to the migration order in the MigrationManager
4. Update validation logic to include the new entity

## Troubleshooting

If you encounter connection issues:

- Verify your environment variables are correctly set
- Ensure you have network access to both Snowflake and Supabase
- Check the logs for detailed error messages

For migration validation errors:

- Review the migration_log table in Supabase for detailed error messages
- Verify the data integrity in the source Snowflake tables
- Check for missing required fields or translation records

## PostGIS Support

For geographic data (particularly for SERVICE_AREA), the tool utilizes Supabase's built-in PostGIS extension:

- GeoJSON data from Snowflake is properly formatted for PostGIS
- Spatial queries can be performed directly on the migrated data

## Batching Strategy

To handle large datasets efficiently, the migration process uses batching:

- Default batch size is defined in the config
- Override with the `-b, --batch-size` option
- Each batch is processed and validated independently
- Migration logs track success/failure counts per batch

## Security Considerations

The migration tool is designed to work with Supabase's Row-Level Security (RLS) features:

- Default RLS policies are created during database setup
- Service role key is used for the migration process to bypass RLS
- Consider implementing custom RLS policies post-migration

## Future Enhancements

Planned future enhancements to the migration tool:

- Support for additional languages beyond English
- Migration of deferred tables
- Enhanced validation reporting
- Performance optimizations for large datasets
