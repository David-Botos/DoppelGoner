#!/usr/bin/env node
import dotenv from "dotenv";
import { Command } from "commander";
import { SnowflakeClient } from "./services/snowflake-client";
import { IdConverter } from "./utils/uuid-utils";
import { MigrationManager } from "./managers/migrationManager";
import { migrationConfig } from "./config/config";
import { PostgresLoader } from "./3-Load/postgres-loader";

// Load environment variables
dotenv.config();

// Initialize CLI
const program = new Command();

program
  .name("db-migration")
  .description("Database migration tool for Snowflake to Supabase ETL")
  .version("1.0.0");

program
  .command("migrate")
  .description("Run the migration process")
  .option("-e, --entity <entity>", "Specific entity to migrate")
  .option("-b, --batch-size <size>", "Batch size for processing", parseInt)
  .option("-l, --limit <limit>", "Limit number of records", parseInt)
  .option("-o, --offset <offset>", "Offset for pagination", parseInt)
  .option("--locale <locale>", "Locale for translations", "en")
  .option("-t, --test", "Run in test mode using test schema", false)
  .option("-s, --schema <schema>", "Specific schema to migrate")
  .option("-a, --all-schemas", "Migrate all configured schemas", false)
  .action(async (options) => {
    try {
      // Initialize services
      const snowflakeClient = new SnowflakeClient();
      const postgresLoader = new PostgresLoader(); // Now a direct concrete class
      const idConverter = new IdConverter();

      // Initialize migration manager
      const migrationManager = new MigrationManager(
        snowflakeClient,
        postgresLoader,
        idConverter
      );

      // Run migration with provided options
      await migrationManager.runCliMigration({
        entity: options.entity,
        batchSize: options.batchSize || migrationConfig.batchSize,
        limit: options.limit,
        offset: options.offset || 0,
        locale: options.locale || "en",
        testMode: options.test || false,
        schema: options.schema,
        allSchemas: options.allSchemas || false,
      });

      console.log("Migration completed successfully");
      process.exit(0);
    } catch (error) {
      console.error("Migration failed:", error);
      process.exit(1);
    }
  });

program
  .command("list-schemas")
  .description("List all available schemas")
  .action(async () => {
    try {
      // Initialize snowflake client
      const snowflakeClient = new SnowflakeClient();

      // Connect to ensure we have access
      await snowflakeClient.connect();

      // Get all schemas
      const schemas = snowflakeClient.getSchemas();

      console.log("Available schemas:");
      schemas.forEach((schema) => {
        console.log(`- ${schema}`);
      });

      // Disconnect
      await snowflakeClient.disconnect();

      process.exit(0);
    } catch (error) {
      console.error("Failed to list schemas:", error);
      process.exit(1);
    }
  });

program
  .command("validate")
  .description("Validate migrated data")
  .option("-e, --entity <entity>", "Specific entity to validate")
  .action(async (options) => {
    // Implementation for validation command
    console.log("Validation not yet implemented");
  });

program.parse(process.argv);
