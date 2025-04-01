#!/usr/bin/env node
import dotenv from "dotenv";
import { Command } from "commander";
import { SnowflakeClient } from "./services/snowflake-client";
// import { SupabaseLoader } from "./3-Load/supabase-loader";
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
  .action(async (options) => {
    try {
      // Initialize services
      const snowflakeClient = new SnowflakeClient();
      // const supabaseLoader = new SupabaseLoader();
      const postgresLoader = new PostgresLoader();
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
      });

      console.log("Migration completed successfully");
      process.exit(0);
    } catch (error) {
      console.error("Migration failed:", error);
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
