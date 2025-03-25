#!/usr/bin/env node
"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const dotenv_1 = __importDefault(require("dotenv"));
const commander_1 = require("commander");
const snowflake_client_1 = require("./services/snowflake-client");
const supabase_loader_1 = require("./3-Load/supabase-loader");
const uuid_utils_1 = require("./utils/uuid-utils");
const migrationManager_1 = require("./managers/migrationManager");
const config_1 = require("./config/config");
// Load environment variables
dotenv_1.default.config();
// Initialize CLI
const program = new commander_1.Command();
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
    .action(async (options) => {
    try {
        // Initialize services
        const snowflakeClient = new snowflake_client_1.SnowflakeClient();
        const supabaseLoader = new supabase_loader_1.SupabaseLoader();
        const idConverter = new uuid_utils_1.IdConverter();
        // Initialize migration manager
        const migrationManager = new migrationManager_1.MigrationManager(snowflakeClient, supabaseLoader, idConverter);
        // Run migration with provided options
        await migrationManager.runCliMigration({
            entity: options.entity,
            batchSize: options.batchSize || config_1.migrationConfig.batchSize,
            limit: options.limit || 1000,
            offset: options.offset || 0,
            locale: options.locale || "en",
        });
        console.log("Migration completed successfully");
        process.exit(0);
    }
    catch (error) {
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
