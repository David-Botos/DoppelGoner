"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
// Main entry point for the migration application
const snowflake_1 = require("./connectors/snowflake");
const postgres_1 = require("./connectors/postgres");
const migration_service_1 = require("./services/migration-service");
const logger_1 = require("./utils/logger");
async function main() {
    try {
        logger_1.logger.info("Starting Snowflake to RDS migration");
        // Connect to databases
        await snowflake_1.snowflakeConnector.connect();
        logger_1.logger.info("Connected to Snowflake");
        // Run the migration
        const results = await migration_service_1.migrationService.migrateAll();
        // Log summary
        let totalRecords = 0;
        let totalSuccess = 0;
        let totalFailure = 0;
        results.forEach((result) => {
            totalRecords += result.recordsProcessed;
            totalSuccess += result.successCount;
            totalFailure += result.failureCount;
        });
        logger_1.logger.info("Migration Summary:");
        logger_1.logger.info(`Total records processed: ${totalRecords}`);
        logger_1.logger.info(`Successfully migrated: ${totalSuccess}`);
        logger_1.logger.info(`Failed to migrate: ${totalFailure}`);
        logger_1.logger.info(`Success rate: ${((totalSuccess / totalRecords) * 100).toFixed(2)}%`);
        // Close connections
        await snowflake_1.snowflakeConnector.close();
        await postgres_1.postgresConnector.close();
        logger_1.logger.info("Migration completed");
    }
    catch (error) {
        logger_1.logger.error("Migration failed:", error);
        // Ensure connections are closed even in case of error
        try {
            await snowflake_1.snowflakeConnector.close();
            await postgres_1.postgresConnector.close();
        }
        catch (closeError) {
            logger_1.logger.error("Error closing connections:", closeError);
        }
        process.exit(1);
    }
}
// Execute if this is the main module
if (require.main === module) {
    main().catch((error) => {
        console.error("Unhandled error:", error);
        process.exit(1);
    });
}
