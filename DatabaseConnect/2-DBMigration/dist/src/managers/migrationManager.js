"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.MigrationManager = void 0;
const organizationExtractor_1 = require("../1-Extract/organizationExtractor");
const organizationTransformer_1 = require("../2-Transform/organizationTransformer");
const config_1 = require("../config/config");
/**
 * MigrationManager orchestrates the ETL process by coordinating
 * extractors, transformers, and loaders
 */
class MigrationManager {
    constructor(snowflakeClient, supabaseLoader, idConverter) {
        this.snowflakeClient = snowflakeClient;
        this.supabaseLoader = supabaseLoader;
        this.idConverter = idConverter;
        // Initialize maps
        this.extractors = new Map();
        this.transformers = new Map();
        // Register extractors and transformers
        this.registerExtractorsAndTransformers();
    }
    /**
     * Register all extractors and transformers for different entity types
     */
    registerExtractorsAndTransformers() {
        // Register organization extractor and transformer
        this.extractors.set("organization", new organizationExtractor_1.OrganizationExtractor(this.snowflakeClient));
        this.transformers.set("organization", new organizationTransformer_1.OrganizationTransformer(this.idConverter));
        // Additional extractors and transformers will be registered here
        // as they are implemented for other entity types
        // Example:
        // this.extractors.set('service', new ServiceExtractor(this.snowflakeClient));
        // this.transformers.set('service', new ServiceTransformer(this.idConverter));
    }
    /**
     * Execute migration for a specific entity type
     */
    async migrateEntity(entityType, batchSize = config_1.migrationConfig.batchSize, limit = 1000, offset = 0, locale = "en") {
        const startTime = new Date();
        const errors = [];
        let successCount = 0;
        let failureCount = 0;
        try {
            // Get the appropriate extractor and transformer
            const extractor = this.extractors.get(entityType);
            const transformer = this.transformers.get(entityType);
            if (!extractor || !transformer) {
                throw new Error(`No extractor or transformer found for entity type: ${entityType}`);
            }
            // 1. Extract data from Snowflake
            console.log(`Extracting ${entityType} data from Snowflake...`);
            const dataMap = await extractor.extract(limit, offset, locale);
            console.log(`Extracted ${dataMap.size} ${entityType} records`);
            // 2. Transform data
            console.log(`Transforming ${entityType} data...`);
            const transformedData = transformer.transform(dataMap);
            console.log(`Transformed ${transformedData.length} ${entityType} records`);
            // 3. Load data into Supabase
            console.log(`Loading ${entityType} data into Supabase...`);
            const loadResult = await this.supabaseLoader.upsertData(entityType, transformedData, "id", batchSize);
            successCount = loadResult.success;
            failureCount = transformedData.length - loadResult.success;
            errors.push(...loadResult.errors);
            // 4. Log migration results
            const endTime = new Date();
            await this.supabaseLoader.logMigration(extractor.sourceTables.main, entityType, transformedData.length, successCount, failureCount, errors.map((e) => e.message), startTime, endTime);
            // 5. Validate migration if enabled
            if (config_1.migrationConfig.enableValidation) {
                const validationResult = await this.supabaseLoader.validateRecordCount(entityType, successCount);
                console.log(`Validation result: ${validationResult.message}`);
            }
            else {
                console.log("Validation skipped as per configuration");
            }
            return {
                success: successCount,
                failure: failureCount,
                errors,
                startTime,
                endTime,
            };
        }
        catch (error) {
            const endTime = new Date();
            const e = error instanceof Error ? error : new Error(String(error));
            errors.push(e);
            console.error(`Error migrating ${entityType}:`, e.message);
            // Log migration failure
            await this.supabaseLoader.logMigration(`UNKNOWN_${entityType.toUpperCase()}`, entityType, 0, successCount, failureCount + 1, [e.message], startTime, endTime);
            return {
                success: successCount,
                failure: failureCount + 1,
                errors,
                startTime,
                endTime,
            };
        }
    }
    /**
     * Execute migration for all entities in the specified order
     */
    async migrateAll(limit = 1000, offset = 0, locale = "en") {
        const results = new Map();
        // Get batch size from migration config
        const batchSize = config_1.migrationConfig.batchSize;
        // Get tables to migrate from migration config
        const tablesToMigrate = config_1.migrationConfig.tables;
        // Log migration settings
        console.log("Starting migration with the following settings:");
        console.log(`Batch Size: ${batchSize}`);
        console.log(`Limit: ${limit}`);
        console.log(`Offset: ${offset}`);
        console.log(`Locale: ${locale}`);
        console.log(`Validation Enabled: ${config_1.migrationConfig.enableValidation}`);
        console.log(`Migration Order: ${tablesToMigrate.join(", ")}`);
        // Migrate entities in the order specified in migrationConfig
        for (const entityType of tablesToMigrate) {
            console.log(`Starting migration for: ${entityType}`);
            const result = await this.migrateEntity(entityType, batchSize, limit, offset, locale);
            results.set(entityType, result);
            console.log(`Completed migration for: ${entityType}`);
            console.log(`Success: ${result.success}, Failure: ${result.failure}`);
            if (result.errors.length > 0) {
                console.log(`Errors: ${result.errors.map((e) => e.message).join("\n")}`);
            }
            // Skip validation if disabled in config
            if (!config_1.migrationConfig.enableValidation) {
                console.log("Validation skipped as per configuration");
            }
        }
        return results;
    }
    /**
     * Running a migration with CLI
     */
    async runCliMigration(args) {
        const { entity, batchSize = config_1.migrationConfig.batchSize, limit = 1000, offset = 0, locale = "en", } = args;
        console.log("Starting migration with parameters:");
        console.log(`Entity: ${entity || "ALL"}`);
        console.log(`Batch Size: ${batchSize}`);
        console.log(`Limit: ${limit}`);
        console.log(`Offset: ${offset}`);
        console.log(`Locale: ${locale}`);
        try {
            if (entity) {
                // Migrate specific entity
                const result = await this.migrateEntity(entity, batchSize, limit, offset, locale);
                console.log(`Migration completed for ${entity}`);
                console.log(`Success: ${result.success}, Failure: ${result.failure}`);
                const duration = (result.endTime.getTime() - result.startTime.getTime()) / 1000;
                console.log(`Duration: ${duration.toFixed(2)} seconds`);
            }
            else {
                // Migrate all entities
                const results = await this.migrateAll(limit, offset, locale);
                console.log("Migration completed for all entities");
                let totalSuccess = 0;
                let totalFailure = 0;
                let totalDuration = 0;
                results.forEach((result, entityType) => {
                    console.log(`Entity: ${entityType}`);
                    console.log(`Success: ${result.success}, Failure: ${result.failure}`);
                    const duration = (result.endTime.getTime() - result.startTime.getTime()) / 1000;
                    console.log(`Duration: ${duration.toFixed(2)} seconds`);
                    totalSuccess += result.success;
                    totalFailure += result.failure;
                    totalDuration += duration;
                });
                console.log("=== Summary ===");
                console.log(`Total Success: ${totalSuccess}`);
                console.log(`Total Failure: ${totalFailure}`);
                console.log(`Total Duration: ${totalDuration.toFixed(2)} seconds`);
            }
        }
        catch (error) {
            console.error("Migration failed:", error);
            process.exit(1);
        }
    }
}
exports.MigrationManager = MigrationManager;
