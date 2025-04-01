import { SnowflakeClient } from "../services/snowflake-client";
// import { SupabaseLoader } from "../3-Load/supabase-loader";
import { Extractor } from "../1-Extract/extractor";
import { Transformer } from "../2-Transform/transformer";
import { IdConverter } from "../utils/uuid-utils";
import { OrganizationExtractor } from "../1-Extract/organizationExtractor";
import { OrganizationTransformer } from "../2-Transform/organizationTransformer";
import { MigratedData, SourceData, SourceDataTranslations } from "../types";
import { migrationConfig } from "../config/config";
import { ServiceExtractor } from "../1-Extract/serviceExtractor";
import { ServiceTransformer } from "../2-Transform/serviceTransformer";
import { PostgresLoader } from "../3-Load/postgres-loader";
import { PREDEFINED_CONSTRAINTS } from "../utils/constraint-utils";

/**
 * MigrationManager orchestrates the ETL process by coordinating
 * extractors, transformers, and loaders
 */
export class MigrationManager {
  private snowflakeClient: SnowflakeClient;
  // private supabaseLoader: SupabaseLoader;
  private postgresLoader: PostgresLoader;
  private idConverter: IdConverter;

  // Storing extractors and transformers for different entity types
  private extractors: Map<string, Extractor<any, any>>;
  private transformers: Map<string, Transformer<any, any, any>>;

  constructor(
    snowflakeClient: SnowflakeClient,
    postgresLoader: PostgresLoader,
    idConverter: IdConverter
  ) {
    this.snowflakeClient = snowflakeClient;
    this.postgresLoader = postgresLoader;
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
  private registerExtractorsAndTransformers(): void {
    // Register organization extractor and transformer
    this.extractors.set(
      "organization",
      new OrganizationExtractor(this.snowflakeClient)
    );

    this.extractors.set("service", new ServiceExtractor(this.snowflakeClient));

    this.transformers.set(
      "organization",
      new OrganizationTransformer(this.idConverter)
    );
    this.transformers.set("service", new ServiceTransformer(this.idConverter));

    // Additional extractors and transformers will be registered here
    // as they are implemented for other entity types
  }

  /**
   * Execute migration for a specific entity type
   */
  async migrateEntity(
    entityType: string,
    batchSize: number = migrationConfig.batchSize,
    limit?: number,
    offset: number = 0,
    locale: string = "en",
    testMode: boolean = false
  ): Promise<{
    success: number;
    failure: number;
    errors: Error[];
    startTime: Date;
    endTime: Date;
  }> {
    const startTime = new Date();
    const errors: Error[] = [];
    let successCount = 0;
    let failureCount = 0;

    try {
      // Get the appropriate extractor and transformer
      const extractor = this.extractors.get(entityType);
      const transformer = this.transformers.get(entityType);

      if (!extractor || !transformer) {
        throw new Error(
          `No extractor or transformer found for entity type: ${entityType}`
        );
      }

      // 1. Extract data from Snowflake
      console.log(`Extracting ${entityType} data from Snowflake...`);
      const dataMap = await extractor.extract(offset, locale, limit);
      console.log(`Extracted ${dataMap.size} ${entityType} records`);

      // 2. Transform data
      console.log(`Transforming ${entityType} data...`);
      const transformedData = await transformer.transform(dataMap);
      console.log(
        `Transformed ${transformedData.length} ${entityType} records`
      );

      // 3. Load data into Supabase
      console.log(`Loading ${entityType} data into Supabase...`);
      // Log a sample record for debugging
      if (transformedData.length > 0) {
        console.log(
          "Sample transformed record:",
          JSON.stringify(transformedData[0], null, 2)
        );
      }

      // Use the sourceTable information from the extractor when available
      const sourceTable = extractor.sourceTables?.main || "unknown";

      // Fetch the proper predefined constraints from the util
      const constraints = PREDEFINED_CONSTRAINTS.entityType;

      // Set the schema based on test mode
      const schema = testMode ? "test" : "public";

      // Use upsertData method with source table information
      // Skip table existence check since we're sure the tables exist
      const loadResult = await this.postgresLoader.upsertData(
        entityType,
        transformedData,
        "id",
        batchSize,
        sourceTable,
        false,
        constraints,
        schema
      );

      successCount = loadResult.success;
      failureCount = transformedData.length - loadResult.success;
      errors.push(...loadResult.errors);

      // If there were failures, log them
      if (failureCount > 0) {
        console.warn(
          `${failureCount} records failed to migrate. Check the failed_migration_records table.`
        );

        // Optionally, get a count of distinct error types
        const failedRecords = await this.postgresLoader.getFailedRecords(
          entityType
        );

        if (failedRecords.length > 0) {
          const errorCounts = failedRecords.reduce((acc, record) => {
            const error = record.error_message;
            acc[error] = (acc[error] || 0) + 1;
            return acc;
          }, {});

          console.log("Error frequency:");
          Object.entries(errorCounts)
            .sort((a, b) => (b[1] as number) - (a[1] as number))
            .forEach(([error, count]) => {
              console.log(`- ${count} occurrences: ${error}`);
            });
        }
      }

      // 4. Validation is now handled automatically in the loader

      const endTime = new Date();
      return {
        success: successCount,
        failure: failureCount,
        errors,
        startTime,
        endTime,
      };
    } catch (error) {
      const endTime = new Date();
      const e = error instanceof Error ? error : new Error(String(error));
      errors.push(e);

      console.error(`Error migrating ${entityType}:`, e.message);

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
  async migrateAll(
    limit: number = 1000,
    offset: number = 0,
    locale: string = "en",
    testMode: boolean = false
  ): Promise<
    Map<
      string,
      {
        success: number;
        failure: number;
        errors: Error[];
        startTime: Date;
        endTime: Date;
      }
    >
  > {
    const results = new Map();

    // Get batch size from migration config
    const batchSize = migrationConfig.batchSize;

    // Get tables to migrate from migration config
    const tablesToMigrate = migrationConfig.tables;

    // Log migration settings
    console.log("Starting migration with the following settings:");
    console.log(`Batch Size: ${batchSize}`);
    console.log(`Limit: ${limit}`);
    console.log(`Offset: ${offset}`);
    console.log(`Locale: ${locale}`);
    console.log(`Validation Enabled: ${migrationConfig.enableValidation}`);
    console.log(`Migration Order: ${tablesToMigrate.join(", ")}`);
    console.log(`Test Mode: ${testMode}`);

    // Migrate entities in the order specified in migrationConfig
    for (const entityType of tablesToMigrate) {
      console.log(`Starting migration for: ${entityType}`);

      const result = await this.migrateEntity(
        entityType,
        batchSize,
        limit,
        offset,
        locale,
        testMode
      );

      results.set(entityType, result);

      console.log(`Completed migration for: ${entityType}`);
      console.log(`Success: ${result.success}, Failure: ${result.failure}`);

      if (result.errors.length > 0) {
        console.log(
          `Errors: ${result.errors.map((e) => e.message).join("\n")}`
        );
      }
    }

    return results;
  }

  /**
   * Running a migration with CLI
   */
  async runCliMigration(args: {
    entity?: string;
    batchSize?: number;
    limit?: number;
    offset?: number;
    locale?: string;
    testMode?: boolean;
  }): Promise<void> {
    const {
      entity,
      batchSize,
      limit,
      offset = 0,
      locale = "en",
      testMode = false,
    } = args;

    console.log("Starting migration with parameters:");
    console.log(`Entity: ${entity || "ALL"}`);
    console.log(`Batch Size: ${batchSize}`);
    console.log(`Limit: ${limit === undefined ? "ALL" : limit}`);
    console.log(`Offset: ${offset}`);
    console.log(`Locale: ${locale}`);
    console.log(`Test Mode: ${testMode}`);

    try {
      if (entity) {
        // Migrate specific entity
        const result = await this.migrateEntity(
          entity,
          batchSize,
          limit,
          offset,
          locale,
          testMode
        );

        console.log(`Migration completed for ${entity}`);
        console.log(`Success: ${result.success}, Failure: ${result.failure}`);

        const duration =
          (result.endTime.getTime() - result.startTime.getTime()) / 1000;
        console.log(`Duration: ${duration.toFixed(2)} seconds`);
      } else {
        // Migrate all entities
        const results = await this.migrateAll(limit, offset, locale, testMode);

        console.log("Migration completed for all entities");

        let totalSuccess = 0;
        let totalFailure = 0;
        let totalDuration = 0;

        results.forEach((result, entityType) => {
          console.log(`Entity: ${entityType}`);
          console.log(`Success: ${result.success}, Failure: ${result.failure}`);

          const duration =
            (result.endTime.getTime() - result.startTime.getTime()) / 1000;
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
    } catch (error) {
      console.error("Migration failed:", error);
      process.exit(1);
    }
  }
}
