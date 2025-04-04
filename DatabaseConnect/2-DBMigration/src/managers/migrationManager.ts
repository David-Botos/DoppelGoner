import { SnowflakeClient } from "../services/snowflake-client";
import { Extractor } from "../1-Extract/extractor";
import { Transformer, TransformOptions } from "../2-Transform/transformer";
import { IdConverter } from "../utils/uuid-utils";
import { OrganizationExtractor } from "../1-Extract/organizationExtractor";
import { OrganizationTransformer } from "../2-Transform/organizationTransformer";
import { MigratedData, SourceData, SourceDataTranslations } from "../types";
import { migrationConfig } from "../config/config";
import { ServiceExtractor } from "../1-Extract/serviceExtractor";
import { ServiceTransformer } from "../2-Transform/serviceTransformer";
import { PostgresLoader } from "../3-Load/postgres-loader";
import { PREDEFINED_CONSTRAINTS } from "../utils/constraint-utils";
import { LocationExtractor } from "../1-Extract/locationExtractor";
import { LocationTransformer } from "../2-Transform/locationTransformer";
import { ServiceAtLocationExtractor } from "../1-Extract/ServiceAtLocationExtractor";
import { ServiceAtLocationTransformer } from "../2-Transform/ServiceAtLocationTransformer";
import { PhoneExtractor } from "../1-Extract/phoneExtractor";
import { PhoneTransformer } from "../2-Transform/phoneTransformer";
import { AddressExtractor } from "../1-Extract/addressExtractor";
import { AddressTransformer } from "../2-Transform/addressTransformer";

/**
 * MigrationManager orchestrates the ETL process by coordinating
 * extractors, transformers, and data managers
 */
export class MigrationManager {
  private snowflakeClient: SnowflakeClient;
  private loader: PostgresLoader;
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
    this.loader = postgresLoader;
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

    this.extractors.set(
      "location",
      new LocationExtractor(this.snowflakeClient)
    );
    this.extractors.set(
      "service_at_location",
      new ServiceAtLocationExtractor(this.snowflakeClient)
    );
    this.extractors.set("phone", new PhoneExtractor(this.snowflakeClient));
    this.extractors.set("address", new AddressExtractor(this.snowflakeClient));

    this.transformers.set(
      "organization",
      new OrganizationTransformer(this.idConverter)
    );
    this.transformers.set("service", new ServiceTransformer(this.idConverter));
    this.transformers.set(
      "location",
      new LocationTransformer(this.idConverter)
    );
    this.transformers.set(
      "service_at_location",
      new ServiceAtLocationTransformer(this.idConverter)
    );
    this.transformers.set("phone", new PhoneTransformer(this.idConverter));
    this.transformers.set("address", new AddressTransformer(this.idConverter));

    // Additional extractors and transformers will be registered here
    // as they are implemented for other entity types
  }

  /**
   * Execute migration for a specific entity type with batch optimization
   */
  async migrateEntity(
    entityType: string,
    batchSize: number = migrationConfig.batchSize,
    limit?: number,
    offset: number = 0,
    locale: string = "en",
    testMode: boolean = false,
    schema?: string
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
      console.log(
        `Starting migration for ${entityType} with schema: ${
          schema || "default"
        }`
      );

      // If schema is provided, set it in the SnowflakeClient
      if (schema) {
        await this.snowflakeClient.setSchema(schema);
      }

      // Get the appropriate extractor and transformer
      const extractor = this.extractors.get(entityType);
      const transformer = this.transformers.get(entityType);

      if (!extractor || !transformer) {
        throw new Error(
          `No extractor or transformer found for entity type: ${entityType}`
        );
      }

      // Set the schema based on test mode
      const pgSchema = testMode ? "test" : "public";
      console.log(
        `Extracting ${entityType} data from Snowflake schema ${this.snowflakeClient.getCurrentSchema()} to Postgres schema ${pgSchema}...`
      );

      // 1. Extract data from Snowflake
      const dataMap = await extractor.extract(offset, locale, limit);
      console.log(`Extracted ${dataMap.size} ${entityType} records`);

      // 2. Transform data using batch processing
      console.log(`Transforming ${entityType} data in batches...`);
      console.time("transform");
      const transformationBatchSize = migrationConfig.batchSize || 100;

      const transformOptions: TransformOptions = {
        batchSize: transformationBatchSize,
        pgSchema: pgSchema,
      };

      // Pass the schema to the transform method
      const transformedData = await transformer.transform(
        dataMap,
        transformOptions
      );

      console.timeEnd("transform");
      console.log(
        `Transformed ${transformedData.length} ${entityType} records`
      );

      // NEW: Process invalid records from transformation
      const invalidRecords = transformer.getInvalidRecords();
      if (invalidRecords.length > 0) {
        console.log(
          `Processing ${invalidRecords.length} invalid records from transformer`
        );

        // Log each invalid record to failed_migration_records
        for (const { record, errors: validationErrors } of invalidRecords) {
          await this.loader.logFailedRecord(
            entityType,
            record.original_id || null,
            record.original_translations_id || null,
            validationErrors.join(", "),
            record,
            pgSchema
          );

          // Increment failure count for statistics
          failureCount++;
        }
      }

      // 3. Load data into Postgres
      console.log(
        `Loading ${transformedData.length} valid ${entityType} records into ${pgSchema} schema...`
      );

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

      // Use upsertData method with optimized batch size
      console.time("load");
      const loadResult = await this.loader.upsertData(
        entityType,
        transformedData,
        "id",
        batchSize,
        sourceTable,
        false,
        pgSchema
      );
      console.timeEnd("load");

      successCount = loadResult.success;
      failureCount += transformedData.length - loadResult.success;
      errors.push(...loadResult.errors);

      // Show total failures including both validation and loading failures
      const totalFailures = failureCount + invalidRecords.length;

      // If there were failures, log them
      if (totalFailures > 0) {
        console.warn(
          `${totalFailures} records failed to migrate (${invalidRecords.length} validation errors, ${failureCount} load errors). Check the failed_migration_records table.`
        );

        // Get a count of distinct error types
        const failedRecords = await this.loader.getFailedRecords(entityType);

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

      const endTime = new Date();
      const elapsedMs = endTime.getTime() - startTime.getTime();
      console.log(`Total migration time: ${elapsedMs / 1000} seconds`);

      return {
        success: successCount,
        failure: failureCount + invalidRecords.length, // Include validation failures in the count
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
  async migrateAllEntities(
    limit?: number,
    offset: number = 0,
    locale: string = "en",
    testMode: boolean = false,
    schema?: string
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
    console.log(`Limit: ${limit === undefined ? "ALL" : limit}`);
    console.log(`Offset: ${offset}`);
    console.log(`Locale: ${locale}`);
    console.log(`Validation Enabled: ${migrationConfig.enableValidation}`);
    console.log(`Migration Order: ${tablesToMigrate.join(", ")}`);
    console.log(`Test Mode: ${testMode}`);
    console.log(`Schema: ${schema || this.snowflakeClient.getCurrentSchema()}`);

    // Set schema if provided
    if (schema) {
      await this.snowflakeClient.setSchema(schema);
    }

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
   * Migrate all schemas
   */
  async migrateAllSchemas(
    limit?: number,
    offset: number = 0,
    locale: string = "en",
    testMode: boolean = false
  ): Promise<
    Map<
      string, // schema name
      Map<
        string, // entity type
        {
          success: number;
          failure: number;
          errors: Error[];
          startTime: Date;
          endTime: Date;
        }
      >
    >
  > {
    const allResults = new Map();
    const schemas = this.snowflakeClient.getSchemas();

    console.log(
      `Preparing to migrate ${schemas.length} schemas: ${schemas.join(", ")}`
    );

    for (const schema of schemas) {
      console.log(`\n=== Starting migration for schema: ${schema} ===\n`);

      const results = await this.migrateAllEntities(
        limit,
        offset,
        locale,
        testMode,
        schema
      );

      allResults.set(schema, results);

      // Summary for this schema
      let schemaSuccess = 0;
      let schemaFailure = 0;

      results.forEach((result) => {
        schemaSuccess += result.success;
        schemaFailure += result.failure;
      });

      console.log(`\n=== Completed migration for schema: ${schema} ===`);
      console.log(
        `Total Success: ${schemaSuccess}, Total Failure: ${schemaFailure}\n`
      );
    }

    return allResults;
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
    schema?: string;
    allSchemas?: boolean;
  }): Promise<void> {
    const {
      entity,
      batchSize,
      limit,
      offset = 0,
      locale = "en",
      testMode = false,
      schema,
      allSchemas = false,
    } = args;

    console.log("Starting migration with parameters:");
    console.log(`Entity: ${entity || "ALL"}`);
    console.log(`Batch Size: ${batchSize}`);
    console.log(`Limit: ${limit === undefined ? "ALL" : limit}`);
    console.log(`Offset: ${offset}`);
    console.log(`Locale: ${locale}`);
    console.log(`Test Mode: ${testMode}`);
    console.log(`Schema: ${schema || "Default"}`);
    console.log(`All Schemas: ${allSchemas}`);

    try {
      // If specific schema is requested, set it
      if (schema && !allSchemas) {
        await this.snowflakeClient.setSchema(schema);
      }

      if (allSchemas) {
        // Migrate all schemas
        console.log("Starting migration for all schemas");
        const results = await this.migrateAllSchemas(
          limit,
          offset,
          locale,
          testMode
        );

        console.log("\n=== Overall Migration Summary ===");

        let totalSuccess = 0;
        let totalFailure = 0;

        results.forEach((entityResults, schema) => {
          let schemaSuccess = 0;
          let schemaFailure = 0;

          entityResults.forEach((result) => {
            schemaSuccess += result.success;
            schemaFailure += result.failure;
          });

          console.log(
            `Schema ${schema}: Success=${schemaSuccess}, Failure=${schemaFailure}`
          );

          totalSuccess += schemaSuccess;
          totalFailure += schemaFailure;
        });

        console.log(
          `Grand Total: Success=${totalSuccess}, Failure=${totalFailure}`
        );
      } else if (entity) {
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
        // Migrate all entities in current schema
        const results = await this.migrateAllEntities(
          limit,
          offset,
          locale,
          testMode
        );

        console.log(
          `Migration completed for all entities in schema ${this.snowflakeClient.getCurrentSchema()}`
        );

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
