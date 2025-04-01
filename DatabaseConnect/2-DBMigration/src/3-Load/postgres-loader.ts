import { PoolClient } from "pg";
import { BaseLoader } from "./loader";
import { PostgresMetadataManager } from "./postgres-metadata-manager";
import { MetadataManager } from "./metadata-manager";
import { PostgresClient } from "../services/postgres-client";
import { batchRecords } from "../utils/loader-utils";
import {
  TableConstraints,
  validateRecordsAgainstConstraints,
  getTableConstraints,
  PREDEFINED_CONSTRAINTS,
} from "../utils/constraint-utils";
import { v4 as uuidv4 } from "uuid";

/**
 * PostgreSQL implementation of the BaseLoader abstract class
 * With enhanced batch processing and constraint validation
 */
export class PostgresLoader extends BaseLoader {
  private client: PostgresClient;
  private metadataManager: MetadataManager;
  private constraintsCache: Map<string, TableConstraints> = new Map();

  constructor(client?: PostgresClient, metadataManager?: MetadataManager) {
    super();
    this.client = client || new PostgresClient();

    this.metadataManager =
      metadataManager || new PostgresMetadataManager(this.client.getPool());
  }

  /**
   * Get the PostgreSQL client instance
   * @returns PostgresClient instance
   */
  getClient(): PostgresClient {
    return this.client;
  }

  /**
   * Implementation of the core data loading functionality for PostgreSQL
   * With improved batch handling
   */
  protected async loadDataInternal<T extends Record<string, any>>(
    tableName: string,
    data: T[],
    batchSize: number = 1000,
    schema: string = "public"
  ): Promise<{ success: number; errors: Error[] }> {
    let successCount = 0;
    const errors: Error[] = [];

    // Get table constraints
    const constraints = await this.getTableConstraintsWithCache(tableName);

    // Validate records against constraints
    const { validRecords, invalidRecordsWithErrors } =
      validateRecordsAgainstConstraints(data, constraints);

    // Process invalid records
    for (const {
      record,
      errors: validationErrors,
    } of invalidRecordsWithErrors) {
      const errorMsg = `Validation failed: ${validationErrors.join(", ")}`;
      errors.push(new Error(errorMsg));

      try {
        await this.logFailedRecord(
          tableName,
          record.original_id || null,
          record.original_translations_id || null,
          errorMsg,
          record
        );
      } catch (logError) {
        console.error(`Failed to log invalid record: ${logError}`);
      }
    }

    // Process valid records in batches
    if (validRecords.length > 0) {
      // Break data into batches
      const batches = batchRecords(validRecords, batchSize);

      for (const batch of batches) {
        try {
          if (batch.length === 0) continue;

          // Get column names from the first record
          const columns = Object.keys(batch[0]);

          // Prepare values and placeholder strings
          const values: any[] = [];
          const placeholders: string[] = [];

          batch.forEach((record, rowIndex) => {
            const rowPlaceholders: string[] = [];

            columns.forEach((column, colIndex) => {
              const paramIndex = rowIndex * columns.length + colIndex + 1;
              rowPlaceholders.push(`$${paramIndex}`);
              values.push(record[column]);
            });

            placeholders.push(`(${rowPlaceholders.join(", ")})`);
          });

          // Build the INSERT query
          const query = `
            INSERT INTO ${schema}.${tableName} (${columns.join(", ")})
            VALUES ${placeholders.join(", ")}
            RETURNING id
          `;

          // Execute the query
          const result = await this.client.getPool().query(query, values);
          if (result && result.rowCount) {
            successCount += result.rowCount;
          }

          // Track metadata for each successfully inserted record
          await this.client.executeTransaction(async (dbClient) => {
            for (const record of batch) {
              try {
                await this.trackMetadataWithClient(
                  dbClient,
                  record.id,
                  tableName,
                  "insert",
                  "all",
                  "Imported from Snowflake",
                  JSON.stringify(record),
                  "ETL Process",
                  record.original_id || null
                );
              } catch (metadataError) {
                console.error(
                  `Failed to track metadata for record ${record.id}: ${metadataError}`
                );
                // Don't fail the whole process for metadata tracking failures
              }
            }
          });
        } catch (error) {
          const errorMsg =
            error instanceof Error ? error : new Error(String(error));
          console.error(`Error inserting batch into ${tableName}:`, errorMsg);
          errors.push(errorMsg);

          // If batch fails, try each record individually as fallback
          for (const record of batch) {
            try {
              const singleResult = await this.insertSingleRecord(
                tableName,
                record
              );
              if (singleResult.success) {
                successCount++;
              } else {
                errors.push(new Error(singleResult.error || "Unknown error"));

                await this.logFailedRecord(
                  tableName,
                  record.original_id || null,
                  record.original_translations_id || null,
                  singleResult.error || "Unknown error",
                  record
                );
              }
            } catch (singleError) {
              const singleErrorMsg =
                singleError instanceof Error
                  ? singleError.message
                  : String(singleError);
              errors.push(new Error(singleErrorMsg));

              await this.logFailedRecord(
                tableName,
                record.original_id || null,
                record.original_translations_id || null,
                singleErrorMsg,
                record
              );
            }
          }
        }
      }
    }

    return { success: successCount, errors };
  }

  /**
   * Insert a single record as fallback when batch insert fails
   */
  private async insertSingleRecord<T extends Record<string, any>>(
    tableName: string,
    record: T
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const columns = Object.keys(record);
      const placeholders = columns.map((_, i) => `$${i + 1}`);
      const values = columns.map((col) => record[col]);

      const query = `
        INSERT INTO ${tableName} (${columns.join(", ")})
        VALUES (${placeholders.join(", ")})
        RETURNING id
      `;

      const result = await this.client.getPool().query(query, values);

      if (result && result.rowCount && result.rowCount > 0) {
        // Track metadata
        await this.trackMetadata(
          record.id,
          tableName,
          "insert",
          "all",
          "Imported from Snowflake",
          JSON.stringify(record),
          "ETL Process",
          record.original_id || null
        );

        return { success: true };
      } else {
        return {
          success: false,
          error: "Failed to insert record - no rows affected",
        };
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * PostgreSQL-specific implementation of upsert functionality
   * TODO: migrate how the batching works in this implementation to the abstract class(?)
   */
  override async upsertData<T extends Record<string, any>>(
    tableName: string,
    data: T[],
    onConflict: string = "id",
    batchSize: number = 1000,
    sourceTable: string = "unknown",
    skipTableCheck: boolean = false,
    tableConstraints?: TableConstraints,
    schema: string = "public"
  ): Promise<{ success: number; errors: Error[] }> {
    // Pre-processing logic
    const startTime = new Date();
    let successCount = 0;
    const errors: Error[] = [];

    try {
      // Verify table exists (if not skipped)
      if (!skipTableCheck) {
        const exists = await this.tableExists(tableName, schema);
        if (!exists) {
          throw new Error(
            `Table ${tableName} does not exist in schema ${schema}`
          );
        }
      }

      // Get table constraints (either from parameter, cache, or database)
      const constraints =
        tableConstraints ||
        (await this.getTableConstraintsWithCache(tableName));

      // Validate records against constraints
      const { validRecords, invalidRecordsWithErrors } =
        validateRecordsAgainstConstraints(data, constraints);

      // Process invalid records - log them to failed_migration_records
      await this.client.executeTransaction(async (dbClient) => {
        for (const {
          record,
          errors: validationErrors,
        } of invalidRecordsWithErrors) {
          const errorMsg = `Validation failed: ${validationErrors.join(", ")}`;
          errors.push(new Error(errorMsg));

          await this.logFailedRecordWithClient(
            dbClient,
            tableName,
            record.original_id || null,
            record.original_translations_id || null,
            errorMsg,
            record
          );
        }
      });

      // Process valid records in batches
      if (validRecords.length > 0) {
        const batches = batchRecords(validRecords, batchSize);

        for (const batch of batches) {
          try {
            if (batch.length === 0) continue;

            // Process batch using transaction
            const batchResult = await this.processBatchUpsert(
              tableName,
              batch,
              onConflict,
              schema
            );
            successCount += batchResult.successCount;

            // Add any batch errors to our error collection
            if (batchResult.errors.length > 0) {
              errors.push(...batchResult.errors);
            }
          } catch (batchError) {
            console.error(
              `Error processing batch for ${tableName}:`,
              batchError
            );

            // If batch processing fails, try each record individually
            for (const record of batch) {
              try {
                const result = await this.processSingleRecordUpsert(
                  tableName,
                  record,
                  onConflict,
                  schema
                );
                if (result.success) {
                  successCount++;
                } else if (result.error) {
                  errors.push(new Error(result.error));
                  await this.logFailedRecord(
                    tableName,
                    record.original_id || null,
                    record.original_translations_id || null,
                    result.error,
                    record
                  );
                }
              } catch (singleError) {
                const errorMsg =
                  singleError instanceof Error
                    ? singleError.message
                    : String(singleError);
                errors.push(new Error(errorMsg));

                await this.logFailedRecord(
                  tableName,
                  record.original_id || null,
                  record.original_translations_id || null,
                  errorMsg,
                  record
                );
              }
            }
          }
        }
      }

      // Post-processing logic (logging completion)
      const endTime = new Date();
      await this.logMigration(
        sourceTable,
        tableName,
        data.length,
        successCount,
        data.length - successCount,
        errors.map((e) => e.message),
        startTime,
        endTime
      );

      return { success: successCount, errors };
    } catch (error) {
      // Error handling for overall process failure
      const endTime = new Date();
      const e = error instanceof Error ? error : new Error(String(error));

      await this.logMigration(
        sourceTable,
        tableName,
        data.length,
        0,
        data.length,
        [e.message],
        startTime,
        endTime
      );

      return { success: 0, errors: [e] };
    }
  }

  /**
   * Process a batch of records for upsert in a single transaction
   */
  private async processBatchUpsert<T extends Record<string, any>>(
    tableName: string,
    batch: T[],
    onConflict: string,
    schema: string = "public"
  ): Promise<{ successCount: number; errors: Error[] }> {
    let successCount = 0;
    const errors: Error[] = [];
    const returnedIds: string[] = [];

    // Execute as a transaction
    await this.client.executeTransaction(async (dbClient) => {
      if (batch.length === 0) return;

      console.log(`using schema: ${schema} for ${tableName}`);

      // Get column names from the first record
      const columns = Object.keys(batch[0]);

      // Prepare values and placeholder strings
      const values: any[] = [];
      const placeholders: string[] = [];

      // Generate placeholders and values for all records in batch
      batch.forEach((record, rowIndex) => {
        const rowPlaceholders: string[] = [];

        columns.forEach((column, colIndex) => {
          const paramIndex = rowIndex * columns.length + colIndex + 1;
          rowPlaceholders.push(`$${paramIndex}`);
          values.push(record[column]);
        });

        placeholders.push(`(${rowPlaceholders.join(", ")})`);
      });

      // Build the UPDATE part (for the ON CONFLICT clause)
      const updateColumns = columns
        .filter((col) => col !== onConflict)
        .map((col) => `${col} = EXCLUDED.${col}`);

      // Build the complete UPSERT query with RETURNING
      const query = `
        INSERT INTO ${schema}.${tableName} (${columns.join(", ")})
        VALUES ${placeholders.join(", ")}
        ON CONFLICT (${onConflict}) DO UPDATE SET
        ${updateColumns.join(", ")}
        RETURNING id
      `;

      // Execute the query
      const result = await dbClient.query(query, values);
      const ids = result.rows.map((row) => row.id);
      returnedIds.push(...ids);
      successCount = ids.length;

      // Track metadata for each successfully upserted record
      for (let i = 0; i < batch.length; i++) {
        const record = batch[i];
        const recordId = record.id;

        if (ids.includes(recordId)) {
          try {
            await this.trackMetadataWithClient(
              dbClient,
              recordId,
              tableName,
              "upsert",
              "all",
              "Previous values in PostgreSQL",
              JSON.stringify(record),
              "ETL Process",
              record.original_id || null
            );
          } catch (metadataError) {
            console.error(
              `Failed to track metadata for record ${recordId}: ${metadataError}`
            );
            // Don't fail the whole batch for metadata errors
          }
        }
      }
    });

    return { successCount, errors };
  }

  /**
   * Process a single record for upsert (fallback method)
   */
  private async processSingleRecordUpsert<T extends Record<string, any>>(
    tableName: string,
    record: T,
    onConflict: string,
    schema: string = "public"
  ): Promise<{ success: boolean; error?: string }> {
    try {
      // Get column names from the record
      const columns = Object.keys(record);

      // Prepare values and placeholder strings
      const values: any[] = [];
      const placeholders: string[] = [];

      columns.forEach((column, index) => {
        placeholders.push(`$${index + 1}`);
        values.push(record[column]);
      });

      // Build the UPDATE part (for the ON CONFLICT clause)
      const updateColumns = columns
        .filter((col) => col !== onConflict)
        .map((col, index) => `${col} = $${index + columns.length + 1}`);

      // Add values for the UPDATE part
      columns
        .filter((col) => col !== onConflict)
        .forEach((col) => values.push(record[col]));

      // Build the complete UPSERT query
      const query = `
        INSERT INTO ${schema}.${tableName} (${columns.join(", ")})
        VALUES (${placeholders.join(", ")})
        ON CONFLICT (${onConflict}) DO UPDATE SET
        ${updateColumns.join(", ")}
        RETURNING id
      `;

      // Execute the query
      const result = await this.client.getPool().query(query, values);

      if (result && result.rowCount && result.rowCount > 0) {
        // Track metadata for successfully upserted record
        await this.trackMetadata(
          record.id,
          tableName,
          "upsert",
          "all",
          "Previous values in PostgreSQL",
          JSON.stringify(record),
          "ETL Process",
          record.original_id || null
        );

        return { success: true };
      } else {
        return {
          success: false,
          error: "Record not upserted - no rows affected",
        };
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Get table constraints with caching
   */
  private async getTableConstraintsWithCache(
    tableName: string
  ): Promise<TableConstraints> {
    // Check if constraints are in predefined set
    if (PREDEFINED_CONSTRAINTS[tableName]) {
      return PREDEFINED_CONSTRAINTS[tableName];
    }

    // Check if constraints are in cache
    if (this.constraintsCache.has(tableName)) {
      return this.constraintsCache.get(tableName)!;
    }

    // Get constraints from database
    const constraints = await getTableConstraints(
      this.client.getPool(),
      tableName
    );

    // Cache for future use
    this.constraintsCache.set(tableName, constraints);

    return constraints;
  }

  /**
   * Track metadata using a specific database client (for transaction support)
   */
  private async trackMetadataWithClient(
    client: PoolClient,
    resourceId: string,
    resourceType: string,
    actionType: string,
    fieldName: string,
    previousValue: string,
    replacementValue: string,
    updatedBy: string,
    originalId?: string
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const metadataId = uuidv4();
      const now = new Date().toISOString();

      const query = `
        INSERT INTO metadata (
          id, 
          resource_id, 
          resource_type, 
          last_action_date, 
          last_action_type, 
          field_name, 
          previous_value, 
          replacement_value, 
          updated_by, 
          created, 
          last_modified,
          original_id
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
      `;

      const values = [
        metadataId,
        resourceId,
        resourceType,
        now,
        actionType,
        fieldName,
        previousValue,
        replacementValue,
        updatedBy,
        now,
        now,
        originalId || null,
      ];

      await client.query(query, values);

      return {
        success: true,
      };
    } catch (error) {
      console.error("Error tracking metadata with client:", error);
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Log failed record using a specific database client (for transaction support)
   */
  private async logFailedRecordWithClient(
    client: PoolClient,
    tableName: string,
    originalId: string | null,
    originalTranslationsId: string | null,
    errorMessage: string,
    attemptedRecord: any
  ): Promise<void> {
    const query = `
      INSERT INTO failed_migration_records (
        id,
        table_name,
        original_id,
        original_translations_id,
        error_message,
        attempted_record,
        attempted_at,
        resolved,
        retry_count
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    `;

    const values = [
      uuidv4(),
      tableName,
      originalId,
      originalTranslationsId,
      errorMessage,
      JSON.stringify(attemptedRecord),
      new Date().toISOString(),
      false,
      0, // Initialize retry_count to 0
    ];

    await client.query(query, values);
  }

  /**
   * Log failed record to the failed_migration_records table
   */
  async logFailedRecord<T extends Record<string, any>>(
    tableName: string,
    originalId: string | null,
    originalTranslationsId: string | null,
    errorMessage: string,
    attemptedRecord: T
  ): Promise<void> {
    const query = `
      INSERT INTO failed_migration_records (
        id,
        table_name,
        original_id,
        original_translations_id,
        error_message,
        attempted_record,
        attempted_at,
        resolved,
        retry_count
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    `;

    const values = [
      uuidv4(),
      tableName,
      originalId,
      originalTranslationsId,
      errorMessage,
      JSON.stringify(attemptedRecord),
      new Date().toISOString(),
      false,
      0, // Initialize retry_count to 0
    ];

    try {
      await this.client.getPool().query(query, values);
    } catch (error) {
      console.error("Error logging failed record:", error);
      throw error;
    }
  }

  /**
   * Update a record in PostgreSQL
   */
  async updateRecord<T extends Record<string, any>>(
    tableName: string,
    id: string,
    data: Partial<T>
  ): Promise<{ success: boolean; error?: string }> {
    try {
      // First get the current record to compare values
      const existingRecord = await this.client.executeQuery(
        `SELECT * FROM ${tableName} WHERE id = $1`,
        [id]
      );

      if (!existingRecord || existingRecord.length === 0) {
        return {
          success: false,
          error: `Record with id ${id} not found in ${tableName}`,
        };
      }

      // Prepare the SET clause and values for the UPDATE query
      const columns = Object.keys(data);
      const values = Object.values(data);

      const setClauses = columns
        .map((col, i) => `${col} = $${i + 2}`)
        .join(", ");

      // Perform the update
      const query = `
        UPDATE ${tableName} 
        SET ${setClauses} 
        WHERE id = $1
      `;

      const result = await this.client.getPool().query(query, [id, ...values]);

      if (result && result.rowCount && result.rowCount > 0) {
        // Track metadata for each changed field
        for (const [fieldName, newValue] of Object.entries(data)) {
          const oldValue = existingRecord[0][fieldName];

          // Only track if the value actually changed
          if (oldValue !== newValue) {
            await this.trackMetadata(
              id,
              tableName,
              "update",
              fieldName,
              oldValue !== null && oldValue !== undefined
                ? String(oldValue)
                : "null",
              newValue !== null && newValue !== undefined
                ? String(newValue)
                : "null",
              "ETL Process",
              existingRecord[0].original_id || null
            );
          }
        }
      }

      if (result && result.rowCount) {
        return {
          success: result.rowCount > 0,
          error: result.rowCount === 0 ? "Record not updated" : undefined,
        };
      } else {
        return {
          success: false,
          error:
            "On PostgresLoader.updateRecord(), the query returned result.rowCount == nil.  Query executed was: " +
            query,
        };
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Check if a table exists in PostgreSQL (delegated to client)
   */
  async tableExists(
    tableName: string,
    schema: string = "public"
  ): Promise<boolean> {
    return this.client.tableExists(tableName, schema);
  }

  /**
   * Delete all records from a table (delegated to client)
   */
  async deleteAllRecords(
    tableName: string
  ): Promise<{ success: boolean; count: number; error?: string }> {
    return this.client.deleteAllRecords(tableName);
  }

  /**
   * Validate loaded data by comparing record counts
   */
  async validateRecordCount(
    tableName: string,
    expectedCount: number
  ): Promise<{ success: boolean; message: string }> {
    try {
      const result = await this.client
        .getPool()
        .query(`SELECT COUNT(*) FROM ${tableName}`);
      const count = parseInt(result.rows[0].count, 10);

      const success = count === expectedCount;
      return {
        success,
        message: success
          ? `Validation successful: ${count} records found as expected`
          : `Validation failed: Expected ${expectedCount} records, found ${count}`,
      };
    } catch (error) {
      return {
        success: false,
        message: `Error validating record count for ${tableName}: ${
          error instanceof Error ? error.message : String(error)
        }`,
      };
    }
  }

  /**
   * Track metadata for loaded records - delegates to the metadata manager
   */
  async trackMetadata(
    resourceId: string,
    resourceType: string,
    actionType: string,
    fieldName: string,
    previousValue: string,
    replacementValue: string,
    updatedBy: string,
    originalId?: string
  ): Promise<{ success: boolean; error?: string }> {
    return this.metadataManager.trackMetadata(
      resourceId,
      resourceType,
      actionType,
      fieldName,
      previousValue,
      replacementValue,
      updatedBy,
      originalId
    );
  }

  /**
   * Get failed migration records for a specific table
   */
  async getFailedRecords(
    tableName: string,
    resolved: boolean = false
  ): Promise<any[]> {
    return this.metadataManager.getFailedRecords(tableName, resolved);
  }

  /**
   * Mark a failed record as resolved
   */
  async resolveFailedRecord(
    id: string,
    resolvedBy: string,
    notes?: string
  ): Promise<{ success: boolean; error?: string }> {
    return this.metadataManager.resolveFailedRecord(id, resolvedBy, notes);
  }

  /**
   * Log the migration progress to the migration_log table
   */
  async logMigration(
    sourceTable: string,
    targetTable: string,
    recordsMigrated: number,
    successCount: number,
    failureCount: number,
    errorMessages: string[],
    startTime: Date,
    endTime: Date
  ): Promise<void> {
    const executionTimeSeconds =
      (endTime.getTime() - startTime.getTime()) / 1000;

    const query = `
      INSERT INTO migration_log (
        source_table, 
        target_table, 
        records_migrated, 
        success_count, 
        failure_count, 
        error_messages, 
        started_at, 
        completed_at, 
        execution_time_seconds
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    `;

    const values = [
      sourceTable,
      targetTable,
      recordsMigrated,
      successCount,
      failureCount,
      errorMessages.length > 0 ? errorMessages.join("\n") : null,
      startTime.toISOString(),
      endTime.toISOString(),
      executionTimeSeconds,
    ];

    try {
      await this.client.getPool().query(query, values);
    } catch (error) {
      console.error("Error logging migration:", error);
    }
  }

  /**
   * Run a migration task with proper batching, validation, and error handling
   */
  async runMigration<T extends Record<string, any>>(
    sourceData: T[],
    targetTable: string,
    sourceTable: string,
    batchSize: number = 1000,
    onConflict: string = "id"
  ): Promise<{ success: number; errors: Error[] }> {
    console.log(`Starting migration from ${sourceTable} to ${targetTable}`);
    console.log(
      `Processing ${sourceData.length} records with batch size ${batchSize}`
    );

    const result = await this.upsertData(
      targetTable,
      sourceData,
      onConflict,
      batchSize,
      sourceTable
    );

    console.log(
      `Migration completed: ${result.success} successful, ${result.errors.length} failed`
    );

    return result;
  }

  /**
   * Test optimal batch size for a specific table
   */
  async testOptimalBatchSize<T extends Record<string, any>>(
    tableName: string,
    sampleData: T[],
    batchSizesToTest: number[] = [100, 500, 1000, 2000, 5000],
    schema: string = "public"
  ): Promise<{ batchSize: number; timeSeconds: number }[]> {
    const results = [];

    // Create a test table
    const testTableName = `${tableName}_test_${Date.now()}`;

    try {
      // Get the table structure and create test table
      const createTableQuery = await this.client.getPool().query(`
        SELECT pg_get_ddl('${tableName}'::regclass) AS ddl
      `);

      const createTestTableQuery = createTableQuery.rows[0].ddl.replace(
        `CREATE TABLE ${tableName}`,
        `CREATE TABLE ${testTableName}`
      );

      await this.client.getPool().query(createTestTableQuery);

      // Test each batch size
      for (const batchSize of batchSizesToTest) {
        // Clean table before each test
        await this.client.getPool().query(`TRUNCATE TABLE ${testTableName}`);

        const startTime = new Date();
        await this.loadDataInternal(
          testTableName,
          sampleData,
          batchSize,
          schema
        );
        const endTime = new Date();

        const timeSeconds = (endTime.getTime() - startTime.getTime()) / 1000;
        results.push({ batchSize, timeSeconds });

        console.log(`Batch size ${batchSize}: ${timeSeconds} seconds`);
      }

      // Sort by performance
      return results.sort((a, b) => a.timeSeconds - b.timeSeconds);
    } catch (error) {
      console.error("Error testing batch sizes:", error);
      return results;
    } finally {
      // Clean up test table
      try {
        await this.client
          .getPool()
          .query(`DROP TABLE IF EXISTS ${testTableName}`);
      } catch (dropError) {
        console.error("Error dropping test table:", dropError);
      }
    }
  }

  /**
   * Close the PostgreSQL connection (delegated to client)
   */
  close(): void {
    this.client.close();
  }
}
