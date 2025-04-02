import { PoolClient, Pool } from "pg";
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
 * PostgreSQL implementation that handles loading data and managing metadata
 */
export class PostgresLoader {
  private client: PostgresClient;
  private constraintsCache: Map<string, TableConstraints> = new Map();

  constructor(client?: PostgresClient) {
    this.client = client || new PostgresClient();
  }

  /**
   * Get the underlying PostgreSQL pool
   */
  getPool(): Pool {
    return this.client.getPool();
  }

  /**
   * Standard method for loading data with consistent pre/post processing
   */
  async loadData<T extends Record<string, any>>(
    tableName: string,
    data: T[],
    batchSize: number = 100,
    sourceTable: string = "unknown",
    skipTableCheck: boolean = false,
    schema: string = "public"
  ): Promise<{ success: number; errors: Error[] }> {
    // Pre-processing logic (validation, logging start)
    const startTime = new Date();

    try {
      // Verify table exists before attempting to load (if not skipped)
      if (!skipTableCheck) {
        const exists = await this.tableExists(tableName, schema);
        if (!exists) {
          throw new Error(
            `Table ${schema}.${tableName} does not exist in the destination system`
          );
        }
      }

      // Get table constraints
      const constraints = await this.getTableConstraintsWithCache(tableName);

      // Validate records against constraints
      const { validRecords, invalidRecordsWithErrors } =
        validateRecordsAgainstConstraints(data, constraints);

      let successCount = 0;
      const errors: Error[] = [];

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
            record,
            schema
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
                    record.original_id || null,
                    schema
                  );
                } catch (metadataError) {
                  console.error(
                    `Failed to track metadata for record ${record.id}: ${metadataError}`
                  );
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
                  record,
                  schema
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
                    record,
                    schema
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
                  record,
                  schema
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
        endTime,
        schema
      );

      return { success: successCount, errors };
    } catch (error) {
      // Error handling
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
        endTime,
        schema
      );

      return { success: 0, errors: [e] };
    }
  }

  /**
   * Insert a single record as fallback when batch insert fails
   */
  private async insertSingleRecord<T extends Record<string, any>>(
    tableName: string,
    record: T,
    schema: string = "public"
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const columns = Object.keys(record);
      const placeholders = columns.map((_, i) => `$${i + 1}`);
      const values = columns.map((col) => record[col]);

      const query = `
        INSERT INTO ${schema}.${tableName} (${columns.join(", ")})
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
          record.original_id || null,
          schema
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
   * Upsert data into PostgreSQL (insert or update)
   */
  async upsertData<T extends Record<string, any>>(
    tableName: string,
    data: T[],
    onConflict: string = "id",
    batchSize: number = 1000,
    sourceTable: string = "unknown",
    skipTableCheck: boolean = false,
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
          throw new Error(`Table ${schema}.${tableName} does not exist`);
        }
      }

      // Get table constraints (either from parameter, cache, or database)
      const constraints = await this.getTableConstraintsWithCache(tableName);

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
            record,
            schema
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
                    record,
                    schema
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
                  record,
                  schema
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
        endTime,
        schema
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
        endTime,
        schema
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
              record.original_id || null,
              schema
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
          record.original_id || null,
          schema
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
   * Track metadata for loaded/modified records
   */
  async trackMetadata(
    resourceId: string,
    resourceType: string,
    actionType: string,
    fieldName: string,
    previousValue: string,
    replacementValue: string,
    updatedBy: string,
    originalId?: string | null,
    schema: string = "public"
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const metadataId = uuidv4();
      const now = new Date().toISOString();

      const query = `
        INSERT INTO ${schema}.metadata (
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

      await this.client.getPool().query(query, values);

      return {
        success: true,
      };
    } catch (error) {
      console.error("Error tracking metadata:", error);
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
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
    originalId?: string | null,
    schema: string = "public"
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const metadataId = uuidv4();
      const now = new Date().toISOString();

      const query = `
        INSERT INTO ${schema}.metadata (
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
   * Get failed migration records for a specific table
   */
  async getFailedRecords(
    tableName: string,
    resolved: boolean = false
  ): Promise<any[]> {
    try {
      const query = `
        SELECT * FROM failed_migration_records
        WHERE table_name = $1
        AND resolved = $2
        ORDER BY attempted_at DESC
      `;

      const result = await this.client
        .getPool()
        .query(query, [tableName, resolved]);
      return result.rows;
    } catch (error) {
      console.error("Error retrieving failed records:", error);
      return [];
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
    attemptedRecord: any,
    schema: string = "public"
  ): Promise<void> {
    const query = `
      INSERT INTO ${schema}.failed_migration_records (
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
    attemptedRecord: T,
    schema: string = "public"
  ): Promise<void> {
    const query = `
      INSERT INTO ${schema}.failed_migration_records (
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
   * Check if a table exists in PostgreSQL
   */
  async tableExists(
    tableName: string,
    schema: string = "public"
  ): Promise<boolean> {
    return this.client.tableExists(tableName, schema);
  }

  /**
   * Validate loaded data by comparing record counts
   */
  async validateRecordCount(
    tableName: string,
    expectedCount: number,
    schema: string = "public"
  ): Promise<{ success: boolean; message: string }> {
    try {
      const result = await this.client
        .getPool()
        .query(`SELECT COUNT(*) FROM ${schema}.${tableName}`);
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
    endTime: Date,
    schema: string = "public"
  ): Promise<void> {
    const executionTimeSeconds =
      (endTime.getTime() - startTime.getTime()) / 1000;

    const query = `
      INSERT INTO ${schema}.migration_log (
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
   * Close the PostgreSQL connection
   */
  close(): void {
    this.client.close();
  }
}
