import {
  createClient,
  SupabaseClient,
  PostgrestError,
} from "@supabase/supabase-js";
import { supabaseConfig } from "../config/config";
import {
  SupabaseOrganization,
  SupabaseService,
  SupabaseLocation,
  SupabaseServiceAtLocation,
  SupabaseAddress,
  SupabasePhone,
  SupabaseMigrationLog,
} from "../types/supabase-types";
import { BaseLoader } from "./loader";
import {
  batchRecords,
  processSupabaseError,
  supaUpsertRecords,
} from "../utils/loader-utils";
import { SupabaseMetadataManager } from "./supa-metadata-manager";
import { MetadataManager } from "./metadata-manager";

// Type mapping for tables to their respective interfaces
type TableTypes = {
  organization: SupabaseOrganization;
  service: SupabaseService;
  location: SupabaseLocation;
  service_at_location: SupabaseServiceAtLocation;
  address: SupabaseAddress;
  phone: SupabasePhone;
};

/**
 * Supabase implementation of the BaseLoader abstract class
 */
export class SupabaseLoader extends BaseLoader {
  private client: SupabaseClient;
  private metadataManager: MetadataManager;

  constructor(metadataManager?: MetadataManager) {
    super();
    this.client = createClient(supabaseConfig.url, supabaseConfig.key);
    this.metadataManager =
      metadataManager || new SupabaseMetadataManager(this.client);
  }

  /**
   * Implementation of the core data loading functionality for Supabase
   */
  protected async loadDataInternal<T extends Record<string, any>>(
    tableName: string,
    data: T[],
    batchSize: number = 100
  ): Promise<{ success: number; errors: Error[] }> {
    let successCount = 0;
    const errors: Error[] = [];

    // Break data into batches
    const batches = batchRecords(data, batchSize);

    for (const batch of batches) {
      try {
        const { error } = await this.client.from(tableName).insert(batch);

        if (error) {
          errors.push(
            new Error(processSupabaseError(error, "insert", tableName))
          );
        } else {
          successCount += batch.length;

          // Track metadata for each successfully loaded record
          for (const record of batch) {
            try {
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
            } catch (metadataError) {
              console.error(
                `Failed to track metadata for record ${record.id}: ${metadataError}`
              );
              // Don't fail the whole process for metadata tracking failures
            }
          }
        }
      } catch (error) {
        errors.push(error instanceof Error ? error : new Error(String(error)));
      }
    }

    return { success: successCount, errors };
  }

  /**
   * Supabase-specific implementation of upsert functionality
   * Overrides the base implementation for better performance
   */
  override async upsertData<T extends Record<string, any>>(
    tableName: string,
    data: T[],
    onConflict: string = "id",
    batchSize: number = 100,
    sourceTable: string = "unknown",
    skipTableCheck: boolean = false
  ): Promise<{ success: number; errors: Error[] }> {
    // Pre-processing logic
    const startTime = new Date();
    let successCount = 0;
    const errors: Error[] = [];

    try {
      // Verify table exists (if not skipped)
      if (!skipTableCheck) {
        const exists = await this.tableExists(tableName);
        if (!exists) {
          throw new Error(`Table ${tableName} does not exist in Supabase`);
        }
      }

      // Break data into batches
      const batches = batchRecords(data, batchSize);

      for (const batch of batches) {
        try {
          const result = await supaUpsertRecords(
            this.client,
            tableName,
            batch,
            onConflict
          );
          successCount += result.success;
          errors.push(...result.errors);

          // For successful records, track metadata
          if (result.success > 0) {
            for (const record of batch) {
              try {
                await this.trackMetadata(
                  record.id,
                  tableName,
                  "upsert",
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
          }
        } catch (error) {
          errors.push(
            error instanceof Error ? error : new Error(String(error))
          );
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
        endTime
      );

      return { success: 0, errors: [e] };
    }
  }

  /**
   * Type-safe method for loading data into a Supabase table
   */
  async loadTypedData<T extends keyof TableTypes>(
    tableName: T,
    data: TableTypes[T][],
    batchSize: number = 100,
    sourceTable: string = "unknown"
  ): Promise<{ success: number; errors: Error[] }> {
    return this.loadData(tableName, data, batchSize, sourceTable);
  }

  /**
   * Type-safe method for upserting data into a Supabase table
   */
  async upsertTypedData<T extends keyof TableTypes>(
    tableName: T,
    data: TableTypes[T][],
    onConflict: string = "id",
    batchSize: number = 100,
    sourceTable: string = "unknown"
  ): Promise<{ success: number; errors: Error[] }> {
    return this.upsertData(tableName, data, onConflict, batchSize, sourceTable);
  }

  /**
   * Update a record in Supabase
   */
  async updateRecord<T extends keyof TableTypes>(
    tableName: T,
    id: string,
    data: Partial<TableTypes[T]>
  ): Promise<{ success: boolean; error?: string }> {
    try {
      // First get the current record to compare values
      const { data: existingRecord, error: fetchError } = await this.client
        .from(tableName)
        .select("*")
        .eq("id", id)
        .single();

      if (fetchError) {
        return {
          success: false,
          error: fetchError.message,
        };
      }

      // Perform the update
      const { error } = await this.client
        .from(tableName)
        .update(data)
        .eq("id", id);

      if (!error) {
        // Track metadata for each changed field
        for (const [fieldName, newValue] of Object.entries(data)) {
          const oldValue = existingRecord[fieldName];

          // Only track if the value actually changed
          if (oldValue !== newValue) {
            await this.trackMetadata(
              id,
              tableName.toString(),
              "update",
              fieldName,
              oldValue !== null && oldValue !== undefined
                ? String(oldValue)
                : "null",
              newValue !== null && newValue !== undefined
                ? String(newValue)
                : "null",
              "ETL Process",
              existingRecord.original_id || null
            );
          }
        }
      }

      return {
        success: !error,
        error: error?.message,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Delete all records from a table - useful for testing or rollbacks
   */
  async deleteAllRecords(
    tableName: string
  ): Promise<{ success: boolean; count: number; error?: string }> {
    try {
      // First count the records
      const { count: beforeCount, error: countError } = await this.client
        .from(tableName)
        .select("*", { count: "exact", head: true });

      if (countError) {
        return {
          success: false,
          count: 0,
          error: `Failed to count records: ${countError.message}`,
        };
      }

      // Then delete them
      const { error } = await this.client.from(tableName).delete().gt("id", "");

      return {
        success: !error,
        count: beforeCount || 0,
        error: error?.message,
      };
    } catch (error) {
      return {
        success: false,
        count: 0,
        error: error instanceof Error ? error.message : String(error),
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
   * Validate loaded data by comparing record counts
   */
  async validateRecordCount(
    tableName: string,
    expectedCount: number
  ): Promise<{ success: boolean; message: string }> {
    try {
      const { count, error } = await this.client
        .from(tableName)
        .select("*", { count: "exact", head: true });

      if (error) {
        return {
          success: false,
          message: `Failed to get count for ${tableName}: ${error.message}`,
        };
      }

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
   * TODO: Sunset this abstract implementation in favor for the reusable util
   * Check if a table exists in Supabase
   */
  async tableExists(tableName: string): Promise<boolean> {
    try {
      // A more reliable way to check if a table exists is to query the information_schema
      // or simply attempt to get the count from the table
      const { count, error } = await this.client
        .from(tableName)
        .select("*", { count: "exact", head: true })
        .limit(1);

      // If we get an error that contains "relation does not exist", the table doesn't exist
      if (error) {
        if (
          error.message &&
          error.message.includes("relation") &&
          error.message.includes("does not exist")
        ) {
          console.log(`Table ${tableName} does not exist in Supabase`);
          return false;
        }

        // Other errors might be permissions or connectivity issues
        console.error(`Error checking if table exists: ${error.message}`);
        // For non-existence errors, assume the table exists but there's another issue
        return true;
      }

      // No error means the table exists
      return true;
    } catch (error) {
      console.error(`Error checking if table ${tableName} exists:`, error);
      // For unexpected errors, assume the table exists to prevent blocking operations
      return true;
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
    endTime: Date
  ): Promise<void> {
    const executionTimeSeconds =
      (endTime.getTime() - startTime.getTime()) / 1000;

    const migrationLog: Omit<SupabaseMigrationLog, "id"> = {
      source_table: sourceTable,
      target_table: targetTable,
      records_migrated: recordsMigrated,
      success_count: successCount,
      failure_count: failureCount,
      error_messages:
        errorMessages.length > 0 ? errorMessages.join("\n") : undefined,
      started_at: startTime.toISOString(),
      completed_at: endTime.toISOString(),
      execution_time_seconds: executionTimeSeconds,
    };

    try {
      const { error } = await this.client
        .from("migration_log")
        .insert(migrationLog);
      if (error) {
        console.error(`Failed to log migration: ${error.message}`);
      }
    } catch (error) {
      console.error("Error logging migration:", error);
    }
  }

  /**
   * Close the Supabase client (no-op for now)
   */
  close(): void {
    // Clean up any resources if needed
  }
}
