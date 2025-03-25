// Supabase loader for data loading
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
import { MigratedData } from "../types/transformation-types";
import { BaseLoader } from "./loader";
import {
  batchRecords,
  processSupabaseError,
  upsertRecords,
} from "../utils/loader-utils";

// Type mapping for tables to their respective interfaces
type TableTypes = {
  organization: SupabaseOrganization;
  service: SupabaseService;
  location: SupabaseLocation;
  service_at_location: SupabaseServiceAtLocation;
  address: SupabaseAddress;
  phone: SupabasePhone;
};

export class SupabaseLoader implements BaseLoader {
  private client: SupabaseClient;

  constructor() {
    this.client = createClient(supabaseConfig.url, supabaseConfig.key);
  }

  /**
   * Load data into a Supabase table
   * @param tableName The name of the table to load data into
   * @param data The transformed data to load
   * @returns Result with success count and any errors
   */
  async loadData(
    tableName: string,
    data: any[],
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
        }
      } catch (error) {
        errors.push(error instanceof Error ? error : new Error(String(error)));
      }
    }

    return { success: successCount, errors };
  }

  /**
   * Load data into a Supabase table with type safety
   * @param tableName The name of the table to load data into
   * @param data The transformed data to load
   * @returns Result with success count and any errors
   */
  async loadTypedData<T extends keyof TableTypes>(
    tableName: T,
    data: TableTypes[T][],
    batchSize: number = 100
  ): Promise<{ success: number; errors: Error[] }> {
    return this.loadData(tableName, data, batchSize);
  }

  /**
   * Upsert data into a Supabase table (insert or update)
   * @param tableName The name of the table
   * @param data The data to upsert
   * @param onConflict Fields to check for conflicts
   * @returns Result with success count and errors
   */
  async upsertData(
    tableName: string,
    data: any[],
    onConflict: string = "id",
    batchSize: number = 100
  ): Promise<{ success: number; errors: Error[] }> {
    let successCount = 0;
    const errors: Error[] = [];

    // Break data into batches
    const batches = batchRecords(data, batchSize);

    for (const batch of batches) {
      try {
        const result = await upsertRecords(
          this.client,
          tableName,
          batch,
          onConflict
        );
        successCount += result.success;
        errors.push(...result.errors);
      } catch (error) {
        errors.push(error instanceof Error ? error : new Error(String(error)));
      }
    }

    return { success: successCount, errors };
  }

  /**
   * Upsert data into a Supabase table with type safety
   * @param tableName The name of the table
   * @param data The data to upsert
   * @param onConflict Fields to check for conflicts
   * @returns Result with success count and errors
   */
  async upsertTypedData<T extends keyof TableTypes>(
    tableName: T,
    data: TableTypes[T][],
    onConflict: string = "id",
    batchSize: number = 100
  ): Promise<{ success: number; errors: Error[] }> {
    return this.upsertData(tableName, data, onConflict, batchSize);
  }

  /**
   * Log the migration progress to the migration_log table
   * @param sourceTable Source table in Snowflake
   * @param targetTable Target table in Supabase
   * @param recordsMigrated Total number of records attempted to migrate
   * @param successCount Number of records successfully migrated
   * @param failureCount Number of records that failed to migrate
   * @param errorMessages Error messages if any
   * @param startTime Start time of the migration
   * @param endTime End time of the migration
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
   * Check if a table exists in Supabase
   * @param tableName Table name to check
   * @returns Boolean indicating if the table exists
   */
  async tableExists(tableName: string): Promise<boolean> {
    try {
      const { data, error } = await this.client.rpc("table_exists", {
        table_name: tableName,
        schema_name: supabaseConfig.schema,
      });

      if (error) {
        console.error(`Error checking if table exists: ${error.message}`);
        return false;
      }

      return !!data;
    } catch (error) {
      console.error(`Error checking if table ${tableName} exists:`, error);
      return false;
    }
  }

  /**
   * Validate loaded data by comparing record counts
   * @param tableName Table to validate
   * @param expectedCount Expected number of records
   * @returns Validation result with success flag and message
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
   * Update a record in Supabase
   * @param tableName Table containing the record
   * @param id ID of the record to update
   * @param data Update data
   * @returns Success flag and error if any
   */
  async updateRecord<T extends keyof TableTypes>(
    tableName: T,
    id: string,
    data: Partial<TableTypes[T]>
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const { error } = await this.client
        .from(tableName)
        .update(data)
        .eq("id", id);

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
   * Delete records from a table - useful for testing or rollbacks
   * @param tableName Table to delete from
   * @returns Success flag and count of deleted records
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
   * Close the Supabase client
   */
  close(): void {
    // Clean up any resources if needed
  }
}
