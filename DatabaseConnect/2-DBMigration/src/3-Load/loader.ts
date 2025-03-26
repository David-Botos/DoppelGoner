// Base loader interface for data loading
import { MigratedData } from "../types/transformation-types";

/**
 * Base interface for data loaders
 */
export interface BaseLoader {
  /**
   * Load data into a destination system
   * @param tableName The name of the table to load data into
   * @param data The transformed data to load
   * @returns Result with success count and any errors
   */
  loadData<T extends Record<string, any>>(
    tableName: string,
    data: T[],
    batchSize?: number
  ): Promise<{ success: number; errors: Error[] }>;

  /**
   * Log the migration progress
   * @param sourceTable Source table
   * @param targetTable Target table
   * @param recordsMigrated Total number of records attempted to migrate
   * @param successCount Number of records successfully migrated
   * @param failureCount Number of records that failed to migrate
   * @param errorMessages Error messages if any
   * @param startTime Start time of the migration
   * @param endTime End time of the migration
   */
  logMigration(
    sourceTable: string,
    targetTable: string,
    recordsMigrated: number,
    successCount: number,
    failureCount: number,
    errorMessages: string[],
    startTime: Date,
    endTime: Date
  ): Promise<void>;

  /**
   * Validate loaded data
   * @param tableName Table to validate
   * @param expectedCount Expected number of records
   * @returns Validation result
   */
  validateRecordCount(
    tableName: string,
    expectedCount: number
  ): Promise<{ success: boolean; message: string }>;

  /**
   * Check if a table exists
   * @param tableName Table name to check
   * @returns Boolean indicating if the table exists
   */
  tableExists(tableName: string): Promise<boolean>;

  /**
   * Track metadata for loaded records
   * @param resourceId ID of the resource being tracked
   * @param resourceType Type of resource (table name)
   * @param actionType Type of action performed (insert, update, delete)
   * @param fieldName Name of the field being modified (optional for inserts)
   * @param previousValue Previous value of the field ('Imported from Snowflake' for new records)
   * @param replacementValue New value of the field
   * @param updatedBy User or process that made the change
   * @param originalId Original ID from the source system (optional)
   * @returns Success flag and error if any
   */
  trackMetadata(
    resourceId: string,
    resourceType: string,
    actionType: string,
    fieldName: string,
    previousValue: string,
    replacementValue: string,
    updatedBy: string,
    originalId?: string
  ): Promise<{ success: boolean; error?: string }>;

  /**
   * Close the loader and free resources
   */
  close(): void;
}
