/**
 * Base abstract class for data loaders
 * Provides core functionality and common patterns while allowing specific implementations
 */
export abstract class BaseLoader {
  /**
   * Core method to load data into a destination table
   * (to be implemented by specific loader implementations)
   */
  protected abstract loadDataInternal<T extends Record<string, any>>(
    tableName: string,
    data: T[],
    batchSize?: number
  ): Promise<{ success: number; errors: Error[] }>;

  /**
   * Check if a table exists in the destination system
   */
  abstract tableExists(tableName: string): Promise<boolean>;

  /**
   * Close the loader and free resources
   */
  abstract close(): void;

  /**
   * Standard method for loading data with consistent pre/post processing
   * This provides the public API with error handling and logging
   */
  async loadData<T extends Record<string, any>>(
    tableName: string,
    data: T[],
    batchSize: number = 100,
    sourceTable: string = "unknown",
    skipTableCheck: boolean = false
  ): Promise<{ success: number; errors: Error[] }> {
    // Pre-processing logic (validation, logging start)
    const startTime = new Date();

    try {
      // Verify table exists before attempting to load (if not skipped)
      if (!skipTableCheck) {
        const exists = await this.tableExists(tableName);
        if (!exists) {
          throw new Error(
            `Table ${tableName} does not exist in the destination system`
          );
        }
      }

      // Call the implementation-specific loading method
      const result = await this.loadDataInternal(tableName, data, batchSize);

      // Post-processing logic (logging completion)
      const endTime = new Date();
      await this.logMigration(
        sourceTable,
        tableName,
        data.length,
        result.success,
        data.length - result.success,
        result.errors.map((e) => e.message),
        startTime,
        endTime
      );

      return result;
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
   * Load data with upserting behavior (insert or update)
   * Default implementation calls loadDataInternal, but can be overridden
   * for database-specific optimizations
   */
  async upsertData<T extends Record<string, any>>(
    tableName: string,
    data: T[],
    onConflict: string = "id",
    batchSize: number = 100,
    sourceTable: string = "unknown",
    skipTableCheck: boolean = false
  ): Promise<{ success: number; errors: Error[] }> {
    return this.loadData(
      tableName,
      data,
      batchSize,
      sourceTable,
      skipTableCheck
    );
  }

  /**
   * Validate loaded data by comparing record counts
   */
  abstract validateRecordCount(
    tableName: string,
    expectedCount: number
  ): Promise<{ success: boolean; message: string }>;

  /**
   * Track metadata for loaded records
   */
  abstract trackMetadata(
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
   * Log the migration progress
   */
  abstract logMigration(
    sourceTable: string,
    targetTable: string,
    recordsMigrated: number,
    successCount: number,
    failureCount: number,
    errorMessages: string[],
    startTime: Date,
    endTime: Date
  ): Promise<void>;
}
