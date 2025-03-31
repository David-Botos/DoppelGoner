"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.BaseLoader = void 0;
/**
 * Base abstract class for data loaders
 * Provides core functionality and common patterns while allowing specific implementations
 */
class BaseLoader {
    /**
     * Standard method for loading data with consistent pre/post processing
     * This provides the public API with error handling and logging
     */
    async loadData(tableName, data, batchSize = 100, sourceTable = "unknown", skipTableCheck = false) {
        // Pre-processing logic (validation, logging start)
        const startTime = new Date();
        try {
            // Verify table exists before attempting to load (if not skipped)
            if (!skipTableCheck) {
                const exists = await this.tableExists(tableName);
                if (!exists) {
                    throw new Error(`Table ${tableName} does not exist in the destination system`);
                }
            }
            // Call the implementation-specific loading method
            const result = await this.loadDataInternal(tableName, data, batchSize);
            // Post-processing logic (logging completion)
            const endTime = new Date();
            await this.logMigration(sourceTable, tableName, data.length, result.success, data.length - result.success, result.errors.map((e) => e.message), startTime, endTime);
            return result;
        }
        catch (error) {
            // Error handling
            const endTime = new Date();
            const e = error instanceof Error ? error : new Error(String(error));
            await this.logMigration(sourceTable, tableName, data.length, 0, data.length, [e.message], startTime, endTime);
            return { success: 0, errors: [e] };
        }
    }
    /**
     * Load data with upserting behavior (insert or update)
     * Default implementation calls loadDataInternal, but can be overridden
     * for database-specific optimizations
     */
    async upsertData(tableName, data, onConflict = "id", batchSize = 100, sourceTable = "unknown", skipTableCheck = false) {
        return this.loadData(tableName, data, batchSize, sourceTable, skipTableCheck);
    }
}
exports.BaseLoader = BaseLoader;
