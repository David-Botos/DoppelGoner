"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SupabaseLoader = void 0;
const supabase_js_1 = require("@supabase/supabase-js");
const config_1 = require("../config/config");
const loader_1 = require("./loader");
const loader_utils_1 = require("../utils/loader-utils");
const supa_metadata_manager_1 = require("./supa-metadata-manager");
/**
 * Supabase implementation of the BaseLoader abstract class
 */
class SupabaseLoader extends loader_1.BaseLoader {
    constructor(metadataManager) {
        super();
        this.client = (0, supabase_js_1.createClient)(config_1.supabaseConfig.url, config_1.supabaseConfig.key);
        this.metadataManager =
            metadataManager || new supa_metadata_manager_1.SupabaseMetadataManager(this.client);
    }
    /**
     * Implementation of the core data loading functionality for Supabase
     */
    async loadDataInternal(tableName, data, batchSize = 100) {
        let successCount = 0;
        const errors = [];
        // Break data into batches
        const batches = (0, loader_utils_1.batchRecords)(data, batchSize);
        for (const batch of batches) {
            try {
                const { error } = await this.client.from(tableName).insert(batch);
                if (error) {
                    errors.push(new Error((0, loader_utils_1.processSupabaseError)(error, "insert", tableName)));
                }
                else {
                    successCount += batch.length;
                    // Track metadata for each successfully loaded record
                    for (const record of batch) {
                        try {
                            await this.trackMetadata(record.id, tableName, "insert", "all", "Imported from Snowflake", JSON.stringify(record), "ETL Process", record.original_id || null);
                        }
                        catch (metadataError) {
                            console.error(`Failed to track metadata for record ${record.id}: ${metadataError}`);
                            // Don't fail the whole process for metadata tracking failures
                        }
                    }
                }
            }
            catch (error) {
                errors.push(error instanceof Error ? error : new Error(String(error)));
            }
        }
        return { success: successCount, errors };
    }
    /**
     * Supabase-specific implementation of upsert functionality
     * Overrides the base implementation for better performance
     */
    async upsertData(tableName, data, onConflict = "id", batchSize = 100, sourceTable = "unknown", skipTableCheck = false) {
        // Pre-processing logic
        const startTime = new Date();
        let successCount = 0;
        const errors = [];
        try {
            // Verify table exists (if not skipped)
            if (!skipTableCheck) {
                const exists = await this.tableExists(tableName);
                if (!exists) {
                    throw new Error(`Table ${tableName} does not exist in Supabase`);
                }
            }
            // Break data into batches
            const batches = (0, loader_utils_1.batchRecords)(data, batchSize);
            for (const batch of batches) {
                try {
                    const result = await (0, loader_utils_1.supaUpsertRecords)(this.client, tableName, batch, onConflict);
                    successCount += result.success;
                    errors.push(...result.errors);
                    // For successful records, track metadata
                    if (result.success > 0) {
                        for (const record of batch) {
                            try {
                                await this.trackMetadata(record.id, tableName, "upsert", "all", "Imported from Snowflake", JSON.stringify(record), "ETL Process", record.original_id || null);
                            }
                            catch (metadataError) {
                                console.error(`Failed to track metadata for record ${record.id}: ${metadataError}`);
                                // Don't fail the whole process for metadata tracking failures
                            }
                        }
                    }
                }
                catch (error) {
                    errors.push(error instanceof Error ? error : new Error(String(error)));
                }
            }
            // Post-processing logic (logging completion)
            const endTime = new Date();
            await this.logMigration(sourceTable, tableName, data.length, successCount, data.length - successCount, errors.map((e) => e.message), startTime, endTime);
            return { success: successCount, errors };
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
     * Type-safe method for loading data into a Supabase table
     */
    async loadTypedData(tableName, data, batchSize = 100, sourceTable = "unknown") {
        return this.loadData(tableName, data, batchSize, sourceTable);
    }
    /**
     * Type-safe method for upserting data into a Supabase table
     */
    async upsertTypedData(tableName, data, onConflict = "id", batchSize = 100, sourceTable = "unknown") {
        return this.upsertData(tableName, data, onConflict, batchSize, sourceTable);
    }
    /**
     * Update a record in Supabase
     */
    async updateRecord(tableName, id, data) {
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
                        await this.trackMetadata(id, tableName.toString(), "update", fieldName, oldValue !== null && oldValue !== undefined
                            ? String(oldValue)
                            : "null", newValue !== null && newValue !== undefined
                            ? String(newValue)
                            : "null", "ETL Process", existingRecord.original_id || null);
                    }
                }
            }
            return {
                success: !error,
                error: error?.message,
            };
        }
        catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : String(error),
            };
        }
    }
    /**
     * Delete all records from a table - useful for testing or rollbacks
     */
    async deleteAllRecords(tableName) {
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
        }
        catch (error) {
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
    async trackMetadata(resourceId, resourceType, actionType, fieldName, previousValue, replacementValue, updatedBy, originalId) {
        return this.metadataManager.trackMetadata(resourceId, resourceType, actionType, fieldName, previousValue, replacementValue, updatedBy, originalId);
    }
    /**
     * Get failed migration records for a specific table
     */
    async getFailedRecords(tableName, resolved = false) {
        return this.metadataManager.getFailedRecords(tableName, resolved);
    }
    /**
     * Mark a failed record as resolved
     */
    async resolveFailedRecord(id, resolvedBy, notes) {
        return this.metadataManager.resolveFailedRecord(id, resolvedBy, notes);
    }
    /**
     * Validate loaded data by comparing record counts
     */
    async validateRecordCount(tableName, expectedCount) {
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
        }
        catch (error) {
            return {
                success: false,
                message: `Error validating record count for ${tableName}: ${error instanceof Error ? error.message : String(error)}`,
            };
        }
    }
    /**
     * TODO: Sunset this abstract implementation in favor for the reusable util
     * Check if a table exists in Supabase
     */
    async tableExists(tableName) {
        try {
            // A more reliable way to check if a table exists is to query the information_schema
            // or simply attempt to get the count from the table
            const { count, error } = await this.client
                .from(tableName)
                .select("*", { count: "exact", head: true })
                .limit(1);
            // If we get an error that contains "relation does not exist", the table doesn't exist
            if (error) {
                if (error.message &&
                    error.message.includes("relation") &&
                    error.message.includes("does not exist")) {
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
        }
        catch (error) {
            console.error(`Error checking if table ${tableName} exists:`, error);
            // For unexpected errors, assume the table exists to prevent blocking operations
            return true;
        }
    }
    /**
     * Log the migration progress to the migration_log table
     */
    async logMigration(sourceTable, targetTable, recordsMigrated, successCount, failureCount, errorMessages, startTime, endTime) {
        const executionTimeSeconds = (endTime.getTime() - startTime.getTime()) / 1000;
        const migrationLog = {
            source_table: sourceTable,
            target_table: targetTable,
            records_migrated: recordsMigrated,
            success_count: successCount,
            failure_count: failureCount,
            error_messages: errorMessages.length > 0 ? errorMessages.join("\n") : undefined,
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
        }
        catch (error) {
            console.error("Error logging migration:", error);
        }
    }
    /**
     * Close the Supabase client (no-op for now)
     */
    close() {
        // Clean up any resources if needed
    }
}
exports.SupabaseLoader = SupabaseLoader;
