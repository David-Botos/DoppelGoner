"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SupabaseLoader = void 0;
// Supabase loader for data loading
const supabase_js_1 = require("@supabase/supabase-js");
const config_1 = require("../config/config");
const loader_utils_1 = require("../utils/loader-utils");
class SupabaseLoader {
    constructor() {
        this.client = (0, supabase_js_1.createClient)(config_1.supabaseConfig.url, config_1.supabaseConfig.key);
    }
    /**
     * Load data into a Supabase table
     * @param tableName The name of the table to load data into
     * @param data The transformed data to load
     * @returns Result with success count and any errors
     */
    async loadData(tableName, data, batchSize = 100) {
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
                }
            }
            catch (error) {
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
    async loadTypedData(tableName, data, batchSize = 100) {
        return this.loadData(tableName, data, batchSize);
    }
    /**
     * Upsert data into a Supabase table (insert or update)
     * @param tableName The name of the table
     * @param data The data to upsert
     * @param onConflict Fields to check for conflicts
     * @returns Result with success count and errors
     */
    async upsertData(tableName, data, onConflict = "id", batchSize = 100) {
        let successCount = 0;
        const errors = [];
        // Break data into batches
        const batches = (0, loader_utils_1.batchRecords)(data, batchSize);
        for (const batch of batches) {
            try {
                const result = await (0, loader_utils_1.upsertRecords)(this.client, tableName, batch, onConflict);
                successCount += result.success;
                errors.push(...result.errors);
            }
            catch (error) {
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
    async upsertTypedData(tableName, data, onConflict = "id", batchSize = 100) {
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
     * Check if a table exists in Supabase
     * @param tableName Table name to check
     * @returns Boolean indicating if the table exists
     */
    async tableExists(tableName) {
        try {
            const { data, error } = await this.client.rpc("table_exists", {
                table_name: tableName,
                schema_name: config_1.supabaseConfig.schema,
            });
            if (error) {
                console.error(`Error checking if table exists: ${error.message}`);
                return false;
            }
            return !!data;
        }
        catch (error) {
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
     * Update a record in Supabase
     * @param tableName Table containing the record
     * @param id ID of the record to update
     * @param data Update data
     * @returns Success flag and error if any
     */
    async updateRecord(tableName, id, data) {
        try {
            const { error } = await this.client
                .from(tableName)
                .update(data)
                .eq("id", id);
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
     * Delete records from a table - useful for testing or rollbacks
     * @param tableName Table to delete from
     * @returns Success flag and count of deleted records
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
     * Close the Supabase client
     */
    close() {
        // Clean up any resources if needed
    }
}
exports.SupabaseLoader = SupabaseLoader;
