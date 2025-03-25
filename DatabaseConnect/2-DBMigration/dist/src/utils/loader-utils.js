"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.processSupabaseError = processSupabaseError;
exports.batchRecords = batchRecords;
exports.recordExists = recordExists;
exports.upsertRecords = upsertRecords;
/**
 * Process errors from Supabase operations
 * @param error Error object from Supabase
 * @param operation Operation being performed
 * @param tableName Table being operated on
 * @returns Formatted error message
 */
function processSupabaseError(error, operation, tableName) {
    if (!error)
        return "";
    // Handle constraint violations
    if (error.code === "23505") {
        return `Unique constraint violation in ${tableName} during ${operation}: ${error.message}`;
    }
    // Handle foreign key violations
    if (error.code === "23503") {
        return `Foreign key constraint violation in ${tableName} during ${operation}: ${error.message}`;
    }
    // Handle other errors
    return `Error in ${tableName} during ${operation}: ${error.message}`;
}
/**
 * Group records by batch size for processing
 * @param records Records to group
 * @param batchSize Size of each batch
 * @returns Array of batches
 */
function batchRecords(records, batchSize) {
    const batches = [];
    for (let i = 0; i < records.length; i += batchSize) {
        batches.push(records.slice(i, i + batchSize));
    }
    return batches;
}
/**
 * Check if a record exists in Supabase
 * @param supabase Supabase client
 * @param tableName Table to check
 * @param fieldName Field to check by
 * @param value Value to check for
 * @returns Boolean indicating if the record exists
 */
async function recordExists(supabase, tableName, fieldName, value) {
    const { data, error } = await supabase
        .from(tableName)
        .select("id")
        .eq(fieldName, value)
        .single();
    if (error && error.code !== "PGRST116") {
        // PGRST116 means no rows returned
        console.error(`Error checking if record exists: ${error.message}`);
    }
    return !!data;
}
/**
 * Handle upsert operation (insert or update)
 * @param supabase Supabase client
 * @param tableName Table to upsert into
 * @param records Records to upsert
 * @param onConflict Fields to check for conflicts
 * @returns Result with success count and errors
 */
async function upsertRecords(supabase, tableName, records, onConflict = "id") {
    const errors = [];
    try {
        const { data, error } = await supabase
            .from(tableName)
            .upsert(records, { onConflict, returning: "minimal" });
        if (error) {
            errors.push(new Error(processSupabaseError(error, "upsert", tableName)));
            return { success: 0, errors };
        }
        return { success: records.length, errors };
    }
    catch (error) {
        errors.push(error instanceof Error ? error : new Error(String(error)));
        return { success: 0, errors };
    }
}
