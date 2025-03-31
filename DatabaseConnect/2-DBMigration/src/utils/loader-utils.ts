// Utility functions for loaders
import { PostgrestError, SupabaseClient } from "@supabase/supabase-js";

/**
 * Process errors from Supabase operations
 * @param error Error object from Supabase
 * @param operation Operation being performed
 * @param tableName Table being operated on
 * @returns Formatted error message
 */
export function processSupabaseError(
  error: PostgrestError | null,
  operation: string,
  tableName: string
): string {
  if (!error) return "";

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
export function batchRecords<T>(records: T[], batchSize: number): T[][] {
  const batches: T[][] = [];
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
export async function recordExists(
  supabase: any,
  tableName: string,
  fieldName: string,
  value: any
): Promise<boolean> {
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
export async function supaUpsertRecords<T extends Record<string, any>>(
  supabase: any,
  tableName: string,
  records: T[],
  onConflict: string = "id"
): Promise<{ success: number; errors: Error[] }> {
  const errors: Error[] = [];
  let successCount = 0;

  // Process records individually instead of in batch
  for (const record of records) {
    try {
      const { error } = await supabase
        .from(tableName)
        .upsert([record], { onConflict, returning: "minimal" });

      if (error) {
        // Log to failed_migration_records table
        await supabase.from("failed_migration_records").insert({
          table_name: tableName,
          original_id: record.original_id || null,
          original_translations_id: record.original_translations_id || null,
          error_message: processSupabaseError(error, "upsert", tableName),
          attempted_record: record,
        });

        errors.push(
          new Error(processSupabaseError(error, "upsert", tableName))
        );
      } else {
        successCount++;
      }
    } catch (error) {
      // Handle unexpected errors
      const errorMessage =
        error instanceof Error ? error.message : String(error);

      // Log to failed_migration_records table
      await supabase.from("failed_migration_records").insert({
        table_name: tableName,
        original_id: record.original_id || null,
        original_translations_id: record.original_translations_id || null,
        error_message: errorMessage,
        attempted_record: record,
      });

      errors.push(error instanceof Error ? error : new Error(String(error)));
    }
  }

  return { success: successCount, errors };
}

/**
 * Check if a table exists in Supabase
 */
export async function tableExists(
  tableName: string,
  supabaseClient: SupabaseClient
): Promise<boolean> {
  try {
    // A more reliable way to check if a table exists is to query the information_schema
    // or simply attempt to get the count from the table
    const { count, error } = await supabaseClient
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
