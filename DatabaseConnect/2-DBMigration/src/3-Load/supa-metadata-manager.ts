import { SupabaseClient } from "@supabase/supabase-js";
import { MetadataManager } from "./metadata-manager";

/**
 * Supabase implementation of the MetadataManager interface
 */
export class SupabaseMetadataManager implements MetadataManager {
  constructor(private client: SupabaseClient) {}

  /**
   * Track metadata for loaded records in Supabase
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
    try {
      const metadataRecord = {
        id: crypto.randomUUID(), // Generate UUID for metadata record
        resource_id: resourceId,
        resource_type: resourceType,
        last_action_date: new Date().toISOString(),
        last_action_type: actionType,
        field_name: fieldName,
        previous_value: previousValue,
        replacement_value: replacementValue,
        updated_by: updatedBy,
        created: new Date().toISOString(),
        last_modified: new Date().toISOString(),
        original_id: originalId || null,
      };

      const { error } = await this.client
        .from("metadata")
        .insert(metadataRecord);

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
   * Get failed migration records for a specific table
   */
  async getFailedRecords(
    tableName: string,
    resolved: boolean = false
  ): Promise<any[]> {
    try {
      const { data, error } = await this.client
        .from("failed_migration_records")
        .select("*")
        .eq("table_name", tableName)
        .eq("resolved", resolved)
        .order("attempted_at", { ascending: false });

      if (error) {
        console.error(`Error fetching failed records: ${error.message}`);
        return [];
      }

      return data || [];
    } catch (error) {
      console.error("Error retrieving failed records:", error);
      return [];
    }
  }

  /**
   * Mark a failed record as resolved
   */
  async resolveFailedRecord(
    id: string,
    resolvedBy: string,
    notes?: string
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const { error } = await this.client
        .from("failed_migration_records")
        .update({
          resolved: true,
          resolved_at: new Date().toISOString(),
          resolved_by: resolvedBy,
          resolution_notes: notes,
        })
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
}
