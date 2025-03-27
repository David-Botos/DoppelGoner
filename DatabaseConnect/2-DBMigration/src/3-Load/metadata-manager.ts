/**
 * Interface for metadata management operations
 * Separates metadata concerns from the core data loading functionality
 */
export interface MetadataManager {
    /**
     * Track metadata for loaded/modified records
     * @param resourceId ID of the resource being tracked
     * @param resourceType Type of resource (table name)
     * @param actionType Type of action performed (insert, update, delete)
     * @param fieldName Name of the field being modified
     * @param previousValue Previous value of the field
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
     * Get failed migration records for a specific table
     * @param tableName Table name to filter by
     * @param resolved Filter by resolution status
     * @returns Array of failed records
     */
    getFailedRecords(
      tableName: string,
      resolved?: boolean
    ): Promise<any[]>;
    
    /**
     * Mark a failed record as resolved
     * @param id ID of the failed record
     * @param resolvedBy Person or process that resolved the issue
     * @param notes Optional notes about the resolution
     * @returns Success status and error if any
     */
    resolveFailedRecord(
      id: string,
      resolvedBy: string,
      notes?: string
    ): Promise<{ success: boolean; error?: string }>;
  }