import { Pool } from 'pg';
import { MetadataManager } from './metadata-manager';
import { randomUUID } from 'crypto';
import { v4 as uuidv4 } from 'uuid';

/**
 * PostgreSQL implementation of the MetadataManager interface
 */
export class PostgresMetadataManager implements MetadataManager {
  constructor(private pool: Pool) {}

  /**
   * Track metadata for loaded records in PostgreSQL
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
      const metadataId = uuidv4();
      const now = new Date().toISOString();
      
      const query = `
        INSERT INTO metadata (
          id, 
          resource_id, 
          resource_type, 
          last_action_date, 
          last_action_type, 
          field_name, 
          previous_value, 
          replacement_value, 
          updated_by, 
          created, 
          last_modified,
          original_id
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
      `;

      const values = [
        metadataId,
        resourceId,
        resourceType,
        now,
        actionType,
        fieldName,
        previousValue,
        replacementValue,
        updatedBy,
        now,
        now,
        originalId || null
      ];

      await this.pool.query(query, values);
      
      return {
        success: true
      };
    } catch (error) {
      console.error('Error tracking metadata:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error)
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
      const query = `
        SELECT * FROM failed_migration_records
        WHERE table_name = $1
        AND resolved = $2
        ORDER BY attempted_at DESC
      `;
      
      const result = await this.pool.query(query, [tableName, resolved]);
      return result.rows;
    } catch (error) {
      console.error('Error retrieving failed records:', error);
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
      const query = `
        UPDATE failed_migration_records
        SET 
          resolved = true,
          resolved_at = $1,
          resolved_by = $2,
          resolution_notes = $3
        WHERE id = $4
      `;
      
      const now = new Date().toISOString();
      const result = await this.pool.query(query, [now, resolvedBy, notes || null, id]);
      
      if (result.rowCount === 0) {
        return {
          success: false,
          error: `Failed record with id ${id} not found`
        };
      }
      
      return {
        success: true
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Create metadata tables if they don't exist
   * This can be called during initialization to ensure the required tables are available
   */
  async ensureMetadataTables(): Promise<void> {
    try {
      // Create the metadata table
      await this.pool.query(`
        CREATE TABLE IF NOT EXISTS metadata (
          id UUID PRIMARY KEY,
          resource_id VARCHAR(255) NOT NULL,
          resource_type VARCHAR(255) NOT NULL,
          last_action_date TIMESTAMP WITH TIME ZONE NOT NULL,
          last_action_type VARCHAR(50) NOT NULL,
          field_name VARCHAR(255) NOT NULL,
          previous_value TEXT,
          replacement_value TEXT,
          updated_by VARCHAR(255) NOT NULL,
          created TIMESTAMP WITH TIME ZONE NOT NULL,
          last_modified TIMESTAMP WITH TIME ZONE NOT NULL,
          original_id VARCHAR(255)
        );
      `);

      // Create the failed_migration_records table
      await this.pool.query(`
        CREATE TABLE IF NOT EXISTS failed_migration_records (
          id UUID PRIMARY KEY,
          table_name VARCHAR(255) NOT NULL,
          record_data JSONB NOT NULL,
          error_message TEXT NOT NULL,
          attempted_at TIMESTAMP WITH TIME ZONE NOT NULL,
          resolved BOOLEAN DEFAULT false,
          resolved_at TIMESTAMP WITH TIME ZONE,
          resolved_by VARCHAR(255),
          resolution_notes TEXT
        );
      `);
      
      // Create the migration_log table
      await this.pool.query(`
        CREATE TABLE IF NOT EXISTS migration_log (
          id SERIAL PRIMARY KEY,
          source_table VARCHAR(255) NOT NULL,
          target_table VARCHAR(255) NOT NULL,
          records_migrated INTEGER NOT NULL,
          success_count INTEGER NOT NULL,
          failure_count INTEGER NOT NULL,
          error_messages TEXT,
          started_at TIMESTAMP WITH TIME ZONE NOT NULL,
          completed_at TIMESTAMP WITH TIME ZONE NOT NULL,
          execution_time_seconds FLOAT NOT NULL
        );
      `);
      
      console.log('Metadata tables created or verified');
    } catch (error) {
      console.error('Error ensuring metadata tables:', error);
      throw error;
    }
  }
}