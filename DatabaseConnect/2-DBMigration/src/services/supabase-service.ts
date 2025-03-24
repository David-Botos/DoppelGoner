// Supabase database service for data loading
import { createClient, SupabaseClient } from "@supabase/supabase-js";
import { supabaseConfig } from "../config/config";

export class SupabaseService {
  private client: SupabaseClient;

  constructor(config?: Partial<typeof supabaseConfig>) {
    const finalConfig = { ...supabaseConfig, ...config };
    this.client = createClient(finalConfig.url, finalConfig.key);
  }

  /**
   * Get the Supabase client instance
   */
  public getClient(): SupabaseClient {
    return this.client;
  }

  /**
   * Insert records into a table
   * @param table Table name
   * @param records Records to insert
   * @returns Inserted records
   */
  public async insert<T = any>(table: string, records: any[]): Promise<T[]> {
    if (records.length === 0) return [];

    const { data, error } = await this.client
      .from(table)
      .insert(records)
      .select();

    if (error) {
      console.error(`Error inserting into ${table}:`, error);
      throw error;
    }

    return data as T[];
  }

  /**
   * Upsert records into a table (insert or update if exists)
   * @param table Table name
   * @param records Records to upsert
   * @param onConflict Column to check for conflicts
   * @returns Upserted records
   */
  public async upsert<T = any>(
    table: string,
    records: any[],
    onConflict: string = "id"
  ): Promise<T[]> {
    if (records.length === 0) return [];

    const { data, error } = await this.client
      .from(table)
      .upsert(records, { onConflict })
      .select();

    if (error) {
      console.error(`Error upserting into ${table}:`, error);
      throw error;
    }

    return data as T[];
  }

  /**
   * Count records in a table
   * @param table Table name
   * @param filter Optional filter criteria
   * @returns Record count
   */
  public async countRecords(
    table: string,
    filter?: Record<string, any>
  ): Promise<number> {
    let queryBuilder = this.client
      .from(table)
      .select("*", { count: "exact", head: true });

    // Apply filters if provided
    if (filter) {
      Object.entries(filter).forEach(([key, value]) => {
        queryBuilder = queryBuilder.eq(key, value);
      });
    }

    const { count, error } = await queryBuilder;

    if (error) {
      console.error(`Error counting records in ${table}:`, error);
      throw error;
    }

    return count || 0;
  }

  /**
   * Insert a record into the migration_log table
   * @param logEntry Migration log entry
   * @returns Inserted log entry
   */
  public async logMigration(logEntry: {
    source_table: string;
    target_table: string;
    records_migrated: number;
    success_count: number;
    failure_count: number;
    error_messages?: string;
    started_at: string;
    completed_at: string;
    execution_time_seconds: number;
  }): Promise<any> {
    const { data, error } = await this.client
      .from("migration_log")
      .insert(logEntry)
      .select();

    if (error) {
      console.error("Error logging migration:", error);
      throw error;
    }

    return data[0];
  }

  /**
   * Validate connection by running a simple query
   * @returns True if connection is valid
   */
  public async validateConnection(): Promise<boolean> {
    try {
      const { data, error } = await this.client.rpc("version");
      if (error) throw error;
      return true;
    } catch (error) {
      console.error("Failed to validate Supabase connection:", error);
      return false;
    }
  }
}
