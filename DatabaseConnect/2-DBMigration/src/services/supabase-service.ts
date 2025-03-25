// Supabase database service for data loading
import { createClient, SupabaseClient } from "@supabase/supabase-js";
import { supabaseConfig } from "../config/config";
import { SupabaseLoader } from "../3-Load/supabase-loader";

/**
 * Service class to interact with Supabase database
 */
export class SupabaseService {
  private loader: SupabaseLoader;
  private client: SupabaseClient;

  constructor() {
    this.loader = new SupabaseLoader();
    this.client = createClient(supabaseConfig.url, supabaseConfig.key);
  }

  /**
   * Get the Supabase loader instance
   * @returns SupabaseLoader instance
   */
  getLoader(): SupabaseLoader {
    return this.loader;
  }

  /**
   * Get the Supabase client instance
   * @returns SupabaseClient instance
   */
  getClient(): SupabaseClient {
    return this.client;
  }

  /**
   * Execute a raw SQL query
   * @param query SQL query string
   * @param params Query parameters
   * @returns Query result
   */
  async executeRawQuery(query: string, params: any[] = []): Promise<any> {
    try {
      const { data, error } = await this.client.rpc("execute_sql", {
        query_text: query,
        query_params: params,
      });

      if (error) {
        throw new Error(`Error executing query: ${error.message}`);
      }

      return data;
    } catch (error) {
      console.error("Error executing raw query:", error);
      throw error;
    }
  }

  /**
   * Check if all required tables exist
   * @returns Boolean indicating if all tables exist
   */
  async checkTablesExist(tables: string[]): Promise<{
    allExist: boolean;
    missingTables: string[];
  }> {
    const missingTables: string[] = [];

    for (const table of tables) {
      const exists = await this.loader.tableExists(table);
      if (!exists) {
        missingTables.push(table);
      }
    }

    return {
      allExist: missingTables.length === 0,
      missingTables,
    };
  }

  /**
   * Close the service and free resources
   */
  close(): void {
    this.loader.close();
  }
}
