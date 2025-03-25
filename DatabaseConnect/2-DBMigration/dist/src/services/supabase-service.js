"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SupabaseService = void 0;
// Supabase database service for data loading
const supabase_js_1 = require("@supabase/supabase-js");
const config_1 = require("../config/config");
const supabase_loader_1 = require("../3-Load/supabase-loader");
/**
 * Service class to interact with Supabase database
 */
class SupabaseService {
    constructor() {
        this.loader = new supabase_loader_1.SupabaseLoader();
        this.client = (0, supabase_js_1.createClient)(config_1.supabaseConfig.url, config_1.supabaseConfig.key);
    }
    /**
     * Get the Supabase loader instance
     * @returns SupabaseLoader instance
     */
    getLoader() {
        return this.loader;
    }
    /**
     * Get the Supabase client instance
     * @returns SupabaseClient instance
     */
    getClient() {
        return this.client;
    }
    /**
     * Execute a raw SQL query
     * @param query SQL query string
     * @param params Query parameters
     * @returns Query result
     */
    async executeRawQuery(query, params = []) {
        try {
            const { data, error } = await this.client.rpc("execute_sql", {
                query_text: query,
                query_params: params,
            });
            if (error) {
                throw new Error(`Error executing query: ${error.message}`);
            }
            return data;
        }
        catch (error) {
            console.error("Error executing raw query:", error);
            throw error;
        }
    }
    /**
     * Check if all required tables exist
     * @returns Boolean indicating if all tables exist
     */
    async checkTablesExist(tables) {
        const missingTables = [];
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
    close() {
        this.loader.close();
    }
}
exports.SupabaseService = SupabaseService;
