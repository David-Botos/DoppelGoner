"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.SnowflakeClient = void 0;
// Snowflake database service for data extraction
const snowflake_sdk_1 = __importDefault(require("snowflake-sdk"));
const config_1 = require("../config/config");
class SnowflakeClient {
    constructor(config) {
        this.connected = false;
        const finalConfig = { ...config_1.snowflakeConfig, ...config };
        this.connection = snowflake_sdk_1.default.createConnection({
            account: finalConfig.account,
            username: finalConfig.username,
            password: finalConfig.password,
            warehouse: finalConfig.warehouse,
            database: finalConfig.database,
            schema: finalConfig.schema,
            authenticator: finalConfig.authenticator,
        });
    }
    /**
     * Connect to Snowflake
     */
    async connect() {
        if (this.connected)
            return;
        return new Promise((resolve, reject) => {
            this.connection.connect((err, conn) => {
                if (err) {
                    console.error("Failed to connect to Snowflake:", err);
                    reject(err);
                    return;
                }
                this.connected = true;
                console.log("Successfully connected to Snowflake!");
                resolve();
            });
        });
    }
    /**
     * Disconnect from Snowflake
     */
    async disconnect() {
        if (!this.connected)
            return;
        return new Promise((resolve, reject) => {
            this.connection.destroy((err) => {
                if (err) {
                    console.error("Failed to disconnect from Snowflake:", err);
                    reject(err);
                    return;
                }
                this.connected = false;
                console.log("Successfully disconnected from Snowflake!");
                resolve();
            });
        });
    }
    /**
     * Execute a query against Snowflake
     * @param sqlText The SQL query to execute
     * @param binds Optional bind parameters
     * @returns Query result rows
     */
    async query(sqlText, binds) {
        if (!this.connected) {
            await this.connect();
        }
        return new Promise((resolve, reject) => {
            const statement = this.connection.execute({
                sqlText,
                binds,
                complete: (err, stmt, rows) => {
                    if (err) {
                        console.error("Failed to execute query:", err);
                        reject(err);
                        return;
                    }
                    resolve(rows);
                },
            });
        });
    }
    /**
     * Fetch a batch of records from a table
     * @param table Table name
     * @param offset Offset for pagination
     * @param limit Maximum number of records to fetch
     * @param whereClause Optional WHERE clause
     * @returns Batch of records
     */
    async fetchBatch(table, offset = 0, limit = 100, whereClause = "") {
        const where = whereClause ? ` WHERE ${whereClause}` : "";
        const query = `SELECT * FROM ${table}${where} ORDER BY ID LIMIT ${limit} OFFSET ${offset}`;
        return this.query(query);
    }
    /**
     * Fetch English translations for records
     * @param table Table name (without _TRANSLATIONS suffix)
     * @param foreignKeyField Field name for foreign key in translations table
     * @param ids Array of IDs to fetch translations for
     * @returns Translations data
     */
    async fetchTranslations(table, foreignKeyField, ids) {
        if (ids.length === 0)
            return [];
        // Format IDs for SQL IN clause
        const formattedIds = ids.map((id) => `'${id}'`).join(",");
        const query = `
      SELECT * FROM ${table}_TRANSLATIONS 
      WHERE ${foreignKeyField} IN (${formattedIds})
        AND LOCALE = 'en'
        AND IS_CANONICAL = TRUE
    `;
        return this.query(query);
    }
    /**
     * Count records in a table
     * @param table Table name
     * @param whereClause Optional WHERE clause
     * @returns Record count
     */
    async countRecords(table, whereClause = "") {
        const where = whereClause ? ` WHERE ${whereClause}` : "";
        const query = `SELECT COUNT(*) as count FROM ${table}${where}`;
        const result = await this.query(query);
        return result[0]?.COUNT || 0;
    }
    /**
     * Validate connection by running a simple query
     * @returns True if connection is valid
     */
    async validateConnection() {
        try {
            await this.query("SELECT 1 as test");
            return true;
        }
        catch (error) {
            return false;
        }
    }
}
exports.SnowflakeClient = SnowflakeClient;
