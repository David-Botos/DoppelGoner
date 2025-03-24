// Snowflake database service for data extraction
import snowflake from "snowflake-sdk";
import { snowflakeConfig } from "../config/config";

export class SnowflakeClient {
  private connection: snowflake.Connection;
  private connected: boolean = false;

  constructor(config?: Partial<typeof snowflakeConfig>) {
    const finalConfig = { ...snowflakeConfig, ...config };

    this.connection = snowflake.createConnection({
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
  public async connect(): Promise<void> {
    if (this.connected) return;

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
  public async disconnect(): Promise<void> {
    if (!this.connected) return;

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
  public async query<T = any>(sqlText: string, binds?: any[]): Promise<T[]> {
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

          resolve(rows as T[]);
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
  public async fetchBatch<T = any>(
    table: string,
    offset: number = 0,
    limit: number = 100,
    whereClause: string = ""
  ): Promise<T[]> {
    const where = whereClause ? ` WHERE ${whereClause}` : "";
    const query = `SELECT * FROM ${table}${where} ORDER BY ID LIMIT ${limit} OFFSET ${offset}`;

    return this.query<T>(query);
  }

  /**
   * Fetch English translations for records
   * @param table Table name (without _TRANSLATIONS suffix)
   * @param foreignKeyField Field name for foreign key in translations table
   * @param ids Array of IDs to fetch translations for
   * @returns Translations data
   */
  public async fetchTranslations<T = any>(
    table: string,
    foreignKeyField: string,
    ids: string[]
  ): Promise<T[]> {
    if (ids.length === 0) return [];

    // Format IDs for SQL IN clause
    const formattedIds = ids.map((id) => `'${id}'`).join(",");

    const query = `
      SELECT * FROM ${table}_TRANSLATIONS 
      WHERE ${foreignKeyField} IN (${formattedIds})
        AND LOCALE = 'en'
        AND IS_CANONICAL = TRUE
    `;

    return this.query<T>(query);
  }

  /**
   * Count records in a table
   * @param table Table name
   * @param whereClause Optional WHERE clause
   * @returns Record count
   */
  public async countRecords(
    table: string,
    whereClause: string = ""
  ): Promise<number> {
    const where = whereClause ? ` WHERE ${whereClause}` : "";
    const query = `SELECT COUNT(*) as count FROM ${table}${where}`;

    const result = await this.query<{ COUNT: number }>(query);
    return result[0]?.COUNT || 0;
  }

  /**
   * Validate connection by running a simple query
   * @returns True if connection is valid
   */
  public async validateConnection(): Promise<boolean> {
    try {
      await this.query("SELECT 1 as test");
      return true;
    } catch (error) {
      return false;
    }
  }
}
