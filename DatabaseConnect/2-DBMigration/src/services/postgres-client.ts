import { Pool, PoolClient, QueryResult } from "pg";
import { hetznerPostgresConfig } from "../config/config";
import * as fs from "fs";
import * as path from "path";

/**
 * Client class for PostgreSQL database operations
 * Handles connection management and basic database operations
 */
export class PostgresClient {
  private pool: Pool;

  constructor(config = hetznerPostgresConfig) {
    this.pool = new Pool(config);

    // Handle pool errors
    this.pool.on("error", (err) => {
      console.error("Unexpected error on idle PostgreSQL client", err);
    });
  }

  /**
   * Get the underlying connection pool
   * @returns The PostgreSQL connection pool
   */
  getPool(): Pool {
    return this.pool;
  }

  /**
   * Execute a SQL query with parameters
   * @param query SQL query string
   * @param params Query parameters
   * @returns Query result rows
   */
  async executeQuery<T = any>(query: string, params: any[] = []): Promise<T[]> {
    try {
      const result = await this.pool.query(query, params);
      return result.rows;
    } catch (error) {
      console.error("Error executing query:", error);
      throw error;
    }
  }

  /**
   * Execute SQL from a file, supporting multiple statements
   * @param filePath Path to SQL file
   * @returns Array of query results
   */
  async executeSqlFile(filePath: string): Promise<any[]> {
    const client = await this.pool.connect();

    try {
      const sql = fs.readFileSync(path.resolve(filePath), "utf8");

      // Split the SQL file into individual statements
      const statements = this.splitSqlStatements(sql);

      // Execute each statement and collect results
      const results: QueryResult[] = [];

      for (const statement of statements) {
        // Skip empty statements
        if (!statement.trim()) continue;

        try {
          const result = await client.query(statement);
          results.push(result);
        } catch (err) {
          console.error(`Error executing statement: ${statement}`);
          throw err;
        }
      }

      // Return all results
      return results.map((result) => {
        // Include command and rowCount for non-SELECT queries
        if (!result.rows || result.rows.length === 0) {
          return {
            command: result.command,
            rowCount: result.rowCount,
            rows: [],
          };
        }
        return result.rows;
      });
    } catch (error) {
      console.error(`Error executing SQL file ${filePath}:`, error);
      throw error;
    } finally {
      client.release();
    }
  }

  /**
   * Split SQL string into individual statements
   * Handles comments and semicolons in string literals
   */
  private splitSqlStatements(sql: string): string[] {
    // Remove comments
    sql = sql.replace(/--.*$/gm, "");

    const statements: string[] = [];
    let currentStatement = "";
    let inString = false;

    for (let i = 0; i < sql.length; i++) {
      const char = sql[i];
      const nextChar = sql[i + 1] || "";

      // Handle string delimiters
      if (char === "'" && (i === 0 || sql[i - 1] !== "\\")) {
        inString = !inString;
      }

      // If we're not in a string and we hit a semicolon, we're at the end of a statement
      if (char === ";" && !inString) {
        statements.push(currentStatement + ";");
        currentStatement = "";
        continue;
      }

      currentStatement += char;
    }

    // Add the last statement if it doesn't end with a semicolon
    if (currentStatement.trim()) {
      statements.push(currentStatement);
    }

    return statements;
  }

  /**
   * Execute a transaction with multiple queries
   * @param callback Function to execute within transaction
   * @returns Result of the callback
   */
  async executeTransaction<T = any>(
    callback: (client: PoolClient) => Promise<T>
  ): Promise<T> {
    const client = await this.pool.connect();

    try {
      await client.query("BEGIN");
      const result = await callback(client);
      await client.query("COMMIT");
      return result;
    } catch (error) {
      await client.query("ROLLBACK");
      throw error;
    } finally {
      client.release();
    }
  }

  /**
   * Check if a table exists in PostgreSQL
   * @param tableName Table name to check
   * @param schema Schema name (defaults to 'public')
   * @returns Boolean indicating if table exists
   */
  async tableExists(tableName: string, schema = "public"): Promise<boolean> {
    try {
      const query = `
        SELECT EXISTS (
          SELECT FROM information_schema.tables 
          WHERE table_schema = $1
          AND table_name = $2
        );
      `;

      const result = await this.pool.query(query, [schema, tableName]);
      return result.rows[0].exists;
    } catch (error) {
      console.error(
        `Error checking if table ${schema}.${tableName} exists:`,
        error
      );
      // For unexpected errors, assume the table exists to prevent blocking operations
      return true;
    }
  }

  /**
   * Process a PostgreSQL error into a more readable format
   * @param error PostgreSQL error
   * @param operation Operation name
   * @param tableName Table name
   * @returns Formatted error message
   */
  processPostgresError(
    error: any,
    operation: string,
    tableName: string
  ): string {
    let message = `${operation} operation failed on table ${tableName}: `;

    if (error.code) {
      switch (error.code) {
        case "23505":
          message += `Unique constraint violation (${
            error.constraint || "unknown constraint"
          })`;
          break;
        case "23503":
          message += `Foreign key constraint violation (${
            error.constraint || "unknown constraint"
          })`;
          break;
        case "23502":
          message += `Not null constraint violation (${
            error.column || "unknown column"
          })`;
          break;
        case "22P02":
          message += `Invalid text representation (${
            error.column || "unknown column"
          })`;
          break;
        default:
          message += `${error.message || String(error)}`;
      }
    } else {
      message += `${error.message || String(error)}`;
    }

    return message;
  }

  /**
   * Find records that meet specific criteria
   * @param tableName Table to search
   * @param criteria Object with column-value pairs to match
   * @returns Array of matching records
   */
  async findRecords<T = any>(
    tableName: string,
    criteria: Record<string, any>
  ): Promise<T[]> {
    try {
      // Build WHERE clause
      const conditions: string[] = [];
      const values: any[] = [];
      let paramIndex = 1;

      Object.entries(criteria).forEach(([column, value]) => {
        if (value === null) {
          conditions.push(`${column} IS NULL`);
        } else {
          conditions.push(`${column} = $${paramIndex}`);
          values.push(value);
          paramIndex++;
        }
      });

      const whereClause =
        conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

      const query = `SELECT * FROM ${tableName} ${whereClause}`;
      const result = await this.pool.query(query, values);

      return result.rows;
    } catch (error) {
      console.error(`Error finding records in ${tableName}:`, error);
      throw error;
    }
  }

  /**
   * Delete all records from a table - useful for testing or rollbacks
   * @param tableName Table to delete from
   * @returns Result object with success, count, and error information
   */
  async deleteAllRecords(
    tableName: string
  ): Promise<{ success: boolean; count: number; error?: string }> {
    try {
      // First count the records
      const countResult = await this.pool.query(
        `SELECT COUNT(*) FROM ${tableName}`
      );
      const beforeCount = parseInt(countResult.rows[0].count, 10);

      // Then delete them
      await this.pool.query(`DELETE FROM ${tableName}`);

      return {
        success: true,
        count: beforeCount,
        error: undefined,
      };
    } catch (error) {
      return {
        success: false,
        count: 0,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Close the PostgreSQL connection pool
   */
  close(): void {
    this.pool.end().catch((err) => {
      console.error("Error closing PostgreSQL connection pool:", err);
    });
  }
}
