import { Pool, PoolClient, QueryResult } from "pg";
import { hetznerPostgresConfig } from "../config/config";
import * as fs from "fs";
import * as path from "path";
import { exec } from "child_process";
import { promisify } from "util";
import { TableFormatter } from "../utils/sql-utils";

const execPromise = promisify(exec);

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
  async executeQuery<T = any>(
    query: string,
    params: any[] = [],
    formatOutput: boolean = false
  ): Promise<T[] | string> {
    try {
      const result = await this.pool.query(query, params);

      // Return formatted table if requested
      if (formatOutput) {
        return TableFormatter.formatResults(result);
      }

      return result.rows;
    } catch (error) {
      console.error("Error executing query:", error);
      throw error;
    }
  }

  /**
   * Execute a complex SQL script that may contain PL/pgSQL functions, triggers, etc.
   * @param filePath Path to SQL file
   * @param formatOutput Whether to format the output as a table
   * @returns Result indicating success with formatted output if requested
   */
  async executeComplexSqlScript(
    filePath: string,
    formatOutput: boolean = true
  ): Promise<{ success: boolean; message: string; formattedResults?: string }> {
    const client = await this.pool.connect();

    try {
      const sql = fs.readFileSync(path.resolve(filePath), "utf8");

      // Check if this is a SELECT query that returns results
      const isSelectQuery = /^\s*SELECT\b/i.test(sql.trim());

      // Split script to handle multiple statements
      // Note: This is a simple approach; it won't handle complex SQL correctly
      const statements = this.splitSqlStatements(sql);

      // Detect if there are actual SELECT statements in the script
      const hasSelectStatements = statements.some(
        (stmt) =>
          /^\s*SELECT\b/i.test(stmt.trim()) && !stmt.trim().startsWith("--")
      );

      try {
        let allResults: QueryResult[] = [];

        // For single SELECT queries, try getting formatted results
        if (isSelectQuery || hasSelectStatements) {
          try {
            // Execute each statement and collect results for SELECTs
            for (const statement of statements) {
              if (statement.trim() && !statement.trim().startsWith("--")) {
                const result = await client.query(statement);
                if (result.command === "SELECT") {
                  allResults.push(result);
                }
              }
            }

            let formattedResults = "";
            if (allResults.length > 0 && formatOutput) {
              formattedResults = TableFormatter.formatResults(allResults);
            }

            return {
              success: true,
              message: "SQL script executed successfully",
              formattedResults: formattedResults,
            };
          } catch (e) {
            // If individual statement execution fails, fall back to executing as a single script
            console.warn(
              "Individual statement execution failed, trying as a single script"
            );
            allResults = [];
          }
        }

        // If not a SELECT or individual execution failed, try the transaction approach
        await client.query("BEGIN");
        await client.query(sql);
        await client.query("COMMIT");

        return {
          success: true,
          message: "SQL script executed successfully",
        };
      } catch (error) {
        // Rollback any transaction
        try {
          await client.query("ROLLBACK");
        } catch (e) {
          // Ignore rollback errors
        }

        // Try fallback to psql if enabled
        if (process.env.USE_PSQL_FALLBACK === "true") {
          try {
            const psqlResult = await this.executeSqlWithPsql(filePath);
            return psqlResult;
          } catch (psqlError) {
            return {
              success: false,
              message: `Failed to execute SQL: ${
                error instanceof Error ? error.message : String(error)
              }`,
            };
          }
        } else {
          return {
            success: false,
            message: `Failed to execute SQL: ${
              error instanceof Error ? error.message : String(error)
            }`,
          };
        }
      }
    } catch (error) {
      return {
        success: false,
        message: `Error reading/executing SQL file: ${
          error instanceof Error ? error.message : String(error)
        }`,
      };
    } finally {
      client.release();
    }
  }

  /**
   * Split SQL statements by semicolons
   * This is a simple approach and won't handle all SQL correctly
   */
  private splitSqlStatements(sql: string): string[] {
    // Remove comments first
    const noComments = sql.replace(/--.*$/gm, "");

    // Split by semicolons, but handle quoted strings and dollar-quoted blocks
    const statements: string[] = [];
    let currentStatement = "";
    let inQuotes = false;
    let inDollarQuote = false;
    let dollarTag = "";

    for (let i = 0; i < noComments.length; i++) {
      const char = noComments[i];
      const nextChar = noComments[i + 1] || "";

      // Handle quotes
      if (char === "'" && !inDollarQuote) {
        // Handle escaped quotes
        if (noComments[i - 1] !== "\\") {
          inQuotes = !inQuotes;
        }
      }

      // Handle dollar quoting
      if (char === "$" && !inQuotes) {
        if (!inDollarQuote) {
          // Try to detect start of dollar quote
          let tag = "$";
          let j = i + 1;
          while (j < noComments.length && noComments[j] !== "$") {
            tag += noComments[j];
            j++;
          }
          if (j < noComments.length && noComments[j] === "$") {
            tag += "$";
            inDollarQuote = true;
            dollarTag = tag;
            currentStatement += tag;
            i = j;
            continue;
          }
        } else if (
          noComments.substring(i, i + dollarTag.length) === dollarTag
        ) {
          // End of dollar quote
          inDollarQuote = false;
          currentStatement += dollarTag;
          i += dollarTag.length - 1;
          continue;
        }
      }

      // Handle statement separator
      if (char === ";" && !inQuotes && !inDollarQuote) {
        currentStatement += char;
        statements.push(currentStatement.trim());
        currentStatement = "";
      } else {
        currentStatement += char;
      }
    }

    // Add the last statement if it exists
    if (currentStatement.trim()) {
      statements.push(currentStatement.trim());
    }

    return statements;
  }

  /**
   * Execute a SQL file using the psql command-line utility
   * @param filePath Path to SQL file
   * @returns Result object
   */
  private async executeSqlWithPsql(
    filePath: string
  ): Promise<{ success: boolean; message: string }> {
    try {
      // Extract connection info from connection pool config
      const { host, port, database, user, password } = this.pool.options;

      // Handle password securely
      let command: string;
      let options: any = {};

      if (typeof password === "string") {
        // Option 1: Pass password as environment variable (type-safe way)
        const env = { ...process.env } as Record<string, string>;
        env.PGPASSWORD = password;
        options.env = env;
        command = `psql -h ${host} -p ${port} -d ${database} -U ${user} -f ${filePath}`;
      } else {
        // Option 2: Use connection string without password in environment
        command = `psql "postgresql://${user}:${encodeURIComponent(
          String(password)
        )}@${host}:${port}/${database}" -f ${filePath}`;
      }

      // Execute psql command
      const { stdout, stderr } = await execPromise(command, options);

      if (stderr && !stderr.includes("NOTICE") && !stderr.includes("INFO")) {
        return { success: false, message: stderr.toString() };
      }

      return {
        success: true,
        message: stdout.toString() || "SQL executed successfully via psql",
      };
    } catch (error) {
      return {
        success: false,
        message: `psql execution failed: ${
          error instanceof Error ? error.message : String(error)
        }`,
      };
    }
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
   * Check if a schema exists in PostgreSQL
   * @param schemaName Schema name to check
   * @returns Boolean indicating if schema exists
   */
  async schemaExists(schemaName: string): Promise<boolean> {
    try {
      const query = `
        SELECT EXISTS (
          SELECT FROM information_schema.schemata
          WHERE schema_name = $1
        );
      `;

      const result = await this.pool.query(query, [schemaName]);
      return result.rows[0].exists;
    } catch (error) {
      console.error(`Error checking if schema ${schemaName} exists:`, error);
      return false;
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
