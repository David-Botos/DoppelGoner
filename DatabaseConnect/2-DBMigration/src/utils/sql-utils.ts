import { QueryResult } from "pg";
import chalk from "chalk";

/**
 * Format query results as a table for CLI display
 */
export class TableFormatter {
  /**
   * Format query results as a table
   * @param results Query results to format
   * @returns Formatted string
   */
  static formatResults(results: QueryResult | QueryResult[]): string {
    // Handle array of results (multiple queries)
    if (Array.isArray(results)) {
      return results
        .map((result, index) => {
          return `Query #${index + 1}:\n${this.formatSingleResult(result)}`;
        })
        .join("\n\n");
    }

    // Handle single result
    return this.formatSingleResult(results);
  }

  /**
   * Format a single query result as a table
   * @param result Single query result
   * @returns Formatted string
   */
  private static formatSingleResult(result: QueryResult): string {
    // If no rows, show summary and return
    if (!result.rows || result.rows.length === 0) {
      return chalk.yellow(
        `Command: ${result.command}, Affected rows: ${result.rowCount || 0}`
      );
    }

    // Get column names and widths
    const columns = Object.keys(result.rows[0]);
    const colWidths = this.calculateColumnWidths(result.rows, columns);

    // Create header
    const header = columns
      .map((col, i) => {
        return chalk.cyan(col.padEnd(colWidths[i]));
      })
      .join(" | ");

    // Create separator
    const separator = columns
      .map((_, i) => {
        return "-".repeat(colWidths[i]);
      })
      .join("-+-");

    // Create rows
    const rows = result.rows
      .map((row) => {
        return columns
          .map((col, i) => {
            const value = this.formatValue(row[col]);
            return value.padEnd(colWidths[i]);
          })
          .join(" | ");
      })
      .join("\n");

    // Combine all components
    return [
      header,
      separator,
      rows,
      "",
      chalk.yellow(`${result.rows.length} row(s) returned`),
    ].join("\n");
  }

  /**
   * Format a cell value for display
   * @param value Cell value to format
   * @returns Formatted string value
   */
  private static formatValue(value: any): string {
    if (value === null || value === undefined) {
      return chalk.gray("NULL");
    }

    if (typeof value === "object") {
      if (value instanceof Date) {
        return value.toISOString();
      }
      return JSON.stringify(value);
    }

    return String(value);
  }

  /**
   * Calculate column widths based on data
   * @param rows Row data
   * @param columns Column names
   * @returns Array of column widths
   */
  private static calculateColumnWidths(
    rows: any[],
    columns: string[]
  ): number[] {
    // Initialize with column header lengths
    const widths = columns.map((col) => col.length);

    // Calculate maximum width for each column
    rows.forEach((row) => {
      columns.forEach((col, i) => {
        const value = this.formatValue(row[col]);
        widths[i] = Math.max(widths[i], value.length);
      });
    });

    // Add some padding (min 2 extra spaces)
    return widths.map((w) => Math.min(w + 2, 50)); // Cap at 50 chars per column
  }

  /**
   * Handle non-tabular results (like INSERT, UPDATE, etc.)
   * @param command Command type
   * @param rowCount Number of affected rows
   * @returns Formatted string
   */
  static formatCommand(command: string, rowCount: number | null): string {
    const commandStr = chalk.green(command);
    const rowCountStr = rowCount === null ? "unknown" : rowCount;
    return `${commandStr} affected ${rowCountStr} row(s)`;
  }
}
