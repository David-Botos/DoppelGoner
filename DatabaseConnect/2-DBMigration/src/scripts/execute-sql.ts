#!/usr/bin/env node
import path from "path";
import { PostgresClient } from "../services/postgres-client";
import chalk from "chalk";

async function main() {
  try {
    // Get the file path from command line arguments
    const args = process.argv.slice(2);

    if (args.length === 0) {
      console.error("Error: Please provide a SQL file path");
      console.log(
        "Usage: npx ts-node execute-sql-script.ts <path-to-sql-file>"
      );
      process.exit(1);
    }

    const filePath = args[0];
    const resolvedPath = path.resolve(filePath);

    console.log(`Executing SQL script: ${resolvedPath}`);

    // Create an instance of PostgresClient
    const client = new PostgresClient();

    // Execute the SQL file using the improved method for complex scripts
    // Set formatOutput = true to enable table formatting
    const result = await client.executeComplexSqlScript(resolvedPath, true);

    if (result.success) {
      console.log(chalk.green("SQL execution successful!"));

      // If there are formatted results, display them
      if (result.formattedResults) {
        console.log(chalk.cyan("\nQuery Results:"));
        console.log(result.formattedResults);
      } else {
        console.log(chalk.cyan("\nExecution Summary:"), result.message);
      }
    } else {
      console.error(chalk.red("SQL execution failed:"), result.message);
      process.exit(1);
    }

    // Close the connection pool
    client.close();
  } catch (error) {
    console.error(chalk.red("Error executing SQL script:"), error);
    process.exit(1);
  }
}

// Run the main function
main();
