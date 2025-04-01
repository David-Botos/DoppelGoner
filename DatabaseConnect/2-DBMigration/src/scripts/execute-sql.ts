#!/usr/bin/env node
import path from "path";
import { PostgresClient } from "../services/postgres-client";
import chalk from "chalk";

// Install chalk with: npm install chalk

async function main() {
  try {
    // Get the file path from command line arguments
    const args = process.argv.slice(2);

    if (args.length === 0) {
      console.error("Error: Please provide a SQL file path");
      console.log("Usage: npx ts-node execute-sql-file.ts <path-to-sql-file>");
      process.exit(1);
    }

    const filePath = args[0];
    const resolvedPath = path.resolve(filePath);

    console.log(`Executing SQL file: ${resolvedPath}`);

    // Create an instance of PostgresClient
    const client = new PostgresClient();

    // Execute the SQL file
    const results = await client.executeSqlFile(resolvedPath);

    console.log(chalk.green("SQL execution successful!"));

    // Display results for each statement
    if (results && Array.isArray(results)) {
      console.log(
        chalk.yellow(`File contained ${results.length} SQL statements.`)
      );

      results.forEach((result, index) => {
        // Command results like CREATE, DROP, etc.
        if (result && typeof result === "object" && "command" in result) {
          console.log(
            chalk.cyan(`\nStatement ${index + 1}: ${result.command}`)
          );
          console.log(`Affected rows: ${result.rowCount}`);
          return;
        }

        // Query results
        if (Array.isArray(result) && result.length > 0) {
          console.log(
            chalk.cyan(
              `\nStatement ${index + 1} results (${result.length} rows):`
            )
          );
          console.table(result);
        } else if (Array.isArray(result) && result.length === 0) {
          console.log(
            chalk.cyan(`\nStatement ${index + 1}: Query returned 0 rows`)
          );
        }
      });
    } else {
      console.log("No results returned");
    }

    // Close the connection pool
    client.close();
  } catch (error) {
    console.error(chalk.red("Error executing SQL file:"), error);
    process.exit(1);
  }
}

// Run the main function
main();
