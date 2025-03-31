#!/usr/bin/env node
import path from "path";
import { PostgresClient } from "../services/postgres-client";

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

    // Create an instance of Postgresclient
    const client = new PostgresClient();

    // Execute the SQL file
    const result = await client.executeSqlFile(resolvedPath);

    console.log("SQL execution successful!");
    console.log(`Returned ${result.length} rows`);

    // If you want to display the results
    if (result.length > 0) {
      console.log("Results:");
      console.table(result);
    }

    // Close the connection pool using the existing close() method
    client.close();
  } catch (error) {
    console.error("Error executing SQL file:", error);
    process.exit(1);
  }
}

// Run the main function
main();
