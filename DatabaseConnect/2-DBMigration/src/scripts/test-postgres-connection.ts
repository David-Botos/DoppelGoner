// src/scripts/test-postgres-connection.ts
import { PostgresMetadataManager } from "../3-Load/postgres-metadata-manager";
import { PostgresClient } from "../services/postgres-client";

async function testPostgresConnection() {
  console.log("Testing PostgreSQL connection...");

  // Create a PostgreSQL client instance
  const client = new PostgresClient();

  try {
    // Test a simple query
    const result = await client.executeQuery(
      "SELECT current_database() as db, current_user as user, version()"
    );

    console.log("✅ Successfully connected to PostgreSQL!");
    console.log(`Database: ${result[0].db}`);
    console.log(`User: ${result[0].user}`);
    console.log(`PostgreSQL version: ${result[0].version}`);

    // Test metadata tables setup
    const metadataManager = new PostgresMetadataManager(client["pool"]);
    await metadataManager.ensureMetadataTables();
    console.log("✅ Metadata tables verified");

    // List all tables in the database
    const tables = await client.executeQuery(`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public'
      ORDER BY table_name
    `);

    console.log("\nTables in database:");
    tables.forEach((row, index) => {
      console.log(`${index + 1}. ${row.table_name}`);
    });
  } catch (error) {
    console.error("❌ Failed to connect to PostgreSQL:", error);
  } finally {
    // Close the connection
    client.close();
    console.log("Connection closed");
  }
}

// Run the test function
testPostgresConnection().catch(console.error);
