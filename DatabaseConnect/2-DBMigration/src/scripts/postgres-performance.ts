#!/usr/bin/env node
import { v4 as uuidv4 } from "uuid";
import { PostgresClient } from "../services/postgres-client";

// Define interface for sample data
interface SampleData {
  id: string;
  name: string;
  description: string;
  created: string;
  number_value: number;
  boolean_value: boolean;
}

/**
 * Helper to process query results that might be formatted strings or data arrays
 * @param result Query result (could be formatted string or data array)
 * @returns The result data as an array if possible, empty array otherwise
 */
function ensureDataArray(result: any[] | string): any[] {
  if (typeof result === "string") {
    // If it's a formatted string, output it directly to console
    console.log(result);
    return [];
  }
  return Array.isArray(result) ? result : [];
}

/**
 * Generate sample test data for batch testing
 * @param count Number of records to generate
 * @returns Array of sample data objects
 */
function generateSampleData(count: number = 1000): SampleData[] {
  const data: SampleData[] = [];
  for (let i = 0; i < count; i++) {
    data.push({
      id: uuidv4(),
      name: `Test Record ${i}`,
      description: `This is a test record generated for performance testing with index ${i}`,
      created: new Date().toISOString(),
      number_value: Math.floor(Math.random() * 1000),
      boolean_value: Math.random() > 0.5,
    });
  }
  return data;
}

/**
 * Format time duration in milliseconds to a readable string
 * @param milliseconds Duration in milliseconds
 * @returns Formatted duration string
 */
function formatDuration(milliseconds: number): string {
  if (milliseconds < 1000) {
    return `${milliseconds.toFixed(2)}ms`;
  } else {
    return `${(milliseconds / 1000).toFixed(2)}s`;
  }
}

/**
 * Main diagnostic function for PostgreSQL performance assessment
 */
async function runDiagnostics() {
  console.log("PostgreSQL Performance Diagnostics");
  console.log("=================================");

  // Create PostgresClient instance
  const client = new PostgresClient();

  try {
    // 1. Basic connection test
    console.log("\n1. Testing database connection...");
    const startConnect = Date.now();
    const versionQuery = await client.executeQuery("SELECT version()");
    const connectTime = Date.now() - startConnect;

    const versionData = ensureDataArray(versionQuery);

    console.log(`✅ Connected successfully in ${formatDuration(connectTime)}`);
    if (versionData.length > 0) {
      console.log(`PostgreSQL Version: ${versionData[0].version}`);
    }

    // 2. Database statistics
    console.log("\n2. Gathering database statistics...");

    // Table sizes - use formatting for better readability
    console.log("\nTop 10 tables by size:");
    await client.executeQuery(
      `
      SELECT
        table_schema,
        table_name,
        pg_size_pretty(pg_total_relation_size('"' || table_schema || '"."' || table_name || '"')) as total_size,
        pg_size_pretty(pg_relation_size('"' || table_schema || '"."' || table_name || '"')) as table_size,
        pg_size_pretty(pg_total_relation_size('"' || table_schema || '"."' || table_name || '"') - 
                     pg_relation_size('"' || table_schema || '"."' || table_name || '"')) as index_size
      FROM information_schema.tables
      WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
      ORDER BY pg_total_relation_size('"' || table_schema || '"."' || table_name || '"') DESC
      LIMIT 10;
    `,
      [],
      true
    );

    // Connection statistics - use formatting for better readability
    console.log("\nConnection statistics:");
    const connectionStatsQuery = await client.executeQuery(
      `
      SELECT 
        max_conn.setting as max_connections,
        used.count as used_connections,
        (max_conn.setting::int - used.count::int) as available_connections
      FROM 
        (SELECT setting FROM pg_settings WHERE name = 'max_connections') max_conn,
        (SELECT count(*) FROM pg_stat_activity) used;
    `,
      [],
      true
    );

    // Store connection stats for summary
    const connectionStatsData = ensureDataArray(connectionStatsQuery);
    let connStats = { max_connections: 0, used_connections: 0 };
    if (connectionStatsData.length > 0) {
      connStats = {
        max_connections: parseInt(connectionStatsData[0].max_connections),
        used_connections: parseInt(connectionStatsData[0].used_connections),
      };
    }

    // Cache hit ratio - use formatting for better readability
    console.log("\nCache statistics:");
    const cacheStatsQuery = await client.executeQuery(
      `
      SELECT 
        sum(heap_blks_read) as heap_read,
        sum(heap_blks_hit) as heap_hit,
        sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as cache_hit_ratio
      FROM 
        pg_statio_user_tables;
    `,
      [],
      true
    );

    // Store cache stats for summary
    const cacheStatsData = ensureDataArray(cacheStatsQuery);
    let cacheHitRatio = 0;
    if (cacheStatsData.length > 0) {
      cacheHitRatio = parseFloat(cacheStatsData[0].cache_hit_ratio);
    }

    // 3. Query performance test
    console.log("\n3. Testing query performance...");

    // Performance test function
    async function testQueryPerformance(
      name: string,
      query: string,
      params: any[] = []
    ) {
      const iterations = 5;
      let totalTime = 0;

      console.log(`\nRunning test: ${name}`);

      for (let i = 0; i < iterations; i++) {
        const start = Date.now();
        await client.executeQuery(query, params);
        const elapsed = Date.now() - start;

        console.log(`  Run ${i + 1}: ${formatDuration(elapsed)}`);
        totalTime += elapsed;
      }

      const averageTime = totalTime / iterations;
      console.log(`  Average: ${formatDuration(averageTime)}`);

      return { name, averageTime };
    }

    // Run various query tests
    await testQueryPerformance(
      "Simple SELECT COUNT(*)",
      "SELECT COUNT(*) FROM information_schema.tables"
    );

    // Get a small/medium table for testing
    const tablesQuery = await client.executeQuery(`
      SELECT table_name, 
             (SELECT count(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count
      FROM information_schema.tables t 
      WHERE table_schema = 'public'
      ORDER BY column_count DESC
      LIMIT 5;
    `);

    const tablesData = ensureDataArray(tablesQuery);

    if (tablesData.length > 0) {
      const testTable = tablesData[0].table_name;
      console.log(`\nUsing table "${testTable}" for further testing...`);

      // Count query
      await testQueryPerformance(
        "Table COUNT(*)",
        `SELECT COUNT(*) FROM "${testTable}"`
      );

      // Select all query (limited)
      await testQueryPerformance(
        "SELECT * LIMIT 1000",
        `SELECT * FROM "${testTable}" LIMIT 1000`
      );
    }

    // 4. Batch size performance test
    console.log("\n4. Testing optimal batch size for data loading...");

    // Check if we can create a temporary table
    let canCreateTable = false;
    try {
      // Try to create and then immediately drop a temporary table
      await client.executeQuery(`
        CREATE TEMPORARY TABLE temp_test_permissions (id INT);
        DROP TABLE temp_test_permissions;
      `);
      canCreateTable = true;
    } catch (error) {
      console.log(
        "❌ Unable to create temporary tables. Skipping batch size testing."
      );
      console.log("   You may need elevated privileges for this test.");
    }

    if (canCreateTable) {
      // Generate test data
      const sampleData = generateSampleData(5000);

      // Test batch sizes
      const batchSizesToTest = [100, 500, 1000, 2000, 5000];

      console.log(
        "⏳ Testing different batch sizes, this may take a minute..."
      );

      try {
        // Create a temporary test table
        await client.executeQuery(`
          CREATE TEMPORARY TABLE temp_batch_test (
            id UUID PRIMARY KEY,
            name TEXT,
            description TEXT,
            created TIMESTAMP,
            number_value INTEGER,
            boolean_value BOOLEAN
          );
        `);

        // Test with varying batch sizes
        const testResults = [];

        for (const batchSize of batchSizesToTest) {
          // Clean table before each test
          await client.executeQuery("TRUNCATE TABLE temp_batch_test");

          console.log(`Testing batch size: ${batchSize}`);
          const startTime = Date.now();

          // Use client to insert data in batches
          const batches: SampleData[][] = [];
          for (let i = 0; i < sampleData.length; i += batchSize) {
            batches.push(sampleData.slice(i, i + batchSize));
          }

          for (const batch of batches) {
            // This uses a simple INSERT
            const columns = Object.keys(batch[0]) as (keyof SampleData)[];
            const values: any[] = [];
            const placeholders: string[] = [];

            batch.forEach((record, rowIndex) => {
              const rowPlaceholders: string[] = [];

              columns.forEach((column, colIndex) => {
                const paramIndex = rowIndex * columns.length + colIndex + 1;
                rowPlaceholders.push(`$${paramIndex}`);
                values.push(record[column]);
              });

              placeholders.push(`(${rowPlaceholders.join(", ")})`);
            });

            const query = `
              INSERT INTO temp_batch_test (${columns.join(", ")})
              VALUES ${placeholders.join(", ")}
            `;

            await client.executeQuery(query, values);
          }

          const endTime = Date.now();
          const timeSeconds = (endTime - startTime) / 1000;

          testResults.push({ batchSize, timeSeconds });
          console.log(`  Completed in ${timeSeconds.toFixed(2)} seconds`);
        }

        // Sort and display results - use formatting for better readability
        console.log("\nBatch size test results (sorted by performance):");

        // Sort the results by time (fastest first)
        testResults.sort((a, b) => a.timeSeconds - b.timeSeconds);

        // Display in a formatted table
        console.table(testResults);

        if (testResults.length > 0) {
          console.log(
            `✅ Optimal batch size appears to be: ${testResults[0].batchSize}`
          );
        }
      } catch (error) {
        console.error("Error during batch testing:", error);
      } finally {
        // Drop the temporary table
        await client.executeQuery("DROP TABLE IF EXISTS temp_batch_test");
      }
    }

    // 5. Basic index performance check - use formatting for better readability
    console.log("\n5. Checking index usage...");

    // Get index stats with formatting
    const indexStatsQuery = await client.executeQuery(
      `
      SELECT
        relname as table_name,
        idx_scan as index_scans,
        seq_scan as sequential_scans,
        CASE WHEN seq_scan = 0 THEN 0
          ELSE 100.0 * seq_scan / (seq_scan + idx_scan)
        END as seq_scan_percent
      FROM
        pg_stat_user_tables
      WHERE
        (idx_scan + seq_scan) > 0
      ORDER BY
        seq_scan_percent DESC
      LIMIT 10;
    `,
      [],
      true
    );

    console.log("Tables with potential index issues:");

    // Get raw data for processing in the next step
    const indexStats = await client.executeQuery(`
      SELECT
        relname as table_name,
        idx_scan as index_scans,
        seq_scan as sequential_scans,
        CASE WHEN seq_scan = 0 THEN 0
          ELSE 100.0 * seq_scan / (seq_scan + idx_scan)
        END as seq_scan_percent
      FROM
        pg_stat_user_tables
      WHERE
        (idx_scan + seq_scan) > 0
      ORDER BY
        seq_scan_percent DESC
      LIMIT 10;
    `);

    // Process for warnings
    const indexStatsData = ensureDataArray(indexStats);

    if (indexStatsData.length > 0) {
      // Make sure seq_scan_percent is treated as a number
      const highSeqScanTables = indexStatsData.filter(
        (t) => parseFloat(t.seq_scan_percent) > 50
      );

      if (highSeqScanTables.length > 0) {
        console.log(
          "\nWarning: The following tables show high sequential scan percentages and may benefit from additional indexes:"
        );

        highSeqScanTables.forEach((t) => {
          console.log(
            `- ${t.table_name} (${parseFloat(t.seq_scan_percent).toFixed(
              2
            )}% sequential scans)`
          );
        });
      }
    }

    // 6. Summary
    console.log("\n============== Summary ==============");

    // Connection health
    const connectHealth =
      connectTime < 100 ? "✅ Good" : connectTime < 500 ? "⚠️ Fair" : "❌ Poor";
    console.log(
      `Connection Speed: ${connectHealth} (${formatDuration(connectTime)})`
    );

    // Cache health
    if (cacheHitRatio > 0) {
      const cacheHealth =
        cacheHitRatio > 0.99
          ? "✅ Excellent"
          : cacheHitRatio > 0.95
          ? "✅ Good"
          : cacheHitRatio > 0.9
          ? "⚠️ Fair"
          : "❌ Poor";

      console.log(
        `Cache Hit Ratio: ${cacheHealth} (${(cacheHitRatio * 100).toFixed(2)}%)`
      );
    } else {
      console.log(`Cache Hit Ratio: Unable to determine`);
    }

    // Connection utilization
    if (connStats.max_connections > 0) {
      const connUtilization =
        connStats.used_connections / connStats.max_connections;
      const connHealth =
        connUtilization < 0.5
          ? "✅ Good"
          : connUtilization < 0.8
          ? "⚠️ Fair"
          : "❌ High";

      console.log(
        `Connection Utilization: ${connHealth} (${connStats.used_connections}/${
          connStats.max_connections
        }, ${(connUtilization * 100).toFixed(2)}%)`
      );
    } else {
      console.log(`Connection Utilization: Unable to determine`);
    }
  } catch (error) {
    console.error("Error during diagnostics:", error);
  } finally {
    // Clean up
    client.close();
    console.log("\nDiagnostics completed.");
  }
}

// Run the diagnostics
runDiagnostics().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
