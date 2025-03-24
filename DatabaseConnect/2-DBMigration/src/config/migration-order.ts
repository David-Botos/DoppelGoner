// Defines migration order and dependencies
import { migrationConfig } from "./config";

// Define table dependencies
export interface TableDependency {
  table: string;
  dependencies: string[];
}

// Define the dependencies for each table
export const tableDependencies: TableDependency[] = [
  {
    table: "organization",
    dependencies: [], // No dependencies
  },
  {
    table: "location",
    dependencies: ["organization"],
  },
  {
    table: "service",
    dependencies: ["organization"],
  },
  {
    table: "service_at_location",
    dependencies: ["service", "location"],
  },
  {
    table: "address", // Consolidated address table
    dependencies: ["location"],
  },
  {
    table: "phone",
    dependencies: [
      "location",
      "service",
      "organization",
      "service_at_location",
    ],
  },
];

// Validate the migration order defined in config
export function validateMigrationOrder(): boolean {
  const tables = migrationConfig.tables;

  // Create a set of migrated tables to check dependencies
  const migratedTables = new Set<string>();

  for (const table of tables) {
    // Find the dependencies for this table
    const dependency = tableDependencies.find((d) => d.table === table);
    if (!dependency) {
      console.error(`No dependency information found for table: ${table}`);
      return false;
    }

    // Check if all dependencies have been migrated
    for (const dep of dependency.dependencies) {
      if (!migratedTables.has(dep)) {
        console.error(
          `Invalid migration order: ${table} depends on ${dep}, but ${dep} hasn't been migrated yet`
        );
        return false;
      }
    }

    // Mark this table as migrated
    migratedTables.add(table);
  }

  return true;
}
