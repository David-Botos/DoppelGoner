"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.tableDependencies = void 0;
exports.validateMigrationOrder = validateMigrationOrder;
// Defines migration order and dependencies
const config_1 = require("./config");
// Define the dependencies for each table
exports.tableDependencies = [
    {
        table: 'organization',
        dependencies: [] // No dependencies
    },
    {
        table: 'location',
        dependencies: ['organization']
    },
    {
        table: 'service',
        dependencies: ['organization']
    },
    {
        table: 'service_at_location',
        dependencies: ['service', 'location']
    },
    {
        table: 'physical_address',
        dependencies: ['location']
    },
    {
        table: 'postal_address',
        dependencies: ['location']
    },
    {
        table: 'phone',
        dependencies: ['location', 'service', 'organization', 'service_at_location']
    }
];
// Validate the migration order defined in config
function validateMigrationOrder() {
    const tables = config_1.migrationConfig.tables;
    // Create a set of migrated tables to check dependencies
    const migratedTables = new Set();
    for (const table of tables) {
        // Find the dependencies for this table
        const dependency = exports.tableDependencies.find(d => d.table === table);
        if (!dependency) {
            console.error(`No dependency information found for table: ${table}`);
            return false;
        }
        // Check if all dependencies have been migrated
        for (const dep of dependency.dependencies) {
            if (!migratedTables.has(dep)) {
                console.error(`Invalid migration order: ${table} depends on ${dep}, but ${dep} hasn't been migrated yet`);
                return false;
            }
        }
        // Mark this table as migrated
        migratedTables.add(table);
    }
    return true;
}
