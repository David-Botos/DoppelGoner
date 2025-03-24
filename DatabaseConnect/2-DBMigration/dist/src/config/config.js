"use strict";
// Configuration file for database connections and migration settings
Object.defineProperty(exports, "__esModule", { value: true });
exports.migrationConfig = exports.postgresConfig = exports.snowflakeConfig = void 0;
// Database connection configurations
exports.snowflakeConfig = {
    account: process.env.SNOWFLAKE_ACCOUNT || "",
    username: process.env.SNOWFLAKE_USERNAME || "",
    password: process.env.SNOWFLAKE_PASSWORD || "",
    warehouse: process.env.SNOWFLAKE_WAREHOUSE || "",
    database: "NORSE_STAGING",
    schema: "WA211",
};
exports.postgresConfig = {
    host: process.env.PG_HOST || "localhost",
    port: parseInt(process.env.PG_PORT || "5432"),
    database: process.env.PG_DATABASE || "hsds_migration",
    user: process.env.PG_USER || "postgres",
    password: process.env.PG_PASSWORD || "",
    ssl: process.env.PG_SSL === "true",
};
// Migration configuration
exports.migrationConfig = {
    batchSize: 100, // Number of records to process at once
    logLevel: "info",
    enableValidation: true,
    // Tables to migrate in the specified order
    tables: [
        "organization",
        "location",
        "service",
        "service_at_location",
        "physical_address",
        "postal_address",
        "phone",
    ],
    // Additional validation rules for each table
    validationRules: {
    // Define rules per table
    },
};
