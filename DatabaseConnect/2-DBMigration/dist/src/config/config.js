"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.hetznerPostgresConfig = exports.migrationConfig = exports.supabaseConfig = exports.snowflakeConfig = void 0;
// Configuration file for database connections and migration settings
const dotenv_1 = __importDefault(require("dotenv"));
dotenv_1.default.config();
// Database connection configurations
exports.snowflakeConfig = {
    account: process.env.SNOWFLAKE_ACCOUNT || "",
    username: process.env.SNOWFLAKE_USER || "",
    password: process.env.SNOWFLAKE_PASSWORD || "",
    warehouse: process.env.SNOWFLAKE_WAREHOUSE || "",
    database: "NORSE_STAGING",
    schema: "WHATCOMCOU",
    authenticator: "SNOWFLAKE",
};
exports.supabaseConfig = {
    url: process.env.SUPABASE_URL || "",
    key: process.env.SUPABASE_SERVICE_ROLE_KEY || "",
    schema: "public",
};
// Migration configuration
exports.migrationConfig = {
    batchSize: 1000, // Number of records to process at once
    logLevel: "info",
    enableValidation: true,
    // Tables to migrate in the specified order
    tables: [
        "organization",
        "location",
        "service",
        "service_at_location",
        "address", // Consolidated address table (replacing physical_address & postal_address)
        "phone",
    ],
};
exports.hetznerPostgresConfig = {
    host: process.env.POSTGRES_HOST || "10.0.0.1",
    port: parseInt(process.env.POSTGRES_PORT || "5432", 10),
    database: process.env.POSTGRES_DB || "dataplatform",
    user: process.env.POSTGRES_USER || "postgres",
    password: process.env.POSTGRES_PASSWORD || "",
    ssl: false,
    max: 20,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 2000,
};
