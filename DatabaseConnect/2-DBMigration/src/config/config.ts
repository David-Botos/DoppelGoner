// Configuration file for database connections and migration settings
import dotenv from "dotenv";
dotenv.config();
// Database connection configurations
export const snowflakeConfig = {
  account: process.env.SNOWFLAKE_ACCOUNT || "",
  username: process.env.SNOWFLAKE_USER || "",
  password: process.env.SNOWFLAKE_PASSWORD || "",
  warehouse: process.env.SNOWFLAKE_WAREHOUSE || "",
  database: "NORSE_STAGING",
  schema: ["WA211", "WITHINREAC", "WHATCOMCOU"],
  authenticator: "SNOWFLAKE",
};

export const supabaseConfig = {
  url: process.env.SUPABASE_URL || "",
  key: process.env.SUPABASE_SERVICE_ROLE_KEY || "",
  schema: "public",
};

// Migration configuration
export const migrationConfig = {
  batchSize: 2000, // Number of records to process at once
  logLevel: "info",
  enableValidation: true,
  // Tables to migrate in the specified order
  tables: [
    // "organization",
    "location",
    // "service",
    // "service_at_location",
    // "address", // Consolidated address table (replacing physical_address & postal_address)
    // "phone",
  ],
};

export const hetznerPostgresConfig = {
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
