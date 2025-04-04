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
  // Number of concurrent batches to process
  concurrentBatches: 8,
  // Tables to migrate in the specified order
  tables: [
    "organization",
    "location",
    "service",
    "service_at_location",
    "address",
    "phone",
  ],
};

export const hetznerPostgresConfig = {
  host: process.env.POSTGRES_HOST || "10.0.0.1",
  port: parseInt(process.env.POSTGRES_PORT || "5432", 10),
  database: process.env.POSTGRES_DB || "dataplatform",
  user: process.env.POSTGRES_USER || "postgres",
  password: process.env.POSTGRES_PASSWORD || "",
  ssl: false,
  // Connection pool settings
  max: 20, // Maximum number of connections
  min: 5, // Minimum number of idle connections to maintain
  idleTimeoutMillis: 90000, // 90 seconds - since your server has good memory
  connectionTimeoutMillis: 15000, // 15 seconds - better for Docker networking
  // Query settings
  statement_timeout: 60000, // 60 seconds since you have good memory allocation
  application_name: "etl-migration", // Helpful for identifying connections
  // Performance settings matching your PostgreSQL config
  query_timeout: 120000, // 2 minutes for more complex queries
  // Connection behavior
  keepalive: true,
  keepaliveInitialDelayMillis: 30000, // 30 seconds before first keepalive
  // Add retry logic for connection failures
  max_retries: 5,
  retry_interval: 2000, // 2 seconds between retries
};
