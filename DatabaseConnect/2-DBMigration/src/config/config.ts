// Configuration file for database connections and migration settings

// Database connection configurations
export const snowflakeConfig = {
  account: process.env.SNOWFLAKE_ACCOUNT || "",
  username: process.env.SNOWFLAKE_USER || "",
  password: process.env.SNOWFLAKE_PASSWORD || "",
  warehouse: process.env.SNOWFLAKE_WAREHOUSE || "",
  database: "NORSE_STAGING",
  schema: "WA211",
  authenticator: "SNOWFLAKE",
};

export const supabaseConfig = {
  url: process.env.SUPABASE_URL || "",
  key: process.env.SUPABASE_SERVICE_ROLE_KEY || "",
  schema: "public",
};

// Migration configuration
export const migrationConfig = {
  batchSize: 100, // Number of records to process at once
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
  // Additional validation rules for each table
  validationRules: {
    // Define rules per table
  },
};
