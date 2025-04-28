use dotenv::dotenv;
use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    // Load environment variables from .env file
    dotenv().ok();

    // Get database connection details
    let host = env::var("POSTGRES_HOST").unwrap_or("localhost".into());
    let port = env::var("POSTGRES_PORT").unwrap_or("5432".into());
    let user = env::var("POSTGRES_USER").unwrap_or("postgres".into());
    let pass = env::var("POSTGRES_PASSWORD").unwrap_or("".into());
    let db = env::var("POSTGRES_DB").unwrap_or("postgres".into());

    let db_url = format!("postgres://{}:{}@{}:{}/{}", user, pass, host, port, db);

    println!("Generating SQLx prepare file...");

    // Create a .sqlx directory if it doesn't exist
    let sqlx_dir = Path::new("./.sqlx");
    if !sqlx_dir.exists() {
        fs::create_dir_all(sqlx_dir).expect("Failed to create .sqlx directory");
    }

    // Create a prepare.sql file with type information for SQLx
    let prepare_sql = r#"
-- Define schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS embedding;

-- Tell SQLx about our enum types
DO $$ 
BEGIN
    -- Create job_status type if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM pg_type 
                  JOIN pg_namespace ON pg_namespace.oid = pg_type.typnamespace
                  WHERE pg_type.typname = 'job_status' 
                  AND pg_namespace.nspname = 'embedding') THEN
        CREATE TYPE embedding.job_status AS ENUM ('queued', 'fetched', 'tokenized', 'processing', 'completed', 'failed');
    END IF;
    
    -- Create worker_type type if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM pg_type 
                  JOIN pg_namespace ON pg_namespace.oid = pg_type.typnamespace
                  WHERE pg_type.typname = 'worker_type' 
                  AND pg_namespace.nspname = 'embedding') THEN
        CREATE TYPE embedding.worker_type AS ENUM ('orchestrator', 'inference');
    END IF;
    
    -- Create worker_status type if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM pg_type 
                  JOIN pg_namespace ON pg_namespace.oid = pg_type.typnamespace
                  WHERE pg_type.typname = 'worker_status' 
                  AND pg_namespace.nspname = 'embedding') THEN
        CREATE TYPE embedding.worker_status AS ENUM ('online', 'offline', 'busy');
    END IF;
END $$;

-- Add a query to force SQLx to collect metadata about our types
SELECT 
    'queued'::embedding.job_status as job_status,
    'orchestrator'::embedding.worker_type as worker_type,
    'online'::embedding.worker_status as worker_status;
"#;

    // Write prepare.sql to .sqlx directory
    fs::write(sqlx_dir.join("prepare.sql"), prepare_sql).expect("Failed to write prepare.sql file");

    println!("Created prepare.sql file with custom type definitions");

    // Run cargo sqlx prepare with our customizations
    // Use only standard flags that are supported
    let status = Command::new("cargo")
        .args(&["sqlx", "prepare", "--check", "--database-url", &db_url])
        .env("DATABASE_URL", &db_url)
        .status()
        .expect("Failed to run cargo sqlx prepare");

    if status.success() {
        println!("Successfully generated SQLx prepare file");

        // Check if sqlx-data.json was created
        if Path::new("./sqlx-data.json").exists() {
            println!("✅ sqlx-data.json file created successfully!");
        } else {
            println!("⚠️ Warning: sqlx-data.json was not created.");
        }
    } else {
        println!(
            "❌ Failed to generate SQLx prepare file. Exit code: {:?}",
            status.code()
        );
    }

    std::process::exit(status.code().unwrap_or(1));
}
