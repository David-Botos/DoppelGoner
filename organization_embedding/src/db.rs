// src/db.rs

use anyhow::{Context, Result};
use bb8::Pool;
use bb8_postgres::PostgresConnectionManager;
use log::{debug, info, warn};
use std::time::Duration;
use tokio_postgres::{Config, NoTls};

pub type PgPool = Pool<PostgresConnectionManager<NoTls>>;

/// Reads environment variables and constructs a PostgreSQL config.
/// Matches behavior and settings of the TS `hetznerPostgresConfig`.
fn build_pg_config() -> Config {
    let mut config = Config::new();

    // Get all environment variables with debugging
    let host = std::env::var("POSTGRES_HOST").unwrap_or_else(|e| {
        warn!("POSTGRES_HOST not found in environment: {}", e);
        "10.0.0.1".to_string()
    });

    let port_str = std::env::var("POSTGRES_PORT").unwrap_or_else(|e| {
        warn!("POSTGRES_PORT not found in environment: {}", e);
        "5432".to_string()
    });
    let port = port_str.parse::<u16>().unwrap_or_else(|e| {
        warn!("Invalid POSTGRES_PORT format: {}", e);
        5432
    });

    let dbname = std::env::var("POSTGRES_DB").unwrap_or_else(|e| {
        warn!("POSTGRES_DB not found in environment: {}", e);
        "dataplatform".to_string()
    });

    let user = std::env::var("POSTGRES_USER").unwrap_or_else(|e| {
        warn!("POSTGRES_USER not found in environment: {}", e);
        "postgres".to_string()
    });

    let password = std::env::var("POSTGRES_PASSWORD").unwrap_or_else(|e| {
        warn!("POSTGRES_PASSWORD not found in environment: {}", e);
        "".to_string()
    });

    // Log connection params (except password)
    info!("Database connection parameters:");
    info!("  Host: {}", host);
    info!("  Port: {}", port);
    info!("  Database: {}", dbname);
    info!("  User: {}", user);
    info!(
        "  Password: {}",
        if password.is_empty() {
            "[empty]"
        } else {
            "[set]"
        }
    );

    config
        .host(&host)
        .port(port)
        .dbname(&dbname)
        .user(&user)
        .password(&password);

    // Optional: settings for better diagnostics in PG logs
    config.application_name("deduplication");

    // Add connection timeout to protect against network issues
    config.connect_timeout(Duration::from_secs(10));

    config
}

/// Initializes the database connection pool.
/// This function is now async because we want to properly initialize the pool.
pub async fn connect() -> Result<PgPool> {
    // Get environment variables explicitly to test if they're set
    debug!("Environment variables:");
    for (key, value) in std::env::vars() {
        if key.starts_with("POSTGRES_") {
            debug!(
                "  {}: {}",
                key,
                if key == "POSTGRES_PASSWORD" {
                    "[hidden]"
                } else {
                    &value
                }
            );
        }
    }

    let config = build_pg_config();
    info!("Connecting to PostgreSQL database...");

    // Connection manager with TLS disabled (matches ssl: false in TS)
    let manager = PostgresConnectionManager::new(config, NoTls);

    // Try to build the pool, with more specific error handling
    info!("Building connection pool...");
    let pool = match Pool::builder()
        .max_size(20)
        .min_idle(Some(5))
        .idle_timeout(Some(Duration::from_secs(90)))
        .connection_timeout(Duration::from_secs(15))
        .build(manager)
        .await
    {
        Ok(p) => p,
        Err(e) => {
            warn!("Failed to build connection pool: {}", e);
            return Err(anyhow::anyhow!(
                "Failed to build database connection pool: {}",
                e
            ));
        }
    };

    // Test the connection
    info!("Testing database connection...");
    {
        let conn = match pool.get().await {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to get connection from pool: {}", e);
                return Err(anyhow::anyhow!(
                    "Failed to get initial test connection: {}",
                    e
                ));
            }
        };

        let result = match conn.query_one("SELECT 1", &[]).await {
            Ok(row) => row.get::<_, i32>(0), // Changed from i64 to i32
            Err(e) => {
                warn!("Test query failed: {}", e);
                return Err(anyhow::anyhow!("Failed to execute test query: {}", e));
            }
        };

        if result == 1 {
            info!("Database connection test successful");
        } else {
            warn!(
                "Database connection test returned unexpected value: {}",
                result
            );
            return Err(anyhow::anyhow!("Database connection test failed"));
        }
    }

    info!("Database connection pool initialized successfully");
    Ok(pool)
}

// Helper function to load environment variables from a file if needed
pub fn load_env_from_file(file_path: &str) -> Result<()> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    info!("Loading environment variables from file: {}", file_path);

    let file = File::open(file_path).context(format!("Failed to open env file: {}", file_path))?;

    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line.context("Failed to read line from env file")?;

        // Skip comments and empty lines
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }

        // Parse key=value
        if let Some(idx) = line.find('=') {
            let key = line[..idx].trim();
            let value = line[idx + 1..].trim();

            // Don't set if already present in environment
            if std::env::var(key).is_err() {
                debug!("Setting env var from file: {}", key);
                unsafe {
                    std::env::set_var(key, value);
                }
            } else {
                debug!("Env var already set, skipping: {}", key);
            }
        }
    }

    Ok(())
}
