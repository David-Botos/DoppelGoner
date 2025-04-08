// src/db.rs

use anyhow::Result;
use bb8::Pool;
use bb8_postgres::PostgresConnectionManager;
use std::time::Duration;
use tokio_postgres::{Config, NoTls};

pub type PgPool = Pool<PostgresConnectionManager<NoTls>>;

/// Reads environment variables and constructs a PostgreSQL config.
/// Matches behavior and settings of the TS `hetznerPostgresConfig`.
fn build_pg_config() -> Config {
    let mut config = Config::new();

    config
        .host(&std::env::var("POSTGRES_HOST").unwrap_or_else(|_| "10.0.0.1".to_string()))
        .port(
            std::env::var("POSTGRES_PORT")
                .ok()
                .and_then(|s| s.parse::<u16>().ok())
                .unwrap_or(5432),
        )
        .dbname(&std::env::var("POSTGRES_DB").unwrap_or_else(|_| "dataplatform".to_string()))
        .user(&std::env::var("POSTGRES_USER").unwrap_or_else(|_| "postgres".to_string()))
        .password(&std::env::var("POSTGRES_PASSWORD").unwrap_or_else(|_| "".to_string()));

    // Optional: settings for better diagnostics in PG logs
    config.application_name("deduplication");

    config
}

/// Initializes the database connection pool.
pub fn connect() -> Result<PgPool> {
    let config = build_pg_config();

    // Connection manager with TLS disabled (matches ssl: false in TS)
    let manager = PostgresConnectionManager::new(config, NoTls);

    let pool = Pool::builder()
        .max_size(20) // max connections
        .min_idle(Some(5)) // minimum idle connections
        .idle_timeout(Some(Duration::from_secs(90))) // idle timeout
        .connection_timeout(Duration::from_secs(15)) // wait max 15s for connection
        .build_unchecked(manager); // fallible if async, this one is sync build

    Ok(pool)
}
