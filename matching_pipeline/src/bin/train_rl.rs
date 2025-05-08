// src/bin/train_rl.rs

use anyhow::{Context, Result}; // For easy error handling
use log::{info, warn}; // For logging
use std::path::Path; // For path manipulation (e.g., .env file)
use std::time::Instant; // For timing the run

// Imports from your library crate (dedupe_lib)
// Ensure your lib.rs correctly exports these modules/functions
use dedupe_lib::db::{self, PgPool}; // For database connection and .env loading
use dedupe_lib::reinforcement::process_feedback; // The main function to trigger training

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Initialize logging
    // You can customize the default log level (e.g., "info", "debug")
    // It will read the RUST_LOG environment variable if set.
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    info!("🚀 Starting RL Model Training Run...");
    let start_time = Instant::now();

    // 2. Load environment variables (e.g., for database connection)
    // This attempts to load a .env file from various common locations.
    // Adjust the paths in `env_paths` if your .env file is located elsewhere relative
    // to where you run the compiled binary.
    let env_paths = [
        ".env",             // Project root
        ".env.local",       // Common for local overrides
        "../.env",          // If run from a subdirectory (like target/debug)
        "../../.env",       // If run from a deeper subdirectory
    ];
    let mut loaded_env = false;
    for path_str in env_paths.iter() {
        let path = Path::new(path_str);
        if path.exists() {
            let env_path = path.to_str();
            match db::load_env_from_file(env_path.unwrap()) { // Assuming load_env_from_file is public in db module
                Ok(_) => {
                    info!("✅ Loaded environment variables from {}", path.display());
                    loaded_env = true;
                    break; // Stop searching once an .env file is successfully loaded
                }
                Err(e) => {
                    warn!(
                        "⚠️ Failed to load environment from {}: {}. Continuing without it.",
                        path.display(),
                        e
                    );
                }
            }
        }
    }
    if !loaded_env {
        info!("ℹ️ No .env file found or loaded. Relying on system environment variables for DB connection.");
    }

    // 3. Connect to the database
    // The db::connect() function should use the loaded environment variables
    // (or system environment variables) to establish the database connection pool.
    info!("🔗 Connecting to the database...");
    let pool: PgPool = db::connect()
        .await
        .context("Failed to connect to the database. Ensure DB_URL or related env vars are set.")?;
    info!("✅ Successfully connected to the database.");

    // 4. Call the feedback processing function to trigger training
    info!("🧠 Starting feedback processing and model training cycle...");
    match process_feedback(&pool).await {
        Ok(_) => {
            info!("✅ Feedback processing and model training completed successfully.");
        }
        Err(e) => {
            // Log the error and propagate it to ensure a non-zero exit code on failure
            warn!("❌ An error occurred during feedback processing and training: {:?}", e);
            return Err(e).context("RL model training run failed during feedback processing");
        }
    }

    let elapsed = start_time.elapsed();
    info!(
        "🏁 RL Model Training Run finished in {:.2?}.",
        elapsed
    );

    Ok(())
}
