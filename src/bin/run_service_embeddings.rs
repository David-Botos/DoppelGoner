// src/bin/run_service_embeddings.rs
use anyhow::{Context, Result};
use log::info;
use std::path::Path;

// Import modules from the main crate
use dedupe_lib::{db, services::embed_services::embed_services};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    info!("Starting embed_services test");

    // Try to load .env file if it exists
    let env_paths = [".env", ".env.local", "../.env"];
    let mut loaded_env = false;

    for path in env_paths.iter() {
        if Path::new(path).exists() {
            if let Err(e) = db::load_env_from_file(path) {
                println!("Failed to load environment from {}: {}", path, e);
            } else {
                println!("Loaded environment variables from {}", path);
                loaded_env = true;
                break;
            }
        }
    }

    if !loaded_env {
        println!("No .env file found, using environment variables from system");
    }

    // Connect to the database
    let pool = db::connect()
        .await
        .context("Failed to connect to database")?;
    info!("Successfully connected to the database");

    // Run just the embed_services module
    info!("Running embed_services in isolation...");
    embed_services(&pool)
        .await
        .context("Failed to embed services")?;

    info!("Successfully completed embedding services");

    Ok(())
}
