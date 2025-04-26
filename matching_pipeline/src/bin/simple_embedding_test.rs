// src/bin/simple_embedding_test.rs
use anyhow::{Context, Result};
use log::{debug, error, info, warn};
use pgvector::Vector;
use std::path::Path;
use std::time::Instant;

// Import modules from the main crate
use dedupe_lib::service_embedding::config::{CONFIG_PATH, MODEL_PATH, TOKENIZER_PATH};
use dedupe_lib::service_embedding::data_fetcher::DataFetcher;
use dedupe_lib::{db, service_embedding::inference::InferenceEngine, service_embedding::tokenizer};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    info!("Starting simple embedding test");

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

    // Create data fetcher
    let data_fetcher = DataFetcher::new(pool.clone(), 10, 1);

    // Count services needing embeddings
    let total_count = data_fetcher.count_services_needing_embeddings().await?;
    info!("Found {} services without embeddings", total_count);

    if total_count == 0 {
        info!("No services to embed, exiting");
        return Ok(());
    }

    // Fetch a small batch for testing
    let batch_size = 10;
    let service_ids = data_fetcher
        .fetch_services_needing_embeddings(batch_size)
        .await?;
    info!("Fetched {} service IDs for testing", service_ids.len());

    if service_ids.is_empty() {
        info!("No service IDs found, exiting");
        return Ok(());
    }

    // Fetch service data
    let services = data_fetcher.fetch_service_data(&service_ids).await?;
    info!("Fetched data for {} services", services.len());

    // Load tokenizer
    info!("Loading tokenizer from {}", TOKENIZER_PATH);
    let tokenizer =
        tokenizer::load_tokenizer(TOKENIZER_PATH).context("Failed to load tokenizer")?;

    // Initialize tokenization manager
    let tokenization_manager = tokenizer::TokenizationManager::new(tokenizer.clone());

    // Tokenize services
    let tokenized_batch = tokenization_manager.tokenize_services(services).await?;
    info!("Tokenized {} services", tokenized_batch.service_ids.len());

    // Initialize inference engine
    info!("Initializing inference engine");
    let inference_engine = InferenceEngine::new(
        Some(MODEL_PATH),
        Some(CONFIG_PATH),
        false, // Don't force CPU
    )
    .await
    .context("Failed to initialize inference engine")?;

    // Generate embeddings
    info!("Generating embeddings");
    let (service_ids, embeddings, _) = inference_engine
        .generate_embeddings(tokenized_batch)
        .await
        .context("Failed to generate embeddings")?;

    info!("Generated {} embeddings", embeddings.len());

    // Save embeddings to database
    info!("Saving embeddings to database");
    let start = Instant::now();

    // Get a client from the pool
    let mut client = pool
        .get()
        .await
        .context("Failed to get database connection")?;

    // Start a transaction
    let transaction = client
        .transaction()
        .await
        .context("Failed to start transaction")?;

    let mut success_count = 0;

    for (i, (id, embedding)) in service_ids.iter().zip(embeddings.iter()).enumerate() {
        // Convert Vec<f32> to pgvector::Vector
        let pgvector = Vector::from(embedding.clone());

        // Update the service record with the new embedding
        match transaction
            .execute(
                "UPDATE service SET embedding_v2 = $1, embedding_v2_updated_at = NOW() WHERE id = $2",
                &[&pgvector, id],
            )
            .await
        {
            Ok(rows) => {
                if rows > 0 {
                    success_count += 1;
                    info!("Successfully updated embedding for service {}", id);
                } else {
                    warn!("No rows affected for service ID {}, possibly invalid ID", id);
                }
            },
            Err(e) => {
                error!("Error updating embedding for service {}: {}", id, e);
            }
        }
    }

    // Commit the transaction
    if success_count > 0 {
        match transaction.commit().await {
            Ok(_) => {
                info!(
                    "Successfully committed transaction with {} embeddings",
                    success_count
                );
            }
            Err(e) => {
                error!("Failed to commit transaction: {}", e);
                return Err(anyhow::anyhow!("Failed to commit transaction: {}", e));
            }
        }
    } else {
        warn!("No successful updates, rolling back transaction");
        if let Err(e) = transaction.rollback().await {
            warn!("Error rolling back transaction: {}", e);
        }
    }

    info!(
        "Saved {} embeddings in {:.2?}",
        success_count,
        start.elapsed()
    );

    // Verify the embeddings were saved
    let client = pool
        .get()
        .await
        .context("Failed to get database connection")?;
    let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = service_ids
        .iter()
        .map(|id| id as &(dyn tokio_postgres::types::ToSql + Sync))
        .collect();

    let placeholders: Vec<String> = service_ids
        .iter()
        .enumerate()
        .map(|(i, _)| format!("${}", i + 1))
        .collect();

    let query = format!(
        "SELECT id, embedding_v2_updated_at FROM service WHERE id IN ({}) AND embedding_v2 IS NOT NULL",
        placeholders.join(",")
    );

    let rows = client
        .query(&query, &params[..])
        .await
        .context("Failed to verify saved embeddings")?;

    info!(
        "Verified {} services have embeddings in database",
        rows.len()
    );

    for row in rows {
        let id: String = row.get("id");
        let updated_at: Option<chrono::DateTime<chrono::Utc>> = row.get("embedding_v2_updated_at");
        info!("Service {}: updated at {:?}", id, updated_at);
    }

    Ok(())
}
