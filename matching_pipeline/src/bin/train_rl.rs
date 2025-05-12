// src/bin/train_rl.rs

use anyhow::{Context, Result};
use log::{info, warn};
use std::path::Path;
use std::time::Instant;

use dedupe_lib::db::{self, fetch_recent_feedback_items, PgPool};
// Import the new main RL cycle function and the Orchestrator
use dedupe_lib::reinforcement::{feedback_processor, MatchingOrchestrator};

// This could be a method on MatchingOrchestrator or a free function
pub async fn run_reinforcement_learning_cycle(
    pool: &PgPool,
    orchestrator: &mut MatchingOrchestrator, // Get mutable access to update models
) -> Result<()> {
    info!("Starting reinforcement learning cycle...");

    // 1. Fetch recent feedback items
    let feedback_items = fetch_recent_feedback_items(pool)
        .await
        .context("Failed to fetch recent feedback items")?;

    if feedback_items.is_empty() {
        info!("No new feedback items to process. Models are up-to-date.");
        return Ok(());
    }
    info!("Fetched {} new feedback items.", feedback_items.len());

    // 2. Prepare training data using feedback_processor
    // The prepare_pairwise_training_data_batched function from the previous step is used here.
    // It takes &PgClient, so we get one from the pool.
    let client = pool
        .get()
        .await
        .context("Failed to get client from pool for training data prep")?;
    let training_examples =
        feedback_processor::prepare_pairwise_training_data_batched(&client, &feedback_items)
            .await
            .context("Failed to prepare pairwise training data")?;
    drop(client); // Release client

    // 3. Train ContextModel
    // We need the 'was_correct' labels from feedback_items aligned with training_examples
    if !training_examples.is_empty() {
        info!(
            "Training ContextModel with {} examples...",
            training_examples.len()
        );
        // orchestrator.context_model is public or has a method to access it mutably
        // For simplicity, let's assume direct mutable access or a specific training method.
        // The current ContextModel::train_with_data expects &[TrainingExample].
        // The `was_correct` labels aren't explicitly passed to it, which is a design flaw if it's a typical supervised classifier.
        // However, TrainingExample.best_method IS derived from was_correct indirectly.
        // The `collect_training_data` in `ContextModel` also tries to establish `best_method`.
        // Let's assume `TrainingExample.best_method` (derived from `FeedbackItem.method_type` where `was_correct` is true)
        // is the target for the RandomForest in ContextModel.
        orchestrator
            .context_model
            .train_with_data(&training_examples)
            .context("Failed to train ContextModel with prepared data")?;
        info!(
            "ContextModel training completed. New version: {}",
            orchestrator.context_model.version
        );
        orchestrator
            .context_model
            .save_to_db(pool)
            .await
            .context("Failed to save ContextModel")?;
    } else {
        info!("No training examples generated for ContextModel.");
    }

    // 4. Update ConfidenceTuner
    // The ConfidenceTuner learns from whether its confidence assignments led to correct outcomes.
    // Each FeedbackItem tells us the method_type, the system's confidence, and if it was_correct.
    info!("Updating ConfidenceTuner...");
    for item in &feedback_items {
        let reward = if item.was_correct { 1.0 } else { 0.0 }; // Simple reward
        orchestrator
            .confidence_tuner
            .update(&item.method_type, item.confidence, reward)
            .context(format!(
                "Failed to update confidence tuner for method {}",
                item.method_type
            ))?;
    }
    orchestrator.confidence_tuner.version += 1; // Manually increment version after a batch of updates
    orchestrator
        .confidence_tuner
        .save_to_db(pool)
        .await
        .context("Failed to save ConfidenceTuner")?;
    info!(
        "ConfidenceTuner update completed. New version: {}",
        orchestrator.confidence_tuner.version
    );

    info!("Reinforcement learning cycle completed.");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    info!("🚀 Starting RL Model Training Run...");
    let start_time = Instant::now();

    // --- Load Environment Variables ---
    let env_paths = [".env", ".env.local", "../.env", "../../.env"];
    let mut loaded_env = false;
    for path_str in env_paths.iter() {
        let path = Path::new(path_str);
        if path.exists() {
            match db::load_env_from_file(path.to_str().unwrap_or_default()) {
                Ok(_) => {
                    info!("✅ Loaded environment variables from {}", path.display());
                    loaded_env = true;
                    break;
                }
                Err(e) => warn!(
                    "⚠️ Failed to load environment from {}: {}.",
                    path.display(),
                    e
                ),
            }
        }
    }
    if !loaded_env {
        info!("ℹ️ No .env file found or loaded. Relying on system environment variables.");
    }

    // --- Connect to Database ---
    info!("🔗 Connecting to the database...");
    let pool: PgPool = db::connect()
        .await
        .context("Failed to connect to the database.")?;
    info!("✅ Successfully connected to the database.");

    // --- Initialize Matching Orchestrator ---
    // The orchestrator loads its models (ContextModel, ConfidenceTuner) internally from the DB or creates new ones.
    info!("🔧 Initializing Matching Orchestrator (loading models)...");
    let mut orchestrator = MatchingOrchestrator::new(&pool) // `new` now takes &PgPool
        .await
        .context("Failed to initialize MatchingOrchestrator")?;
    info!("✅ Matching Orchestrator initialized.");
    info!(
        "   ContextModel version: {}",
        orchestrator.context_model.version
    );
    info!(
        "   ConfidenceTuner version: {}",
        orchestrator.confidence_tuner.version
    );

    // --- Run the Reinforcement Learning Cycle ---
    info!("🧠 Starting reinforcement learning cycle...");
    // The run_reinforcement_learning_cycle function now takes a mutable reference to the orchestrator
    match run_reinforcement_learning_cycle(&pool, &mut orchestrator).await {
        Ok(_) => {
            info!("✅ Reinforcement learning cycle completed successfully.");
            info!(
                "   ContextModel new version: {}",
                orchestrator.context_model.version
            );
            info!(
                "   ConfidenceTuner new version: {}",
                orchestrator.confidence_tuner.version
            );
        }
        Err(e) => {
            warn!(
                "❌ An error occurred during the reinforcement learning cycle: {:?}",
                e
            );
            return Err(e).context("RL model training run failed during RL cycle");
        }
    }

    let elapsed = start_time.elapsed();
    info!("🏁 RL Model Training Run finished in {:.2?}.", elapsed);

    Ok(())
}
