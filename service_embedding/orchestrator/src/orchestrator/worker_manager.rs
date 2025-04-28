// orchestrator/src/orchestrator/worker_manager.rs

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use crate::orchestrator::service::OrchestratorService;
use crate::orchestrator::stats::PipelineStats;

/// Commands that can be sent to the worker manager
pub enum WorkerCommand {
    /// Run the pipeline once and report results
    RunOnce,

    /// Pause the pipeline
    Pause,

    /// Resume a paused pipeline
    Resume,

    /// Shutdown the worker manager
    Shutdown,

    /// Get current pipeline stats
    GetStats,
}

/// Status of the worker manager
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkerManagerStatus {
    /// Worker is running and processing jobs
    Running,

    /// Worker is paused
    Paused,

    /// Worker is shutting down
    ShuttingDown,

    /// Worker has shut down
    ShutDown,
}

/// Response from the worker manager
pub struct WorkerResponse {
    /// Status of the worker manager
    pub status: WorkerManagerStatus,

    /// Pipeline stats from the last run
    pub stats: Option<PipelineStats>,

    /// Error message if an operation failed
    pub error: Option<String>,
}

/// Worker manager that runs the pipeline as a background process
pub struct WorkerManager {
    /// Command sender to control the worker
    command_tx: mpsc::Sender<WorkerCommand>,

    /// Response receiver for worker operations
    response_rx: Arc<Mutex<mpsc::Receiver<WorkerResponse>>>,

    /// Handle to the worker task
    worker_handle: JoinHandle<()>,

    /// Current status of the worker
    status: Arc<Mutex<WorkerManagerStatus>>,
}

impl WorkerManager {
    /// Create a new worker manager and start the background task
    pub fn new(orchestrator: Arc<OrchestratorService>, interval_secs: u64) -> Self {
        let (command_tx, command_rx) = mpsc::channel(100);
        let (response_tx, response_rx) = mpsc::channel(100);

        let status = Arc::new(Mutex::new(WorkerManagerStatus::Running));

        // Clone status for worker task
        let worker_status = status.clone();

        // Start worker task
        let worker_handle = tokio::spawn(async move {
            Self::worker_task(
                orchestrator,
                interval_secs,
                command_rx,
                response_tx,
                worker_status,
            )
            .await;
        });

        Self {
            command_tx,
            response_rx: Arc::new(Mutex::new(response_rx)),
            worker_handle,
            status,
        }
    }

    /// Get the current status of the worker manager
    pub async fn get_status(&self) -> WorkerManagerStatus {
        *self.status.lock().await
    }

    /// Run the pipeline once and wait for results
    pub async fn run_once(&self) -> Result<PipelineStats, String> {
        // Send command to run once
        if let Err(e) = self.command_tx.send(WorkerCommand::RunOnce).await {
            return Err(format!("Failed to send command: {}", e));
        }

        // Wait for response
        let mut rx = self.response_rx.lock().await;
        match rx.recv().await {
            Some(response) => match response.stats {
                Some(stats) => Ok(stats),
                None => Err(response
                    .error
                    .unwrap_or_else(|| "No stats returned".to_string())),
            },
            None => Err("Worker has shut down".to_string()),
        }
    }

    /// Pause the pipeline
    pub async fn pause(&self) -> Result<(), String> {
        // Check current status
        let current_status = *self.status.lock().await;
        if current_status == WorkerManagerStatus::Paused {
            return Ok(());
        }

        // Send command to pause
        if let Err(e) = self.command_tx.send(WorkerCommand::Pause).await {
            return Err(format!("Failed to send command: {}", e));
        }

        // Wait for response
        let mut rx = self.response_rx.lock().await;
        match rx.recv().await {
            Some(response) => {
                if response.status == WorkerManagerStatus::Paused {
                    Ok(())
                } else {
                    Err(response
                        .error
                        .unwrap_or_else(|| "Failed to pause worker".to_string()))
                }
            }
            None => Err("Worker has shut down".to_string()),
        }
    }

    /// Resume a paused pipeline
    pub async fn resume(&self) -> Result<(), String> {
        // Check current status
        let current_status = *self.status.lock().await;
        if current_status == WorkerManagerStatus::Running {
            return Ok(());
        }

        // Send command to resume
        if let Err(e) = self.command_tx.send(WorkerCommand::Resume).await {
            return Err(format!("Failed to send command: {}", e));
        }

        // Wait for response
        let mut rx = self.response_rx.lock().await;
        match rx.recv().await {
            Some(response) => {
                if response.status == WorkerManagerStatus::Running {
                    Ok(())
                } else {
                    Err(response
                        .error
                        .unwrap_or_else(|| "Failed to resume worker".to_string()))
                }
            }
            None => Err("Worker has shut down".to_string()),
        }
    }

    /// Get the current pipeline stats
    pub async fn get_stats(&self) -> Result<PipelineStats, String> {
        // Send command to get stats
        if let Err(e) = self.command_tx.send(WorkerCommand::GetStats).await {
            return Err(format!("Failed to send command: {}", e));
        }

        // Wait for response
        let mut rx = self.response_rx.lock().await;
        match rx.recv().await {
            Some(response) => match response.stats {
                Some(stats) => Ok(stats),
                None => Err(response
                    .error
                    .unwrap_or_else(|| "No stats available".to_string())),
            },
            None => Err("Worker has shut down".to_string()),
        }
    }

    /// Shutdown the worker manager
    pub async fn shutdown(&self) -> Result<(), String> {
        // Update status
        {
            let mut status = self.status.lock().await;
            *status = WorkerManagerStatus::ShuttingDown;
        }

        // Send command to shutdown
        if let Err(e) = self.command_tx.send(WorkerCommand::Shutdown).await {
            return Err(format!("Failed to send command: {}", e));
        }

        Ok(())
    }

    /// Worker task that runs the pipeline
    async fn worker_task(
        orchestrator: Arc<OrchestratorService>,
        interval_secs: u64,
        mut command_rx: mpsc::Receiver<WorkerCommand>,
        response_tx: mpsc::Sender<WorkerResponse>,
        status: Arc<Mutex<WorkerManagerStatus>>,
    ) {
        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
        let mut paused = false;

        loop {
            tokio::select! {
                // Check for commands
                Some(command) = command_rx.recv() => {
                    match command {
                        WorkerCommand::RunOnce => {
                            // Run the pipeline once
                            info!("Running pipeline once on command");
                            let result = orchestrator.run_embedding_pipeline().await;

                            match result {
                                Ok(stats) => {
                                    if let Err(e) = response_tx.send(WorkerResponse {
                                        status: *status.lock().await,
                                        stats: Some(stats),
                                        error: None,
                                    }).await {
                                        error!("Failed to send response: {}", e);
                                    }
                                },
                                Err(e) => {
                                    error!("Pipeline run failed: {}", e);
                                    if let Err(e) = response_tx.send(WorkerResponse {
                                        status: *status.lock().await,
                                        stats: None,
                                        error: Some(format!("Pipeline error: {}", e)),
                                    }).await {
                                        error!("Failed to send response: {}", e);
                                    }
                                }
                            }
                        },
                        WorkerCommand::Pause => {
                            // Pause the pipeline
                            info!("Pausing pipeline");
                            paused = true;

                            // Update status
                            {
                                let mut status_guard = status.lock().await;
                                *status_guard = WorkerManagerStatus::Paused;
                            }

                            if let Err(e) = response_tx.send(WorkerResponse {
                                status: WorkerManagerStatus::Paused,
                                stats: None,
                                error: None,
                            }).await {
                                error!("Failed to send response: {}", e);
                            }
                        },
                        WorkerCommand::Resume => {
                            // Resume the pipeline
                            info!("Resuming pipeline");
                            paused = false;

                            // Update status
                            {
                                let mut status_guard = status.lock().await;
                                *status_guard = WorkerManagerStatus::Running;
                            }

                            if let Err(e) = response_tx.send(WorkerResponse {
                                status: WorkerManagerStatus::Running,
                                stats: None,
                                error: None,
                            }).await {
                                error!("Failed to send response: {}", e);
                            }
                        },
                        WorkerCommand::Shutdown => {
                            // Shutdown the pipeline
                            info!("Shutting down pipeline worker");

                            // Update status
                            {
                                let mut status_guard = status.lock().await;
                                *status_guard = WorkerManagerStatus::ShutDown;
                            }

                            if let Err(e) = response_tx.send(WorkerResponse {
                                status: WorkerManagerStatus::ShutDown,
                                stats: None,
                                error: None,
                            }).await {
                                error!("Failed to send response: {}", e);
                            }

                            break;
                        },
                        WorkerCommand::GetStats => {
                            // Get current stats
                            let stats = orchestrator.get_stats().await;

                            if let Err(e) = response_tx.send(WorkerResponse {
                                status: *status.lock().await,
                                stats: Some(stats),
                                error: None,
                            }).await {
                                error!("Failed to send response: {}", e);
                            }
                        }
                    }
                }

                // Run pipeline at interval unless paused
                _ = interval.tick() => {
                    if !paused {
                        // Reset stale jobs
                        if let Err(e) = orchestrator.reset_stale_jobs().await {
                            error!("Failed to reset stale jobs: {}", e);
                        }

                        // Run the pipeline
                        info!("Running pipeline on schedule");
                        match orchestrator.run_embedding_pipeline().await {
                            Ok(stats) => {
                                info!(
                                    "Pipeline completed: processed {} jobs, tokenized {} documents, generated {} embeddings in {}ms",
                                    stats.jobs_processed, stats.documents_tokenized, stats.embeddings_generated, stats.total_time_ms
                                );
                            },
                            Err(e) => {
                                error!("Pipeline run failed: {}", e);
                            }
                        }
                    }
                }
            }
        }

        info!("Worker task has shutdown");
    }
}
