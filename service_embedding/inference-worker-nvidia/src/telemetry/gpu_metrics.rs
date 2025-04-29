// inference_worker/src/telemetry/gpu_metrics.rs
use anyhow::Result;
use std::{
    collections::VecDeque,
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};
use tracing::{debug, info, warn};

// For CUDA metrics using tch-rs
#[cfg(feature = "cuda")]
use tch::Cuda;

#[derive(Clone, Debug)]
pub struct GPUMetricSample {
    pub timestamp: Instant,
    pub memory_used_mb: f64,
    pub utilization: f64,
    pub batch_size: Option<usize>,
    pub operation: Option<String>,
}

/// GPU metrics collection for monitoring resource usage
pub struct GPUMetrics {
    // Common metrics
    pub total_memory_mb: f64,
    pub memory_used_mb: f64,
    pub memory_free_mb: f64,
    pub utilization: f64,
    pub compute_utilization: f64,
    pub memory_utilization: f64,
    pub last_updated: Instant,
    pub update_interval: Duration,
    pub peak_memory_used_mb: f64,
    pub peak_utilization: f64,
    pub memory_allocation_count: usize,
    pub memory_deallocation_count: usize,
    pub peak_batch_memory_mb: f64,
    pub metrics_history: VecDeque<GPUMetricSample>,
    pub memory_throttling_events: usize,

    // Manual memory tracking
    memory_last_used: AtomicUsize,
}

impl GPUMetrics {
    pub fn new() -> Self {
        let mut metrics = Self {
            total_memory_mb: 0.0,
            memory_used_mb: 0.0,
            memory_free_mb: 0.0,
            utilization: 0.0,
            compute_utilization: 0.0,
            memory_utilization: 0.0,
            last_updated: Instant::now(),
            update_interval: Duration::from_secs(1),
            peak_memory_used_mb: 0.0,
            peak_utilization: 0.0,
            memory_allocation_count: 0,
            memory_deallocation_count: 0,
            peak_batch_memory_mb: 0.0,
            metrics_history: VecDeque::with_capacity(100),
            memory_throttling_events: 0,
            memory_last_used: AtomicUsize::new(0),
        };

        // Initialize with system values
        let _ = metrics.update();
        metrics
    }

    pub fn update(&mut self) -> Result<()> {
        // If we've updated recently, don't do it again
        if self.last_updated.elapsed() < self.update_interval {
            return Ok(());
        }

        // Update metrics based on available backend
        #[cfg(feature = "cuda")]
        self.update_cuda_metrics()?;

        #[cfg(not(feature = "cuda"))]
        {
            // Fallback for CPU or when no GPU features are enabled
            // Use placeholder values or retrieve from our memory tracker
            let used_bytes = self.memory_last_used.load(Ordering::Relaxed);
            let used_mb = used_bytes as f64 / 1024.0 / 1024.0;
            
            if self.total_memory_mb == 0.0 {
                // First time initialization with placeholder
                self.total_memory_mb = 16384.0; // 16GB placeholder
            }
            
            self.memory_used_mb = used_mb;
            self.memory_free_mb = self.total_memory_mb - self.memory_used_mb;
            self.utilization = (self.memory_used_mb / self.total_memory_mb) * 100.0; // Estimate utilization from memory
            self.compute_utilization = self.utilization * 0.8; // Rough estimate
            self.memory_utilization = self.utilization;
        }

        // Update peak values
        if self.memory_used_mb > self.peak_memory_used_mb {
            self.peak_memory_used_mb = self.memory_used_mb;
        }

        if self.utilization > self.peak_utilization {
            self.peak_utilization = self.utilization;
        }

        // Add to history
        self.metrics_history.push_back(GPUMetricSample {
            timestamp: Instant::now(),
            memory_used_mb: self.memory_used_mb,
            utilization: self.utilization,
            batch_size: None,
            operation: Some("regular_update".to_string()),
        });

        // Trim history if too large
        if self.metrics_history.len() > 100 {
            self.metrics_history.pop_front();
        }

        // Detect if memory is critically high (over 90%)
        if self.memory_used_mb / self.total_memory_mb > 0.9 {
            self.memory_throttling_events += 1;
            tracing::warn!(
                "GPU memory critically high: {}MB used ({}% of {}MB)",
                self.memory_used_mb,
                (self.memory_used_mb / self.total_memory_mb * 100.0).round(),
                self.total_memory_mb
            );
        }

        self.last_updated = Instant::now();
        debug!(
            "GPU metrics updated: {}MB used, {}% utilization",
            self.memory_used_mb, self.utilization
        );

        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn update_cuda_metrics(&mut self) -> Result<()> {
        // Use tch::Cuda functions to get memory info
        if Cuda::is_available() {
            // Get device properties
            let device_index = 0;  // Default to first GPU
            
            // Try to get info directly from pytorch
            if let Some(mem_info) = tch::Cuda::memory_stats(device_index) {
                if let (Some(allocated), Some(reserved)) = (mem_info.get("allocated.all.current"), mem_info.get("reserved.all.current")) {
                    // Convert bytes to megabytes
                    let used_mb = *allocated as f64 / 1024.0 / 1024.0;
                    let reserved_mb = *reserved as f64 / 1024.0 / 1024.0;
                    
                    self.memory_used_mb = used_mb;
                    
                    // If we don't have total memory yet, try to get it or use reserved as fallback
                    if self.total_memory_mb == 0.0 {
                        let total_mb = if let Some(properties) = tch::Cuda::device_properties(device_index) {
                            if let Some(total_memory) = properties.get("total_memory") {
                                *total_memory as f64 / 1024.0 / 1024.0
                            } else {
                                reserved_mb * 1.1  // Estimate total as slightly more than reserved
                            }
                        } else {
                            reserved_mb * 1.1
                        };
                        
                        self.total_memory_mb = total_mb;
                    }
                    
                    self.memory_free_mb = self.total_memory_mb - self.memory_used_mb;
                    self.memory_utilization = (self.memory_used_mb / self.total_memory_mb) * 100.0;
                    
                    // Estimate compute utilization from memory usage (simplistic)
                    self.compute_utilization = self.memory_utilization * 0.8;
                    self.utilization = (self.compute_utilization + self.memory_utilization) / 2.0;
                    
                    return Ok(());
                }
            }
            
            // Fallback to nvidia-smi if available
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args([
                    "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
                    "--format=csv,noheader,nounits",
                ])
                .output()
            {
                if output.status.success() {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        let parts: Vec<&str> = output_str.trim().split(',').collect();
                        if parts.len() >= 5 {
                            if let (Ok(total), Ok(used), Ok(free), Ok(gpu_util), Ok(mem_util)) = (
                                parts[0].trim().parse::<f64>(),
                                parts[1].trim().parse::<f64>(),
                                parts[2].trim().parse::<f64>(),
                                parts[3].trim().parse::<f64>(),
                                parts[4].trim().parse::<f64>(),
                            ) {
                                self.total_memory_mb = total;
                                self.memory_used_mb = used;
                                self.memory_free_mb = free;
                                self.compute_utilization = gpu_util;
                                self.memory_utilization = mem_util;
                                self.utilization = (gpu_util + mem_util) / 2.0;
                                return Ok(());
                            }
                        }
                    }
                }
            }
        }
        
        // If all methods failed, use our manual tracking
        let used_bytes = self.memory_last_used.load(Ordering::Relaxed);
        let used_mb = used_bytes as f64 / 1024.0 / 1024.0;
        
        // Provide reasonable defaults
        if self.total_memory_mb == 0.0 {
            self.total_memory_mb = 16384.0;  // 16GB as default
        }
        
        self.memory_used_mb = used_mb;
        self.memory_free_mb = self.total_memory_mb - self.memory_used_mb;
        self.memory_utilization = (self.memory_used_mb / self.total_memory_mb) * 100.0;
        self.compute_utilization = self.memory_utilization * 0.8;
        self.utilization = (self.compute_utilization + self.memory_utilization) / 2.0;
        
        Ok(())
    }

    // Record memory allocation
    pub fn record_allocation(&mut self, bytes: usize) {
        let current = self.memory_last_used.load(Ordering::Relaxed);
        self.memory_last_used.store(current + bytes, Ordering::Relaxed);

        self.memory_allocation_count += 1;

        // Update bytes to MB and record
        let mb = bytes as f64 / 1024.0 / 1024.0;

        let memory_used_mb = self.get_memory_used_mb();
        let utilization = self.get_utilization();

        // Update metrics_history
        self.metrics_history.push_back(GPUMetricSample {
            timestamp: Instant::now(),
            memory_used_mb: memory_used_mb,
            utilization: utilization,
            batch_size: None,
            operation: Some(format!("alloc_{}mb", mb.round())),
        });

        // Trim history if too large
        if self.metrics_history.len() > 100 {
            self.metrics_history.pop_front();
        }
    }

    // Record memory deallocation
    pub fn record_deallocation(&mut self, bytes: usize) {
        let current = self.memory_last_used.load(Ordering::Relaxed);
        let new_value = if bytes > current { 0 } else { current - bytes };
        self.memory_last_used.store(new_value, Ordering::Relaxed);

        self.memory_deallocation_count += 1;

        // Update bytes to MB and record
        let mb = bytes as f64 / 1024.0 / 1024.0;

        let memory_used_mb = self.get_memory_used_mb();
        let utilization = self.get_utilization();

        // Update metrics_history
        self.metrics_history.push_back(GPUMetricSample {
            timestamp: Instant::now(),
            memory_used_mb: memory_used_mb,
            utilization: utilization,
            batch_size: None,
            operation: Some(format!("dealloc_{}mb", mb.round())),
        });

        // Trim history if too large
        if self.metrics_history.len() > 100 {
            self.metrics_history.pop_front();
        }
    }

    pub fn get_memory_total_mb(&mut self) -> f64 {
        let _ = self.update();
        self.total_memory_mb
    }

    pub fn get_memory_used_mb(&mut self) -> f64 {
        let _ = self.update();
        self.memory_used_mb
    }

    pub fn get_memory_free_mb(&mut self) -> f64 {
        let _ = self.update();
        self.memory_free_mb
    }

    pub fn get_utilization(&mut self) -> f64 {
        let _ = self.update();
        self.utilization
    }

    pub fn get_compute_utilization(&mut self) -> f64 {
        let _ = self.update();
        self.compute_utilization
    }

    pub fn get_memory_utilization(&mut self) -> f64 {
        let _ = self.update();
        self.memory_utilization
    }

    pub fn record_batch_processing(&mut self, batch_size: usize, memory_mb: f64) {
        if memory_mb > self.peak_batch_memory_mb {
            self.peak_batch_memory_mb = memory_mb;
        }

        let memory_used_mb = self.get_memory_used_mb();
        let utilization = self.get_utilization();

        // Update metrics_history
        self.metrics_history.push_back(GPUMetricSample {
            timestamp: Instant::now(),
            memory_used_mb: memory_used_mb,
            utilization: utilization,
            batch_size: Some(batch_size),
            operation: Some("batch_processing".to_string()),
        });

        // Trim history if too large
        if self.metrics_history.len() > 100 {
            self.metrics_history.pop_front();
        }
    }

    // New getter methods
    pub fn get_peak_memory_mb(&mut self) -> f64 {
        self.peak_memory_used_mb
    }

    pub fn get_peak_utilization(&mut self) -> f64 {
        self.peak_utilization
    }

    pub fn get_allocation_count(&self) -> usize {
        self.memory_allocation_count
    }

    pub fn get_deallocation_count(&self) -> usize {
        self.memory_deallocation_count
    }

    pub fn get_throttling_events(&self) -> usize {
        self.memory_throttling_events
    }

    pub fn get_recent_metrics(&self, count: usize) -> Vec<GPUMetricSample> {
        self.metrics_history
            .iter()
            .rev()
            .take(count)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }
}