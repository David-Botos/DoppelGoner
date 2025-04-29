// inference_worker/src/telemetry/gpu_metrics.rs
use anyhow::Result;
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};
use tracing::{debug, info, warn};

// For CUDA metrics
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::CudaDevice;

// For Metal metrics (simplified since Metal doesn't provide as direct access)
#[cfg(feature = "metal")]
use std::sync::atomic::{AtomicUsize, Ordering};

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

    // Backend-specific implementation
    #[cfg(feature = "cuda")]
    cuda_device: Option<std::sync::Arc<CudaDevice>>,

    #[cfg(feature = "metal")]
    metal_last_used_memory: AtomicUsize,
}

impl GPUMetrics {
    pub fn new() -> Self {
        let metrics = Self {
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

            #[cfg(feature = "cuda")]
            cuda_device: Self::initialize_cuda_device(),

            #[cfg(feature = "metal")]
            metal_last_used_memory: AtomicUsize::new(0),
        };

        // Initialize with system values
        metrics
    }

    #[cfg(feature = "cuda")]
    fn initialize_cuda_device() -> Option<std::sync::Arc<CudaDevice>> {
        match CudaDevice::new(0) {
            Ok(device) => {
                info!("CUDA device initialized for metrics collection");
                Some(device)
            }
            Err(e) => {
                warn!("Failed to initialize CUDA device for metrics: {}", e);
                None
            }
        }
    }

    pub fn update(&mut self) -> Result<()> {
        // If we've updated recently, don't do it again
        if self.last_updated.elapsed() < self.update_interval {
            return Ok(());
        }

        // Update metrics based on available backend
        #[cfg(feature = "cuda")]
        self.update_cuda_metrics()?;

        #[cfg(feature = "metal")]
        self.update_metal_metrics()?;

        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            // Fallback for CPU or when no GPU features are enabled
            // Use placeholder values
            self.total_memory_mb = 16384.0; // 16GB placeholder
            self.memory_used_mb = 4096.0; // 4GB placeholder
            self.memory_free_mb = self.total_memory_mb - self.memory_used_mb;
            self.utilization = 25.0; // 25% placeholder
            self.compute_utilization = 30.0; // 30% placeholder
            self.memory_utilization = 20.0; // 20% placeholder
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
        if let Some(_device) = &self.cuda_device {
            // Use nvidia-smi to get memory and utilization info
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
            
            // Fallback to just getting utilization if we couldn't get full memory info
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args([
                    "--query-gpu=utilization.gpu,utilization.memory",
                    "--format=csv,noheader,nounits",
                ])
                .output()
            {
                if output.status.success() {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        let parts: Vec<&str> = output_str.trim().split(',').collect();
                        if parts.len() >= 2 {
                            if let Ok(gpu_util) = parts[0].trim().parse::<f64>() {
                                if let Ok(mem_util) = parts[1].trim().parse::<f64>() {
                                    self.compute_utilization = gpu_util;
                                    self.memory_utilization = mem_util;
                                    self.utilization = (gpu_util + mem_util) / 2.0;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // If we couldn't get info from nvidia-smi, use fallback values
        // This happens if either nvidia-smi is not available or we couldn't parse its output
        if self.total_memory_mb == 0.0 {
            debug!("Using fallback values for CUDA metrics (nvidia-smi failed)");
            self.total_memory_mb = 16384.0;  // 16GB as default
            self.memory_used_mb = 4096.0;    // 4GB as default usage
            self.memory_free_mb = 12288.0;   // Remaining memory
            self.utilization = 25.0;         // 25% default utilization
            self.compute_utilization = 30.0; // 30% default compute util
            self.memory_utilization = 20.0;  // 20% default memory util
        }

        Ok(())
    }

    #[cfg(feature = "metal")]
    fn update_metal_metrics(&mut self) -> Result<()> {
        // For Metal, we don't have direct access to GPU metrics like in CUDA
        // We'll use a combination of estimates and system APIs

        // Try to get memory info from system APIs (on macOS)
        if let Ok(output) = std::process::Command::new("system_profiler")
            .args(["SPDisplaysDataType"])
            .output()
        {
            if output.status.success() {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    // Parse system_profiler output to extract GPU memory
                    // This is a simple heuristic that looks for VRAM in the output
                    for line in output_str.lines() {
                        if line.contains("VRAM") || line.contains("Metal") {
                            let parts: Vec<&str> = line.split(':').collect();
                            if parts.len() >= 2 {
                                let value_part = parts[1].trim();
                                if value_part.ends_with("MB") {
                                    if let Ok(mb) =
                                        value_part.trim_end_matches("MB").trim().parse::<f64>()
                                    {
                                        self.total_memory_mb = mb;
                                        // Estimate used based on atomic counter
                                        let used =
                                            self.metal_last_used_memory.load(Ordering::Relaxed)
                                                as f64
                                                / 1024.0
                                                / 1024.0;
                                        self.memory_used_mb = used;
                                        self.memory_free_mb =
                                            self.total_memory_mb - self.memory_used_mb;
                                        break;
                                    }
                                } else if value_part.ends_with("GB") {
                                    if let Ok(gb) =
                                        value_part.trim_end_matches("GB").trim().parse::<f64>()
                                    {
                                        self.total_memory_mb = gb * 1024.0;
                                        // Estimate used based on atomic counter
                                        let used =
                                            self.metal_last_used_memory.load(Ordering::Relaxed)
                                                as f64
                                                / 1024.0
                                                / 1024.0;
                                        self.memory_used_mb = used;
                                        self.memory_free_mb =
                                            self.total_memory_mb - self.memory_used_mb;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // If we couldn't get memory info from system APIs, use fallback values
        if self.total_memory_mb == 0.0 {
            // Default values for Apple Silicon
            self.total_memory_mb = 8192.0; // 8GB unified memory as default
            self.memory_used_mb = 2048.0; // Estimate
            self.memory_free_mb = 6144.0;
        }

        // For utilization on Metal, we don't have direct metrics
        // We'll estimate based on memory usage
        self.memory_utilization = 100.0 * self.memory_used_mb / self.total_memory_mb;
        self.compute_utilization = self.memory_utilization * 0.8; // Rough estimate
        self.utilization = (self.compute_utilization + self.memory_utilization) / 2.0;

        Ok(())
    }

    // Record memory allocation for Metal tracking
    #[cfg(feature = "metal")]
    pub fn record_allocation(&mut self, bytes: usize) {
        let current = self.metal_last_used_memory.load(Ordering::Relaxed);
        self.metal_last_used_memory
            .store(current + bytes, Ordering::Relaxed);

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

    // Record memory deallocation for Metal tracking
    #[cfg(feature = "metal")]
    pub fn record_deallocation(&mut self, bytes: usize) {
        let current = self.metal_last_used_memory.load(Ordering::Relaxed);
        let new_value = if bytes > current { 0 } else { current - bytes };
        self.metal_last_used_memory
            .store(new_value, Ordering::Relaxed);

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

// For tracking GPU memory across allocation and function calls
// Especially useful for Metal where we don't have direct metrics
#[cfg(feature = "metal")]
pub struct GPUMemoryTracker {
    metrics: std::sync::Arc<std::sync::Mutex<GPUMetrics>>,
}

#[cfg(feature = "metal")]
impl GPUMemoryTracker {
    pub fn new(metrics: std::sync::Arc<std::sync::Mutex<GPUMetrics>>) -> Self {
        Self { metrics }
    }

    pub fn track_allocation(&self, bytes: usize) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.record_allocation(bytes);
        }
    }

    pub fn track_deallocation(&self, bytes: usize) {
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.record_deallocation(bytes);
        }
    }
}