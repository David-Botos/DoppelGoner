// benches/batch_processing_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use inference_worker::inference::batch::BatchOptimizer;
use inference_worker::telemetry::gpu_metrics::GPUMetrics;
use inference_worker::types::types::{TokenizedDocument, WorkerCapabilities};
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;
use uuid::Uuid;

// Setup benchmark state
fn setup_benchmark_state() -> (Arc<Mutex<GPUMetrics>>, WorkerCapabilities) {
    let gpu_metrics = Arc::new(Mutex::new(GPUMetrics::new()));

    let capabilities = WorkerCapabilities {
        gpu_type: Some("Benchmark GPU".to_string()),
        gpu_memory_mb: Some(8192),
        supports_cuda: false,
        supports_metal: false,
        cpu_cores: 4,
        optimal_batch_size: 16,
        max_batch_size: 32,
        embedding_dimensions: Some(384),
    };

    (gpu_metrics, capabilities)
}

// Benchmark BatchOptimizer
fn benchmark_batch_optimizer(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (gpu_metrics, _) = setup_benchmark_state();

    let mut group = c.benchmark_group("BatchOptimizer");

    // Benchmark batch suggestion
    group.bench_function("suggest_batch_size", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut optimizer = BatchOptimizer::new(1, 64, 16, 500.0, gpu_metrics.clone());

                // Create test documents of varying sizes
                let documents: Vec<TokenizedDocument> = (0..100)
                    .map(|i| TokenizedDocument {
                        service_id: format!("service-{}", i),
                        tokenized_text: format!("document {}", i),
                        token_count: 10 + (i % 100),
                        job_id: Uuid::new_v4(),
                    })
                    .collect();

                black_box(
                    optimizer
                        .suggest_batch_size(documents.as_slice())
                        .await
                        .unwrap(),
                )
            })
        });
    });

    // Benchmark batch creation
    group.bench_function("create_optimal_batches", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut optimizer = BatchOptimizer::new(1, 64, 16, 500.0, gpu_metrics.clone());

                // Create test documents
                let documents: Vec<TokenizedDocument> = (0..100)
                    .map(|i| TokenizedDocument {
                        service_id: format!("service-{}", i),
                        tokenized_text: format!("document {}", i),
                        token_count: 10 + (i % 100),
                        job_id: Uuid::new_v4(),
                    })
                    .collect();

                black_box(optimizer.create_optimal_batches(documents).await.unwrap())
            })
        });
    });

    group.finish();
}

// Benchmark optimization strategies
fn benchmark_optimization_strategies(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (gpu_metrics, _) = setup_benchmark_state();

    let mut group = c.benchmark_group("OptimizationStrategies");

    // Benchmark with different batch sizes
    for batch_size in [8, 16, 32, 64].iter() {
        group.bench_function(format!("batch_size_{}", batch_size), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let mut optimizer =
                        BatchOptimizer::new(1, 128, *batch_size, 500.0, gpu_metrics.clone());

                    // Simulate processing batches
                    for _ in 0..10 {
                        optimizer.record_batch_result(true, 200.0);
                    }

                    black_box(optimizer.optimize_if_needed().await.unwrap())
                })
            });
        });
    }

    // Benchmark with different target latencies
    for latency in [100.0, 250.0, 500.0, 1000.0].iter() {
        group.bench_function(format!("target_latency_{}", latency), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let mut optimizer =
                        BatchOptimizer::new(1, 64, 16, *latency, gpu_metrics.clone());

                    // Simulate processing batches
                    for i in 0..10 {
                        let batch_size = 8 + (i % 8) * 4; // 8, 12, 16, 20, ...
                        optimizer.record_batch_performance(batch_size, latency * 0.8);
                    }

                    black_box(optimizer.optimize_if_needed().await.unwrap())
                })
            });
        });
    }

    group.finish();
}

// Benchmark GPU metrics collection
fn benchmark_gpu_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPUMetrics");

    group.bench_function("update_metrics", |b| {
        b.iter(|| {
            let mut gpu_metrics = GPUMetrics::new();
            black_box(gpu_metrics.update())
        });
    });

    #[cfg(feature = "metal")]
    group.bench_function("record_allocation", |b| {
        b.iter(|| {
            let mut gpu_metrics = GPUMetrics::new();
            black_box(gpu_metrics.record_allocation(1024 * 1024)) // 1MB allocation
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_batch_optimizer,
    benchmark_optimization_strategies,
    benchmark_gpu_metrics
);
criterion_main!(benches);
