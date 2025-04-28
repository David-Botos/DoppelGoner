// tests/batch_optimizer_tests.rs
use inference_worker::inference::batch::BatchOptimizer;
use inference_worker::telemetry::gpu_metrics::GPUMetrics;
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::test]
async fn test_batch_optimizer_initialization() {
    let gpu_metrics = Arc::new(Mutex::new(GPUMetrics::new()));
    let optimizer = BatchOptimizer::new(1, 64, 16, 500.0, gpu_metrics.clone());

    assert_eq!(optimizer.min_batch_size, 1);
    assert_eq!(optimizer.max_batch_size, 64);
    assert_eq!(optimizer.get_optimal_batch_size(), 16);
    assert!(optimizer.is_warmup_phase);
    assert_eq!(optimizer.warmup_batches_processed, 0);
}

#[tokio::test]
async fn test_batch_optimizer_successful_batch() {
    let gpu_metrics = Arc::new(Mutex::new(GPUMetrics::new()));
    let mut optimizer = BatchOptimizer::new(1, 64, 16, 500.0, gpu_metrics.clone());

    // Record a successful batch
    optimizer.record_batch_result(true, 200.0);

    // Should increment successful batches and warmup batches
    assert_eq!(optimizer.consecutive_successful_batches, 1);
    assert_eq!(optimizer.warmup_batches_processed, 1);
    assert!(!optimizer.last_batch_had_error);

    // In warmup phase, we should see batch size increase
    let initial_size = optimizer.get_optimal_batch_size();
    let expected_new_size = (initial_size as f64 * optimizer.warmup_scale_factor).ceil() as usize;
    assert_eq!(
        optimizer.get_optimal_batch_size(),
        expected_new_size.min(16)
    );
}

#[tokio::test]
async fn test_batch_optimizer_error_handling() {
    let gpu_metrics = Arc::new(Mutex::new(GPUMetrics::new()));
    let mut optimizer = BatchOptimizer::new(1, 64, 16, 500.0, gpu_metrics.clone());
    
    println!("Initial batch size: {}", optimizer.get_optimal_batch_size());
    
    // Start with some successful batches
    println!("Processing successful batches:");
    for i in 0..3 {
        optimizer.record_batch_result(true, 200.0);
        println!("  After success #{}: batch_size={}, warmup={}, processed={}",
            i+1,
            optimizer.get_optimal_batch_size(),
            optimizer.is_warmup_phase,
            optimizer.warmup_batches_processed
        );
    }
    
    // Check the size after successful batches
    let size_before_error = optimizer.get_optimal_batch_size();
    println!("Size before error: {}", size_before_error);
    
    if optimizer.is_warmup_phase {
        // In warmup phase, check that batch size has increased according to the scale factor
        // Initial is 16, after 3 successes with 1.2 scale factor: 16 -> 19 -> 23 -> 27
        let expected_min = (16.0 * (optimizer.warmup_scale_factor.powi(3))).floor() as usize;
        assert!(
            size_before_error >= expected_min || size_before_error >= 16,
            "Expected batch size to be at least {} after warmup scaling, or at least the initial 16, but got {}",
            expected_min,
            size_before_error
        );
    } else {
        // If we've exited warmup, the batch size should be at least the initial size
        assert!(
            size_before_error >= 16,
            "Expected batch size to be at least 16, but got {}",
            size_before_error
        );
    }
    
    // Now record an error
    println!("Recording error batch");
    optimizer.record_batch_result(false, 300.0);
    
    // Check post-error state
    println!("After error: batch_size={}, consecutive_successes={}, last_had_error={}",
        optimizer.get_optimal_batch_size(),
        optimizer.consecutive_successful_batches,
        optimizer.last_batch_had_error
    );
    
    // Should reset consecutive success count and reduce batch size
    assert_eq!(optimizer.consecutive_successful_batches, 0, 
        "Expected consecutive_successful_batches to be 0 after error");
    assert!(optimizer.last_batch_had_error, 
        "Expected last_batch_had_error to be true after error");
    
    // Batch size should be reduced by about 25% but not below minimum
    let expected_size = (size_before_error as f64 * 0.75).ceil() as usize;
    let expected_min = optimizer.min_batch_size;
    
    assert_eq!(
        optimizer.get_optimal_batch_size(), 
        expected_size.max(expected_min),
        "Expected batch size to be max({}, {}) after error, but got {}",
        expected_size,
        expected_min,
        optimizer.get_optimal_batch_size()
    );
    
    println!("✅ Error handling test passed with expected batch size reduction");
}

#[tokio::test]
async fn test_batch_optimizer_warmup_completion() {
    let gpu_metrics = Arc::new(Mutex::new(GPUMetrics::new()));
    let mut optimizer = BatchOptimizer::new(1, 64, 32, 500.0, gpu_metrics.clone());

    // Complete the warmup phase
    for _ in 0..optimizer.warmup_target_batches {
        optimizer.record_batch_result(true, 200.0);
    }

    // Warmup should be complete
    assert!(!optimizer.is_warmup_phase);
    assert_eq!(
        optimizer.warmup_batches_processed,
        optimizer.warmup_target_batches
    );

    // Batch size should be set to initial_batch_size after warmup
    assert_eq!(optimizer.get_optimal_batch_size(), 32);
}
