// tests/cuda_acceleration_test.rs
//
// This test specifically tests CUDA acceleration for embedding inference
// focusing particularly on whether layer normalization works correctly on CUDA.

use anyhow::Result;
use candle_core::{Device, Tensor};
use inference_worker::inference::model::{BgeEmbeddingModel, BgeTokenizer, get_best_device};
use inference_worker::telemetry::gpu_metrics::GPUMetrics;
use inference_worker::types::types::{TokenizedDocument, WorkerCapabilities};
use inference_worker::inference::engine::InferenceEngine;
use std::sync::Arc;
use std::time::{Instant};
use tokio::sync::Mutex;
use uuid::Uuid;

#[cfg(feature = "cuda")]
#[tokio::test]
async fn test_cuda_layer_normalization() -> Result<()> {
    // Skip this test if CUDA is not actually available at runtime
    let cuda_device = match Device::cuda_if_available(0) {
        Ok(device) if device.is_cuda() => {
            println!("✅ CUDA device found and available for testing");
            device
        },
        _ => {
            println!("⚠️ No CUDA device available, skipping test");
            return Ok(());
        }
    };

    // Create CPU device for comparison
    let cpu_device = Device::Cpu;

    // Set up GPU metrics
    let gpu_metrics = Arc::new(Mutex::new(GPUMetrics::new()));

    // Test data that will exercise layer normalization - intentionally varied inputs
    let test_texts = vec![
        // Regular text
        "This is a standard input for embedding generation.",
        // Text with extreme values (long repetitive content to create large activations)
        "This text repeats words many many many many many many many many many many many many many times to create extreme activation values that will need significant normalization.",
        // Very short text
        "Short.",
        // Mixed case and special characters
        "UPPERCASE text with $pecial CH@R@CTERS and numbers 12345!",
    ];

    println!("Testing layer normalization in GPU vs CPU inference with {} test samples", test_texts.len());

    // Model path - adjust based on your setup
    let model_path = "./models/bge-small-en-v1.5";
    let model_id = "bge-small-en-v1.5";

    // PART 1: Load models on both devices
    println!("\n=== Loading models on CPU and CUDA ===");
    
    // Load model on CUDA
    println!("Loading model on CUDA device...");
    let cuda_start = Instant::now();
    let cuda_model = BgeEmbeddingModel::load(model_path, &cuda_device, model_id)?;
    let cuda_load_time = cuda_start.elapsed();
    println!("✅ Model loaded on CUDA in {:?}", cuda_load_time);
    
    // Load model on CPU
    println!("Loading model on CPU device...");
    let cpu_start = Instant::now();
    let cpu_model = BgeEmbeddingModel::load(model_path, &cpu_device, model_id)?;
    let cpu_load_time = cpu_start.elapsed();
    println!("✅ Model loaded on CPU in {:?}", cpu_load_time);
    
    // Load tokenizer (shared)
    let tokenizer = BgeTokenizer::load(model_path)?;
    
    // PART 2: Generate embeddings on both devices
    println!("\n=== Generating embeddings ===");
    
    // Tokenize test data
    let (input_ids, attention_masks) = tokenizer.tokenize_batch(&test_texts, 512)?;
    
    // Generate on CUDA
    println!("Generating embeddings on CUDA...");
    let cuda_inference_start = Instant::now();
    let cuda_embeddings = cuda_model.generate_embeddings(&input_ids, &attention_masks)?;
    let cuda_inference_time = cuda_inference_start.elapsed();
    println!("✅ CUDA inference completed in {:?}", cuda_inference_time);
    
    // Generate on CPU
    println!("Generating embeddings on CPU...");
    let cpu_inference_start = Instant::now();
    let cpu_embeddings = cpu_model.generate_embeddings(&input_ids, &attention_masks)?;
    let cpu_inference_time = cpu_inference_start.elapsed();
    println!("✅ CPU inference completed in {:?}", cpu_inference_time);
    
    // PART 3: Verify embeddings are normalized correctly
    println!("\n=== Verifying layer normalization ===");
    
    // Check embedding dimensions
    let embedding_dim = cuda_embeddings[0].len();
    println!("Embedding dimension: {}", embedding_dim);
    
    // Verify each embedding has unit norm (L2 normalized)
    for (i, embedding) in cuda_embeddings.iter().enumerate() {
        // Calculate L2 norm
        let squared_sum: f32 = embedding.iter().map(|x| x * x).sum();
        let norm = squared_sum.sqrt();
        
        // Check if norm is close to 1.0 (normalized)
        assert!((norm - 1.0).abs() < 1e-5, 
                "CUDA embedding {} not properly normalized, L2 norm = {}", i, norm);
                
        println!("✅ CUDA embedding {} has correct L2 norm: {:.6}", i, norm);
    }
    
    // Verify CPU embeddings have unit norm as well (for comparison)
    for (i, embedding) in cpu_embeddings.iter().enumerate() {
        let squared_sum: f32 = embedding.iter().map(|x| x * x).sum();
        let norm = squared_sum.sqrt();
        
        assert!((norm - 1.0).abs() < 1e-5, 
                "CPU embedding {} not properly normalized, L2 norm = {}", i, norm);
                
        println!("✅ CPU embedding {} has correct L2 norm: {:.6}", i, norm);
    }
    
    // PART 4: Compare CUDA vs CPU results
    println!("\n=== Comparing CUDA vs CPU results ===");
    
    let mut max_diff = 0.0;
    let mut avg_diff = 0.0;
    let mut total_values = 0;
    
    for (i, (cuda_emb, cpu_emb)) in cuda_embeddings.iter().zip(cpu_embeddings.iter()).enumerate() {
        assert_eq!(cuda_emb.len(), cpu_emb.len(), 
                  "Embedding {} dimensions don't match between GPU and CPU", i);
        
        for (j, (cuda_val, cpu_val)) in cuda_emb.iter().zip(cpu_emb.iter()).enumerate() {
            let diff = (cuda_val - cpu_val).abs();
            max_diff = max_diff.max(diff);
            avg_diff += diff;
            total_values += 1;
            
            // Values should be close but not identical due to floating point differences
            // We use a relatively loose tolerance for this test
            assert!(diff < 0.01, 
                   "Too much difference at position [{}, {}]: CUDA={}, CPU={}, diff={}",
                   i, j, cuda_val, cpu_val, diff);
        }
    }
    
    avg_diff /= total_values as f32;
    println!("Maximum difference between CUDA and CPU: {:.6}", max_diff);
    println!("Average difference between CUDA and CPU: {:.6}", avg_diff);
    println!("✅ CUDA and CPU results are sufficiently similar");
    
    // PART 5: Test end-to-end with InferenceEngine
    println!("\n=== Testing InferenceEngine with CUDA ===");
    
    // Create capabilities
    let capabilities = WorkerCapabilities {
        gpu_type: Some("NVIDIA GPU".to_string()),
        gpu_memory_mb: Some(8192),
        supports_cuda: true,
        supports_metal: false,
        cpu_cores: 4,
        optimal_batch_size: 16,
        max_batch_size: 32,
        embedding_dimensions: Some(embedding_dim as i32),
    };
    
    // Create and initialize engine
    let mut engine = InferenceEngine::new(
        model_id.to_string(),
        model_path.to_string(),
        capabilities,
        gpu_metrics.clone(),
    );
    
    println!("Initializing inference engine...");
    engine.initialize().await?;
    println!("✅ Inference engine initialized");
    
    // Create test documents
    let documents: Vec<TokenizedDocument> = test_texts.iter().enumerate().map(|(i, text)| {
        TokenizedDocument {
            service_id: format!("test-{}", i),
            tokenized_text: text.to_string(),
            token_count: text.split_whitespace().count(),
            job_id: Uuid::new_v4(),
        }
    }).collect();
    
    // Process batch
    println!("Processing batch through engine...");
    let engine_start = Instant::now();
    let results = engine.process_batch(&documents, model_id).await?;
    let engine_time = engine_start.elapsed();
    println!("✅ Engine processed batch in {:?}", engine_time);
    
    // Verify engine results
    println!("Verifying engine results...");
    assert_eq!(results.len(), test_texts.len(), "Should have one result per test text");
    
    for result in &results {
        let squared_sum: f32 = result.embedding.iter().map(|x| x * x).sum();
        let norm = squared_sum.sqrt();
        
        assert!((norm - 1.0).abs() < 1e-5, 
                "Engine embedding not properly normalized, L2 norm = {}", norm);
    }
    println!("✅ All engine embeddings have correct L2 normalization");
    
    // PART 6: Performance comparison
    let speedup = cpu_inference_time.as_secs_f64() / cuda_inference_time.as_secs_f64();
    println!("\n=== Performance Summary ===");
    println!("CUDA inference time: {:?}", cuda_inference_time);
    println!("CPU inference time: {:?}", cpu_inference_time);
    println!("CUDA speedup factor: {:.2}x", speedup);
    
    if speedup > 1.0 {
        println!("✅ CUDA acceleration confirmed: {:.2}x faster than CPU", speedup);
    } else {
        println!("⚠️ No speedup detected. This may be due to:");
        println!("   - Small batch size (too little work to offset CUDA overhead)");
        println!("   - Small model size");
        println!("   - Test environment constraints");
    }
    
    // Final summary
    println!("\n=== Test Summary ===");
    println!("✅ Successfully loaded model on CUDA");
    println!("✅ Layer normalization working correctly on CUDA");
    println!("✅ Embeddings properly L2 normalized (unit norm verified)");
    println!("✅ CUDA and CPU results are consistent");
    println!("✅ End-to-end engine test successful with CUDA");
    println!("✅ Performance comparison completed");
    
    Ok(())
}

// Skip test when CUDA is not available
#[cfg(not(feature = "cuda"))]
#[tokio::test]
async fn test_cuda_layer_normalization() -> Result<()> {
    println!("CUDA feature not enabled - test skipped");
    Ok(())
}