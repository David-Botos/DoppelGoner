// src/results.rs

use crate::models::{MatchGroup, MatchingMethod, LLMReviewResult};
use crate::db::PgPool;
use anyhow::{Result, Context};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Semaphore;
use uuid::Uuid;
use std::time::{Duration, Instant};
use futures::StreamExt;
// use tokio_postgres::Row;

/// Run LLM review on match groups to validate and provide reasoning
///
/// This function takes match groups that are candidates for merging and asks an LLM
/// to evaluate whether they represent the same real-world entity.
// pub async fn run_llm_review(pool: &PgPool, unified_groups: &[MatchGroup]) -> Result<Vec<LLMReviewResult>> {
//     // This is a placeholder implementation
//     // In a real system, you would:
//     // 1. Select groups that need review (e.g., below confidence threshold)
//     // 2. For each group, fetch the full details of involved records
//     // 3. Format a prompt for the LLM with this context
//     // 4. Call an LLM API (Mistral, OpenAI, etc.)
//     // 5. Parse the result and update the match groups

//     // For now, we'll just return the groups unmodified
//     let results = unified_groups
//         .iter()
//         .map(|group| LLMReviewResult {
//             group: group.clone(),
//             reasoning: Some("Automated placeholder review".to_string()),
//             is_valid: true, // All are valid by default
//         })
//         .collect();

//     Ok(results)
// }

/// Save the cluster results to database for human review using concurrent processing
///
/// This persists the match groups to a database table where human reviewers
/// can access them through a review interface. Uses controlled concurrency and Arc
/// to efficiently share data across tasks.
pub async fn save_clusters(pool: &PgPool, reviewed_results: &[LLMReviewResult]) -> Result<()> {
    println!("   Saving {} match groups to database...", reviewed_results.len());
    
    // Filter out only the indices of valid clusters to avoid excessive cloning
    let valid_indices: Vec<usize> = reviewed_results.iter()
        .enumerate()
        .filter(|(_, result)| result.is_valid && result.group.record_ids.len() > 1)
        .map(|(idx, _)| idx)
        .collect();
    
    println!("   Found {} valid match groups to process", valid_indices.len());
    
    if valid_indices.is_empty() {
        println!("   No valid match groups to save, skipping database operations");
        return Ok(());
    }
    
    // Configuration for concurrent processing
    const BATCH_SIZE: usize = 100;
    const MAX_CONCURRENT_BATCHES: usize = 16; // Control concurrent database connections
    
    let total_batches = (valid_indices.len() + BATCH_SIZE - 1) / BATCH_SIZE;
    println!("   Processing in {} batches with maximum {} concurrent workers", 
             total_batches, MAX_CONCURRENT_BATCHES);
    
    // Share data across tasks using Arc
    let pool = Arc::new(pool.clone());
    let shared_results = Arc::new(reviewed_results.to_vec());
    
    // Use semaphore to limit concurrent operations
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_BATCHES));
    
    // Statistics tracking
    let start_time = Instant::now();
    let mut success_count = 0;
    let mut error_count = 0;
    
    // Process batches with controlled concurrency
    let mut tasks = Vec::new();
    
    for batch_idx in 0..total_batches {
        let start_idx = batch_idx * BATCH_SIZE;
        let end_idx = std::cmp::min(start_idx + BATCH_SIZE, valid_indices.len());
        let batch_indices = valid_indices[start_idx..end_idx].to_vec();
        
        // Clone Arcs for the task (only clones the pointer, not the data)
        let semaphore_clone = Arc::clone(&semaphore);
        let pool_clone = Arc::clone(&pool);
        let results_clone = Arc::clone(&shared_results);
        
        // Create a batch processing task
        let task = tokio::spawn(async move {
            // Acquire permit from semaphore (will wait if max concurrency reached)
            let _permit = semaphore_clone.acquire().await.expect("Semaphore was closed");
            
            let batch_start = Instant::now();
            println!("   Starting batch {}/{} ({} groups)...", 
                    batch_idx + 1, total_batches, batch_indices.len());
            
            let result = process_batch_by_indices(&pool_clone, &results_clone, 
                                               &batch_indices, batch_idx + 1, total_batches).await;
            
            let elapsed = batch_start.elapsed();
            match result {
                Ok((success, errors)) => {
                    println!("   Completed batch {}/{} in {:.2}s: {} succeeded, {} failed", 
                            batch_idx + 1, total_batches, elapsed.as_secs_f32(), 
                            success, errors);
                    (success, errors)
                },
                Err(e) => {
                    eprintln!("   Failed batch {}/{}: {}", batch_idx + 1, total_batches, e);
                    (0, batch_indices.len())
                }
            }
        });
        
        tasks.push(task);
        
        // Optional: small delay between spawning tasks to avoid connection spike
        if batch_idx % 4 == 3 {  // Every 4 batches
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    
    // Collect results from all tasks
    for task in tasks {
        match task.await {
            Ok((s, e)) => {
                success_count += s;
                error_count += e;
            },
            Err(e) => {
                eprintln!("Task panicked: {}", e);
                error_count += BATCH_SIZE; // Assume worst case
            }
        }
    }
    
    let total_time = start_time.elapsed();
    println!("   Database update complete in {:.2}s", total_time.as_secs_f32());
    println!("   Stats: {} match groups saved successfully, {} failed", 
             success_count, error_count);
    
    if error_count > 0 {
        println!("   Warning: Some errors occurred during saving. Check logs for details.");
    }
    
    Ok(())
}

/// Process a batch of match groups identified by their indices
/// Process a batch of match groups identified by their indices
async fn process_batch_by_indices(
    pool: &PgPool, 
    all_results: &[LLMReviewResult],
    indices: &[usize], 
    batch_number: usize,
    total_batches: usize
) -> Result<(usize, usize)> {
    let mut client = pool.get().await.context("Failed to get DB connection for batch")?;
    let mut success_count = 0;
    let mut error_count = 0;
    
    // Begin transaction for this batch
    let transaction = client.transaction().await.context("Failed to start transaction")?;
    
    // Create prepared statements for better performance with bulk operations
    let cluster_stmt = transaction
        .prepare(
            "INSERT INTO match_clusters 
             (confidence, notes, reasoning, is_reviewed) 
             VALUES ($1, $2, $3, $4)
             RETURNING id"
        )
        .await
        .context("Failed to prepare cluster statement")?;
        
    let entity_stmt = transaction
        .prepare(
            "INSERT INTO cluster_entities 
             (cluster_id, entity_type, entity_id) 
             VALUES ($1, $2, $3)"
        )
        .await
        .context("Failed to prepare entity statement")?;
        
    let method_stmt = transaction
        .prepare(
            "INSERT INTO matching_methods 
             (cluster_id, method_name, confidence) 
             VALUES ($1, $2, $3)"
        )
        .await
        .context("Failed to prepare method statement")?;
    
    for (idx, &result_idx) in indices.iter().enumerate() {
        // Get the actual result from the shared data using its index
        let result = &all_results[result_idx];
        
        // Progress indicator for large batches
        if (idx + 1) % 25 == 0 || idx + 1 == indices.len() {
            println!("   Batch {}/{}: Progress {}/{}", 
                    batch_number, total_batches, idx + 1, indices.len());
        }
        
        // 1. Insert the match cluster using prepared statement
        let cluster_row = transaction
            .query_one(
                &cluster_stmt,
                &[
                    &result.group.confidence,
                    &result.group.notes,
                    &result.reasoning,
                    &result.group.is_reviewed,
                ],
            )
            .await;
            
        match cluster_row {
            Ok(row) => {
                let cluster_id = row.get::<_, Uuid>(0);
                
                // 2. Insert each entity in the cluster
                let mut entity_success = true;
                
                for record_id in &result.group.record_ids {
                    // Parse entity type and ID from the prefixed record_id
                    let parts: Vec<&str> = record_id.split(':').collect();
                    if parts.len() != 2 {
                        eprintln!("Invalid record ID format: {}", record_id);
                        entity_success = false;
                        continue;
                    }

                    let entity_type = parts[0];
                    let entity_id = parts[1];
                    
                    // Use prepared statement for entity insertion
                    let entity_result = transaction
                        .execute(
                            &entity_stmt,
                            &[&cluster_id, &entity_type, &entity_id],
                        )
                        .await;
                    
                    if let Err(e) = entity_result {
                        eprintln!("Error saving entity {}: {}", record_id, e);
                        entity_success = false;
                    }
                }
                
                // 3. Insert the matching method(s)
                match &result.group.method {
                    MatchingMethod::Merged(methods) => {
                        // For merged methods, insert each contributing method
                        for method in methods {
                            let method_name = method_to_string(method);
                            
                            // Use prepared statement for method insertion
                            let method_result = transaction
                                .execute(
                                    &method_stmt,
                                    &[&cluster_id, &method_name, &result.group.confidence],
                                )
                                .await;
                                
                            if let Err(e) = method_result {
                                eprintln!("Error saving method {}: {}", method_name, e);
                                entity_success = false;
                            }
                        }
                    },
                    _ => {
                        let method_name = method_to_string(&result.group.method);
                        
                        // Use prepared statement for method insertion
                        let method_result = transaction
                            .execute(
                                &method_stmt,
                                &[&cluster_id, &method_name, &result.group.confidence],
                            )
                            .await;
                            
                        if let Err(e) = method_result {
                            eprintln!("Error saving method {}: {}", method_name, e);
                            entity_success = false;
                        }
                    }
                }
                
                if entity_success {
                    success_count += 1;
                } else {
                    error_count += 1;
                }
            },
            Err(e) => {
                error_count += 1;
                eprintln!("Error saving match cluster: {}", e);
            }
        }
    }
    
    // Commit the batch transaction with timeout
    match tokio::time::timeout(
        Duration::from_secs(30),
        transaction.commit()
    ).await {
        Ok(commit_result) => match commit_result {
            Ok(_) => Ok((success_count, error_count)),
            Err(e) => {
                eprintln!("Error committing batch {}: {}", batch_number, e);
                Err(anyhow::anyhow!("Failed to commit transaction: {}", e))
            }
        },
        Err(_) => {
            eprintln!("Timeout committing batch {}", batch_number);
            Err(anyhow::anyhow!("Transaction commit timeout after 30 seconds"))
        }
    }
}

// Helper function to convert MatchingMethod to string
fn method_to_string(method: &MatchingMethod) -> String {
    match method {
        MatchingMethod::Email => "Email".to_string(),
        MatchingMethod::Phone => "Phone".to_string(),
        MatchingMethod::Url => "URL".to_string(),
        MatchingMethod::Address => "Address".to_string(),
        MatchingMethod::Semantic => "Semantic".to_string(),
        MatchingMethod::TaxonomyRegion => "TaxonomyRegion".to_string(),
        MatchingMethod::LLMReview => "LLMReview".to_string(),
        MatchingMethod::HumanVerified => "HumanVerified".to_string(),
        MatchingMethod::Merged(_) => "Merged".to_string(),
    }
}

/// Merge overlapping match groups into unified groups
/// 
/// When multiple matching methods identify the same records as duplicates,
/// this function combines them into a single unified match group.
pub fn merge_match_groups(groups: Vec<MatchGroup>) -> Vec<MatchGroup> {
    // If there are no groups to merge, return empty vec
    if groups.is_empty() {
        return Vec::new();
    }
    
    // Track which records are in which groups using a union-find data structure
    let mut record_to_group_map: HashMap<String, usize> = HashMap::new();
    let mut unified_groups: Vec<HashSet<String>> = Vec::new();
    let mut group_methods: Vec<HashSet<MatchingMethod>> = Vec::new();
    let mut group_notes: Vec<Vec<String>> = Vec::new();
    let mut group_confidences: Vec<Vec<f32>> = Vec::new();
    
    // Process each match group
    for group in groups {
        let record_ids = group.record_ids;
        let method = group.method;
        let confidence = group.confidence;
        let notes = group.notes.unwrap_or_default();
        
        // Find existing groups that contain any of the current records
        let mut related_group_indices: HashSet<usize> = HashSet::new();
        
        for record_id in &record_ids {
            if let Some(&group_idx) = record_to_group_map.get(record_id) {
                related_group_indices.insert(group_idx);
            }
        }
        
        match related_group_indices.len() {
            0 => {
                // No existing group contains these records, create a new one
                let new_group_idx = unified_groups.len();
                
                // Initialize new collections
                let mut new_record_set: HashSet<String> = HashSet::new();
                let mut new_method_set: HashSet<MatchingMethod> = HashSet::new();
                let mut new_notes_vec: Vec<String> = Vec::new();
                let mut new_confidence_vec: Vec<f32> = Vec::new();
                
                // Add data from this group
                new_record_set.extend(record_ids.clone());
                new_method_set.insert(method);
                new_notes_vec.push(notes);
                new_confidence_vec.push(confidence);
                
                // Update maps for each record
                for record_id in &record_ids {
                    record_to_group_map.insert(record_id.clone(), new_group_idx);
                }
                
                // Add new collections to their respective vecs
                unified_groups.push(new_record_set);
                group_methods.push(new_method_set);
                group_notes.push(new_notes_vec);
                group_confidences.push(new_confidence_vec);
            }
            1 => {
                // Records belong to a single existing group
                let group_idx = *related_group_indices.iter().next().unwrap();
                
                // Update the existing group with the new records
                for record_id in &record_ids {
                    unified_groups[group_idx].insert(record_id.clone());
                    record_to_group_map.insert(record_id.clone(), group_idx);
                }
                
                // Update the method set
                group_methods[group_idx].insert(method);
                
                // Add the notes
                group_notes[group_idx].push(notes);
                
                // Add the confidence
                group_confidences[group_idx].push(confidence);
            }
            _ => {
                // Records span multiple existing groups, we need to merge them
                // Take the first group as the base
                let base_group_idx = *related_group_indices.iter().next().unwrap();
                
                // Collect all records, methods, notes, and confidences from related groups
                let mut all_records = HashSet::new();
                let mut all_methods = HashSet::new();
                let mut all_notes = Vec::new();
                let mut all_confidences = Vec::new();
                
                // Add current group's data
                for record_id in &record_ids {
                    all_records.insert(record_id.clone());
                }
                all_methods.insert(method);
                all_notes.push(notes);
                all_confidences.push(confidence);
                
                // Track indices to mark for removal (except base_group_idx)
                let mut indices_to_clear = Vec::new();
                
                // Process all related groups
                for &group_idx in &related_group_indices {
                    // Add records from this group
                    for record_id in &unified_groups[group_idx] {
                        all_records.insert(record_id.clone());
                    }
                    
                    // Add methods from this group
                    for method in &group_methods[group_idx] {
                        all_methods.insert(method.clone());
                    }
                    
                    // Add notes and confidences
                    all_notes.extend(group_notes[group_idx].clone());
                    all_confidences.extend(group_confidences[group_idx].clone());
                    
                    // Mark non-base groups for removal
                    if group_idx != base_group_idx {
                        indices_to_clear.push(group_idx);
                    }
                }
                
                // Update record_to_group_map for all records
                for record_id in &all_records {
                    record_to_group_map.insert(record_id.clone(), base_group_idx);
                }
                
                // Update the base group with merged data
                unified_groups[base_group_idx] = all_records;
                group_methods[base_group_idx] = all_methods;
                group_notes[base_group_idx] = all_notes;
                group_confidences[base_group_idx] = all_confidences;
                
                // Clear the groups that were merged (except the base)
                for idx in indices_to_clear {
                    unified_groups[idx].clear();
                    group_methods[idx].clear();
                    group_notes[idx].clear();
                    group_confidences[idx].clear();
                }
            }
        }
    }
    
    // Convert back to MatchGroup structs, skipping empty groups
    let mut result = Vec::new();
    
    for i in 0..unified_groups.len() {
        if !unified_groups[i].is_empty() {
            // Convert HashSet to Vec
            let record_ids: Vec<String> = unified_groups[i].iter().cloned().collect();
            
            // Skip single-record groups
            if record_ids.len() <= 1 {
                continue;
            }
            
            // Determine the combined method
            let combined_method = if group_methods[i].len() > 1 {
                // If multiple methods, use Merged variant
                MatchingMethod::Merged(group_methods[i].iter().cloned().collect())
            } else {
                // If just one method, use it directly
                group_methods[i].iter().next().unwrap().clone()
            };
            
            // Combine the notes
            let combined_notes = if !group_notes[i].is_empty() {
                Some(group_notes[i].join("; "))
            } else {
                None
            };
            
            // Calculate the average confidence
            let avg_confidence = if !group_confidences[i].is_empty() {
                group_confidences[i].iter().sum::<f32>() / group_confidences[i].len() as f32
            } else {
                0.0
            };
            
            // Create the merged MatchGroup
            result.push(MatchGroup {
                record_ids,
                method: combined_method,
                confidence: avg_confidence,
                notes: combined_notes,
                is_reviewed: false, // Reset review status for merged groups
            });
        }
    }
    
    result
}