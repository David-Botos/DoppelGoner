// src/services/data_fetcher.rs

use anyhow::{Context, Result};
use bb8::Pool;
use bb8_postgres::PostgresConnectionManager;
use futures::{StreamExt, stream};
use log::{debug, error, info, warn};
use std::time::Instant;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::Mutex;
use tokio_postgres::NoTls;

use crate::services::types::{ServiceData, TaxonomyInfo};

/// Component for fetching services and taxonomy data from PostgreSQL
///
/// Responsible for retrieving services that need embeddings and their
/// associated taxonomy information. Supports efficient batch processing
/// and parallel fetching for improved performance.
pub struct DataFetcher {
    pool: Pool<PostgresConnectionManager<NoTls>>,
    batch_size: usize,
    concurrent_fetches: usize,
}

impl DataFetcher {
    /// Create a new DataFetcher with the specified configuration
    ///
    /// # Arguments
    ///
    /// * `pool` - Database connection pool
    /// * `batch_size` - Number of services to fetch in a single batch
    /// * `concurrent_fetches` - Number of batches to fetch concurrently
    pub fn new(
        pool: Pool<PostgresConnectionManager<NoTls>>,
        batch_size: usize,
        concurrent_fetches: usize,
    ) -> Self {
        Self {
            pool,
            batch_size,
            concurrent_fetches,
        }
    }

    pub async fn fetch_taxonomy_info_in_bulk(
        &self,
        client: &tokio_postgres::Client,
        service_ids: &[String],
    ) -> Result<HashMap<String, Vec<TaxonomyInfo>>> {
        debug!(
            "Fetching taxonomy info for {} services in bulk",
            service_ids.len()
        );

        if service_ids.is_empty() {
            return Ok(HashMap::new());
        }

        // Generate placeholders for SQL query
        let placeholders: Vec<String> = service_ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("${}", i + 1))
            .collect();

        let query = format!(
            "SELECT st.service_id, tt.term, tt.description
             FROM service_taxonomy st
             JOIN taxonomy_term tt ON st.taxonomy_term_id = tt.id
             WHERE st.service_id IN ({})",
            placeholders.join(",")
        );

        let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = service_ids
            .iter()
            .map(|id| id as &(dyn tokio_postgres::types::ToSql + Sync))
            .collect();

        let rows = client
            .query(&query, &params[..])
            .await
            .context("Failed to fetch taxonomy info in bulk")?;

        // Group results by service_id
        let mut result: HashMap<String, Vec<TaxonomyInfo>> = HashMap::new();

        for row in rows {
            let service_id: String = row.get("service_id");
            let term: String = row.get("term");
            let description: Option<String> = row.get("description");

            let taxonomy_info = TaxonomyInfo { term, description };

            result
                .entry(service_id)
                .or_insert_with(Vec::new)
                .push(taxonomy_info);
        }

        debug!("Found taxonomy terms for {} services", result.len());
        Ok(result)
    }

    /// Count the total number of services that need embeddings
    ///
    /// Returns the number of services where embedding_v2 IS NULL
    pub async fn count_services_needing_embeddings(&self) -> Result<i64> {
        debug!("Counting services that need embeddings");
        let start = Instant::now();

        let client = self
            .pool
            .get()
            .await
            .context("Failed to get database connection for counting services")?;

        let row = client
            .query_one(
                "SELECT COUNT(*) FROM service WHERE embedding_v2 IS NULL",
                &[],
            )
            .await
            .context("Failed to count services needing embeddings")?;

        let count: i64 = row.get(0);

        debug!(
            "Found {} services needing embeddings in {:.2?}",
            count,
            start.elapsed()
        );

        Ok(count)
    }

    /// Fetch a batch of service IDs that need embeddings with locking to prevent duplicate processing
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of service IDs to fetch
    pub async fn fetch_services_needing_embeddings(&self, limit: usize) -> Result<Vec<String>> {
        debug!("Fetching up to {} service IDs that need embeddings", limit);
        let start = Instant::now();

        let mut client = self
            .pool
            .get()
            .await
            .context("Failed to get database connection for fetching service IDs")?;

        // Start a transaction to ensure atomicity
        let transaction = client
            .transaction()
            .await
            .context("Failed to start transaction")?;

        // Use FOR UPDATE SKIP LOCKED to prevent other processes from selecting the same rows
        // This is a PostgreSQL-specific feature for concurrent batch processing
        let rows = transaction
            .query(
                "SELECT id FROM service 
             WHERE embedding_v2 IS NULL 
             FOR UPDATE SKIP LOCKED
             LIMIT $1",
                &[&(limit as i64)],
            )
            .await
            .context("Failed to fetch service IDs")?;

        let ids: Vec<String> = rows.iter().map(|row| row.get::<_, String>("id")).collect();

        if !ids.is_empty() {
            debug!("Acquired lock on {} service IDs for embedding", ids.len());

            // Optionally, you could add a temporary marker to these services
            // This would require adding a column like 'embedding_in_progress' to your schema
            // If you have such a column, uncomment this code:
            /*
            let batch_id = uuid::Uuid::new_v4().to_string();
            transaction
                .execute(
                    "UPDATE service SET embedding_in_progress = $1, embedding_started_at = NOW()
                     WHERE id = ANY($2) AND embedding_v2 IS NULL",
                    &[&batch_id, &ids],
                )
                .await
                .context("Failed to mark services as in-progress")?;
            */
        }

        // Commit the transaction to release the locks on rows we're not using
        // but maintain locks on the ones we've selected
        transaction
            .commit()
            .await
            .context("Failed to commit transaction")?;

        debug!(
            "Fetched {} service IDs in {:.2?}",
            ids.len(),
            start.elapsed()
        );

        Ok(ids)
    }

    /// Fetch taxonomy information for a specific service
    ///
    /// # Arguments
    ///
    /// * `client` - Database client
    /// * `service_id` - ID of the service to fetch taxonomy information for
    pub async fn fetch_taxonomy_info(
        &self,
        client: &tokio_postgres::Client,
        service_id: &str,
    ) -> Result<Vec<TaxonomyInfo>> {
        debug!("Fetching taxonomy info for service: {}", service_id);

        let query = "
            SELECT tt.term, tt.description
            FROM service_taxonomy st
            JOIN taxonomy_term tt ON st.taxonomy_term_id = tt.id
            WHERE st.service_id = $1
        ";

        let rows = client.query(query, &[&service_id]).await.context(format!(
            "Failed to fetch taxonomy info for service: {}",
            service_id
        ))?;

        let mut taxonomy_info = Vec::with_capacity(rows.len());
        for row in rows {
            taxonomy_info.push(TaxonomyInfo {
                term: row.get("term"),
                description: row.get("description"),
            });
        }

        debug!(
            "Found {} taxonomy terms for service {}",
            taxonomy_info.len(),
            service_id
        );
        Ok(taxonomy_info)
    }

    /// Fetch complete service data for a batch of service IDs
    ///
    /// Retrieves name, description, and taxonomy information for each service
    ///
    /// # Arguments
    ///
    /// * `service_ids` - List of service IDs to fetch data for
    pub async fn fetch_service_data(&self, service_ids: &[String]) -> Result<Vec<ServiceData>> {
        if service_ids.is_empty() {
            debug!("No service IDs provided, returning empty result");
            return Ok(Vec::new());
        }

        debug!("Fetching data for {} services", service_ids.len());
        let start = Instant::now();

        let client = self
            .pool
            .get()
            .await
            .context("Failed to get database connection for fetching service data")?;

        // Generate placeholders for the SQL query
        let placeholders: Vec<String> = service_ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("${}", i + 1))
            .collect();

        let query = format!(
            "SELECT id, name, description FROM service WHERE id IN ({})",
            placeholders.join(",")
        );

        let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = service_ids
            .iter()
            .map(|id| id as &(dyn tokio_postgres::types::ToSql + Sync))
            .collect();

        let rows = client
            .query(&query, &params[..])
            .await
            .context("Failed to fetch service data")?;

        debug!(
            "Fetched basic data for {} services in {:.2?}",
            rows.len(),
            start.elapsed()
        );

        // Fetch service data and taxonomy info for each service
        let mut service_data = Vec::with_capacity(rows.len());
        let taxonomy_start = Instant::now();

        for row in rows {
            let id = row.get::<_, String>("id");
            let name = row.get::<_, Option<String>>("name").unwrap_or_default();
            let description = row
                .get::<_, Option<String>>("description")
                .unwrap_or_default();

            // Fetch taxonomy information for this service
            let taxonomy_info = match self.fetch_taxonomy_info(&client, &id).await {
                Ok(info) => info,
                Err(e) => {
                    warn!(
                        "Failed to fetch taxonomy info for service {}: {}. Continuing with empty taxonomy.",
                        id, e
                    );
                    Vec::new()
                }
            };

            service_data.push(ServiceData {
                id,
                name,
                description,
                taxonomy_info,
            });
        }

        debug!(
            "Fetched taxonomy data for all services in {:.2?}",
            taxonomy_start.elapsed()
        );

        debug!(
            "Completed fetching all service data in {:.2?}",
            start.elapsed()
        );

        Ok(service_data)
    }

    /// Fetch service data in parallel batches for improved performance
    ///
    /// Splits service IDs into batches and processes them concurrently
    ///
    /// # Arguments
    ///
    /// * `service_ids` - List of all service IDs to fetch data for
    // Modify the fetch_service_data_in_batches method to use bulk taxonomy fetching
    pub async fn fetch_service_data_in_batches(
        &self,
        service_ids: &[String],
    ) -> Result<Vec<Vec<ServiceData>>> {
        if service_ids.is_empty() {
            debug!("No service IDs provided, returning empty result");
            return Ok(Vec::new());
        }

        debug!(
            "Fetching data for {} services in batches of {} with {} concurrent fetchers",
            service_ids.len(),
            self.batch_size,
            self.concurrent_fetches
        );

        // Split IDs into batches
        let batches: Vec<Vec<String>> = service_ids
            .chunks(self.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        debug!("Split into {} batches", batches.len());

        // Get a client for fetching the taxonomy data
        let client = self
            .pool
            .get()
            .await
            .context("Failed to get database connection for fetching taxonomy data")?;

        // Fetch all taxonomy data in a single query for all service IDs
        let taxonomy_data = self
            .fetch_taxonomy_info_in_bulk(&client, service_ids)
            .await?;
        debug!("Fetched taxonomy data for {} services", taxonomy_data.len());

        // Drop the client as we don't need it anymore for global operations
        drop(client);

        let start = Instant::now();
        let processed_batches = Arc::new(Mutex::new(0));
        let total_batches = batches.len();

        // Process batches in parallel
        let results = stream::iter(batches)
            .map(|batch| {
                let fetcher = self.clone(); // Clone the DataFetcher which has pool access
                let processed_batches = processed_batches.clone();
                let batch_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
                let taxonomy_data = taxonomy_data.clone();
                let pool = self.pool.clone(); // Clone the pool, not the connection

                async move {
                    let batch_start = Instant::now();
                    debug!("Starting batch {} with {} services", batch_id, batch.len());

                    // Each task gets its own connection from the pool
                    let client = match pool.get().await {
                        Ok(client) => client,
                        Err(e) => {
                            return Err(anyhow::anyhow!(
                                "Failed to get database connection: {}",
                                e
                            ));
                        }
                    };

                    // Generate placeholders for the SQL query
                    let placeholders: Vec<String> = batch
                        .iter()
                        .enumerate()
                        .map(|(i, _)| format!("${}", i + 1))
                        .collect();

                    let query = format!(
                        "SELECT id, name, description FROM service WHERE id IN ({})",
                        placeholders.join(",")
                    );

                    let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = batch
                        .iter()
                        .map(|id| id as &(dyn tokio_postgres::types::ToSql + Sync))
                        .collect();

                    // Execute query to get basic service data
                    let rows = match client.query(&query, &params[..]).await {
                        Ok(rows) => rows,
                        Err(e) => {
                            return Err(anyhow::anyhow!("Failed to fetch service data: {}", e));
                        }
                    };

                    // Process service data and add taxonomy info from our already fetched data
                    let mut service_data = Vec::with_capacity(rows.len());

                    for row in rows {
                        let id = row.get::<_, String>("id");
                        let name = row.get::<_, Option<String>>("name").unwrap_or_default();
                        let description = row
                            .get::<_, Option<String>>("description")
                            .unwrap_or_default();

                        // Get taxonomy info from our pre-fetched data
                        let taxonomy_info = taxonomy_data.get(&id).cloned().unwrap_or_default();

                        service_data.push(ServiceData {
                            id,
                            name,
                            description,
                            taxonomy_info,
                        });
                    }

                    // Update progress
                    let mut processed = processed_batches.lock().await;
                    *processed += 1;

                    debug!(
                        "Completed batch {}/{} ({}) in {:.2?}",
                        *processed,
                        total_batches,
                        batch_id,
                        batch_start.elapsed()
                    );

                    Ok(service_data)
                }
            })
            .buffer_unordered(self.concurrent_fetches)
            .collect::<Vec<Result<Vec<ServiceData>>>>()
            .await;

        debug!("Completed fetching all batches in {:.2?}", start.elapsed());

        // Check for errors and collect successful results
        let mut service_data_batches = Vec::with_capacity(results.len());
        let mut error_count = 0;

        for result in results {
            match result {
                Ok(batch) => {
                    if !batch.is_empty() {
                        service_data_batches.push(batch);
                    }
                }
                Err(e) => {
                    error_count += 1;
                    error!("Error fetching batch: {}", e);
                }
            }
        }

        if error_count > 0 {
            warn!(
                "{} out of {} batches had errors",
                error_count, total_batches
            );
        }

        info!(
            "Successfully fetched {} batches with {} total services in {:.2?}",
            service_data_batches.len(),
            service_data_batches
                .iter()
                .map(|batch| batch.len())
                .sum::<usize>(),
            start.elapsed()
        );

        Ok(service_data_batches)
    }
}

impl Clone for DataFetcher {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            batch_size: self.batch_size,
            concurrent_fetches: self.concurrent_fetches,
        }
    }
}
