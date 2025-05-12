// // tests/entity_organizations_integration_tests.rs

// // Make sure your crate is accessible. If your library is named `dedupe_lib`:
// // extern crate dedupe_lib;
// // use dedupe_lib::{db, entity_organizations, models};

// // If your main crate is `dedupe` and `lib.rs` is the entry point:
// use dedupe_lib::{db::{self, PgPool}, entity_organizations, models::{self, Entity, EntityFeature, OrganizationId, EntityId}};

// use anyhow::Result;
// use chrono::Utc;
// use deadpool_postgres::Pool; // Or bb8_postgres::Pool if you are using that
// use std::collections::HashSet;
// use uuid::Uuid;

// // --- Test Configuration ---
// // TODO: Potentially move these to a shared test_utils module or config file
// const TEST_DB_URL_ENV_VAR: &str = "TEST_DATABASE_URL"; // e.g., postgres://user:pass@host:port/test_db?schema=mock_hsds
//                                                       // Ensure this schema is 'mock_hsds'

// async fn setup_db_pool() -> Result<PgPool> {
//     // Load .env file for test database URL (if you use one for tests)
//     // dotenv::dotenv().ok(); // Or specific test .env file

//     let db_url = std::env::var(TEST_DB_URL_ENV_VAR)
//         .expect(&format!("{} must be set for integration tests", TEST_DB_URL_ENV_VAR));

//     // Assuming your db::connect() function or a similar one can be used/adapted for tests
//     // Or, create a new pool specifically for tests:
//     let config = db_url.parse::<tokio_postgres::Config>()?;
//     let mgr_config = deadpool_postgres::ManagerConfig {
//         recycling_method: deadpool_postgres::RecyclingMethod::Fast,
//     };
//     let mgr = deadpool_postgres::Manager::from_config(config, tokio_postgres::NoTls, mgr_config);
//     let pool = deadpool_postgres::Pool::builder(mgr)
//         .max_size(10) // Adjust as needed for tests
//         .build()
//         .expect("Failed to create test database pool");

//     // Important: Ensure the search_path is set to mock_hsds for all connections from this pool
//     // This can be done via connection options in the URL `?options=-csearch_path%3Dmock_hsds,public`
//     // or by executing `SET search_path TO mock_hsds, public;` on each new connection.
//     // Deadpool-postgres has `post_create` hook for this.
//     // For simplicity here, we assume the URL handles it or it's the default for the test DB user.

//     Ok(pool)
// }

// async fn clear_tables(pool: &PgPool) -> Result<()> {
//     let client = pool.get().await?;
//     // Clear tables in reverse order of dependencies or use TRUNCATE ... CASCADE
//     client.batch_execute("
//         TRUNCATE TABLE mock_hsds.entity_feature RESTART IDENTITY CASCADE;
//         TRUNCATE TABLE mock_hsds.entity RESTART IDENTITY CASCADE;
//         TRUNCATE TABLE mock_hsds.organization RESTART IDENTITY CASCADE;
//         TRUNCATE TABLE mock_hsds.service RESTART IDENTITY CASCADE;
//         TRUNCATE TABLE mock_hsds.phone RESTART IDENTITY CASCADE;
//         TRUNCATE TABLE mock_hsds.location RESTART IDENTITY CASCADE;
//         TRUNCATE TABLE mock_hsds.contact RESTART IDENTITY CASCADE;
//         -- Add other relevant mock_hsds tables if they are involved
//     ").await?;
//     Ok(())
// }

// // Helper to insert a mock organization
// async fn insert_mock_organization(pool: &PgPool, id: &str, name: &str) -> Result<String> {
//     let client = pool.get().await?;
//     let org_id = id.to_string();
//     client.execute(
//         "INSERT INTO mock_hsds.organization (id, name, description, email, url, x_network_id, x_api_id, created_at, updated_at)
//          VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
//         &[&org_id, &name, &Some("Test Desc"), &Some("test@example.com"), &Some("http://example.com"),
//           &Some("network1"), &Some("api1"), &Utc::now().naive_utc(), &Utc::now().naive_utc()]
//     ).await?;
//     Ok(org_id)
// }

// // --- Tests for extract_entities ---

// #[tokio::test]
// async fn test_extract_entities_empty_db() -> Result<()> {
//     let pool = setup_db_pool().await?;
//     clear_tables(&pool).await?;

//     let extracted = entity_organizations::extract_entities(&pool).await?;
//     assert!(extracted.is_empty(), "Should extract no entities from an empty organization table");

//     let client = pool.get().await?;
//     let count: i64 = client.query_one("SELECT COUNT(*) FROM mock_hsds.entity", &[]).await?.get(0);
//     assert_eq!(count, 0, "Entity table should be empty");

//     Ok(())
// }

// #[tokio::test]
// async fn test_extract_entities_new_organizations() -> Result<()> {
//     let pool = setup_db_pool().await?;
//     clear_tables(&pool).await?;

//     // Insert some mock organizations
//     let org1_id = insert_mock_organization(&pool, "org_1", "Organization One").await?;
//     let org2_id = insert_mock_organization(&pool, "org_2", "Organization Two").await?;

//     let extracted_entities = entity_organizations::extract_entities(&pool).await?;

//     assert_eq!(extracted_entities.len(), 2, "Should extract two new entities");

//     let client = pool.get().await?;
//     let rows = client.query("SELECT id, organization_id, name FROM mock_hsds.entity ORDER BY name", &[]).await?;
//     assert_eq!(rows.len(), 2, "Two entities should be in the database");

//     let entity1_db_id: String = rows[0].get("id");
//     let entity1_org_id: String = rows[0].get("organization_id");
//     let entity1_name: Option<String> = rows[0].get("name");

//     let entity2_db_id: String = rows[1].get("id");
//     let entity2_org_id: String = rows[1].get("organization_id");
//     let entity2_name: Option<String> = rows[1].get("name");

//     // Check if extracted entities match DB (existence and basic fields)
//     // This assumes a certain order or you might need to find them by org_id in `extracted_entities`
//     let extracted_org1 = extracted_entities.iter().find(|e| e.organization_id.0 == org1_id).expect("Org1 entity not found in extracted list");
//     let extracted_org2 = extracted_entities.iter().find(|e| e.organization_id.0 == org2_id).expect("Org2 entity not found in extracted list");

//     assert_eq!(extracted_org1.name.as_deref(), Some("Organization One"));
//     assert_eq!(extracted_org2.name.as_deref(), Some("Organization Two"));

//     if entity1_name.as_deref() == Some("Organization One") {
//         assert_eq!(entity1_org_id, org1_id);
//         assert_eq!(entity2_name.as_deref(), Some("Organization Two"));
//         assert_eq!(entity2_org_id, org2_id);
//     } else {
//         assert_eq!(entity1_name.as_deref(), Some("Organization Two"));
//         assert_eq!(entity1_org_id, org2_id);
//         assert_eq!(entity2_name.as_deref(), Some("Organization One"));
//         assert_eq!(entity2_org_id, org1_id);
//     }

//     Ok(())
// }

// #[tokio::test]
// async fn test_extract_entities_idempotency() -> Result<()> {
//     let pool = setup_db_pool().await?;
//     clear_tables(&pool).await?;

//     let org1_id = insert_mock_organization(&pool, "org_idemp_1", "Idempotent Org 1").await?;

//     // First run
//     let extracted_first_run = entity_organizations::extract_entities(&pool).await?;
//     assert_eq!(extracted_first_run.len(), 1, "Should extract one entity on first run");
//     assert_eq!(extracted_first_run[0].organization_id.0, org1_id);

//     let client = pool.get().await?;
//     let count_after_first: i64 = client.query_one("SELECT COUNT(*) FROM mock_hsds.entity", &[]).await?.get(0);
//     assert_eq!(count_after_first, 1, "Entity table should have 1 record after first run");
//     let entity_id_first_run: String = client.query_one("SELECT id FROM mock_hsds.entity WHERE organization_id = $1", &[&org1_id]).await?.get("id");

//     // Second run - should not create new entities
//     let extracted_second_run = entity_organizations::extract_entities(&pool).await?;
//     // The function returns ALL entities (existing + new). So, count should still be 1.
//     assert_eq!(extracted_second_run.len(), 1, "Should still report one entity on second run (existing)");
//     assert_eq!(extracted_second_run[0].organization_id.0, org1_id);
//     assert_eq!(extracted_second_run[0].id.0, entity_id_first_run, "Entity ID should be the same on second run");

//     let count_after_second: i64 = client.query_one("SELECT COUNT(*) FROM mock_hsds.entity", &[]).await?.get(0);
//     assert_eq!(count_after_second, 1, "Entity table should still have 1 record after second run");

//     let entity_id_second_run: String = client.query_one("SELECT id FROM mock_hsds.entity WHERE organization_id = $1", &[&org1_id]).await?.get("id");
//     assert_eq!(entity_id_first_run, entity_id_second_run, "Entity ID in DB should remain unchanged");

//     Ok(())
// }

// #[tokio::test]
// async fn test_extract_entities_mixed_existing_and_new() -> Result<()> {
//     let pool = setup_db_pool().await?;
//     clear_tables(&pool).await?;

//     // 1. Pre-existing organization and entity
//     let org_existing_id = insert_mock_organization(&pool, "org_exist_1", "Existing Org").await?;
//     let existing_entity_id = Uuid::new_v4().to_string();
//     let client = pool.get().await?;
//     client.execute(
//         "INSERT INTO mock_hsds.entity (id, organization_id, name, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)",
//         &[&existing_entity_id, &org_existing_id, &"Existing Org", &Utc::now().naive_utc(), &Utc::now().naive_utc()]
//     ).await?;

//     // 2. New organization to be processed
//     let org_new_id = insert_mock_organization(&pool, "org_new_1", "New Org").await?;

//     let extracted_entities = entity_organizations::extract_entities(&pool).await?;

//     // Should return both existing and newly created entities
//     assert_eq!(extracted_entities.len(), 2, "Should extract/find two entities in total");

//     let db_entities = client.query("SELECT id, organization_id, name FROM mock_hsds.entity ORDER BY name", &[]).await?;
//     assert_eq!(db_entities.len(), 2, "Two entities should be in the database");

//     let found_existing = extracted_entities.iter().find(|e| e.organization_id.0 == org_existing_id).expect("Existing entity not found");
//     assert_eq!(found_existing.id.0, existing_entity_id, "Existing entity ID mismatch");
//     assert_eq!(found_existing.name.as_deref(), Some("Existing Org"));

//     let found_new = extracted_entities.iter().find(|e| e.organization_id.0 == org_new_id).expect("New entity not found");
//     assert_eq!(found_new.name.as_deref(), Some("New Org"));
//     assert_ne!(found_new.id.0, existing_entity_id, "New entity should have a different ID");

//     Ok(())
// }

// // --- Tests for link_entity_features ---

// // Helper to insert mock features for an organization
// async fn insert_mock_service(pool: &PgPool, org_id: &str, service_name: &str) -> Result<String> {
//     let client = pool.get().await?;
//     let service_id = Uuid::new_v4().to_string();
//     client.execute(
//         "INSERT INTO mock_hsds.service (id, organization_id, name, description, created_at, updated_at)
//          VALUES ($1, $2, $3, $4, $5, $6)",
//         &[&service_id, &org_id, &service_name, &Some("Test service"), &Utc::now().naive_utc(), &Utc::now().naive_utc()]
//     ).await?;
//     Ok(service_id)
// }

// async fn insert_mock_phone(pool: &PgPool, org_id: &str, phone_number: &str) -> Result<String> {
//     let client = pool.get().await?;
//     let phone_id = Uuid::new_v4().to_string();
//     client.execute(
//         "INSERT INTO mock_hsds.phone (id, organization_id, number, created_at, updated_at)
//          VALUES ($1, $2, $3, $4, $5)",
//         &[&phone_id, &org_id, &phone_number, &Utc::now().naive_utc(), &Utc::now().naive_utc()]
//     ).await?;
//     Ok(phone_id)
// }

// async fn insert_mock_location(pool: &PgPool, org_id: &str, loc_name: &str) -> Result<String> {
//     let client = pool.get().await?;
//     let loc_id = Uuid::new_v4().to_string();
//     // Assuming a simplified location table for mock, add/remove fields as per your actual mock_hsds.location
//     client.execute(
//         "INSERT INTO mock_hsds.location (id, organization_id, name, created_at, updated_at)
//          VALUES ($1, $2, $3, $4, $5)",
//         &[&loc_id, &org_id, &loc_name, &Utc::now().naive_utc(), &Utc::now().naive_utc()]
//     ).await?;
//     Ok(loc_id)
// }

// async fn insert_mock_contact(pool: &PgPool, org_id: &str, contact_name: &str) -> Result<String> {
//     let client = pool.get().await?;
//     let contact_id = Uuid::new_v4().to_string();
//     client.execute(
//         "INSERT INTO mock_hsds.contact (id, organization_id, name, title, created_at, updated_at)
//          VALUES ($1, $2, $3, $4, $5, $6)",
//         &[&contact_id, &org_id, &contact_name, &Some("Test Title"), &Utc::now().naive_utc(), &Utc::now().naive_utc()]
//     ).await?;
//     Ok(contact_id)
// }

// #[tokio::test]
// async fn test_link_entity_features_no_features() -> Result<()> {
//     let pool = setup_db_pool().await?;
//     clear_tables(&pool).await?;

//     let org_id = insert_mock_organization(&pool, "org_feat_none", "Org With No Features").await?;
//     let entities = entity_organizations::extract_entities(&pool).await?; // Creates the entity

//     let linked_count = entity_organizations::link_entity_features(&pool, &entities).await?;
//     assert_eq!(linked_count, 0, "Should link 0 features if none exist");

//     let client = pool.get().await?;
//     let count: i64 = client.query_one("SELECT COUNT(*) FROM mock_hsds.entity_feature", &[]).await?.get(0);
//     assert_eq!(count, 0, "entity_feature table should be empty");
//     Ok(())
// }

// #[tokio::test]
// async fn test_link_entity_features_single_entity_all_feature_types() -> Result<()> {
//     let pool = setup_db_pool().await?;
//     clear_tables(&pool).await?;

//     let org_id_str = "org_feat_all";
//     let org_id = insert_mock_organization(&pool, org_id_str, "Org With All Features").await?;

//     // Create the entity for this organization
//     let entities = entity_organizations::extract_entities(&pool).await?;
//     let entity = entities.iter().find(|e| e.organization_id.0 == org_id).expect("Entity not found");

//     // Insert mock features
//     let service_id1 = insert_mock_service(&pool, &org_id, "Service A").await?;
//     let service_id2 = insert_mock_service(&pool, &org_id, "Service B").await?;
//     let phone_id1 = insert_mock_phone(&pool, &org_id, "555-0001").await?;
//     let location_id1 = insert_mock_location(&pool, &org_id, "Main Location").await?;
//     let contact_id1 = insert_mock_contact(&pool, &org_id, "John Doe").await?;

//     let linked_count = entity_organizations::link_entity_features(&pool, &entities).await?;
//     assert_eq!(linked_count, 5, "Should link 5 features");

//     let client = pool.get().await?;
//     let rows = client.query("SELECT entity_id, table_name, table_id FROM mock_hsds.entity_feature WHERE entity_id = $1 ORDER BY table_name, table_id", &[&entity.id.0]).await?;
//     assert_eq!(rows.len(), 5, "Five features should be in the database for this entity");

//     let mut found_features = HashSet::new();
//     for row in rows {
//         let table_name: String = row.get("table_name");
//         let table_id: String = row.get("table_id");
//         found_features.insert((table_name, table_id));
//     }

//     assert!(found_features.contains(&("service".to_string(), service_id1)));
//     assert!(found_features.contains(&("service".to_string(), service_id2)));
//     assert!(found_features.contains(&("phone".to_string(), phone_id1)));
//     assert!(found_features.contains(&("location".to_string(), location_id1)));
//     assert!(found_features.contains(&("contact".to_string(), contact_id1)));

//     Ok(())
// }

// #[tokio::test]
// async fn test_link_entity_features_multiple_entities() -> Result<()> {
//     let pool = setup_db_pool().await?;
//     clear_tables(&pool).await?;

//     // Org 1 with some features
//     let org1_id = insert_mock_organization(&pool, "org_multi_1", "Multi Org 1").await?;
//     let service1_id = insert_mock_service(&pool, &org1_id, "Service X").await?;
//     let phone1_id = insert_mock_phone(&pool, &org1_id, "555-0011").await?;

//     // Org 2 with some features
//     let org2_id = insert_mock_organization(&pool, "org_multi_2", "Multi Org 2").await?;
//     let location2_id = insert_mock_location(&pool, &org2_id, "Branch Office").await?;

//     // Org 3 with no features
//     let org3_id = insert_mock_organization(&pool, "org_multi_3", "Multi Org 3 (No Features)").await?;

//     let entities = entity_organizations::extract_entities(&pool).await?;
//     assert_eq!(entities.len(), 3);

//     let entity1 = entities.iter().find(|e| e.organization_id.0 == org1_id).unwrap();
//     let entity2 = entities.iter().find(|e| e.organization_id.0 == org2_id).unwrap();
//     // Entity3 will also be in `entities`

//     let linked_count = entity_organizations::link_entity_features(&pool, &entities).await?;
//     assert_eq!(linked_count, 3, "Should link a total of 3 features across all entities");

//     let client = pool.get().await?;

//     // Check features for entity1
//     let features_e1 = client.query("SELECT table_name, table_id FROM mock_hsds.entity_feature WHERE entity_id = $1", &[&entity1.id.0]).await?;
//     assert_eq!(features_e1.len(), 2);
//     let mut found_e1 = HashSet::new();
//     for row in features_e1 { found_e1.insert((row.get::<_,String>("table_name"), row.get::<_,String>("table_id"))); }
//     assert!(found_e1.contains(&("service".to_string(), service1_id)));
//     assert!(found_e1.contains(&("phone".to_string(), phone1_id)));

//     // Check features for entity2
//     let features_e2 = client.query("SELECT table_name, table_id FROM mock_hsds.entity_feature WHERE entity_id = $1", &[&entity2.id.0]).await?;
//     assert_eq!(features_e2.len(), 1);
//     assert_eq!(features_e2[0].get::<String, _>("table_name"), "location");
//     assert_eq!(features_e2[0].get::<String, _>("table_id"), location2_id);

//     // Check total features in DB
//     let total_db_features: i64 = client.query_one("SELECT COUNT(*) FROM mock_hsds.entity_feature", &[]).await?.get(0);
//     assert_eq!(total_db_features, 3);

//     Ok(())
// }

// #[tokio::test]
// async fn test_link_entity_features_idempotency() -> Result<()> {
//     let pool = setup_db_pool().await?;
//     clear_tables(&pool).await?;

//     let org_id = insert_mock_organization(&pool, "org_idemp_feat", "Idempotent Feature Org").await?;
//     let service_id = insert_mock_service(&pool, &org_id, "Idempotent Service").await?;

//     let entities = entity_organizations::extract_entities(&pool).await?;

//     // First run
//     let linked_count_first = entity_organizations::link_entity_features(&pool, &entities).await?;
//     assert_eq!(linked_count_first, 1, "Should link 1 feature on first run");

//     let client = pool.get().await?;
//     let count_after_first: i64 = client.query_one("SELECT COUNT(*) FROM mock_hsds.entity_feature", &[]).await?.get(0);
//     assert_eq!(count_after_first, 1, "entity_feature table should have 1 record after first run");

//     // Second run
//     let linked_count_second = entity_organizations::link_entity_features(&pool, &entities).await?;
//     // The function returns total features (existing + new). Since no new ones, it's the count of existing.
//     assert_eq!(linked_count_second, 1, "Should report 1 feature on second run (existing)");

//     let count_after_second: i64 = client.query_one("SELECT COUNT(*) FROM mock_hsds.entity_feature", &[]).await?.get(0);
//     assert_eq!(count_after_second, 1, "entity_feature table should still have 1 record after second run");

//     Ok(())
// }

// #[tokio::test]
// async fn test_link_entity_features_some_existing_some_new() -> Result<()> {
//     let pool = setup_db_pool().await?;
//     clear_tables(&pool).await?;

//     let org_id = insert_mock_organization(&pool, "org_mix_feat", "Mixed Feature Org").await?;
//     let entities = entity_organizations::extract_entities(&pool).await?;
//     let entity = entities.iter().find(|e| e.organization_id.0 == org_id).unwrap();

//     // Pre-existing feature
//     let existing_service_id = insert_mock_service(&pool, &org_id, "Existing Service").await?;
//     let client = pool.get().await?;
//     client.execute(
//         "INSERT INTO mock_hsds.entity_feature (id, entity_id, table_name, table_id, created_at) VALUES ($1, $2, $3, $4, $5)",
//         &[&Uuid::new_v4().to_string(), &entity.id.0, &"service", &existing_service_id, &Utc::now().naive_utc()]
//     ).await?;

//     // New feature to be linked
//     let new_phone_id = insert_mock_phone(&pool, &org_id, "555-7777").await?;

//     let linked_count = entity_organizations::link_entity_features(&pool, &entities).await?;
//     // Should count existing (1) + new (1) = 2
//     assert_eq!(linked_count, 2, "Should report 2 features (1 existing, 1 new)");

//     let db_features = client.query("SELECT table_name, table_id FROM mock_hsds.entity_feature WHERE entity_id = $1 ORDER BY table_name", &[&entity.id.0]).await?;
//     assert_eq!(db_features.len(), 2, "Two features should be in the database for this entity");

//     let mut found_features = HashSet::new();
//     for row in db_features {
//         found_features.insert((row.get::<_,String>("table_name"), row.get::<_,String>("table_id")));
//     }
//     assert!(found_features.contains(&("service".to_string(), existing_service_id)));
//     assert!(found_features.contains(&("phone".to_string(), new_phone_id)));

//     Ok(())
// }
