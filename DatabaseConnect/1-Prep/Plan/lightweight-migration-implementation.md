# Lightweight Migration Implementation Plan: Snowflake to RDS

This document outlines a pragmatic approach to migrate 100 sample records from Snowflake to RDS following the HSDS schema with minimal effort.

## 1. RDS Instance Configuration

### Database Setup SQL Script

```sql
-- RDS setup script that can be run directly from AWS console
-- This script creates the core HSDS tables needed for the initial migration

-- Create database and schema
CREATE DATABASE hsds_migration;
USE hsds_migration;

-- Core Organizations table
CREATE TABLE organization (
  id CHAR(36) PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  alternate_name VARCHAR(255),
  description TEXT,
  email VARCHAR(255),
  url VARCHAR(255),
  tax_status VARCHAR(255),
  tax_id VARCHAR(255),
  year_incorporated CHAR(4),
  legal_status VARCHAR(255),
  parent_organization_id CHAR(36),
  last_modified TIMESTAMP,
  created TIMESTAMP
);

-- Core Services table
CREATE TABLE service (
  id CHAR(36) PRIMARY KEY,
  organization_id CHAR(36) NOT NULL,
  program_id CHAR(36),
  name VARCHAR(255) NOT NULL,
  alternate_name VARCHAR(255),
  description TEXT,
  url VARCHAR(255),
  email VARCHAR(255),
  status VARCHAR(50) NOT NULL,
  interpretation_services TEXT,
  application_process TEXT,
  wait_time VARCHAR(255),
  fees_description TEXT,
  accreditations TEXT,
  licenses VARCHAR(255),
  minimum_age INT,
  maximum_age INT,
  last_modified TIMESTAMP,
  created TIMESTAMP,
  FOREIGN KEY (organization_id) REFERENCES organization(id)
);

-- Core Location table
CREATE TABLE location (
  id CHAR(36) PRIMARY KEY,
  organization_id CHAR(36) NOT NULL,
  name VARCHAR(255),
  alternate_name VARCHAR(255),
  description TEXT,
  transportation TEXT,
  latitude DECIMAL(10,6),
  longitude DECIMAL(10,6),
  location_type VARCHAR(50),
  last_modified TIMESTAMP,
  created TIMESTAMP,
  FOREIGN KEY (organization_id) REFERENCES organization(id)
);

-- Core Service at Location table
CREATE TABLE service_at_location (
  id CHAR(36) PRIMARY KEY,
  service_id CHAR(36) NOT NULL,
  location_id CHAR(36) NOT NULL,
  description TEXT,
  last_modified TIMESTAMP,
  created TIMESTAMP,
  FOREIGN KEY (service_id) REFERENCES service(id),
  FOREIGN KEY (location_id) REFERENCES location(id)
);

-- Core Phone table
CREATE TABLE phone (
  id CHAR(36) PRIMARY KEY,
  location_id CHAR(36),
  service_id CHAR(36),
  organization_id CHAR(36),
  contact_id CHAR(36),
  service_at_location_id CHAR(36),
  number VARCHAR(50) NOT NULL,
  extension VARCHAR(20),
  type VARCHAR(20),
  language VARCHAR(10),
  description TEXT,
  last_modified TIMESTAMP,
  created TIMESTAMP,
  FOREIGN KEY (location_id) REFERENCES location(id),
  FOREIGN KEY (service_id) REFERENCES service(id),
  FOREIGN KEY (organization_id) REFERENCES organization(id),
  FOREIGN KEY (service_at_location_id) REFERENCES service_at_location(id)
);

-- Core Physical Address table
CREATE TABLE physical_address (
  id CHAR(36) PRIMARY KEY,
  location_id CHAR(36) NOT NULL,
  attention VARCHAR(255),
  address_1 VARCHAR(255) NOT NULL,
  address_2 VARCHAR(255),
  city VARCHAR(255) NOT NULL,
  region VARCHAR(255),
  state_province VARCHAR(100) NOT NULL,
  postal_code VARCHAR(20) NOT NULL,
  country CHAR(2) NOT NULL,
  last_modified TIMESTAMP,
  created TIMESTAMP,
  FOREIGN KEY (location_id) REFERENCES location(id)
);

-- Create a simple migration log table
CREATE TABLE migration_log (
  id INT AUTO_INCREMENT PRIMARY KEY,
  source_table VARCHAR(100) NOT NULL,
  target_table VARCHAR(100) NOT NULL,
  records_processed INT NOT NULL,
  status VARCHAR(50) NOT NULL,
  message TEXT,
  execution_time FLOAT NOT NULL,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### RDS Configuration Steps

1. **Launch RDS Instance from AWS Console**:
   - Select MySQL or PostgreSQL engine (PostgreSQL recommended for JSONB support)
   - Choose db.t3.micro instance for testing (minimal cost)
   - Set storage to 20GB (minimum)
   - Enable public access for development (disable in production)
   - Create new security group for migration and configure inbound rules

2. **Run Setup Script**:
   - Connect to the database using Query Editor in AWS Console
   - Paste and execute the above SQL script
   - This creates the minimum tables needed for initial testing

## 2. Database Connection Configuration

### Connection Configuration File

Create a `config.py` file with the following structure:

```python
# Database connection configurations

# Snowflake connection parameters
SNOWFLAKE_CONFIG = {
    'user': 'YOUR_SNOWFLAKE_USER',
    'password': 'YOUR_SNOWFLAKE_PASSWORD',
    'account': 'YOUR_SNOWFLAKE_ACCOUNT',
    'warehouse': 'YOUR_WAREHOUSE',
    'database': 'NORSE_STAGING',
    'schema': 'WA211'
}

# RDS connection parameters
RDS_CONFIG = {
    'host': 'YOUR_RDS_ENDPOINT',
    'user': 'YOUR_RDS_USER',
    'password': 'YOUR_RDS_PASSWORD',
    'database': 'hsds_migration',
    'port': 5432  # For PostgreSQL, use 3306 for MySQL
}

# Migration control parameters
MIGRATION_CONFIG = {
    'batch_size': 100,  # Number of records to process in each batch
    'log_level': 'INFO',
    'tables_to_migrate': [
        'ORGANIZATION',
        'SERVICE',
        'LOCATION',
        'SERVICE_AT_LOCATION',
        'PHONE',
        'ADDRESS'
    ]
}
```

### Dependencies Installation

Create a `requirements.txt` file:

```
snowflake-connector-python==3.0.4
psycopg2-binary==2.9.6  # For PostgreSQL
# pymysql==1.0.3        # Uncomment for MySQL
pandas==1.5.3
sqlalchemy==2.0.20
python-dotenv==1.0.0
tqdm==4.66.1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 3. Data Transfer Implementation

### Basic Migration Script

Create `migrate.py` with the following structure:

```python
#!/usr/bin/env python3

import time
import logging
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from tqdm import tqdm
import uuid
from datetime import datetime
import os
from config import SNOWFLAKE_CONFIG, RDS_CONFIG, MIGRATION_CONFIG

# Set up logging
logging.basicConfig(
    level=getattr(logging, MIGRATION_CONFIG['log_level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("migration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("migration")

def create_snowflake_connection():
    """Create and return a Snowflake connection."""
    try:
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_CONFIG['user'],
            password=SNOWFLAKE_CONFIG['password'],
            account=SNOWFLAKE_CONFIG['account'],
            warehouse=SNOWFLAKE_CONFIG['warehouse'],
            database=SNOWFLAKE_CONFIG['database'],
            schema=SNOWFLAKE_CONFIG['schema']
        )
        logger.info("Successfully connected to Snowflake")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {e}")
        raise

def create_rds_engine():
    """Create and return a SQLAlchemy engine for RDS."""
    try:
        # Adjust connection string as needed for PostgreSQL or MySQL
        conn_string = f"postgresql://{RDS_CONFIG['user']}:{RDS_CONFIG['password']}@{RDS_CONFIG['host']}:{RDS_CONFIG['port']}/{RDS_CONFIG['database']}"
        # For MySQL: conn_string = f"mysql+pymysql://{RDS_CONFIG['user']}:{RDS_CONFIG['password']}@{RDS_CONFIG['host']}:{RDS_CONFIG['port']}/{RDS_CONFIG['database']}"
        engine = create_engine(conn_string)
        logger.info("Successfully created RDS engine")
        return engine
    except Exception as e:
        logger.error(f"Failed to create RDS engine: {e}")
        raise

def log_migration(source_table, target_table, records, status, message, execution_time, rds_engine):
    """Log migration statistics to the database."""
    try:
        query = """
        INSERT INTO migration_log (source_table, target_table, records_processed, status, message, execution_time)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        with rds_engine.connect() as conn:
            conn.execute(sa.text(query), 
                        [source_table, target_table, records, status, message, execution_time])
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to log migration: {e}")

def migrate_organizations(sf_conn, rds_engine, limit=100):
    """Migrate organizations from Snowflake to RDS."""
    start_time = time.time()
    
    try:
        logger.info(f"Starting migration of {limit} ORGANIZATION records")
        
        # Extract data from Snowflake
        query = f"""
        SELECT 
            o.ID, 
            o.NAME, 
            o.ALTERNATE_NAME, 
            ot.DESCRIPTION, 
            o.EMAIL, 
            o.WEBSITE as URL, 
            o.LEGAL_STATUS, 
            o.YEAR_INCORPORATED,
            o.PARENT_ORGANIZATION_ID, 
            o.LAST_MODIFIED, 
            o.CREATED
        FROM 
            ORGANIZATION o
        LEFT JOIN 
            ORGANIZATION_TRANSLATIONS ot ON o.ID = ot.ORGANIZATION_ID AND ot.IS_CANONICAL = TRUE
        LIMIT {limit}
        """
        
        cursor = sf_conn.cursor()
        cursor.execute(query)
        df = cursor.fetch_pandas_all()
        
        # Clean and transform data
        # Convert columns to appropriate types
        df['URL'] = df['URL'].fillna('')
        df['EMAIL'] = df['EMAIL'].fillna('')
        df['DESCRIPTION'] = df['DESCRIPTION'].fillna('')
        
        # Transform timestamps
        for col in ['LAST_MODIFIED', 'CREATED']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Load data to RDS
        with rds_engine.connect() as conn:
            # Clear existing data (optional for testing)
            conn.execute(sa.text("DELETE FROM organization"))
            conn.commit()
        
        # Prepare data for insertion
        insert_df = df.rename(columns={
            'NAME': 'name',
            'ALTERNATE_NAME': 'alternate_name',
            'DESCRIPTION': 'description',
            'EMAIL': 'email',
            'URL': 'url',
            'LEGAL_STATUS': 'legal_status',
            'YEAR_INCORPORATED': 'year_incorporated',
            'PARENT_ORGANIZATION_ID': 'parent_organization_id',
            'LAST_MODIFIED': 'last_modified',
            'CREATED': 'created'
        })
        
        # Add missing columns with default values
        insert_df['tax_status'] = None
        insert_df['tax_id'] = None
        
        # Insert data
        insert_df.to_sql('organization', rds_engine, if_exists='append', index=False)
        
        execution_time = time.time() - start_time
        logger.info(f"Successfully migrated {len(df)} ORGANIZATION records in {execution_time:.2f} seconds")
        
        # Log migration
        log_migration('ORGANIZATION', 'organization', len(df), 'SUCCESS', 
                     f"Migrated {len(df)} records successfully", execution_time, rds_engine)
        
        return len(df)
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Failed to migrate ORGANIZATION: {e}")
        log_migration('ORGANIZATION', 'organization', 0, 'ERROR', 
                     f"Migration failed: {str(e)}", execution_time, rds_engine)
        raise

def migrate_services(sf_conn, rds_engine, limit=100):
    """Migrate services from Snowflake to RDS."""
    start_time = time.time()
    
    try:
        logger.info(f"Starting migration of {limit} SERVICE records")
        
        # Extract data from Snowflake
        query = f"""
        SELECT 
            s.ID, 
            s.ORGANIZATION_ID, 
            s.PROGRAM_ID, 
            st.NAME, 
            st.ALTERNATE_NAME, 
            st.DESCRIPTION, 
            s.URL, 
            s.EMAIL, 
            s.STATUS, 
            st.INTERPRETATION_SERVICES, 
            st.APPLICATION_PROCESS, 
            st.FEES_DESCRIPTION, 
            st.ACCREDITATIONS, 
            s.MINIMUM_AGE, 
            s.MAXIMUM_AGE, 
            s.LAST_MODIFIED, 
            s.CREATED
        FROM 
            SERVICE s
        LEFT JOIN 
            SERVICE_TRANSLATIONS st ON s.ID = st.SERVICE_ID AND st.IS_CANONICAL = TRUE
        LIMIT {limit}
        """
        
        cursor = sf_conn.cursor()
        cursor.execute(query)
        df = cursor.fetch_pandas_all()
        
        # Clean and transform data
        # (Similar transformation logic as in migrate_organizations)
        
        # Load data to RDS
        with rds_engine.connect() as conn:
            # Clear existing data (optional for testing)
            conn.execute(sa.text("DELETE FROM service"))
            conn.commit()
        
        # Prepare data for insertion and insert
        # (Similar to organization migration)
        
        execution_time = time.time() - start_time
        logger.info(f"Successfully migrated {len(df)} SERVICE records in {execution_time:.2f} seconds")
        
        # Log migration
        log_migration('SERVICE', 'service', len(df), 'SUCCESS', 
                     f"Migrated {len(df)} records successfully", execution_time, rds_engine)
        
        return len(df)
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Failed to migrate SERVICE: {e}")
        log_migration('SERVICE', 'service', 0, 'ERROR', 
                     f"Migration failed: {str(e)}", execution_time, rds_engine)
        raise

# Similar functions for other entities: locations, service_at_location, etc.

def main():
    """Main migration function."""
    sf_conn = None
    try:
        # Connect to Snowflake
        sf_conn = create_snowflake_connection()
        
        # Create RDS engine
        rds_engine = create_rds_engine()
        
        # Migrate core entities
        org_count = migrate_organizations(sf_conn, rds_engine, MIGRATION_CONFIG['batch_size'])
        service_count = migrate_services(sf_conn, rds_engine, MIGRATION_CONFIG['batch_size'])
        
        # Add migration functions for other entities
        
        logger.info(f"Migration completed successfully!")
        logger.info(f"Organization records: {org_count}")
        logger.info(f"Service records: {service_count}")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
    finally:
        if sf_conn:
            sf_conn.close()
            logger.info("Snowflake connection closed")

if __name__ == "__main__":
    main()
```

### Execution Instructions

1. Configure connection parameters in `config.py`
2. Run migration script for initial test:
   ```bash
   python migrate.py
   ```
3. Check logs and verify data in RDS

## Recommended Implementation Approach

### Step 1: Set Up Environment & Database

1. Create RDS instance through AWS Console
2. Run the SQL setup script
3. Configure database access security

### Step 2: Implement & Test Migration Scripts

1. Implement the `config.py` file with connection parameters
2. Update `migrate.py` with full functionality for all required tables
3. Run migration with limit=10 for initial testing
4. Verify data integrity in target database

### Step 3: Run 100-Record Migration

1. Set `batch_size` to 100 in config
2. Run migration script
3. Validate data in RDS:
   - Verify record counts
   - Check data integrity (especially IDs and relationships)
   - Ensure timestamp and geographic data transferred correctly

### Step 4: Iterate and Improve

1. Add migration functions for additional tables as needed
2. Improve error handling and data validation
3. Optimize performance for larger datasets

## Recommendations for Scaling Up

After successful migration of 100 records, the following steps are recommended for full migration:

1. Implement incremental loading with checkpoints
2. Add parallel processing for independent tables
3. Implement data validation functions to verify integrity
4. Add retry logic for failed migrations
5. Consider a more robust ETL tool like Apache Airflow for production migration

## Troubleshooting Common Issues

1. **Connection Timeouts**: Increase timeout settings in connection parameters
2. **Memory Issues**: Process data in smaller batches
3. **Foreign Key Violations**: Ensure tables are migrated in the correct order
4. **Data Type Mismatches**: Add explicit type casting in SQL queries or pandas transformations
5. **Missing Dependencies**: Check all required libraries are installed

## Conclusion

This lightweight approach provides a starting point for migrating a small sample of data from Snowflake to RDS following the HSDS schema. By focusing on just 100 records initially, you can validate the migration process with minimal effort before scaling up to a full migration.
