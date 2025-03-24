import snowflake.connector
import os
from datetime import datetime
import getpass
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Snowflake connection parameters
conn_params = {
    'user': os.environ.get('SNOWFLAKE_USER', 'YOUR_USERNAME'),
    'password': os.environ.get('SNOWFLAKE_PASSWORD', 'YOUR_PASSWORD'),
    'account': os.environ.get('SNOWFLAKE_ACCOUNT', 'YOUR_ACCOUNT'),
    'warehouse': os.environ.get('SNOWFLAKE_WAREHOUSE', 'YOUR_WAREHOUSE'),
    'database': 'NORSE_STAGING',
    'role': os.environ.get('SNOWFLAKE_ROLE', 'YOUR_ROLE'),
    # MFA authentication parameters
    'authenticator': 'snowflake'
}

# Schema to extract
schema = 'WA211'

def get_table_data_samples():
    """Connect to Snowflake and extract the first 3 entries of each table in the WA211 schema."""
    
    # Create markdown content
    markdown = f"# Snowflake Table Data Samples: NORSE_STAGING.{schema}\n\n"
    markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    try:
        # For interactive MFA, we should get credentials at runtime
        if not os.environ.get('SNOWFLAKE_USER'):
            conn_params['user'] = input("Enter Snowflake username: ")
        
        if not os.environ.get('SNOWFLAKE_PASSWORD'):
            conn_params['password'] = getpass.getpass("Enter Snowflake password: ")
            
        print("Connecting to Snowflake... (You may be prompted for MFA verification)")
        
        # Connect to Snowflake - this will trigger MFA prompt if enabled
        conn = snowflake.connector.connect(**conn_params)
        cursor = conn.cursor()
        
        print("Connection successful! Extracting table data samples...")
        
        # Get all tables in the schema
        cursor.execute(f"SHOW TABLES IN SCHEMA NORSE_STAGING.{schema}")
        tables = cursor.fetchall()
        
        if not tables:
            markdown += f"No tables found in schema {schema}.\n\n"
            return markdown
        
        # For each table
        for table in tables:
            table_name = table[1]  # Table name is in the second column
            
            # Skip tables that start with APP_
            if table_name.startswith("APP_"):
                print(f"Skipping table: {table_name} (starts with APP_)")
                continue
                
            print(f"Processing table: {table_name}")
            markdown += f"## Table: {table_name}\n\n"
            
            # First get count of records
            cursor.execute(f"SELECT COUNT(*) FROM NORSE_STAGING.{schema}.{table_name}")
            count = cursor.fetchone()[0]
            
            if count == 0:
                markdown += f"**Note: This table is empty and should be ignored for migration.**\n\n"
                markdown += "---\n\n"
                continue
            
            # Get column information to build a proper display
            cursor.execute(f"SHOW COLUMNS IN TABLE NORSE_STAGING.{schema}.{table_name}")
            columns = cursor.fetchall()
            column_names = [col[0] for col in columns]
            
            # Get first 3 entries (or fewer if there are less than 3)
            cursor.execute(f"SELECT * FROM NORSE_STAGING.{schema}.{table_name} LIMIT 3")
            records = cursor.fetchall()
            
            if len(records) < 3:
                markdown += f"**Note: This table contains only {len(records)} records.**\n\n"
            
            # Create a pandas DataFrame for better formatting
            df = pd.DataFrame(records, columns=column_names)
            
            # Convert DataFrame to markdown table
            markdown_table = df.to_markdown(index=False)
            markdown += markdown_table + "\n\n"
            
            if len(records) < 3:
                markdown += f"**Note: This table contains fewer than 3 records but still has data and should be included in migration.**\n\n"
            
            markdown += "---\n\n"
        
        # Close connection
        cursor.close()
        conn.close()
        print("Data extraction complete!")
        
        return markdown
        
    except snowflake.connector.errors.DatabaseError as e:
        # Handle MFA errors specifically
        if "authentication" in str(e).lower():
            return f"# Authentication Error\n\nAn authentication error occurred: {str(e)}\n\nPlease ensure you have completed the MFA verification process."
        else:
            return f"# Database Error\n\nA database error occurred: {str(e)}"
    except Exception as e:
        return f"# Error Extracting Data\n\nAn error occurred: {str(e)}"

def save_to_file(content, filename="snowflake_table_samples.md"):
    """Save the markdown content to a file."""
    with open(filename, "w") as f:
        f.write(content)
    print(f"Data samples exported to {filename}")

if __name__ == "__main__":
    # Get the markdown content
    print(f"Starting Snowflake data sample extraction for schema {schema}...")
    markdown_content = get_table_data_samples()
    
    # Save to file
    save_to_file(markdown_content)