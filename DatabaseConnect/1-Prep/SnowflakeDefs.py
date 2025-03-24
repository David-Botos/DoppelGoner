import snowflake.connector
import os
from datetime import datetime
import getpass
from dotenv import load_dotenv

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

# Schemas to extract
schemas = ['WA211', 'WHATCOMCOU', 'WITHINREAC']

def get_table_definitions():
    """Connect to Snowflake and extract basic table definitions for the specified schemas."""
    
    # Create markdown content
    markdown = f"# Snowflake Database Schema: NORSE_STAGING\n\n"
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
        
        print("Connection successful! Extracting schema information...")
        
        # For each schema
        for schema in schemas:
            print(f"Processing schema: {schema}")
            markdown += f"## Schema: {schema}\n\n"
            
            # Get all tables in the schema
            cursor.execute(f"SHOW TABLES IN SCHEMA NORSE_STAGING.{schema}")
            tables = cursor.fetchall()
            
            if not tables:
                markdown += f"No tables found in schema {schema}.\n\n"
                continue
            
            # For each table
            for table in tables:
                table_name = table[1]  # Table name is in the second column
                print(f"  Processing table: {table_name}")
                markdown += f"### Table: {table_name}\n\n"
                
                # Get table DDL - this is the part that's working in the UI
                cursor.execute(f"SELECT GET_DDL('TABLE', 'NORSE_STAGING.{schema}.{table_name}')")
                ddl = cursor.fetchone()[0]
                markdown += f"```sql\n{ddl}\n```\n\n"
                
                # Get basic column info from SHOW COLUMNS instead of INFORMATION_SCHEMA
                markdown += "#### Columns\n\n"
                cursor.execute(f"SHOW COLUMNS IN TABLE NORSE_STAGING.{schema}.{table_name}")
                columns = cursor.fetchall()
                
                markdown += "| Column Name | Data Type | Nullable | Default | Comment |\n"
                markdown += "|-------------|-----------|----------|---------|--------|\n"
                
                for col in columns:
                    col_name = col[0]  # Column name is typically the first column
                    data_type = col[1]  # Data type is typically the second column
                    is_nullable = col[2] if len(col) > 2 else "YES"  # Nullable info if available
                    default = col[3] if len(col) > 3 else ""  # Default value if available
                    comment = col[4] if len(col) > 4 else ""  # Comment if available
                    
                    # Escape pipe characters in markdown table
                    col_name = str(col_name).replace("|", "\\|")
                    data_type = str(data_type).replace("|", "\\|")
                    default = str(default).replace("|", "\\|") if default else ""
                    comment = str(comment).replace("|", "\\|") if comment else ""
                    
                    markdown += f"| {col_name} | {data_type} | {is_nullable} | {default} | {comment} |\n"
                
                markdown += "\n---\n\n"
        
        # Close connection
        cursor.close()
        conn.close()
        print("Schema extraction complete!")
        
        return markdown
        
    except snowflake.connector.errors.DatabaseError as e:
        # Handle MFA errors specifically
        if "authentication" in str(e).lower():
            return f"# Authentication Error\n\nAn authentication error occurred: {str(e)}\n\nPlease ensure you have completed the MFA verification process."
        else:
            return f"# Database Error\n\nA database error occurred: {str(e)}"
    except Exception as e:
        return f"# Error Extracting Schema\n\nAn error occurred: {str(e)}"
def save_to_file(content, filename="snowflake_schema.md"):
    """Save the markdown content to a file."""
    with open(filename, "w") as f:
        f.write(content)
    print(f"Schema exported to {filename}")

if __name__ == "__main__":
    # Get the markdown content
    print("Starting Snowflake schema extraction...")
    markdown_content = get_table_definitions()
    
    # Save to file
    save_to_file(markdown_content)