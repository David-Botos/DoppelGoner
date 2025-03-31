import { Pool } from 'pg';

/**
 * Interface to define table constraints
 */
export interface TableConstraints {
  tableName: string;
  columns: ColumnConstraint[];
}

/**
 * Interface to define column-specific constraints
 */
export interface ColumnConstraint {
  name: string;
  type: string;
  nullable: boolean;
  isPrimaryKey?: boolean;
  isForeignKey?: boolean;
  references?: {
    table: string;
    column: string;
  };
  maxLength?: number;
  defaultValue?: any;
}

/**
 * Interface for validation result
 */
export interface ValidationResult<T> {
  validRecords: T[];
  invalidRecords: T[];
  invalidRecordsWithErrors: Array<{
    record: T;
    errors: string[];
  }>;
}

/**
 * Get database constraints for a table by querying PostgreSQL's information schema
 * @param pool PostgreSQL connection pool
 * @param tableName Table to get constraints for
 * @returns Table constraints object
 */
export async function getTableConstraints(pool: Pool, tableName: string): Promise<TableConstraints> {
  // Query to get column information
  const columnQuery = `
    SELECT 
      column_name, 
      data_type, 
      is_nullable,
      character_maximum_length,
      column_default
    FROM 
      information_schema.columns
    WHERE 
      table_schema = 'public' AND 
      table_name = $1
  `;
  
  // Query to get primary key information
  const pkQuery = `
    SELECT 
      kcu.column_name
    FROM 
      information_schema.table_constraints tc
    JOIN 
      information_schema.key_column_usage kcu 
      ON tc.constraint_name = kcu.constraint_name
    WHERE 
      tc.constraint_type = 'PRIMARY KEY' AND
      tc.table_schema = 'public' AND
      tc.table_name = $1
  `;
  
  // Query to get foreign key information
  const fkQuery = `
    SELECT 
      kcu.column_name,
      ccu.table_name AS foreign_table_name,
      ccu.column_name AS foreign_column_name
    FROM 
      information_schema.table_constraints tc
    JOIN 
      information_schema.key_column_usage kcu 
      ON tc.constraint_name = kcu.constraint_name
    JOIN 
      information_schema.constraint_column_usage ccu 
      ON tc.constraint_name = ccu.constraint_name
    WHERE 
      tc.constraint_type = 'FOREIGN KEY' AND
      tc.table_schema = 'public' AND
      tc.table_name = $1
  `;
  
  try {
    // Execute queries to get table constraints
    const [columnResults, pkResults, fkResults] = await Promise.all([
      pool.query(columnQuery, [tableName]),
      pool.query(pkQuery, [tableName]),
      pool.query(fkQuery, [tableName])
    ]);
    
    // Extract primary key columns
    const primaryKeys = pkResults.rows.map(row => row.column_name);
    
    // Extract foreign key relationships
    const foreignKeys = fkResults.rows.reduce((acc, row) => {
      acc[row.column_name] = {
        table: row.foreign_table_name,
        column: row.foreign_column_name
      };
      return acc;
    }, {} as Record<string, { table: string; column: string }>);
    
    // Build the constraints object
    const columns = columnResults.rows.map(row => ({
      name: row.column_name,
      type: row.data_type,
      nullable: row.is_nullable === 'YES',
      isPrimaryKey: primaryKeys.includes(row.column_name),
      isForeignKey: foreignKeys[row.column_name] !== undefined,
      references: foreignKeys[row.column_name],
      maxLength: row.character_maximum_length,
      defaultValue: row.column_default
    }));
    
    return {
      tableName,
      columns
    };
  } catch (error) {
    console.error(`Error fetching constraints for table ${tableName}:`, error);
    throw error;
  }
}

/**
 * Validate records against table constraints and split into valid and invalid sets
 * @param records Array of records to validate
 * @param constraints Table constraints to validate against
 * @returns Object containing valid and invalid records
 */
export function validateRecordsAgainstConstraints<T extends Record<string, any>>(
  records: T[],
  constraints: TableConstraints
): ValidationResult<T> {
  const validRecords: T[] = [];
  const invalidRecords: T[] = [];
  const invalidRecordsWithErrors: Array<{ record: T; errors: string[] }> = [];
  
  for (const record of records) {
    const validationErrors = getValidationErrors(record, constraints);
    
    if (validationErrors.length === 0) {
      validRecords.push(record);
    } else {
      invalidRecords.push(record);
      invalidRecordsWithErrors.push({
        record,
        errors: validationErrors
      });
    }
  }
  
  return { validRecords, invalidRecords, invalidRecordsWithErrors };
}

/**
 * Get validation errors for a record against table constraints
 * @param record Record to validate
 * @param constraints Table constraints to validate against
 * @returns Array of validation error messages
 */
export function getValidationErrors<T extends Record<string, any>>(
  record: T,
  constraints: TableConstraints
): string[] {
  const errors: string[] = [];
  
  // Check all required (non-nullable) columns
  for (const column of constraints.columns) {
    // Skip id if it's a primary key (it might be generated)
    if (column.name === 'id' && column.isPrimaryKey) {
      continue;
    }
    
    // Check required fields
    if (!column.nullable && (record[column.name] === undefined || record[column.name] === null)) {
      errors.push(`${column.name} is required and cannot be null`);
      continue;
    }
    
    // Skip further checks if value is null/undefined and nullable
    if ((record[column.name] === undefined || record[column.name] === null) && column.nullable) {
      continue;
    }
    
    // Validate data types and formats based on column type
    switch (column.type) {
      case 'text':
      case 'character varying':
      case 'character':
        if (typeof record[column.name] !== 'string') {
          errors.push(`${column.name} must be a string`);
        } else if (column.maxLength && record[column.name].length > column.maxLength) {
          errors.push(`${column.name} exceeds maximum length of ${column.maxLength}`);
        }
        break;
        
      case 'integer':
        if (!Number.isInteger(Number(record[column.name]))) {
          errors.push(`${column.name} must be an integer`);
        }
        break;
        
      case 'numeric':
      case 'decimal':
      case 'float':
        if (isNaN(Number(record[column.name]))) {
          errors.push(`${column.name} must be a number`);
        }
        break;
        
      case 'boolean':
        if (typeof record[column.name] !== 'boolean' && 
            record[column.name] !== 'true' && 
            record[column.name] !== 'false' && 
            record[column.name] !== '0' && 
            record[column.name] !== '1' && 
            record[column.name] !== 0 && 
            record[column.name] !== 1) {
          errors.push(`${column.name} must be a boolean value`);
        }
        break;
        
      case 'date':
      case 'timestamp without time zone':
      case 'timestamp with time zone':
        if (isNaN(Date.parse(record[column.name]))) {
          errors.push(`${column.name} must be a valid date/timestamp`);
        }
        break;
        
      case 'jsonb':
      case 'json':
        try {
          if (typeof record[column.name] === 'string') {
            JSON.parse(record[column.name]);
          } else if (typeof record[column.name] !== 'object') {
            errors.push(`${column.name} must be a valid JSON object or string`);
          }
        } catch (e) {
          errors.push(`${column.name} must be a valid JSON`);
        }
        break;
        
      case 'uuid':
        const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
        if (typeof record[column.name] !== 'string' || !uuidPattern.test(record[column.name])) {
          errors.push(`${column.name} must be a valid UUID`);
        }
        break;
        
      case 'USER-DEFINED':
        // For custom types like geometry, we don't validate strictly
        break;
    }
    
    // If it's a foreign key, we could potentially check referential integrity here
    // but that's typically better handled by the database
  }
  
  return errors;
}

/**
 * Create predefined constraints for a specific table
 * This can be used when you know the constraints ahead of time and want to avoid database lookups
 * @param tableName Table name
 * @param columnDefinitions Column definitions
 * @returns TableConstraints object
 */
export function createPredefinedConstraints(
  tableName: string,
  columnDefinitions: ColumnConstraint[]
): TableConstraints {
  return {
    tableName,
    columns: columnDefinitions
  };
}

/**
 * Predefined constraints for common HSDS tables
 * This can be extended as needed for optimization
 */
export const PREDEFINED_CONSTRAINTS: Record<string, TableConstraints> = {
  organization: createPredefinedConstraints('organization', [
    { name: 'id', type: 'character', nullable: false, isPrimaryKey: true },
    { name: 'name', type: 'text', nullable: false },
    { name: 'description', type: 'text', nullable: true },
    { name: 'email', type: 'text', nullable: true },
    { name: 'url', type: 'text', nullable: true },
    { name: 'tax_status', type: 'text', nullable: true },
    { name: 'tax_id', type: 'text', nullable: true },
    { name: 'year_incorporated', type: 'character', nullable: true },
    { name: 'legal_status', type: 'text', nullable: true },
    { name: 'parent_organization_id', type: 'character', nullable: true, isForeignKey: true, references: { table: 'organization', column: 'id' } },
    { name: 'created', type: 'timestamp without time zone', nullable: true },
    { name: 'last_modified', type: 'timestamp without time zone', nullable: true },
    { name: 'alternate_name', type: 'text', nullable: true },
    { name: 'original_id', type: 'text', nullable: true },
    { name: 'original_translations_id', type: 'text', nullable: true }
  ]),
  
  service: createPredefinedConstraints('service', [
    { name: 'id', type: 'character', nullable: false, isPrimaryKey: true },
    { name: 'organization_id', type: 'character', nullable: false, isForeignKey: true, references: { table: 'organization', column: 'id' } },
    { name: 'program_id', type: 'character', nullable: true, isForeignKey: true, references: { table: 'program', column: 'id' } },
    { name: 'name', type: 'text', nullable: false },
    { name: 'alternate_name', type: 'text', nullable: true },
    { name: 'description', type: 'text', nullable: true },
    { name: 'short_description', type: 'text', nullable: true },
    { name: 'url', type: 'text', nullable: true },
    { name: 'email', type: 'text', nullable: true },
    { name: 'status', type: 'text', nullable: false },
    { name: 'interpretation_services', type: 'text', nullable: true },
    { name: 'application_process', type: 'text', nullable: true },
    { name: 'wait_time', type: 'text', nullable: true },
    { name: 'fees_description', type: 'text', nullable: true },
    { name: 'accreditations', type: 'text', nullable: true },
    { name: 'licenses', type: 'text', nullable: true },
    { name: 'minimum_age', type: 'integer', nullable: true },
    { name: 'maximum_age', type: 'integer', nullable: true },
    { name: 'created', type: 'timestamp without time zone', nullable: true },
    { name: 'last_modified', type: 'timestamp without time zone', nullable: true },
    { name: 'original_id', type: 'text', nullable: true },
    { name: 'original_translations_id', type: 'text', nullable: true },
    { name: 'alert', type: 'text', nullable: true },
    { name: 'eligibility_description', type: 'text', nullable: true }
  ])
};