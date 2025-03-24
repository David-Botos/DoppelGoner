// Types for data transformation
import * as SnowflakeTypes from './snowflake-types';
import * as SupabaseTypes from './supabase-types';

// Generic transformation function type
export interface TransformFunction<Source, Target> {
  (sourceRecord: Source, translations?: any): Target;
}

// Transformation functions for each table
export interface TransformationMap {
  organization: TransformFunction<
    SnowflakeTypes.SnowflakeOrganization, 
    SupabaseTypes.SupabaseOrganization
  >;
  
  service: TransformFunction<
    SnowflakeTypes.SnowflakeService, 
    SupabaseTypes.SupabaseService
  >;
  
  location: TransformFunction<
    SnowflakeTypes.SnowflakeLocation, 
    SupabaseTypes.SupabaseLocation
  >;
  
  service_at_location: TransformFunction<
    SnowflakeTypes.SnowflakeServiceAtLocation, 
    SupabaseTypes.SupabaseServiceAtLocation
  >;
  
  address: TransformFunction<
    SnowflakeTypes.SnowflakeAddress, 
    SupabaseTypes.SupabaseAddress
  >;
  
  phone: TransformFunction<
    SnowflakeTypes.SnowflakePhone, 
    SupabaseTypes.SupabasePhone
  >;
}

// Migration result type
export interface MigrationResult {
  sourceTable: string;
  targetTable: string;
  recordsProcessed: number;
  recordsSucceeded: number;
  recordsFailed: number;
  errors: string[];
  startTime: Date;
  endTime: Date;
  executionTimeSeconds: number;
}

// Translation map type
export interface TranslationMap {
  [recordId: string]: any;
}