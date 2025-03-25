// Type definitions for Supabase database tables
import { MigratedData } from "./transformation-types";

// Organization table type
export interface SupabaseOrganization extends MigratedData {
  name: string;
  alternate_name?: string;
  description?: string;
  email?: string;
  url?: string;
  year_incorporated?: string;
  legal_status?: string;
  parent_organization_id: string | null;
}

// Service table type
export interface SupabaseService extends MigratedData {
  organization_id: string;
  program_id?: string;
  name: string;
  alternate_name?: string;
  description?: string;
  short_description?: string;
  url?: string;
  email?: string;
  status: string;
  interpretation_services?: string;
  application_process?: string;
  wait_time?: string;
  fees_description?: string;
  accreditations?: string;
  licenses?: string;
  minimum_age?: number;
  maximum_age?: number;
  eligibility_description?: string;
  alert?: string;
}

// Location table type
export interface SupabaseLocation extends MigratedData {
  organization_id: string;
  name?: string;
  alternate_name?: string;
  description?: string;
  short_description?: string;
  transportation?: string;
  latitude?: number;
  longitude?: number;
  location_type?: string;
}

// ServiceAtLocation table type
export interface SupabaseServiceAtLocation extends MigratedData {
  service_id: string;
  location_id: string;
  description?: string;
}

// Address table type (consolidated)
export interface SupabaseAddress extends MigratedData {
  location_id: string;
  attention?: string;
  address_1: string;
  address_2?: string;
  city: string;
  region?: string;
  state_province: string;
  postal_code: string;
  country: string;
  address_type: string;
}

// Phone table type
export interface SupabasePhone extends MigratedData {
  location_id?: string;
  service_id?: string;
  organization_id?: string;
  contact_id?: string;
  service_at_location_id?: string;
  number: string;
  extension?: string;
  type?: string;
  language?: string;
  description?: string;
  priority?: number;
}

// Migration log table type
export interface SupabaseMigrationLog {
  id: number;
  source_table: string;
  target_table: string;
  records_migrated: number;
  success_count: number;
  failure_count: number;
  error_messages?: string;
  started_at: string;
  completed_at: string;
  execution_time_seconds: number;
}
