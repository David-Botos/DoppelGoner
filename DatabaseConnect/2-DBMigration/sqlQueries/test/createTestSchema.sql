-- Drop test schema if it exists
DROP SCHEMA IF EXISTS test CASCADE;

-- Create test schema
CREATE SCHEMA test;

-- Create tables in test schema
CREATE TABLE test.accessibility_for_disabilities (
    id character(36) NOT NULL,
    location_id character(36) NOT NULL,
    accessibility text NOT NULL,
    details text,
    last_modified timestamp without time zone,
    created timestamp without time zone,
    original_id text,
    CONSTRAINT test_accessibility_for_disabilities_pkey PRIMARY KEY (id)
);

CREATE TABLE test.address (
    id character(36) NOT NULL,
    location_id character(36) NOT NULL,
    attention text,
    address_1 text NOT NULL,
    address_2 text,
    city text NOT NULL,
    region text,
    state_province text NOT NULL,
    postal_code text NOT NULL,
    country character(2) NOT NULL,
    address_type text NOT NULL,
    last_modified timestamp without time zone,
    created timestamp without time zone,
    original_id text,
    CONSTRAINT test_address_pkey PRIMARY KEY (id)
);

CREATE TABLE test.contact (
    id character(36) NOT NULL,
    organization_id character(36),
    service_id character(36),
    service_at_location_id character(36),
    name text,
    title text,
    department text,
    email text,
    last_modified timestamp without time zone,
    created timestamp without time zone,
    original_id text,
    original_translations_id text,
    CONSTRAINT test_contact_pkey PRIMARY KEY (id)
);

CREATE TABLE test.failed_migration_records (
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    table_name text NOT NULL,
    original_id text,
    original_translations_id text,
    error_message text NOT NULL,
    attempted_record jsonb NOT NULL,
    attempted_at timestamp without time zone NOT NULL DEFAULT now(),
    resolved boolean NOT NULL DEFAULT false,
    resolved_at timestamp without time zone,
    resolved_by text,
    resolution_notes text,
    retry_count integer NOT NULL DEFAULT 0,
    last_retry_at timestamp without time zone,
    CONSTRAINT test_failed_migration_records_pkey PRIMARY KEY (id)
);

CREATE TABLE test.funding (
    id character(36) NOT NULL,
    organization_id character(36),
    service_id character(36),
    source text,
    last_modified timestamp without time zone,
    created timestamp without time zone,
    original_id text,
    CONSTRAINT test_funding_pkey PRIMARY KEY (id)
);

CREATE TABLE test.language (
    id character(36) NOT NULL,
    service_id character(36),
    location_id character(36),
    language text NOT NULL,
    last_modified timestamp without time zone,
    created timestamp without time zone,
    original_id text,
    CONSTRAINT test_language_pkey PRIMARY KEY (id)
);

CREATE TABLE test.organization (
    id character(36) NOT NULL,
    name text NOT NULL,
    alternate_name text,
    description text,
    email text,
    url text,
    tax_status text,
    tax_id text,
    year_incorporated character(4),
    legal_status text,
    parent_organization_id character(36),
    last_modified timestamp without time zone,
    created timestamp without time zone,
    original_id text,
    original_translations_id text,
    CONSTRAINT test_organization_pkey PRIMARY KEY (id)
);

CREATE TABLE test.location (
    id character(36) NOT NULL,
    organization_id character(36) NOT NULL,
    name text,
    alternate_name text,
    description text,
    short_description text,
    transportation text,
    latitude numeric(10,6),
    longitude numeric(10,6),
    location_type text,
    last_modified timestamp without time zone,
    created timestamp without time zone,
    original_id text,
    original_translations_id text,
    CONSTRAINT test_location_pkey PRIMARY KEY (id)
);

CREATE TABLE test.metadata (
    id uuid NOT NULL,
    resource_id text NOT NULL,
    resource_type text NOT NULL,
    last_action_date timestamp with time zone NOT NULL,
    last_action_type text NOT NULL,
    field_name text NOT NULL,
    previous_value text,
    replacement_value text,
    updated_by text NOT NULL,
    created timestamp with time zone NOT NULL,
    last_modified timestamp with time zone NOT NULL,
    original_id text,
    CONSTRAINT test_metadata_pkey PRIMARY KEY (id)
);

CREATE SEQUENCE IF NOT EXISTS test.migration_log_id_seq;

CREATE TABLE test.migration_log (
    id integer NOT NULL DEFAULT nextval('test.migration_log_id_seq'::regclass),
    source_table text NOT NULL,
    target_table text NOT NULL,
    records_migrated integer NOT NULL,
    success_count integer NOT NULL,
    failure_count integer NOT NULL,
    error_messages text,
    started_at timestamp without time zone NOT NULL,
    completed_at timestamp without time zone NOT NULL,
    execution_time_seconds numeric(10,2) NOT NULL,
    CONSTRAINT test_migration_log_pkey PRIMARY KEY (id)
);

CREATE TABLE test.other_attribute (
    id character(36) NOT NULL,
    link_id character(36) NOT NULL,
    link_type text NOT NULL,
    taxonomy_term_id character(36),
    last_modified timestamp without time zone,
    created timestamp without time zone,
    original_id text,
    CONSTRAINT test_other_attribute_pkey PRIMARY KEY (id)
);

CREATE TABLE test.phone (
    id character(36) NOT NULL,
    location_id character(36),
    service_id character(36),
    organization_id character(36),
    contact_id character(36),
    service_at_location_id character(36),
    number text NOT NULL,
    extension text,
    type text,
    language text DEFAULT 'en'::character varying,
    description text,
    priority integer,
    last_modified timestamp without time zone,
    created timestamp without time zone,
    original_id text,
    original_translations_id text,
    CONSTRAINT test_phone_pkey PRIMARY KEY (id)
);

CREATE TABLE test.program (
    id character(36) NOT NULL,
    organization_id character(36) NOT NULL,
    name text NOT NULL,
    alternate_name text,
    last_modified timestamp without time zone,
    created timestamp without time zone,
    original_id text,
    original_translations_id text,
    CONSTRAINT test_program_pkey PRIMARY KEY (id)
);

CREATE TABLE test.required_document (
    id character(36) NOT NULL,
    service_id character(36) NOT NULL,
    document text NOT NULL,
    last_modified timestamp without time zone,
    created timestamp without time zone,
    original_id text,
    CONSTRAINT test_required_document_pkey PRIMARY KEY (id)
);

CREATE TABLE test.schedule (
    id character(36) NOT NULL,
    service_id character(36),
    location_id character(36),
    service_at_location_id character(36),
    valid_from date,
    valid_to date,
    dtstart date,
    freq text,
    interval integer,
    byday text,
    description text,
    last_modified timestamp without time zone,
    created timestamp without time zone,
    original_id text,
    CONSTRAINT test_schedule_pkey PRIMARY KEY (id)
);

CREATE TABLE test.service (
    id character(36) NOT NULL,
    organization_id character(36) NOT NULL,
    program_id character(36),
    name text NOT NULL,
    alternate_name text,
    description text,
    short_description text,
    url text,
    email text,
    status text NOT NULL,
    interpretation_services text,
    application_process text,
    wait_time text,
    fees_description text,
    accreditations text,
    licenses text,
    minimum_age integer,
    maximum_age integer,
    eligibility_description text,
    alert text,
    last_modified timestamp without time zone,
    created timestamp without time zone,
    original_id text,
    original_translations_id text,
    CONSTRAINT test_service_pkey PRIMARY KEY (id)
);

CREATE TABLE test.service_area (
    id character(36) NOT NULL,
    service_id character(36) NOT NULL,
    service_area character varying(255),
    description text,
    extent geometry,
    extent_type character varying(50),
    last_modified timestamp without time zone,
    created timestamp without time zone,
    original_id character varying(100),
    original_translations_id character varying(100),
    CONSTRAINT test_service_area_pkey PRIMARY KEY (id)
);

CREATE TABLE test.service_at_location (
    id character(36) NOT NULL,
    service_id character(36) NOT NULL,
    location_id character(36) NOT NULL,
    description text,
    last_modified timestamp without time zone,
    created timestamp without time zone,
    original_id text,
    original_translations_id text,
    CONSTRAINT test_service_at_location_pkey PRIMARY KEY (id)
);

CREATE TABLE test.taxonomy_term (
    id character(36) NOT NULL,
    term text NOT NULL,
    description text,
    parent_id character(36),
    taxonomy text,
    language text,
    last_modified timestamp without time zone,
    created timestamp without time zone,
    original_id text,
    CONSTRAINT test_taxonomy_term_pkey PRIMARY KEY (id)
);

CREATE TABLE test.service_taxonomy (
    id character(36) NOT NULL,
    service_id character(36) NOT NULL,
    taxonomy_term_id character(36) NOT NULL,
    last_modified timestamp without time zone,
    created timestamp without time zone,
    original_id text,
    CONSTRAINT test_service_taxonomy_pkey PRIMARY KEY (id)
);

-- Add Foreign Key constraints within test schema only
ALTER TABLE test.accessibility_for_disabilities
    ADD CONSTRAINT test_accessibility_for_disabilities_location_id_fkey
    FOREIGN KEY (location_id) REFERENCES test.location(id);

ALTER TABLE test.address
    ADD CONSTRAINT test_address_location_id_fkey
    FOREIGN KEY (location_id) REFERENCES test.location(id);

ALTER TABLE test.contact
    ADD CONSTRAINT test_contact_organization_id_fkey
    FOREIGN KEY (organization_id) REFERENCES test.organization(id),
    ADD CONSTRAINT test_contact_service_id_fkey
    FOREIGN KEY (service_id) REFERENCES test.service(id),
    ADD CONSTRAINT test_contact_service_at_location_id_fkey
    FOREIGN KEY (service_at_location_id) REFERENCES test.service_at_location(id);

ALTER TABLE test.funding
    ADD CONSTRAINT test_funding_organization_id_fkey
    FOREIGN KEY (organization_id) REFERENCES test.organization(id),
    ADD CONSTRAINT test_funding_service_id_fkey
    FOREIGN KEY (service_id) REFERENCES test.service(id);

ALTER TABLE test.language
    ADD CONSTRAINT test_language_location_id_fkey
    FOREIGN KEY (location_id) REFERENCES test.location(id),
    ADD CONSTRAINT test_language_service_id_fkey
    FOREIGN KEY (service_id) REFERENCES test.service(id);

ALTER TABLE test.location
    ADD CONSTRAINT test_location_organization_id_fkey
    FOREIGN KEY (organization_id) REFERENCES test.organization(id);

ALTER TABLE test.organization
    ADD CONSTRAINT test_organization_parent_organization_id_fkey
    FOREIGN KEY (parent_organization_id) REFERENCES test.organization(id);

ALTER TABLE test.other_attribute
    ADD CONSTRAINT test_other_attribute_taxonomy_term_id_fkey
    FOREIGN KEY (taxonomy_term_id) REFERENCES test.taxonomy_term(id);

ALTER TABLE test.phone
    ADD CONSTRAINT test_phone_location_id_fkey
    FOREIGN KEY (location_id) REFERENCES test.location(id),
    ADD CONSTRAINT test_phone_organization_id_fkey
    FOREIGN KEY (organization_id) REFERENCES test.organization(id),
    ADD CONSTRAINT test_phone_service_at_location_id_fkey
    FOREIGN KEY (service_at_location_id) REFERENCES test.service_at_location(id),
    ADD CONSTRAINT test_phone_service_id_fkey
    FOREIGN KEY (service_id) REFERENCES test.service(id);

ALTER TABLE test.program
    ADD CONSTRAINT test_program_organization_id_fkey
    FOREIGN KEY (organization_id) REFERENCES test.organization(id);

ALTER TABLE test.required_document
    ADD CONSTRAINT test_required_document_service_id_fkey
    FOREIGN KEY (service_id) REFERENCES test.service(id);

ALTER TABLE test.schedule
    ADD CONSTRAINT test_schedule_location_id_fkey
    FOREIGN KEY (location_id) REFERENCES test.location(id),
    ADD CONSTRAINT test_schedule_service_at_location_id_fkey
    FOREIGN KEY (service_at_location_id) REFERENCES test.service_at_location(id),
    ADD CONSTRAINT test_schedule_service_id_fkey
    FOREIGN KEY (service_id) REFERENCES test.service(id);

ALTER TABLE test.service
    ADD CONSTRAINT test_service_organization_id_fkey
    FOREIGN KEY (organization_id) REFERENCES test.organization(id);

ALTER TABLE test.service_area
    ADD CONSTRAINT test_service_area_service_id_fkey
    FOREIGN KEY (service_id) REFERENCES test.service(id);

ALTER TABLE test.service_at_location
    ADD CONSTRAINT test_service_at_location_location_id_fkey
    FOREIGN KEY (location_id) REFERENCES test.location(id),
    ADD CONSTRAINT test_service_at_location_service_id_fkey
    FOREIGN KEY (service_id) REFERENCES test.service(id);

ALTER TABLE test.service_taxonomy
    ADD CONSTRAINT test_service_taxonomy_service_id_fkey
    FOREIGN KEY (service_id) REFERENCES test.service(id),
    ADD CONSTRAINT test_service_taxonomy_taxonomy_term_id_fkey
    FOREIGN KEY (taxonomy_term_id) REFERENCES test.taxonomy_term(id);

ALTER TABLE test.taxonomy_term
    ADD CONSTRAINT test_taxonomy_term_parent_id_fkey
    FOREIGN KEY (parent_id) REFERENCES test.taxonomy_term(id);

-- Script to safely truncate only public schema tables
CREATE OR REPLACE FUNCTION truncate_public_schema() RETURNS void AS $$
DECLARE
    r RECORD;
BEGIN
    -- Disable foreign key constraints
    SET session_replication_role = 'replica';
    
    -- Loop through all tables in the public schema
    FOR r IN (
        SELECT tablename FROM pg_tables 
        WHERE schemaname = 'public'
        ORDER BY tablename
    )
    LOOP
        -- Skip PostGIS system tables if they exist
        IF r.tablename NOT IN ('spatial_ref_sys') THEN
            EXECUTE 'TRUNCATE TABLE public."' || r.tablename || '" RESTRICT';
        END IF;
    END LOOP;
    
    -- Re-enable foreign key constraints
    SET session_replication_role = 'origin';
END;
$$ LANGUAGE plpgsql;

-- Script to safely truncate only test schema tables
CREATE OR REPLACE FUNCTION truncate_test_schema() RETURNS void AS $$
DECLARE
    r RECORD;
BEGIN
    -- Disable foreign key constraints
    SET session_replication_role = 'replica';
    
    -- Loop through all tables in the test schema
    FOR r IN (
        SELECT tablename FROM pg_tables 
        WHERE schemaname = 'test'
        ORDER BY tablename
    )
    LOOP
        EXECUTE 'TRUNCATE TABLE test."' || r.tablename || '" RESTRICT';
    END LOOP;
    
    -- Re-enable foreign key constraints
    SET session_replication_role = 'origin';
END;
$$ LANGUAGE plpgsql;