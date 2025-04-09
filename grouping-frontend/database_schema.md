Query #1:
schema   | table_name            | column_name                | data_type                     | nullable   | constraint_type   | default_value            
---------+-----------------------+----------------------------+-------------------------------+------------+-------------------+--------------------------
public   | address               | id                         | character(36)                 | NOT NULL   | PRIMARY KEY       | NULL           
public   | address               | location_id                | character(36)                 | NOT NULL   | FOREIGN KEY       | NULL           
public   | address               | attention                  | text                          | NULL       | NULL    | NULL           
public   | address               | address_1                  | text                          | NOT NULL   | NULL    | NULL           
public   | address               | address_2                  | text                          | NULL       | NULL    | NULL           
public   | address               | city                       | text                          | NOT NULL   | NULL    | NULL           
public   | address               | region                     | text                          | NULL       | NULL    | NULL           
public   | address               | state_province             | text                          | NOT NULL   | NULL    | NULL           
public   | address               | postal_code                | text                          | NOT NULL   | NULL    | NULL           
public   | address               | country                    | character(2)                  | NOT NULL   | NULL    | NULL           
public   | address               | address_type               | text                          | NOT NULL   | NULL    | NULL           
public   | address               | last_modified              | timestamp without time zone   | NULL       | NULL    | NULL           
public   | address               | created                    | timestamp without time zone   | NULL       | NULL    | NULL           
public   | address               | original_id                | text                          | NULL       | NULL    | NULL           
public   | cluster_entities      | id                         | uuid                          | NOT NULL   | PRIMARY KEY       | gen_random_uuid()        
public   | cluster_entities      | cluster_id                 | uuid                          | NOT NULL   | FOREIGN KEY       | NULL                      
public   | cluster_entities      | entity_type                | text                          | NOT NULL   | UNIQUE            | NULL           
public   | cluster_entities      | entity_id                  | text                          | NOT NULL   | UNIQUE            | NULL           
public   | cluster_entities      | created_at                 | timestamp with time zone      | NOT NULL   | NULL    | now()                    
public   | location              | id                         | character(36)                 | NOT NULL   | PRIMARY KEY       | NULL           
public   | location              | organization_id            | character(36)                 | NOT NULL   | FOREIGN KEY       | NULL           
public   | location              | name                       | text                          | NULL       | NULL    | NULL           
public   | location              | alternate_name             | text                          | NULL       | NULL    | NULL           
public   | location              | description                | text                          | NULL       | NULL    | NULL           
public   | location              | short_description          | text                          | NULL       | NULL    | NULL           
public   | location              | transportation             | text                          | NULL       | NULL    | NULL           
public   | location              | latitude                   | numeric(10,6)                 | NULL       | NULL    | NULL           
public   | location              | longitude                  | numeric(10,6)                 | NULL       | NULL    | NULL           
public   | location              | location_type              | text                          | NULL       | NULL    | NULL           
public   | location              | last_modified              | timestamp without time zone   | NULL       | NULL    | NULL           
public   | location              | created                    | timestamp without time zone   | NULL       | NULL    | NULL           
public   | location              | original_id                | text                          | NULL       | NULL    | NULL           
public   | location              | original_translations_id   | text                          | NULL       | NULL    | NULL           
public   | match_clusters        | id                         | uuid                          | NOT NULL   | PRIMARY KEY       | gen_random_uuid()        
public   | match_clusters        | confidence                 | real                          | NOT NULL   | NULL    | NULL           
public   | match_clusters        | notes                      | text                          | NULL       | NULL    | NULL           
public   | match_clusters        | reasoning                  | text                          | NULL       | NULL    | NULL           
public   | match_clusters        | is_reviewed                | boolean                       | NOT NULL   | NULL    | false                    
public   | match_clusters        | review_result              | boolean                       | NULL       | NULL    | NULL           
public   | match_clusters        | reviewed_by                | text                          | NULL       | NULL    | NULL           
public   | match_clusters        | reviewed_at                | timestamp with time zone      | NULL       | NULL    | NULL           
public   | match_clusters        | created_at                 | timestamp with time zone      | NOT NULL   | NULL    | now()                    
public   | match_clusters        | updated_at                 | timestamp with time zone      | NOT NULL   | NULL    | now()                    
public   | matching_methods      | id                         | uuid                          | NOT NULL   | PRIMARY KEY       | gen_random_uuid()        
public   | matching_methods      | cluster_id                 | uuid                          | NOT NULL   | FOREIGN KEY       | NULL           
public   | matching_methods      | method_name                | text                          | NOT NULL   | NULL    | NULL           
public   | matching_methods      | confidence                 | real                          | NOT NULL   | NULL    | NULL           
public   | matching_methods      | details                    | jsonb                         | NULL       | NULL    | NULL           
public   | matching_methods      | created_at                 | timestamp with time zone      | NOT NULL   | NULL    | now()                    
public   | organization          | id                         | character(36)                 | NOT NULL   | PRIMARY KEY       | NULL           
public   | organization          | name                       | text                          | NOT NULL   | NULL    | NULL           
public   | organization          | alternate_name             | text                          | NULL       | NULL    | NULL           
public   | organization          | description                | text                          | NULL       | NULL    | NULL           
public   | organization          | email                      | text                          | NULL       | NULL    | NULL           
public   | organization          | url                        | text                          | NULL       | NULL    | NULL           
public   | organization          | tax_status                 | text                          | NULL       | NULL    | NULL           
public   | organization          | tax_id                     | text                          | NULL       | NULL    | NULL           
public   | organization          | year_incorporated          | character(4)                  | NULL       | NULL    | NULL           
public   | organization          | legal_status               | text                          | NULL       | NULL    | NULL           
public   | organization          | parent_organization_id     | character(36)                 | NULL       | FOREIGN KEY       | NULL           
public   | organization          | last_modified              | timestamp without time zone   | NULL       | NULL    | NULL           
public   | organization          | created                    | timestamp without time zone   | NULL       | NULL    | NULL           
public   | organization          | original_id                | text                          | NULL       | NULL    | NULL           
public   | organization          | original_translations_id   | text                          | NULL       | NULL    | NULL           
public   | phone                 | id                         | character(36)                 | NOT NULL   | PRIMARY KEY       | NULL           
public   | phone                 | location_id                | character(36)                 | NULL       | FOREIGN KEY       | NULL           
public   | phone                 | service_id                 | character(36)                 | NULL       | FOREIGN KEY       | NULL           
public   | phone                 | organization_id            | character(36)                 | NULL       | FOREIGN KEY       | NULL           
public   | phone                 | contact_id                 | character(36)                 | NULL       | NULL    | NULL           
public   | phone                 | service_at_location_id     | character(36)                 | NULL       | FOREIGN KEY       | NULL           
public   | phone                 | number                     | text                          | NOT NULL   | NULL    | NULL           
public   | phone                 | extension                  | text                          | NULL       | NULL    | NULL           
public   | phone                 | type                       | text                          | NULL       | NULL    | NULL           
public   | phone                 | language                   | text                          | NULL       | NULL    | 'en'::character varying  
public   | phone                 | description                | text                          | NULL       | NULL    | NULL           
public   | phone                 | priority                   | integer                       | NULL       | NULL    | NULL           
public   | phone                 | last_modified              | timestamp without time zone   | NULL       | NULL    | NULL           
public   | phone                 | created                    | timestamp without time zone   | NULL       | NULL    | NULL           
public   | phone                 | original_id                | text                          | NULL       | NULL    | NULL           
public   | phone                 | original_translations_id   | text                          | NULL       | NULL    | NULL           
public   | service               | id                         | character(36)                 | NOT NULL   | PRIMARY KEY       | NULL           
public   | service               | organization_id            | character(36)                 | NOT NULL   | FOREIGN KEY       | NULL           
public   | service               | program_id                 | character(36)                 | NULL       | NULL    | NULL           
public   | service               | name                       | text                          | NOT NULL   | NULL    | NULL           
public   | service               | alternate_name             | text                          | NULL       | NULL    | NULL           
public   | service               | description                | text                          | NULL       | NULL    | NULL           
public   | service               | short_description          | text                          | NULL       | NULL    | NULL           
public   | service               | url                        | text                          | NULL       | NULL    | NULL           
public   | service               | email                      | text                          | NULL       | NULL    | NULL           
public   | service               | status                     | text                          | NOT NULL   | NULL    | NULL           
public   | service               | interpretation_services    | text                          | NULL       | NULL    | NULL           
public   | service               | application_process        | text                          | NULL       | NULL    | NULL           
public   | service               | wait_time                  | text                          | NULL       | NULL    | NULL           
public   | service               | fees_description           | text                          | NULL       | NULL    | NULL           
public   | service               | accreditations             | text                          | NULL       | NULL    | NULL           
public   | service               | licenses                   | text                          | NULL       | NULL    | NULL           
public   | service               | minimum_age                | integer                       | NULL       | NULL    | NULL           
public   | service               | maximum_age                | integer                       | NULL       | NULL    | NULL           
public   | service               | eligibility_description    | text                          | NULL       | NULL    | NULL           
public   | service               | alert                      | text                          | NULL       | NULL    | NULL           
public   | service               | last_modified              | timestamp without time zone   | NULL       | NULL    | NULL           
public   | service               | created                    | timestamp without time zone   | NULL       | NULL    | NULL           
public   | service               | original_id                | text                          | NULL       | NULL    | NULL           
public   | service               | original_translations_id   | text                          | NULL       | NULL    | NULL           
public   | service_at_location   | id                         | character(36)                 | NOT NULL   | PRIMARY KEY       | NULL           
public   | service_at_location   | service_id                 | character(36)                 | NOT NULL   | FOREIGN KEY       | NULL           
public   | service_at_location   | location_id                | character(36)                 | NOT NULL   | FOREIGN KEY       | NULL           
public   | service_at_location   | description                | text                          | NULL       | NULL    | NULL           
public   | service_at_location   | last_modified              | timestamp without time zone   | NULL       | NULL    | NULL           
public   | service_at_location   | created                    | timestamp without time zone   | NULL       | NULL    | NULL           
public   | service_at_location   | original_id                | text                          | NULL       | NULL    | NULL           
public   | service_at_location   | original_translations_id   | text                          | NULL       | NULL    | NULL           