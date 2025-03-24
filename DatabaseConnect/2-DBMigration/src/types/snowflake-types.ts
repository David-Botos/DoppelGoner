// Type definitions for Snowflake database tables

// Organization table types
export interface SnowflakeOrganization {
    ID: string;
    NAME: string;
    ALTERNATE_NAME: string | null;
    EMAIL: string | null;
    WEBSITE: string | null;
    TAX_STATUS: string | null;
    TAX_ID: string | null;
    YEAR_INCORPORATED: string | null;
    LEGAL_STATUS: string | null;
    PARENT_ORGANIZATION_ID: string | null;
    LAST_MODIFIED: string;
    CREATED: string;
  }
  
  export interface SnowflakeOrganizationTranslation {
    ID: string;
    ORGANIZATION_ID: string;
    LOCALE: string;
    DESCRIPTION: string | null;
    IS_CANONICAL: boolean;
    LAST_MODIFIED: string;
    CREATED: string;
  }
  
  // Location table types
  export interface SnowflakeLocation {
    ID: string;
    ORGANIZATION_ID: string;
    NAME: string | null;
    ALTERNATE_NAME: string | null;
    LATITUDE: number | null;
    LONGITUDE: number | null;
    LOCATION_TYPE: string | null;
    LAST_MODIFIED: string;
    CREATED: string;
  }
  
  export interface SnowflakeLocationTranslation {
    ID: string;
    LOCATION_ID: string;
    LOCALE: string;
    DESCRIPTION: string | null;
    SHORT_DESCRIPTION: string | null;
    TRANSPORTATION: string | null;
    IS_CANONICAL: boolean;
    LAST_MODIFIED: string;
    CREATED: string;
  }
  
  // Service table types
  export interface SnowflakeService {
    ID: string;
    ORGANIZATION_ID: string;
    PROGRAM_ID: string | null;
    URL: string | null;
    EMAIL: string | null;
    STATUS: string;
    MINIMUM_AGE: number | null;
    MAXIMUM_AGE: number | null;
    LAST_MODIFIED: string;
    CREATED: string;
  }
  
  export interface SnowflakeServiceTranslation {
    ID: string;
    SERVICE_ID: string;
    LOCALE: string;
    NAME: string | null;
    ALTERNATE_NAME: string | null;
    DESCRIPTION: string | null;
    SHORT_DESCRIPTION: string | null;
    INTERPRETATION_SERVICES: string | null;
    APPLICATION_PROCESS: string | null;
    FEES_DESCRIPTION: string | null;
    ACCREDITATIONS: string | null;
    ELIGIBILITY_DESCRIPTION: string | null;
    ALERT: string | null;
    IS_CANONICAL: boolean;
    LAST_MODIFIED: string;
    CREATED: string;
  }
  
  // ServiceAtLocation table types
  export interface SnowflakeServiceAtLocation {
    ID: string;
    SERVICE_ID: string;
    LOCATION_ID: string;
    LAST_MODIFIED: string;
    CREATED: string;
  }
  
  export interface SnowflakeServiceAtLocationTranslation {
    ID: string;
    SERVICE_AT_LOCATION_ID: string;
    LOCALE: string;
    DESCRIPTION: string | null;
    IS_CANONICAL: boolean;
    LAST_MODIFIED: string;
    CREATED: string;
  }
  
  // Address table types
  export interface SnowflakeAddress {
    ID: string;
    LOCATION_ID: string;
    ATTENTION: string | null;
    ADDRESS_1: string;
    ADDRESS_2: string | null;
    CITY: string;
    REGION: string | null;
    STATE_PROVINCE: string;
    POSTAL_CODE: string;
    COUNTRY: string;
    ADDRESS_TYPE: string;
    LAST_MODIFIED: string;
    CREATED: string;
  }
  
  // Phone table types
  export interface SnowflakePhone {
    ID: string;
    LOCATION_ID: string | null;
    SERVICE_ID: string | null;
    ORGANIZATION_ID: string | null;
    CONTACT_ID: string | null;
    SERVICE_AT_LOCATION_ID: string | null;
    NUMBER: string;
    EXTENSION: string | null;
    TYPE: string | null;
    PRIORITY: number | null;
    LAST_MODIFIED: string;
    CREATED: string;
  }
  
  export interface SnowflakePhoneTranslation {
    ID: string;
    PHONE_ID: string;
    LOCALE: string;
    DESCRIPTION: string | null;
    IS_CANONICAL: boolean;
    LAST_MODIFIED: string;
    CREATED: string;
  }