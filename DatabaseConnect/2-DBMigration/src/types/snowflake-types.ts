// Type definitions for Snowflake database tables
import { SourceData, SourceDataTranslations } from "./transformation-types";

// Organization table types
export interface SnowflakeOrganization extends SourceData {
  NAME: string;
  ALTERNATE_NAME: string | null;
  EMAIL: string | null;
  WEBSITE: string | null;
  YEAR_INCORPORATED: string | null;
  LEGAL_STATUS: string | null;
  PARENT_ORGANIZATION_ID: string | null;
  LAST_MODIFIED: string;
  CREATED: string;
}

export interface SnowflakeOrganizationTranslation
  extends SourceDataTranslations {
  ORGANIZATION_ID: string;
  LOCALE: string;
  DESCRIPTION: string | null;
  IS_CANONICAL: boolean;
}

// Location table types
export interface SnowflakeLocation extends SourceData {
  ORGANIZATION_ID: string;
  NAME: string | null;
  ALTERNATE_NAME: string | null;
  LATITUDE: number | null;
  LONGITUDE: number | null;
  LOCATION_TYPE: string | null;
  LAST_MODIFIED: string;
  CREATED: string;
}

export interface SnowflakeLocationTranslation extends SourceDataTranslations {
  LOCATION_ID: string;
  LOCALE: string;
  DESCRIPTION: string | null;
  SHORT_DESCRIPTION: string | null;
  TRANSPORTATION: string | null;
  IS_CANONICAL: boolean;
}

// Service table types
export interface SnowflakeService extends SourceData {
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

export interface SnowflakeServiceTranslation extends SourceDataTranslations {
  SERVICE_ID: string;
  LOCALE: string;
  NAME: string;
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
}

// ServiceAtLocation table types
export interface SnowflakeServiceAtLocation extends SourceData {
  SERVICE_ID: string;
  LOCATION_ID: string;
  LAST_MODIFIED: string;
  CREATED: string;
}

export interface SnowflakeServiceAtLocationTranslation
  extends SourceDataTranslations {
  SERVICE_AT_LOCATION_ID: string;
  LOCALE: string;
  DESCRIPTION: string | null;
  IS_CANONICAL: boolean;
}

// Address table types
export interface SnowflakeAddress extends SourceData {
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
export interface SnowflakePhone extends SourceData {
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

export interface SnowflakePhoneTranslation extends SourceDataTranslations {
  PHONE_ID: string;
  LOCALE: string;
  DESCRIPTION: string | null;
  IS_CANONICAL: boolean;
}
