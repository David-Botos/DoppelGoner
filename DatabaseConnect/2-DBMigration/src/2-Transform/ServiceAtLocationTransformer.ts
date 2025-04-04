import {
  SnowflakeServiceAtLocation,
  SnowflakeServiceAtLocationTranslation,
  MigratedData,
} from "../types";
import { IdConverter } from "../utils/uuid-utils";
import {
  Transformer,
  TransformOptions,
  RelationshipDefinition,
} from "./transformer";
import { v4 as uuidv4 } from "uuid";

// Define the Postgres target type
export interface PostgresServiceAtLocation extends MigratedData {
  id: string;
  service_id: string;
  location_id: string;
  description?: string;
  last_modified: string;
  created: string;
  original_id: string;
  original_translations_id?: string;
}

export class ServiceAtLocationTransformer extends Transformer<
  SnowflakeServiceAtLocation,
  SnowflakeServiceAtLocationTranslation,
  PostgresServiceAtLocation
> {
  constructor(idConverter: IdConverter) {
    super(idConverter);
  }

  /**
   * Define relationships for service_at_location entity
   * Service and Location are the main foreign keys
   */
  protected getRelationships(): RelationshipDefinition[] {
    return [
      {
        fieldName: "service_id",
        sourceField: "SERVICE_ID",
        tableName: "service",
        idField: "original_id",
        transformId: (id) => id?.toString() || null,
      },
      {
        fieldName: "location_id",
        sourceField: "LOCATION_ID",
        tableName: "location",
        idField: "original_id",
        transformId: (id) => id?.toString() || null,
      },
    ];
  }

  /**
   * Post-process transformed records to filter out invalid records
   * This is called by the base class transform method
   */
  protected async postBatchProcess(
    records: PostgresServiceAtLocation[]
  ): Promise<PostgresServiceAtLocation[]> {
    // Track invalid records (those missing required service_id or location_id)
    records.forEach((record) => {
      if (!record.service_id || record.service_id.trim() === "") {
        this.addInvalidRecord(record, "Missing required service_id");
      }
      if (!record.location_id || record.location_id.trim() === "") {
        this.addInvalidRecord(record, "Missing required location_id");
      }
    });

    // Only return valid records after tracking invalid ones
    const validRecords = records.filter(
      (record) =>
        record.service_id &&
        record.service_id.trim() !== "" &&
        record.location_id &&
        record.location_id.trim() !== ""
    );

    const invalidCount = records.length - validRecords.length;
    if (invalidCount > 0) {
      console.warn(
        `Found ${invalidCount} records with missing required IDs (tracked in invalidRecords)`
      );
    }

    return validRecords;
  }

  /**
   * Transform a single service_at_location record using resolved relationships
   * @param source Source service_at_location data
   * @param translation Service_at_location translation data
   * @param resolvedRelationships Map of resolved relationship IDs
   * @returns Transformed service_at_location record
   */
  protected transformSingleRecord(
    source: SnowflakeServiceAtLocation,
    translation: SnowflakeServiceAtLocationTranslation | null,
    resolvedRelationships?: Map<string, string>
  ): PostgresServiceAtLocation {
    const newId = uuidv4();

    // Get service and location IDs from resolved relationships
    const serviceId = resolvedRelationships?.get("service_id") || "";
    const locationId = resolvedRelationships?.get("location_id") || "";

    // Log warnings if IDs are missing
    if (!serviceId && source.SERVICE_ID) {
      console.warn(
        `No PostgreSQL service ID found for Snowflake service ID: ${source.SERVICE_ID}`
      );
    }

    if (!locationId && source.LOCATION_ID) {
      console.warn(
        `No PostgreSQL location ID found for Snowflake location ID: ${source.LOCATION_ID}`
      );
    }

    return {
      id: newId,
      service_id: serviceId,
      location_id: locationId,
      description: translation?.DESCRIPTION || undefined,
      last_modified: source.LAST_MODIFIED
        ? new Date(source.LAST_MODIFIED).toISOString()
        : new Date().toISOString(),
      created: source.CREATED
        ? new Date(source.CREATED).toISOString()
        : new Date().toISOString(),
      original_id:
        this.idConverter.convertToUuid(source.ID) || source.ID.toString(),
      original_translations_id: translation?.ID
        ? this.idConverter.convertToUuid(translation.ID) || undefined
        : undefined,
    };
  }
}
