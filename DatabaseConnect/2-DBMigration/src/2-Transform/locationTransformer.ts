import {
  SnowflakeLocation,
  SnowflakeLocationTranslation,
  PostgresLocation,
} from "../types";
import { IdConverter } from "../utils/uuid-utils";
import {
  Transformer,
  TransformOptions,
  RelationshipDefinition,
} from "./transformer";
import { v4 as uuidv4 } from "uuid";

export class LocationTransformer extends Transformer<
  SnowflakeLocation,
  SnowflakeLocationTranslation,
  PostgresLocation
> {
  constructor(idConverter: IdConverter) {
    super(idConverter);
  }

  /**
   * Define relationships for location entity
   * Organization is the main foreign key relationship for locations
   */
  protected getRelationships(): RelationshipDefinition[] {
    return [
      {
        fieldName: "organization_id",
        sourceField: "ORGANIZATION_ID",
        tableName: "organization",
        idField: "original_id",
        transformId: (id) => id?.toString() || null,
      },
    ];
  }

  /**
   * Transform a batch of location records with optimized organization lookups
   * @param dataMap Map of source data and translations
   * @param batchSizeOrOptions Batch size or transformation options
   * @param pgSchema Database schema to use
   * @returns Array of transformed locations
   */
  async transform(
    dataMap: Map<
      string,
      { main: SnowflakeLocation; translations: SnowflakeLocationTranslation[] }
    >,
    batchSizeOrOptions: number | TransformOptions = 100,
    pgSchema: string = "public"
  ): Promise<PostgresLocation[]> {
    // Handle legacy method signature for backward compatibility
    const options: TransformOptions =
      typeof batchSizeOrOptions === "number"
        ? { batchSize: batchSizeOrOptions, pgSchema, debug: true }
        : { ...batchSizeOrOptions, debug: true };

    console.log(
      `Starting to transform ${dataMap.size} location records using schema: ${
        options.pgSchema || pgSchema
      }`
    );

    // Call the base class implementation with our options
    const transformedRecords = await super.transform(dataMap, options);

    // Instead of silently filtering, check each record and track invalid ones
    transformedRecords.forEach((record) => {
      if (!record.organization_id || record.organization_id.trim() === "") {
        this.addInvalidRecord(record, "Missing required organization_id");
      }
    });

    // Only return valid records after tracking invalid ones
    const validRecords = transformedRecords.filter(
      (record) => record.organization_id && record.organization_id.trim() !== ""
    );

    const invalidCount = transformedRecords.length - validRecords.length;
    if (invalidCount > 0) {
      console.warn(
        `Found ${invalidCount} records with missing organization IDs (tracked in invalidRecords)`
      );
    }

    return validRecords;
  }

  /**
   * Post-process transformed records to filter out invalid records
   * This is called by the base class transform method
   */
  protected async postBatchProcess(
    records: PostgresLocation[]
  ): Promise<PostgresLocation[]> {
    // Additional post-processing could be added here if needed
    return records;
  }

  /**
   * Transform a single location record using resolved relationships
   * @param source Source location data
   * @param translation Location translation data
   * @param resolvedRelationships Map of resolved relationship IDs
   * @returns Transformed location record
   */
  protected transformSingleRecord(
    source: SnowflakeLocation,
    translation: SnowflakeLocationTranslation | null,
    resolvedRelationships?: Map<string, string>
  ): PostgresLocation {
    const newId = uuidv4();

    // Get organization ID from resolved relationships if available
    const organizationId = resolvedRelationships?.get("organization_id") || "";

    // Log warning if organization ID is missing
    if (!organizationId && source.ORGANIZATION_ID) {
      console.warn(
        `No PostgreSQL organization ID found for Snowflake organization ID: ${source.ORGANIZATION_ID}`
      );
    }

    return {
      id: newId,
      organization_id: organizationId,
      name: source.NAME || undefined,
      alternate_name: source.ALTERNATE_NAME || undefined,
      description: translation?.DESCRIPTION || undefined,
      short_description: translation?.SHORT_DESCRIPTION || undefined,
      transportation: translation?.TRANSPORTATION || undefined,
      latitude: source.LATITUDE || undefined,
      longitude: source.LONGITUDE || undefined,
      location_type: source.LOCATION_TYPE || undefined,
      last_modified: new Date(source.LAST_MODIFIED).toISOString(),
      created: new Date(source.CREATED).toISOString(),
      original_id:
        this.idConverter.convertToUuid(source.ID) || source.ID.toString(),
      original_translations_id: translation?.ID
        ? this.idConverter.convertToUuid(translation.ID)
        : null,
    };
  }
}
