import {
  SnowflakeService,
  SnowflakeServiceTranslation,
  PostgresService,
} from "../types";
import { IdConverter } from "../utils/uuid-utils";
import {
  Transformer,
  TransformOptions,
  RelationshipDefinition,
} from "./transformer";
import { v4 as uuidv4 } from "uuid";

export class ServiceTransformer extends Transformer<
  SnowflakeService,
  SnowflakeServiceTranslation,
  PostgresService
> {
  constructor(idConverter: IdConverter) {
    super(idConverter);
  }

  /**
   * Define relationships for service entity
   * Organization is the main foreign key relationship for services
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
   * Transform a batch of service records with optimized organization lookups
   * @param dataMap Map of source data and translations
   * @param batchSizeOrOptions Batch size or transformation options
   * @param pgSchema Database schema to use
   * @returns Array of transformed services
   */
  async transform(
    dataMap: Map<
      string,
      { main: SnowflakeService; translations: SnowflakeServiceTranslation[] }
    >,
    batchSizeOrOptions: number | TransformOptions = 100,
    pgSchema: string = "public"
  ): Promise<PostgresService[]> {
    // Handle legacy method signature for backward compatibility
    const options: TransformOptions =
      typeof batchSizeOrOptions === "number"
        ? { batchSize: batchSizeOrOptions, pgSchema, debug: true }
        : { ...batchSizeOrOptions, debug: true };

    console.log(
      `Starting to transform ${dataMap.size} service records using schema: ${
        options.pgSchema || pgSchema
      }`
    );

    // Call the base class implementation with our options
    const transformedRecords = await super.transform(dataMap, options);

    // Filter out records with empty organization IDs (same as original implementation)
    const validRecords = transformedRecords.filter(
      (record) => record.organization_id && record.organization_id.trim() !== ""
    );

    const invalidCount = transformedRecords.length - validRecords.length;
    if (invalidCount > 0) {
      console.warn(
        `Filtered out ${invalidCount} records with missing organization IDs`
      );
    }

    return validRecords;
  }

  /**
   * Post-process transformed records to filter out invalid records
   * This is called by the base class transform method
   */
  protected async postBatchProcess(
    records: PostgresService[]
  ): Promise<PostgresService[]> {
    // Additional post-processing could be added here if needed
    return records;
  }

  /**
   * Transform a single service record using resolved relationships
   * @param source Source service data
   * @param translation Service translation data
   * @param resolvedRelationships Map of resolved relationship IDs
   * @returns Transformed service record
   */
  protected transformSingleRecord(
    source: SnowflakeService,
    translation: SnowflakeServiceTranslation | null,
    resolvedRelationships?: Map<string, string>
  ): PostgresService {
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
      program_id: source.PROGRAM_ID
        ? this.idConverter.convertToUuid(source.PROGRAM_ID) ||
          source.PROGRAM_ID.toString()
        : undefined,
      name: translation?.NAME || "",
      alternate_name: translation?.ALTERNATE_NAME || undefined,
      description: translation?.DESCRIPTION || undefined,
      short_description: translation?.SHORT_DESCRIPTION || undefined,
      url: source.URL || undefined,
      email: source.EMAIL || undefined,
      status: source.STATUS,
      interpretation_services:
        translation?.INTERPRETATION_SERVICES || undefined,
      application_process: translation?.APPLICATION_PROCESS || undefined,
      wait_time: undefined,
      fees_description: translation?.FEES_DESCRIPTION || undefined,
      accreditations: translation?.ACCREDITATIONS || undefined,
      licenses: undefined,
      minimum_age: source.MINIMUM_AGE || undefined,
      maximum_age: source.MAXIMUM_AGE || undefined,
      eligibility_description:
        translation?.ELIGIBILITY_DESCRIPTION || undefined,
      alert: translation?.ALERT || undefined,
      last_modified: new Date(source.LAST_MODIFIED).toISOString(),
      created: new Date(source.CREATED).toISOString(),
      original_id:
        this.idConverter.convertToUuid(source.ID) || source.ID.toString(),
      original_translations_id: translation?.ID
        ? this.idConverter.convertToUuid(translation.ID)
        : undefined,
    };
  }
}
