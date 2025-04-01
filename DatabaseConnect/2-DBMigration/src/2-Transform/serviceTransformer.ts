import { Transformer } from "./transformer";
import {
  SnowflakeService,
  SnowflakeServiceTranslation,
  PostgresOrganization,
  PostgresService,
} from "../types";
import { IdConverter } from "../utils/uuid-utils";
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
   * Transform a batch of service records with optimized organization lookups
   * @param dataMap Map of source data and translations
   * @param batchSize Number of records to process in each batch
   * @returns Array of transformed services
   */
  async transform(
    dataMap: Map<
      string,
      { main: SnowflakeService; translations: SnowflakeServiceTranslation[] }
    >,
    batchSize: number = 100
  ): Promise<PostgresService[]> {
    const entries = Array.from(dataMap.entries());
    const transformedRecords: PostgresService[] = [];

    // Process records in batches
    for (let i = 0; i < entries.length; i += batchSize) {
      const batch = entries.slice(i, i + batchSize);
      console.log(
        `Processing batch ${i / batchSize + 1} of ${Math.ceil(
          entries.length / batchSize
        )}`
      );

      // Collect all organization IDs in this batch for a single lookup
      const orgIds = batch
        .map(([_, { main }]) =>
          main.ORGANIZATION_ID ? main.ORGANIZATION_ID.toString() : null
        )
        .filter(Boolean) as string[];

      // Batch lookup organization IDs
      const orgIdMap = await this.batchLookupIds(
        "organization_id",
        orgIds,
        "organization"
      );

      // Process each record in the batch with the pre-fetched organization data
      const batchResults = await Promise.all(
        batch.map(async ([_, { main, translations }]) => {
          // For each service, get the first translation
          const translation = translations.length > 0 ? translations[0] : null;

          // Create the transformed record with organization info from the batch lookup
          const originalOrgId = main.ORGANIZATION_ID
            ? main.ORGANIZATION_ID.toString()
            : "";
          const newOrgFKId = originalOrgId
            ? orgIdMap.get(originalOrgId) || ""
            : "";

          return this.transformSingleRecordWithOrgId(
            main,
            translation,
            newOrgFKId
          );
        })
      );

      transformedRecords.push(...batchResults);
    }

    return transformedRecords;
  }

  /**
   * Transform single record with pre-fetched organization ID
   */
  private async transformSingleRecordWithOrgId(
    source: SnowflakeService,
    translation: SnowflakeServiceTranslation | null,
    organizationId: string
  ): Promise<PostgresService> {
    const newId = uuidv4();

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

  /**
   * Original method required for implementation, but superseded by the optimized transform method
   */
  protected async transformSingleRecord(
    source: SnowflakeService,
    translation: SnowflakeServiceTranslation | null
  ): Promise<PostgresService> {
    const newId = uuidv4();

    // Get organization by original ID
    const orgQuery = `SELECT id, original_id FROM organization WHERE original_id = $1 LIMIT 1`;
    const orgResults = await this.postgresClient.executeQuery(orgQuery, [
      source.ORGANIZATION_ID.toString(),
    ]);

    let newOrgFKId = "";
    if (orgResults.length > 0) {
      newOrgFKId = orgResults[0].id;
    }

    return {
      id: newId,
      organization_id: newOrgFKId,
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
