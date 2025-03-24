import { SnowflakeClient } from "../services/snowflake-client";
import { SnowflakeService, SnowflakeServiceTranslation } from "../types";
import { SupabaseService } from "../types";
import { BaseExtractor } from "./base-extractor";

export class ServiceExtractor extends BaseExtractor<
  SnowflakeService,
  SnowflakeServiceTranslation,
  SupabaseService
> {
  constructor(snowflakeClient: SnowflakeClient) {
    super(snowflakeClient, "SERVICE", "SERVICE_TRANSLATIONS", "SERVICE_ID");
  }

  public async extractBatch(
    offset: number,
    limit: number
  ): Promise<SnowflakeService[]> {
    return this.dataService.fetchBatch<SnowflakeService>(
      this.tableName,
      offset,
      limit
    );
  }

  public async extractTranslations(
    ids: string[],
    locale: string = "en"
  ): Promise<SnowflakeServiceTranslation[]> {
    if (ids.length === 0) return [];

    // Format IDs for SQL IN clause
    const formattedIds = ids.map((id) => `'${id}'`).join(",");

    const query = `
            SELECT * FROM ${this.translationTableName}
            WHERE ${this.foreignKeyField} IN (${formattedIds})
                AND LOCALE = '${locale}'
                AND IS_CANONICAL = TRUE
        `;

    return this.dataService.query<SnowflakeServiceTranslation>(query);
  }

  public joinTranslations(
    records: SnowflakeService[],
    translations: SnowflakeServiceTranslation[]
  ): SupabaseService[] {
    if (translations.length === 0) {
      return records.map((record) => this.mapToSupabase(record));
    }

    // Create a map for quick lookup of translations by service ID
    const translationMap = new Map<string, SnowflakeServiceTranslation>();

    for (const translation of translations) {
      translationMap.set(translation.SERVICE_ID, translation);
    }

    // Join translations with records and transform to Supabase format
    return records.map((record) => {
      const translation = translationMap.get(record.ID);
      return this.mapToSupabase(record, translation);
    });
  }

  private mapToSupabase(
    record: SnowflakeService,
    translation?: SnowflakeServiceTranslation
  ): SupabaseService {
    return {
      id: record.ID, // Will need UUID conversion in full implementation
      organization_id: record.ORGANIZATION_ID, // Will need UUID conversion
      program_id: record.PROGRAM_ID || undefined,
      name: translation?.NAME || "",
      alternate_name: translation?.ALTERNATE_NAME || undefined,
      description: translation?.DESCRIPTION || undefined,
      short_description: translation?.SHORT_DESCRIPTION || undefined,
      url: record.URL || undefined,
      email: record.EMAIL || undefined,
      status: record.STATUS,
      interpretation_services:
        translation?.INTERPRETATION_SERVICES || undefined,
      application_process: translation?.APPLICATION_PROCESS || undefined,
      wait_time: undefined, // New field per HSDS
      fees_description: translation?.FEES_DESCRIPTION || undefined,
      accreditations: translation?.ACCREDITATIONS || undefined,
      licenses: undefined, // New field per HSDS
      minimum_age: record.MINIMUM_AGE || undefined,
      maximum_age: record.MAXIMUM_AGE || undefined,
      eligibility_description:
        translation?.ELIGIBILITY_DESCRIPTION || undefined,
      alert: translation?.ALERT || undefined,
      last_modified: record.LAST_MODIFIED,
      created: record.CREATED,
      original_id: record.ID,
      original_translations_id: translation?.ID,
    };
  }

  public async countRecords(filter?: string): Promise<number> {
    return this.dataService.countRecords(this.tableName, filter);
  }
}
