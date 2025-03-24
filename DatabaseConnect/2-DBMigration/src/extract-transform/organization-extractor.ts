import { SnowflakeClient } from "../services/snowflake-client";
import {
  SnowflakeOrganization,
  SnowflakeOrganizationTranslation,
  SupabaseOrganization,
} from "../types";
import { BaseExtractor } from "./base-extractor";

export class OrganizationExtractor extends BaseExtractor<
  SnowflakeOrganization,
  SnowflakeOrganizationTranslation,
  SupabaseOrganization
> {
  constructor(snowflakeClient: SnowflakeClient) {
    super(
      snowflakeClient,
      "ORGANIZATION",
      "ORGANIZATION_TRANSLATIONS",
      "ORGANIZATION_ID"
    );
  }

  public async extractBatch(
    offset: number,
    limit: number
  ): Promise<SnowflakeOrganization[]> {
    return this.dataService.fetchBatch<SnowflakeOrganization>(
      this.tableName,
      offset,
      limit
    );
  }

  public async extractTranslations(
    ids: string[],
    locale: string = "en"
  ): Promise<SnowflakeOrganizationTranslation[]> {
    if (ids.length === 0) return [];

    // Format IDs for SQL IN clause
    const formattedIds = ids.map((id) => `'${id}'`).join(",");

    const query = `
            SELECT * FROM ${this.translationTableName}
            WHERE ${this.foreignKeyField} IN (${formattedIds})
                AND LOCALE = '${locale}'
                AND IS_CANONICAL = TRUE
        `;

    return this.dataService.query<SnowflakeOrganizationTranslation>(query);
  }

  public joinTranslations(
    records: SnowflakeOrganization[],
    translations: SnowflakeOrganizationTranslation[]
  ): SupabaseOrganization[] {
    if (translations.length === 0) {
      // Convert records to Supabase format without translations
      return records.map((record) => this.transformToSupabaseFormat(record));
    }

    // Create a map for quick lookup of translations by organization ID
    const translationMap = new Map<string, SnowflakeOrganizationTranslation>();
    for (const translation of translations) {
      translationMap.set(translation.ORGANIZATION_ID, translation);
    }

    // Join translations with records and transform to Supabase format
    return records.map((record) => {
      const translation = translationMap.get(record.ID);
      return this.transformToSupabaseFormat(record, translation);
    });
  }

  private transformToSupabaseFormat(
    record: SnowflakeOrganization,
    translation?: SnowflakeOrganizationTranslation
  ): SupabaseOrganization {
    return {
      id: record.ID,
      name: record.NAME,
      alternate_name: record.ALTERNATE_NAME || undefined,
      description: translation?.DESCRIPTION || undefined,
      email: record.EMAIL || undefined,
      url: record.WEBSITE || undefined,
      tax_status: record.TAX_STATUS || undefined,
      tax_id: record.TAX_ID || undefined,
      year_incorporated: record.YEAR_INCORPORATED || undefined,
      legal_status: record.LEGAL_STATUS || undefined,
      parent_organization_id: record.PARENT_ORGANIZATION_ID || undefined,
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
