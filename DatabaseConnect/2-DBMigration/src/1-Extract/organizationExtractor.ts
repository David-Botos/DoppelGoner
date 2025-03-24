import { SnowflakeClient } from "../services/snowflake-client";
import {
  SnowflakeOrganization,
  SnowflakeOrganizationTranslation,
  SourceData,
  SourceDataTranslations,
} from "../types";
import { Extractor } from "./extractor";

export class OrganizationExtractor extends Extractor<
  SnowflakeOrganization,
  SnowflakeOrganizationTranslation
> {
  constructor(protected snowflakeClient: SnowflakeClient) {
    super(snowflakeClient, {
      main: "ORGANIZATION",
      translations: "ORGANIZATION_TRANSLATIONS",
    });
  }

  async extractMainRecords(
    limit: number,
    offset: number
  ): Promise<SnowflakeOrganization[]> {
    const query = `
      SELECT 
        ID as id,
        NAME,
        ALTERNATE_NAME,
        EMAIL,
        WEBSITE,
        TAX_STATUS,
        TAX_ID,
        YEAR_INCORPORATED,
        LEGAL_STATUS,
        PARENT_ORGANIZATION_ID,
        LAST_MODIFIED,
        CREATED
      FROM ${this.sourceTables.main}
      ORDER BY CREATED DESC
      LIMIT ${limit}
      OFFSET ${offset}
    `;

    return this.snowflakeClient.query<SnowflakeOrganization>(query);
  }

  async extractTranslationRecords(
    ids: string[],
    locale: string
  ): Promise<SnowflakeOrganizationTranslation[]> {
    // Format the IDs for SQL IN clause
    const formattedIds = ids.map((id) => `'${id}'`).join(", ");

    const query = `
      SELECT 
        ID as id,
        ORGANIZATION_ID as parent_id,
        LOCALE,
        DESCRIPTION,
        IS_CANONICAL,
        LAST_MODIFIED,
        CREATED
      FROM ${this.sourceTables.translations}
      WHERE ORGANIZATION_ID IN (${formattedIds})
      AND LOCALE = '${locale}'
    `;

    return this.snowflakeClient.query<SnowflakeOrganizationTranslation>(query);
  }
}
