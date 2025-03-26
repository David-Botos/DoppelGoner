import { SnowflakeClient } from "../services/snowflake-client";
import {
  SnowflakeOrganization,
  SnowflakeOrganizationTranslation,
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
        ID,
        NAME,
        ALTERNATE_NAME,
        EMAIL,
        WEBSITE,
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
    const formattedIds = ids.map((id) => `'${id}'`).join(", ");
    console.log(
      "fetching translations records for ",
      formattedIds.slice(0, 100),
      "..."
    );

    const query = `
      SELECT 
        ID,
        ORGANIZATION_ID,
        LOCALE,
        DESCRIPTION,
        IS_CANONICAL,
        ORGANIZATION_ID as PARENT_RECORD_ID
      FROM ${this.sourceTables.translations}
      WHERE ORGANIZATION_ID IN (${formattedIds})
      AND LOCALE = '${locale}'
    `;

    return this.snowflakeClient.query<SnowflakeOrganizationTranslation>(query);
  }
}
