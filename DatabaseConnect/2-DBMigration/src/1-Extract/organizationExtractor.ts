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
    offset: number,
    limit?: number
  ): Promise<SnowflakeOrganization[]> {
    // Start building the base query without LIMIT or OFFSET
    let query = `
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
      ORDER BY CREATED DESC`;

    // Add LIMIT and OFFSET in correct order
    if (limit !== undefined) {
      query += `
        LIMIT ${limit}
        OFFSET ${offset}`;
    }

    console.log("SQL query being sent: ", query);
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
