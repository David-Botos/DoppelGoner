import { SnowflakeClient } from "../services/snowflake-client";
import { SnowflakePhone, SnowflakePhoneTranslation } from "../types";
import { Extractor } from "./extractor";

export class PhoneExtractor extends Extractor<
  SnowflakePhone,
  SnowflakePhoneTranslation
> {
  constructor(protected snowflakeClient: SnowflakeClient) {
    super(snowflakeClient, {
      main: "PHONE",
      translations: "PHONE_TRANSLATIONS",
    });
  }

  async extractMainRecords(
    offset: number,
    limit?: number
  ): Promise<SnowflakePhone[]> {
    // Building query to extract from PHONE table
    let query = `
      SELECT 
        ID,
        LOCATION_ID,
        SERVICE_ID,
        ORGANIZATION_ID,
        CONTACT_ID,
        SERVICE_AT_LOCATION_ID,
        NUMBER,
        EXTENSION,
        TYPE,
        TENANT_ID,
        RESOURCE_WRITER_ID,
        ORIGINAL_ID,
        LAST_MODIFIED,
        CREATED,
        PRIORITY
      FROM ${this.sourceTables.main}
      ORDER BY CREATED DESC`;

    // Add LIMIT and OFFSET if provided
    if (limit !== undefined) {
      query += ` 
        LIMIT ${limit} 
        OFFSET ${offset}`;
    }

    console.log("SQL query being sent: ", query);
    return this.snowflakeClient.query<SnowflakePhone>(query);
  }

  async extractTranslationRecords(
    ids: string[],
    locale: string
  ): Promise<SnowflakePhoneTranslation[]> {
    const formattedIds = ids.map((id) => `'${id}'`).join(", ");
    console.log(
      "fetching translations records for ",
      formattedIds.slice(0, 100),
      "..."
    );

    const query = `
      SELECT 
        ID,
        PHONE_ID,
        DESCRIPTION,
        IS_CANONICAL,
        LOCALE,
        TENANT_ID,
        RESOURCE_WRITER_ID,
        PHONE_ID as PARENT_RECORD_ID
      FROM ${this.sourceTables.translations}
      WHERE PHONE_ID IN (${formattedIds})
      AND LOCALE = '${locale}'
    `;

    return this.snowflakeClient.query<SnowflakePhoneTranslation>(query);
  }
}
