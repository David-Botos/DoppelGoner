import { SnowflakeClient } from "../services/snowflake-client";
import { SnowflakeService, SnowflakeServiceTranslation } from "../types";
import { Extractor } from "./extractor";

export class ServiceExtractor extends Extractor<
  SnowflakeService,
  SnowflakeServiceTranslation
> {
  constructor(protected snowflakeClient: SnowflakeClient) {
    super(snowflakeClient, {
      main: "SERVICE",
      translations: "SERVICE_TRANSLATIONS",
    });
  }

  async extractMainRecords(
    offset: number,
    limit?: number
  ): Promise<SnowflakeService[]> {
    // Start building the base query without LIMIT or OFFSET
    let query = `
      SELECT 
        ID,
        ORGANIZATION_ID,
        PROGRAM_ID,
        URL,
        EMAIL,
        STATUS,
        MINIMUM_AGE,
        MAXIMUM_AGE,
        ORIGINAL_ID,
        LAST_MODIFIED,
        CREATED,
        PRIORITY
      FROM ${this.sourceTables.main}
      ORDER BY CREATED DESC`;

    // Add LIMIT and OFFSET in correct order
    if (limit !== undefined) {
      query += ` 
        LIMIT ${limit};
        OFFSET ${offset}`;
    }

    console.log("SQL query being sent: ", query);
    return this.snowflakeClient.query<SnowflakeService>(query);
  }

  async extractTranslationRecords(
    ids: string[],
    locale: string
  ): Promise<SnowflakeServiceTranslation[]> {
    const formattedIds = ids.map((id) => `'${id}'`).join(", ");
    console.log(
      "fetching translations records for ",
      formattedIds.slice(0, 100),
      "..."
    );

    const query = `
      SELECT 
        ID,
        SERVICE_ID,
        NAME,
        ALTERNATE_NAME,
        DESCRIPTION,
        SHORT_DESCRIPTION,
        INTERPRETATION_SERVICES,
        APPLICATION_PROCESS,
        FEES_DESCRIPTION,
        ACCREDITATIONS,
        ELIGIBILITY_DESCRIPTION,
        ALERT,
        IS_CANONICAL,
        LOCALE,
        SERVICE_ID as PARENT_RECORD_ID
      FROM ${this.sourceTables.translations}
      WHERE SERVICE_ID IN (${formattedIds})
      AND LOCALE = '${locale}'
    `;

    return this.snowflakeClient.query<SnowflakeServiceTranslation>(query);
  }
}
