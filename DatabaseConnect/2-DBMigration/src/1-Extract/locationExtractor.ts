import { SnowflakeClient } from "../services/snowflake-client";
import { SnowflakeLocation, SnowflakeLocationTranslation } from "../types";
import { Extractor } from "./extractor";

export class LocationExtractor extends Extractor<
  SnowflakeLocation,
  SnowflakeLocationTranslation
> {
  constructor(protected snowflakeClient: SnowflakeClient) {
    super(snowflakeClient, {
      main: "LOCATION",
      translations: "LOCATION_TRANSLATIONS",
    });
  }

  async extractMainRecords(
    offset: number,
    limit?: number
  ): Promise<SnowflakeLocation[]> {
    // Start building the base query without LIMIT or OFFSET
    let query = `
      SELECT 
        ID,
        ORGANIZATION_ID,
        NAME,
        ALTERNATE_NAME,
        LATITUDE,
        LONGITUDE,
        LOCATION_TYPE,
        ORIGINAL_ID,
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
    return this.snowflakeClient.query<SnowflakeLocation>(query);
  }

  async extractTranslationRecords(
    ids: string[],
    locale: string
  ): Promise<SnowflakeLocationTranslation[]> {
    const formattedIds = ids.map((id) => `'${id}'`).join(", ");
    console.log(
      "fetching translations records for ",
      formattedIds.slice(0, 100),
      "..."
    );

    const query = `
      SELECT 
        ID,
        LOCATION_ID,
        DESCRIPTION,
        SHORT_DESCRIPTION,
        TRANSPORTATION,
        IS_CANONICAL,
        LOCALE,
        LOCATION_ID as PARENT_RECORD_ID
      FROM ${this.sourceTables.translations}
      WHERE LOCATION_ID IN (${formattedIds})
      AND LOCALE = '${locale}'
    `;

    return this.snowflakeClient.query<SnowflakeLocationTranslation>(query);
  }
}
