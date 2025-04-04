import { SnowflakeClient } from "../services/snowflake-client";
import {
  SnowflakeServiceAtLocation,
  SnowflakeServiceAtLocationTranslation,
} from "../types";
import { Extractor } from "./extractor";

export class ServiceAtLocationExtractor extends Extractor<
  SnowflakeServiceAtLocation,
  SnowflakeServiceAtLocationTranslation
> {
  constructor(protected snowflakeClient: SnowflakeClient) {
    super(snowflakeClient, {
      main: "SERVICE_AT_LOCATION",
      translations: "SERVICE_AT_LOCATION_TRANSLATIONS",
    });
  }

  async extractMainRecords(
    offset: number,
    limit?: number
  ): Promise<SnowflakeServiceAtLocation[]> {
    // Start building the base query without LIMIT or OFFSET
    let query = `
      SELECT 
        ID,
        SERVICE_ID,
        LOCATION_ID,
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
    return this.snowflakeClient.query<SnowflakeServiceAtLocation>(query);
  }

  async extractTranslationRecords(
    ids: string[],
    locale: string
  ): Promise<SnowflakeServiceAtLocationTranslation[]> {
    const formattedIds = ids.map((id) => `'${id}'`).join(", ");
    console.log(
      "fetching translations records for ",
      formattedIds.slice(0, 100),
      "..."
    );

    const query = `
      SELECT 
        ID,
        SERVICE_AT_LOCATION_ID,
        DESCRIPTION,
        IS_CANONICAL,
        LOCALE,
        SERVICE_AT_LOCATION_ID as PARENT_RECORD_ID
      FROM ${this.sourceTables.translations}
      WHERE SERVICE_AT_LOCATION_ID IN (${formattedIds})
      AND LOCALE = '${locale}'
    `;

    return this.snowflakeClient.query<SnowflakeServiceAtLocationTranslation>(
      query
    );
  }
}
