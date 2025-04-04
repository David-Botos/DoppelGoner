import { SnowflakeClient } from "../services/snowflake-client";
import { SnowflakeAddress } from "../types";
import { Extractor } from "./extractor";

export class AddressExtractor extends Extractor<SnowflakeAddress, any> {
  constructor(protected snowflakeClient: SnowflakeClient) {
    super(snowflakeClient, {
      main: "ADDRESS",
      translations: "", // Empty string as placeholder since we don't have translations
    });
  }

  async extractMainRecords(
    offset: number,
    limit?: number
  ): Promise<SnowflakeAddress[]> {
    let query = `
      SELECT 
        ID,
        LOCATION_ID,
        ATTENTION,
        ADDRESS_1,
        ADDRESS_2,
        CITY,
        REGION,
        STATE_PROVINCE,
        POSTAL_CODE,
        COUNTRY,
        ADDRESS_TYPE,
        ORIGINAL_ID,
        LAST_MODIFIED,
        CREATED
      FROM ${this.sourceTables.main}
      ORDER BY CREATED DESC`;

    if (limit !== undefined) {
      query += ` 
        LIMIT ${limit} 
        OFFSET ${offset}`;
    }

    console.log("SQL query being sent: ", query);
    return this.snowflakeClient.query<SnowflakeAddress>(query);
  }

  // Provide empty implementation since this table has no translations
  async extractTranslationRecords(
    ids: string[],
    locale: string
  ): Promise<any[]> {
    return [];
  }
}
