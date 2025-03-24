import { SnowflakeClient } from "../services/snowflake-client";
import { SourceData, SourceDataTranslations } from "../types";

export abstract class Extractor<
  S extends SourceData,
  T extends SourceDataTranslations
> {
  constructor(
    protected snowflakeClient: SnowflakeClient,
    protected sourceTables: {
      main: string;
      translations: string;
    }
  ) {}

  // Specific SQL scripts will be written that are particular to the table
  abstract extractMainRecords(limit: number, offset: number): Promise<S[]>;
  abstract extractTranslationRecords(
    ids: string[],
    locale: string
  ): Promise<T[]>;

  async extract(
    limit: number,
    offset: number,
    locale: string = "en"
  ): Promise<Map<string, { main: S; translations: T[] }>> {
    // Extract main records
    const mainRecords = await this.extractMainRecords(limit, offset);

    // Get IDs for translation lookup
    const ids = mainRecords.map((record) => record.id);

    // Extract translation records for these IDs
    const translationRecords = await this.extractTranslationRecords(
      ids,
      locale
    );

    // Organize data into a map for easy access
    const dataMap = new Map<string, { main: S; translations: T[] }>();

    mainRecords.forEach((record) => {
      dataMap.set(record.id, {
        main: record,
        translations: [],
      });
    });

    translationRecords.forEach((translation) => {
      const entry = dataMap.get(translation.parent_id);
      if (entry) {
        entry.translations.push(translation);
      }
    });

    return dataMap;
  }
}
