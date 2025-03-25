import { SnowflakeClient } from "../services/snowflake-client";
import { SourceData, SourceDataTranslations } from "../types";

// Export the utility function for reuse elsewhere
export const logJsonBlock = <T>(label: string, data: T): void => {
  console.log(`\n┌─────── ${label} ───────┐`);
  console.log(JSON.stringify(data, null, 2));
  console.log(`└${"─".repeat(label.length + 16)}┘\n`);
};

export abstract class Extractor<
  S extends SourceData,
  T extends SourceDataTranslations
> {
  constructor(
    protected snowflakeClient: SnowflakeClient,
    public sourceTables: {
      main: string;
      translations: string;
    }
  ) {}

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
    if (mainRecords.length > 0) {
      logJsonBlock(
        `First record from ${this.sourceTables.main}`,
        mainRecords[0]
      );
    }

    // Get IDs for translation lookup
    const ids = mainRecords.map((record) => record.id);

    // Extract translation records for these IDs
    const translationRecords = await this.extractTranslationRecords(
      ids,
      locale
    );
    if (translationRecords.length > 0) {
      logJsonBlock(
        `First record from ${this.sourceTables.translations}`,
        translationRecords[0]
      );
    }

    // Organize data into a map for easy access
    const dataMap = new Map<string, { main: S; translations: T[] }>();

    mainRecords.forEach((record) => {
      dataMap.set(record.id, {
        main: record,
        translations: [],
      });
    });

    translationRecords.forEach((translation) => {
      const entry = dataMap.get(translation.PARENT_RECORD_ID);
      if (entry) {
        entry.translations.push(translation);
      }
    });

    return dataMap;
  }
}
