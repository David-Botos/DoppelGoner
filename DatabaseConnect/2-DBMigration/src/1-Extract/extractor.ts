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
      translations?: string; // Make translations optional
    }
  ) {}

  abstract extractMainRecords(offset: number, limit?: number): Promise<S[]>;

  /**
   * Extract translation records for the given IDs and locale
   * This method must be implemented if translations are supported
   */
  abstract extractTranslationRecords(
    ids: string[],
    locale: string
  ): Promise<T[]>;

  /**
   * Check if this extractor supports translations
   */
  protected hasTranslations(): boolean {
    return !!this.sourceTables.translations;
  }

  async extract(
    offset: number,
    locale: string = "en",
    limit?: number
  ): Promise<Map<string, { main: S; translations: T[] }>> {
    // Extract main records
    const mainRecords = await this.extractMainRecords(offset, limit);
    if (mainRecords.length > 0) {
      logJsonBlock(
        `First record from ${this.sourceTables.main}`,
        mainRecords[0]
      );
    }

    const dataMap = new Map<string, { main: S; translations: T[] }>();

    // Initialize the map with main records
    mainRecords.forEach((record) => {
      dataMap.set(record.ID, {
        main: record,
        translations: [],
      });
    });

    // Only extract translations if supported
    if (this.hasTranslations()) {
      // Get IDs for translation lookup
      let ids: string[] = mainRecords.map((record) => record.ID);

      if (ids.length > 0) {
        console.log(
          "These ids were mapped from mainRecords ",
          ids[0],
          ",",
          ids[1] || "",
          "..."
        );

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

        // Add translations to their corresponding main records
        translationRecords.forEach((translation) => {
          const parentId = translation.PARENT_RECORD_ID;
          const entry = dataMap.get(parentId);
          if (entry) {
            entry.translations.push(translation);
          } else {
            console.log(
              `No matching record found for translation with parent ID: ${parentId}`
            );
          }
        });
      }
    }

    return dataMap;
  }
}
