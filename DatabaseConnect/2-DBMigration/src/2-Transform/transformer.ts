import { MigratedData, SourceData, SourceDataTranslations } from "../types";
import { IdConverter } from "../utils/uuid-utils";
import { PostgresClient } from "../services/postgres-client";

export abstract class Transformer<
  S extends SourceData,
  T extends SourceDataTranslations,
  R extends MigratedData
> {
  protected postgresClient: PostgresClient;

  constructor(protected idConverter: IdConverter) {
    this.postgresClient = new PostgresClient();
  }

  protected abstract transformSingleRecord(
    source: S,
    translation: T | null
  ): R | Promise<R>;

  /**
   * Transform records in batches for better performance
   * @param dataMap Map of source data and translations
   * @param batchSize Number of records to process in each batch
   * @returns Array of transformed records
   */
  async transform(
    dataMap: Map<string, { main: S; translations: T[] }>,
    batchSize: number = 100
  ): Promise<R[]> {
    const transformedRecords: R[] = [];
    const entries = Array.from(dataMap.entries());

    // Process records in batches
    for (let i = 0; i < entries.length; i += batchSize) {
      const batch = entries.slice(i, i + batchSize);
      console.log(
        `Processing batch ${i / batchSize + 1} of ${Math.ceil(
          entries.length / batchSize
        )}`
      );

      // Transform batch in parallel
      const batchPromises = batch.map(async ([id, { main, translations }]) => {
        const translation = translations.length > 0 ? translations[0] : null;
        return this.transformSingleRecord(main, translation);
      });

      // Wait for all transformations in the batch to complete
      const batchResults = await Promise.all(batchPromises);
      transformedRecords.push(...batchResults);
    }

    return transformedRecords;
  }

  /**
   * Optimize lookups for related data using a single database query instead of multiple
   * @param fieldName The field containing IDs to look up
   * @param values Array of original IDs to look up
   * @param tableName Target table name
   * @param idField Field name for the ID in the target table
   * @returns Map of original IDs to new UUIDs
   */
  protected async batchLookupIds(
    fieldName: string,
    values: string[],
    tableName: string,
    idField: string = "original_id"
  ): Promise<Map<string, string>> {
    // Remove duplicates
    const uniqueValues = [
      ...new Set(values.filter((v) => v !== undefined && v !== null)),
    ];

    if (uniqueValues.length === 0) {
      return new Map();
    }

    // Create a parameterized query with all values
    const placeholders = uniqueValues.map((_, idx) => `$${idx + 1}`).join(",");
    const query = `SELECT ${idField}, id FROM ${tableName} WHERE ${idField} IN (${placeholders})`;

    try {
      const results = await this.postgresClient.executeQuery(
        query,
        uniqueValues
      );

      // Create a map from original ID to new UUID
      const idMap = new Map<string, string>();
      for (const row of results) {
        idMap.set(row[idField], row.id);
      }

      return idMap;
    } catch (error) {
      console.error(`Error looking up ${fieldName} in ${tableName}:`, error);
      return new Map();
    }
  }
}
