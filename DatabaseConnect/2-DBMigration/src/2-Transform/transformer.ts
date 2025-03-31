import { MigratedData, SourceData, SourceDataTranslations } from "../types";
import { IdConverter } from "../utils/uuid-utils";

export abstract class Transformer<
  S extends SourceData,
  T extends SourceDataTranslations,
  R extends MigratedData
> {
  constructor(protected idConverter: IdConverter) {}

  protected abstract transformSingleRecord(
    source: S,
    translation: T | null
  ): R | Promise<R>;

  async transform(
    dataMap: Map<string, { main: S; translations: T[] }>
  ): Promise<R[]> {
    const transformedRecords: R[] = [];

    // Iterate through each entry in the data map
    for (const [id, { main, translations }] of dataMap.entries()) {
      // Simply use the first translation if available
      const translation = translations.length > 0 ? translations[0] : null;

      // Transform the record and ensure we resolve any promise
      const transformedRecord = await Promise.resolve(
        this.transformSingleRecord(main, translation)
      );

      // Now transformedRecord is guaranteed to be of type R, not Promise<R>
      transformedRecords.push(transformedRecord);
    }

    return transformedRecords;
  }
}
