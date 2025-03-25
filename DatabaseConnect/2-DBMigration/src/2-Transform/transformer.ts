import { MigratedData, SourceData, SourceDataTranslations } from "../types";
import { IdConverter } from "../utils/uuid-utils";

export abstract class Transformer<
  S extends SourceData,
  T extends SourceDataTranslations,
  R extends MigratedData
> {
  constructor(protected idConverter: IdConverter) {}

  protected abstract transformSingleRecord(source: S, translation: T | null): R;

  transform(dataMap: Map<string, { main: S; translations: T[] }>): R[] {
    const transformedRecords: R[] = [];

    // Iterate through each entry in the data map
    dataMap.forEach(({ main, translations }, id) => {
      // Simply use the first translation if available
      const translation = translations.length > 0 ? translations[0] : null;

      // Transform the record using the abstract method that will be implemented by subclasses
      const transformedRecord = this.transformSingleRecord(main, translation);

      // Add the transformed record to our results array
      transformedRecords.push(transformedRecord);
    });

    return transformedRecords;
  }
}
