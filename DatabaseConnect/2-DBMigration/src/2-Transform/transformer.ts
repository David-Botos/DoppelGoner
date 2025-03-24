import { MigratedData, SourceData, SourceDataTranslations } from "../types";

export abstract class Transformer<
  S extends SourceData,
  T extends SourceDataTranslations,
  R extends MigratedData
> {
  constructor(protected idConverter: IdConverter) {}

  protected abstract transformSingleRecord(source: S, translation: T): R;

  
}
