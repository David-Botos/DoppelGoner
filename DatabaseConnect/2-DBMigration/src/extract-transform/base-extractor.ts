import { SnowflakeClient } from "../services/snowflake-client";
import { createTransformer } from "../utils/snowflake-to-supa-semantic-transformer";

export abstract class BaseExtractor<
  SourceType extends Record<string, any> & { ID: string },
  TranslationType extends Record<string, any> & { [key: string]: any },
  TargetType
> {
  protected transformer = createTransformer<
    SourceType,
    TranslationType,
    TargetType
  >(this.mapToSupabase.bind(this));

  constructor(
    protected readonly dataService: SnowflakeClient,
    protected readonly tableName: string,
    protected readonly translationTableName?: string,
    protected readonly foreignKeyField?: string
  ) {}

  protected abstract mapToSupabase(
    record: SourceType,
    translation?: TranslationType
  ): TargetType;

  public joinTranslations(
    records: SourceType[],
    translations: TranslationType[]
  ): TargetType[] {
    return this.transformer.transform(
      records,
      translations,
      "ID",
      this.foreignKeyField as keyof TranslationType
    );
  }

  public abstract extractBatch(
    offset: number,
    limit: number
  ): Promise<SourceType[]>;
  public abstract extractTranslations(
    ids: string[],
    locale?: string
  ): Promise<TranslationType[]>;

  public abstract countRecords(filter?: string): Promise<number>;
}
