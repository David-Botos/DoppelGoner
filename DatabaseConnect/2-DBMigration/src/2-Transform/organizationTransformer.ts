import { Transformer } from "./transformer";
import {
  SnowflakeOrganization,
  SnowflakeOrganizationTranslation,
  PostgresOrganization,
} from "../types";
import { IdConverter } from "../utils/uuid-utils";
import { v4 as uuidv4 } from "uuid";

export class OrganizationTransformer extends Transformer<
  SnowflakeOrganization,
  SnowflakeOrganizationTranslation,
  PostgresOrganization
> {
  constructor(idConverter: IdConverter) {
    super(idConverter);
  }

  /**
   * Batch transform organization records
   * @param dataMap Map of source data and translations
   * @param batchSize Number of records to process in each batch
   * @returns Array of transformed organizations
   */
  async transform(
    dataMap: Map<
      string,
      {
        main: SnowflakeOrganization;
        translations: SnowflakeOrganizationTranslation[];
      }
    >,
    batchSize: number = 100
  ): Promise<PostgresOrganization[]> {
    const transformedRecords: PostgresOrganization[] = [];
    const entries = Array.from(dataMap.entries());

    // Process records in batches
    for (let i = 0; i < entries.length; i += batchSize) {
      const batch = entries.slice(i, i + batchSize);
      console.log(
        `Processing organization batch ${i / batchSize + 1} of ${Math.ceil(
          entries.length / batchSize
        )}`
      );

      // Transform batch in parallel
      const batchResults = await Promise.all(
        batch.map(([_, { main, translations }]) => {
          const translation = translations.length > 0 ? translations[0] : null;
          // Since we know parent_organization_id is always undefined in the source DB,
          // we can use a simpler transformation without database lookups
          return this.transformSingleRecord(main, translation);
        })
      );

      transformedRecords.push(...batchResults);
    }

    return transformedRecords;
  }

  protected transformSingleRecord(
    source: SnowflakeOrganization,
    translation: SnowflakeOrganizationTranslation | null
  ): PostgresOrganization {
    const newId = uuidv4();
    return {
      id: newId,
      name: source.NAME,
      alternate_name: source.ALTERNATE_NAME || undefined,
      description: translation?.DESCRIPTION || undefined,
      email: source.EMAIL || undefined,
      url: source.WEBSITE || undefined,
      year_incorporated: source.YEAR_INCORPORATED || undefined,
      legal_status: source.LEGAL_STATUS || undefined,
      parent_organization_id: null,
      last_modified: new Date(source.LAST_MODIFIED).toISOString(),
      created: new Date(source.CREATED).toISOString(),
      original_id:
        this.idConverter.convertToUuid(source.ID) || source.ID.toString(),
      original_translations_id:
        translation?.id !== undefined
          ? this.idConverter.convertToUuid(translation.id)
          : null,
    };
  }
}
