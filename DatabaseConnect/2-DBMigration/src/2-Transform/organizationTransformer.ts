import {
  SnowflakeOrganization,
  SnowflakeOrganizationTranslation,
  PostgresOrganization,
} from "../types";
import { IdConverter } from "../utils/uuid-utils";
import { Transformer, TransformOptions } from "./transformer";
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
   * Transform organization records using the enhanced batch processing
   * @param dataMap Map of source data and translations
   * @param batchSize Number of records to process in each batch
   * @param pgSchema Database schema to use
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
    batchSizeOrOptions: number | TransformOptions = 100,
    pgSchema: string = "public"
  ): Promise<PostgresOrganization[]> {
    // Handle legacy method signature for backward compatibility
    const options: TransformOptions =
      typeof batchSizeOrOptions === "number"
        ? { batchSize: batchSizeOrOptions, pgSchema }
        : batchSizeOrOptions;

    console.log(
      `Starting to transform ${
        dataMap.size
      } organization records using schema: ${options.pgSchema || pgSchema}`
    );

    // Call the base class implementation with our options
    return super.transform(dataMap, options);
  }

  /**
   * We can define relationships if needed in the future.
   * For organizations, we currently don't have relationships to resolve.
   */
  protected getRelationships() {
    return null;

    // Example of how to define a parent organization relationship if needed in the future:
    /*
    return [
      {
        fieldName: 'parent_organization_id',
        sourceField: 'PARENT_ORGANIZATION_ID',
        tableName: 'organization',
        idField: 'original_id'
      }
    ];
    */
  }

  /**
   * Transform a single organization record
   * @param source Source organization data
   * @param translation Organization translation data
   * @returns Transformed organization record
   */
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
      parent_organization_id: null, // Currently always null in source DB
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
