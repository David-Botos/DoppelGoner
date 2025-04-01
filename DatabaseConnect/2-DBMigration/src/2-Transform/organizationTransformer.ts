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
      parent_organization_id: this.idConverter.convertToUuid(
        source.PARENT_ORGANIZATION_ID
      ),
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
