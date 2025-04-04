import {
  SnowflakePhone,
  SnowflakePhoneTranslation,
  PostgresPhone,
} from "../types";
import { IdConverter } from "../utils/uuid-utils";
import {
  Transformer,
  TransformOptions,
  RelationshipDefinition,
} from "./transformer";
import { v4 as uuidv4 } from "uuid";

export class PhoneTransformer extends Transformer<
  SnowflakePhone,
  SnowflakePhoneTranslation,
  PostgresPhone
> {
  constructor(idConverter: IdConverter) {
    super(idConverter);
  }

  /**
   * Define relationships for phone entity
   * We have several foreign key relationships:
   * location_id, service_id, organization_id, service_at_location_id
   * Note: contact_id is omitted as the CONTACT table is empty
   */
  protected getRelationships(): RelationshipDefinition[] {
    return [
      {
        fieldName: "organization_id",
        sourceField: "ORGANIZATION_ID",
        tableName: "organization",
        idField: "original_id",
        transformId: (id) => id?.toString() || null,
      },
      {
        fieldName: "service_id",
        sourceField: "SERVICE_ID",
        tableName: "service",
        idField: "original_id",
        transformId: (id) => id?.toString() || null,
      },
      {
        fieldName: "location_id",
        sourceField: "LOCATION_ID",
        tableName: "location",
        idField: "original_id",
        transformId: (id) => id?.toString() || null,
      },
      {
        fieldName: "service_at_location_id",
        sourceField: "SERVICE_AT_LOCATION_ID",
        tableName: "service_at_location",
        idField: "original_id",
        transformId: (id) => id?.toString() || null,
      },
    ];
  }

  /**
   * Transform a single phone record using resolved relationships
   * @param source Source phone data
   * @param translation Phone translation data
   * @param resolvedRelationships Map of resolved relationship IDs
   * @returns Transformed phone record
   */
  protected transformSingleRecord(
    source: SnowflakePhone,
    translation: SnowflakePhoneTranslation | null,
    resolvedRelationships?: Map<string, string>
  ): PostgresPhone {
    const newId = uuidv4();

    // Get foreign keys from resolved relationships if available
    const organizationId =
      resolvedRelationships?.get("organization_id") || undefined;
    const serviceId = resolvedRelationships?.get("service_id") || undefined;
    const locationId = resolvedRelationships?.get("location_id") || undefined;
    const serviceAtLocationId =
      resolvedRelationships?.get("service_at_location_id") || undefined;

    return {
      id: newId,
      organization_id: organizationId,
      service_id: serviceId,
      location_id: locationId,
      contact_id: undefined, // Intentionally omitted as per requirements
      service_at_location_id: serviceAtLocationId,
      number: source.NUMBER || "", // NOT NULL constraint in Postgres
      extension: source.EXTENSION || undefined,
      type: source.TYPE || undefined,
      language: translation?.LOCALE || "en", // Default to English
      description: translation?.DESCRIPTION || undefined,
      priority:
        source.PRIORITY !== null && source.PRIORITY !== undefined
          ? Number(source.PRIORITY)
          : undefined,
      last_modified: source.LAST_MODIFIED
        ? new Date(source.LAST_MODIFIED).toISOString()
        : new Date().toISOString(),
      created: source.CREATED
        ? new Date(source.CREATED).toISOString()
        : new Date().toISOString(),
      original_id: source.ID
        ? this.idConverter.convertToUuid(source.ID) || source.ID.toString()
        : new Date().toISOString(),
      original_translations_id: translation?.ID
        ? this.idConverter.convertToUuid(translation.ID)
        : undefined,
    };
  }
}
