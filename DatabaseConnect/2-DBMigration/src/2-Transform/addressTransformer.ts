import { SnowflakeAddress, PostgresAddress, MigratedData } from "../types";
import { IdConverter } from "../utils/uuid-utils";
import {
  Transformer,
  TransformOptions,
  RelationshipDefinition,
} from "./transformer";
import { v4 as uuidv4 } from "uuid";

export class AddressTransformer extends Transformer<
  SnowflakeAddress,
  any, // Using 'any' for the translation type since there are no translations
  PostgresAddress
> {
  constructor(idConverter: IdConverter) {
    super(idConverter);
  }

  /**
   * Define relationships for address entity
   * Location is the main foreign key
   */
  protected getRelationships(): RelationshipDefinition[] {
    return [
      {
        fieldName: "location_id",
        sourceField: "LOCATION_ID",
        tableName: "location",
        idField: "original_id",
        transformId: (id) => id?.toString() || null,
      },
    ];
  }

  /**
   * Post-process transformed records to filter out invalid records
   */
  protected async postBatchProcess(
    records: PostgresAddress[]
  ): Promise<PostgresAddress[]> {
    const requiredFields = [
      { field: "location_id", message: "Missing required location_id" },
      { field: "address_1", message: "Missing required address_1" },
      { field: "city", message: "Missing required city" },
      { field: "state_province", message: "Missing required state_province" },
      { field: "postal_code", message: "Missing required postal_code" },
      { field: "country", message: "Missing required country" },
      { field: "address_type", message: "Missing required address_type" },
    ];

    // Track invalid records
    records.forEach((record) => {
      for (const { field, message } of requiredFields) {
        const value = record[field as keyof PostgresAddress];
        if (!value || (typeof value === "string" && value.trim() === "")) {
          this.addInvalidRecord(record, message);
          break; // Stop after first validation error to avoid duplicates
        }
      }
    });

    // Only return valid records
    const validRecords = records.filter((record) =>
      requiredFields.every(({ field }) => {
        const value = record[field as keyof PostgresAddress];
        return value && (typeof value !== "string" || value.trim() !== "");
      })
    );

    const invalidCount = records.length - validRecords.length;
    if (invalidCount > 0) {
      console.warn(
        `Found ${invalidCount} invalid address records (missing required fields)`
      );
    }

    return validRecords;
  }

  /**
   * Transform a single address record
   */
  protected transformSingleRecord(
    source: SnowflakeAddress,
    translation: null, // No translations for address
    resolvedRelationships?: Map<string, string>
  ): PostgresAddress {
    const newId = uuidv4();
    const locationId = resolvedRelationships?.get("location_id") || "";

    if (!locationId && source.LOCATION_ID) {
      console.warn(
        `No PostgreSQL location ID found for Snowflake location ID: ${source.LOCATION_ID}`
      );
    }

    // Make sure country is exactly 2 characters (PostgreSQL constraint)
    let country = source.COUNTRY || "";
    if (country.length > 2) {
      console.warn(
        `Country code "${country}" exceeds 2 characters, truncating to first 2 characters`
      );
      country = country.substring(0, 2);
    } else if (country.length < 2) {
      console.warn(
        `Country code "${country}" is less than 2 characters, padding with spaces`
      );
      country = country.padEnd(2, " ");
    }

    return {
      id: newId,
      location_id: locationId,
      attention: source.ATTENTION || undefined,
      address_1: source.ADDRESS_1 || "",
      address_2: source.ADDRESS_2 || undefined,
      city: source.CITY || "",
      region: source.REGION || undefined,
      state_province: source.STATE_PROVINCE || "",
      postal_code: source.POSTAL_CODE || "",
      country: country,
      address_type: source.ADDRESS_TYPE || "",
      last_modified: source.LAST_MODIFIED
        ? new Date(source.LAST_MODIFIED).toISOString()
        : new Date().toISOString(),
      created: source.CREATED
        ? new Date(source.CREATED).toISOString()
        : new Date().toISOString(),
      original_id:
        this.idConverter.convertToUuid(source.ID) || source.ID.toString(),
    };
  }
}
