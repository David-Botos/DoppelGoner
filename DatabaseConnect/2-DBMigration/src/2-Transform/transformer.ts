import { MigratedData, SourceData, SourceDataTranslations } from "../types";
import { IdConverter } from "../utils/uuid-utils";
import { PostgresClient } from "../services/postgres-client";

/**
 * Validation result for records with errors
 */
export interface ValidationResult<T> {
  record: T;
  errors: string[];
}

/**
 * Relationship definition for foreign key resolution
 */
export interface RelationshipDefinition {
  /** Field name in the target entity containing the foreign key */
  fieldName: string;
  /** Original source field containing the ID to look up */
  sourceField: string;
  /** Target table name for the lookup */
  tableName: string;
  /** ID field to match in the target table (defaults to "original_id") */
  idField?: string;
  /** Optional callback for custom ID transformation */
  transformId?: (id: any) => string | null;
}

/**
 * Options for the transform process
 */
export interface TransformOptions {
  /** Number of records to process in each batch */
  batchSize?: number;
  /** Database schema to use */
  pgSchema?: string;
  /** Whether to log detailed debug information */
  debug?: boolean;
}

/**
 * Base Transformer class with enhanced batch processing and relationship resolution
 */
export abstract class Transformer<
  S extends SourceData,
  T extends SourceDataTranslations,
  R extends MigratedData
> {
  protected postgresClient: PostgresClient;
  protected invalidRecords: ValidationResult<R>[] = [];

  constructor(protected idConverter: IdConverter) {
    this.postgresClient = new PostgresClient();
  }

  /**
   * Get invalid records that failed validation during transformation
   * @returns Array of invalid records with errors
   */
  public getInvalidRecords(): ValidationResult<R>[] {
    return this.invalidRecords;
  }

  /**
   * Add a record to the invalid records list with an error message
   * @param record The record that failed validation
   * @param error Error message explaining the validation failure
   */
  protected addInvalidRecord(record: R, error: string): void {
    const existingRecord = this.invalidRecords.find(
      (r) => r.record.original_id === record.original_id
    );

    if (existingRecord) {
      existingRecord.errors.push(error);
    } else {
      this.invalidRecords.push({ record, errors: [error] });
    }
  }

  /**
   * Abstract method that subclasses must implement to transform a single record
   * @param source Source data record
   * @param translation Translation data record (if available)
   * @returns Transformed record or Promise of transformed record
   */
  protected abstract transformSingleRecord(
    source: S,
    translation: T | null,
    resolvedRelationships?: Map<string, string>
  ): R | Promise<R>;

  /**
   * Return relationship definitions for this entity
   * Override this method in subclasses to define relationships
   * @returns Array of relationship definitions or null if no relationships
   */
  protected getRelationships(): RelationshipDefinition[] | null {
    return null;
  }

  /**
   * Pre-process batch before transformation
   * Subclasses can override to add custom logic
   * @param batch Batch of records to transform
   * @returns Processed batch
   */
  protected async preBatchProcess(
    batch: Array<[string, { main: S; translations: T[] }]>
  ): Promise<Array<[string, { main: S; translations: T[] }]>> {
    return batch;
  }

  /**
   * Post-process transformed records
   * Subclasses can override to add custom logic
   * @param records Transformed records
   * @returns Processed records
   */
  protected async postBatchProcess(
    records: Awaited<R>[]
  ): Promise<Awaited<R>[]> {
    return records;
  }

  /**
   * Process a single translation set
   * Subclasses can override to customize translation handling
   * @param translations Array of translations
   * @returns Selected translation or null
   */
  protected processTranslation(translations: T[]): T | null {
    return translations.length > 0 ? translations[0] : null;
  }

  /**
   * Main transform method with enhanced batch processing
   * @param dataMap Map of source data and translations
   * @param options Transform options
   * @returns Array of transformed records
   */
  async transform(
    dataMap: Map<string, { main: S; translations: T[] }>,
    options: TransformOptions = {}
  ): Promise<R[]> {
    const { batchSize = 100, pgSchema = "public", debug = false } = options;

    // Reset invalid records at the start of transformation
    this.invalidRecords = [];

    const transformedRecords: R[] = [];
    const entries = Array.from(dataMap.entries());
    const relationships = this.getRelationships();
    const hasRelationships = relationships && relationships.length > 0;

    console.log(
      `Starting to transform ${entries.length} records using schema: ${pgSchema}`
    );

    // Process records in batches
    for (let i = 0; i < entries.length; i += batchSize) {
      const batch = entries.slice(i, i + batchSize);
      console.log(
        `Processing batch ${i / batchSize + 1} of ${Math.ceil(
          entries.length / batchSize
        )}`
      );

      // Allow subclasses to modify the batch before processing
      const processedBatch = await this.preBatchProcess(batch);

      // If we have relationships to resolve, do it batch-wise for efficiency
      let relationshipMaps: Map<string, Map<string, string>> | null = null;

      if (hasRelationships) {
        if (debug) {
          console.log(
            `Resolving ${relationships.length} relationships for batch`
          );
        }
        relationshipMaps = await this.resolveRelationships(
          processedBatch,
          relationships,
          pgSchema
        );
      }

      // Transform batch in parallel with resolved relationships
      const batchPromises = processedBatch.map(
        async ([id, { main, translations }]) => {
          const translation = this.processTranslation(translations);

          // Get resolved relationships for this record if available
          const resolvedRelationships = hasRelationships
            ? new Map<string, string>()
            : undefined;

          if (hasRelationships && relationshipMaps && resolvedRelationships) {
            for (const relationship of relationships) {
              const sourceValue = main[relationship.sourceField];
              if (sourceValue) {
                const sourceIdStr = relationship.transformId
                  ? relationship.transformId(sourceValue)
                  : sourceValue.toString();

                const relationshipMap = relationshipMaps.get(
                  relationship.fieldName
                );
                if (relationshipMap && relationshipMap.has(sourceIdStr)) {
                  const newId = relationshipMap.get(sourceIdStr);
                  if (newId) {
                    resolvedRelationships.set(relationship.fieldName, newId);
                  }
                }
              }
            }
          }

          // Use the correct transformation method
          return this.transformSingleRecord(
            main,
            translation,
            resolvedRelationships
          );
        }
      );

      // Wait for all transformations in the batch to complete
      const promiseResults = await Promise.all(batchPromises);

      // Allow subclasses to modify the results after processing
      const batchResults = await this.postBatchProcess(promiseResults);

      transformedRecords.push(...batchResults);
    }

    console.log(
      `Successfully transformed ${transformedRecords.length} records`
    );

    if (this.invalidRecords.length > 0) {
      console.log(
        `Found ${this.invalidRecords.length} invalid records during transformation`
      );
    }

    return transformedRecords;
  }

  /**
   * Resolve all relationships for a batch of records
   * @param batch Batch of records to process
   * @param relationships Relationship definitions
   * @param schema Database schema
   * @returns Map of field names to maps of original IDs to new UUIDs
   */
  protected async resolveRelationships(
    batch: Array<[string, { main: S; translations: T[] }]>,
    relationships: RelationshipDefinition[],
    schema: string = "public"
  ): Promise<Map<string, Map<string, string>>> {
    const result = new Map<string, Map<string, string>>();

    // Process each relationship in parallel
    await Promise.all(
      relationships.map(async (relationship) => {
        // Extract all values for this relationship from the batch
        const values = batch
          .map(([_, { main }]) => {
            const value = main[relationship.sourceField];
            if (!value) return null;

            return relationship.transformId
              ? relationship.transformId(value)
              : value.toString();
          })
          .filter(Boolean) as string[];

        // Lookup all values at once
        const idMap = await this.batchLookupIds(
          relationship.fieldName,
          values,
          relationship.tableName,
          relationship.idField || "original_id",
          schema
        );

        result.set(relationship.fieldName, idMap);
      })
    );

    return result;
  }

  /**
   * Optimize lookups for related data using a single database query instead of multiple
   * @param fieldName The field containing IDs to look up
   * @param values Array of original IDs to look up
   * @param tableName Target table name
   * @param idField Field name for the ID in the target table
   * @param schema Database schema to use (defaults to 'public')
   * @returns Map of original IDs to new UUIDs
   */
  protected async batchLookupIds(
    fieldName: string,
    values: string[],
    tableName: string,
    idField: string = "original_id",
    schema: string = "public"
  ): Promise<Map<string, string>> {
    // Remove duplicates
    const uniqueValues = [
      ...new Set(values.filter((v) => v !== undefined && v !== null)),
    ];

    if (uniqueValues.length === 0) {
      return new Map();
    }

    // Log the values being looked up
    console.log(
      `Looking up ${uniqueValues.length} unique ${fieldName} values in ${schema}.${tableName}`
    );
    console.log(`Sample values:`, uniqueValues.slice(0, 5));

    // Create a parameterized query with all values that includes schema
    const placeholders = uniqueValues.map((_, idx) => `$${idx + 1}`).join(",");
    const query = `SELECT ${idField}, id FROM ${schema}.${tableName} WHERE ${idField} IN (${placeholders})`;

    try {
      const results = await this.postgresClient.executeQuery(
        query,
        uniqueValues
      );

      // Log the results
      console.log(
        `Found ${results.length} matching records out of ${uniqueValues.length} unique values`
      );
      if (results.length > 0) {
        console.log(`Sample result:`, results[0]);
      }

      // Create a map from original ID to new UUID
      const idMap = new Map<string, string>();
      for (const row of results) {
        idMap.set(row[idField], row.id);
      }

      return idMap;
    } catch (error) {
      console.error(
        `Error looking up ${fieldName} in ${schema}.${tableName}:`,
        error
      );
      console.error(`Query was: ${query}`);
      console.error(`Values were:`, uniqueValues.slice(0, 5), `...`);
      return new Map();
    }
  }
}
