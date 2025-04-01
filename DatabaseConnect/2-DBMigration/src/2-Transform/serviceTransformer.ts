import { Transformer } from "./transformer";
import {
  SnowflakeService,
  SnowflakeServiceTranslation,
  PostgresOrganization,
  PostgresService,
} from "../types";
import { IdConverter } from "../utils/uuid-utils";
import { v4 as uuidv4 } from "uuid";
// import { createClient, SupabaseClient } from "@supabase/supabase-js";
// import { supabaseConfig } from "../config/config";
import { tableExists } from "../utils/loader-utils";
import { getOrgBasedOnOriginalIDFromPostgres } from "../utils/transformer-utils";
import { PostgresClient } from "../services/postgres-client";

export class ServiceTransformer extends Transformer<
  SnowflakeService,
  SnowflakeServiceTranslation,
  PostgresService
> {
  constructor(idConverter: IdConverter) {
    super(idConverter);
  }

  protected async transformSingleRecord(
    source: SnowflakeService,
    translation: SnowflakeServiceTranslation
  ): Promise<PostgresService> {
    const newId = uuidv4();

    const postgresClient = new PostgresClient();

    const supaOrgEntry: PostgresOrganization | undefined =
      await getOrgBasedOnOriginalIDFromPostgres(
        postgresClient,
        source.ORGANIZATION_ID
      );

    let newOrgFKId = "";

    if (supaOrgEntry) {
      newOrgFKId = supaOrgEntry.original_id;
    }

    return {
      id: newId,
      organization_id: newOrgFKId,
      program_id: source.PROGRAM_ID
        ? this.idConverter.convertToUuid(source.PROGRAM_ID) ||
          source.PROGRAM_ID.toString()
        : undefined,
      name: translation.NAME,
      alternate_name: translation?.ALTERNATE_NAME || undefined,
      description: translation?.DESCRIPTION || undefined,
      short_description: translation?.SHORT_DESCRIPTION || undefined,
      url: source.URL || undefined,
      email: source.EMAIL || undefined,
      status: source.STATUS,
      interpretation_services:
        translation?.INTERPRETATION_SERVICES || undefined,
      application_process: translation?.APPLICATION_PROCESS || undefined,
      wait_time: undefined, // Not available in source data
      fees_description: translation?.FEES_DESCRIPTION || undefined,
      accreditations: translation?.ACCREDITATIONS || undefined,
      licenses: undefined, // Not available in source data
      minimum_age: source.MINIMUM_AGE || undefined,
      maximum_age: source.MAXIMUM_AGE || undefined,
      eligibility_description:
        translation?.ELIGIBILITY_DESCRIPTION || undefined,
      alert: translation?.ALERT || undefined,
      last_modified: new Date(source.LAST_MODIFIED).toISOString(),
      created: new Date(source.CREATED).toISOString(),
      original_id:
        this.idConverter.convertToUuid(source.ID) || source.ID.toString(),
      original_translations_id: translation?.ID
        ? this.idConverter.convertToUuid(translation.ID)
        : undefined,
    };
  }
}
