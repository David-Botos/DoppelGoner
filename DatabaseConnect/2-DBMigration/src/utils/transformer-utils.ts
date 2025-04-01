import { SupabaseClient } from "@supabase/supabase-js";
import { PostgresOrganization } from "../types";
import { PostgresClient } from "../services/postgres-client";

export async function getOrgBasedOnOriginalIDFromSupabase(
  supabase: SupabaseClient,
  organization_id: string
): Promise<PostgresOrganization | undefined> {
  const { data, error } = await supabase
    .from("organization")
    .select("*")
    .eq("original_id", organization_id)
    .single();

  if (!error) {
    let result = data as PostgresOrganization;
    return result;
  }
  console.error(
    "error when fetching supa organization with original_id: ",
    organization_id,
    error
  );

  return undefined;
}

export async function getOrgBasedOnOriginalIDFromPostgres(
  postgresClient: PostgresClient,
  organization_id: string
): Promise<PostgresOrganization | undefined> {
  try {
    const query = `
      SELECT * FROM organization
      WHERE original_id = $1
      LIMIT 1
    `;

    const results = await postgresClient.executeQuery<PostgresOrganization>(
      query,
      [organization_id]
    );

    if (results && results.length > 0) {
      return results[0];
    }

    return undefined;
  } catch (error) {
    console.error(
      "Error when fetching organization with original_id: ",
      organization_id,
      error
    );

    return undefined;
  }
}
