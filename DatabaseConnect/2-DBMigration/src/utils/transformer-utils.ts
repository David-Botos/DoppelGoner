import { SupabaseClient } from "@supabase/supabase-js";
import { PostgresOrganization } from "../types";
import { PostgresClient } from "../services/postgres-client";

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

    // Using generic type parameter and keeping formatOutput as false (default)
    // since we need the raw data for processing
    const results = await postgresClient.executeQuery<PostgresOrganization>(
      query,
      [organization_id]
    );

    // Handle potential string response from formatted output
    if (typeof results === "string") {
      console.warn(
        "Received formatted string instead of data array. Check PostgresClient settings."
      );
      return undefined;
    }

    // Check array length
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
