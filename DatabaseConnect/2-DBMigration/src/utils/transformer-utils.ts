import { SupabaseClient } from "@supabase/supabase-js";
import { SupabaseOrganization } from "../types";

export default async function getOrgBasedOnOriginalID(
  supabase: SupabaseClient,
  organization_id: string
): Promise<SupabaseOrganization | undefined> {
  const { data, error } = await supabase
    .from("organization")
    .select("*")
    .eq("original_id", organization_id)
    .single();

  if (!error) {
    let result = data as SupabaseOrganization;
    return result;
  }
  console.error(
    "error when fetching supa organization with original_id: ",
    organization_id,
    error
  );

  return undefined;
}
