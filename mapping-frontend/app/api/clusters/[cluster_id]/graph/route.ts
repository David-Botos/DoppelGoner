// app/api/clusters/[cluster_id]/graph/route.ts
import { GraphData } from "@/lib/types";
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// Mock data for the MVP - in a real application, this would be fetched from a database
const mockGraphs: Record<string, GraphData> = {
  cluster1: {
    nodes: [
      { id: "entity1", name: "Memorial Hospital", organization_id: "org1" },
      { id: "entity2", name: "Memorial Healthcare", organization_id: "org2" },
      { id: "entity3", name: "City Medical Center", organization_id: "org3" },
      {
        id: "entity4",
        name: "Downtown Health Services",
        organization_id: "org4",
      },
      {
        id: "entity5",
        name: "Memorial Hospital System",
        organization_id: "org5",
      },
    ],
    links: [
      {
        id: "eg1",
        source: "entity1",
        target: "entity2",
        method_type: "email",
        confidence_score: 0.92,
        match_values: {
          type: "Email",
          values: {
            original_email1: "info@memorial.org",
            original_email2: "info@memorialhealth.org",
            normalized_shared_email: "info@memorial",
          },
        },
      },
      {
        id: "eg2",
        source: "entity1",
        target: "entity5",
        method_type: "name",
        confidence_score: 0.85,
        match_values: {
          type: "Name",
          values: {
            original_name1: "Memorial Hospital",
            original_name2: "Memorial Hospital System",
            pre_rl_similarity_score: 0.83,
          },
        },
      },
      {
        id: "eg3",
        source: "entity2",
        target: "entity5",
        method_type: "url",
        confidence_score: 0.78,
        match_values: {
          type: "URL",
          values: {
            original_url1: "https://memorialhealth.org",
            original_url2: "https://memorialhospitalsystem.org",
            normalized_shared_url: "memorialhospital",
          },
        },
      },
      {
        id: "eg4",
        source: "entity3",
        target: "entity4",
        method_type: "phone",
        confidence_score: 0.95,
        match_values: {
          type: "Phone",
          values: {
            original_phone1: "555-123-4567",
            original_phone2: "(555) 123-4567",
            normalized_shared_phone: "5551234567",
          },
        },
      },
      {
        id: "eg5",
        source: "entity2",
        target: "entity3",
        method_type: "name",
        confidence_score: 0.45,
        match_values: {
          type: "Name",
          values: {
            original_name1: "Memorial Healthcare",
            original_name2: "City Medical Center",
            pre_rl_similarity_score: 0.32,
          },
        },
      },
    ],
  },
  cluster2: {
    nodes: [
      { id: "entity6", name: "City College", organization_id: "org6" },
      { id: "entity7", name: "City University", organization_id: "org7" },
      { id: "entity8", name: "Metro University", organization_id: "org8" },
    ],
    links: [
      {
        id: "eg6",
        source: "entity6",
        target: "entity7",
        method_type: "email",
        confidence_score: 0.88,
        match_values: {
          type: "Email",
          values: {
            original_email1: "admissions@citycollege.edu",
            original_email2: "admissions@cityuniversity.edu",
            normalized_shared_email: "admissions@city",
          },
        },
      },
      {
        id: "eg7",
        source: "entity7",
        target: "entity8",
        method_type: "name",
        confidence_score: 0.72,
        match_values: {
          type: "Name",
          values: {
            original_name1: "City University",
            original_name2: "Metro University",
            pre_rl_similarity_score: 0.68,
          },
        },
      },
    ],
  },
  cluster3: {
    nodes: [
      {
        id: "entity9",
        name: "Community Services Inc",
        organization_id: "org9",
      },
      {
        id: "entity10",
        name: "Community Services Network",
        organization_id: "org10",
      },
      {
        id: "entity11",
        name: "City Community Services",
        organization_id: "org11",
      },
      {
        id: "entity12",
        name: "Metro Community Help",
        organization_id: "org12",
      },
    ],
    links: [
      {
        id: "eg8",
        source: "entity9",
        target: "entity10",
        method_type: "name",
        confidence_score: 0.91,
        match_values: {
          type: "Name",
          values: {
            original_name1: "Community Services Inc",
            original_name2: "Community Services Network",
            pre_rl_similarity_score: 0.88,
          },
        },
      },
      {
        id: "eg9",
        source: "entity10",
        target: "entity11",
        method_type: "phone",
        confidence_score: 0.97,
        match_values: {
          type: "Phone",
          values: {
            original_phone1: "555-987-6543",
            original_phone2: "555-987-6543",
            normalized_shared_phone: "5559876543",
          },
        },
      },
      {
        id: "eg10",
        source: "entity11",
        target: "entity12",
        method_type: "url",
        confidence_score: 0.65,
        match_values: {
          type: "URL",
          values: {
            original_url1: "https://citycs.org",
            original_url2: "https://metrocommunity.org",
            normalized_shared_url: "community",
          },
        },
      },
    ],
  },
};

export async function GET(
  request: NextRequest,
  { params }: { params: { cluster_id: string } }
) {
  const { cluster_id } = await params;
  const clusterId = cluster_id;

  try {
    // In a real app, you would fetch from a database here
    // e.g.,
    // const nodes = await db.query('SELECT e.id, e.name, e.organization_id FROM public.entity e JOIN ...');
    // const links = await db.query('SELECT eg.id, eg.entity_id_1 as source, eg.entity_id_2 as target, ... FROM public.entity_group eg WHERE ...');

    const graphData = mockGraphs[clusterId];

    if (!graphData) {
      return NextResponse.json(
        { error: `Cluster with ID ${clusterId} not found` },
        { status: 404 }
      );
    }

    return NextResponse.json(graphData);
  } catch (error) {
    console.error(`Error fetching graph for cluster ${clusterId}:`, error);
    return NextResponse.json(
      { error: "Failed to fetch graph data" },
      { status: 500 }
    );
  }
}
