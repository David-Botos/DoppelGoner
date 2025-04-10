// app/api/clusters/[id]/route.ts
import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/lib/db/prisma';
import { EntityType } from '@/types/entity';

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const clusterId = params.id;
    
    const cluster = await prisma.match_clusters.findUnique({
      where: { id: clusterId },
      select: {
        id: true,
        confidence: true,
        notes: true,
        reasoning: true,
        is_reviewed: true,
        review_result: true,
        reviewed_by: true,
        reviewed_at: true,
        created_at: true,
        updated_at: true,
        entities: true,
        methods: true,
      }
    });
    
    if (!cluster) {
      return NextResponse.json(
        { error: 'Cluster not found' },
        { status: 404 }
      );
    }

    // Fetch entity data for each entity in the cluster
    const entitiesWithData = await Promise.all(
      cluster.entities.map(async (entity) => {
        const entityData = await getEntityData(
          entity.entity_type as EntityType,
          entity.entity_id
        );

        return {
          ...entity,
          entity_data: entityData,
        };
      })
    );
    
    return NextResponse.json({
      ...cluster,
      entities: entitiesWithData,
    });
  } catch (error) {
    console.error(`Error fetching cluster ${params.id}:`, error);
    return NextResponse.json(
      { error: 'Failed to fetch cluster' },
      { status: 500 }
    );
  }
}

// Helper function to get entity data
async function getEntityData(type: EntityType, id: string) {
  switch (type) {
    case EntityType.ORGANIZATION:
      return prisma.organization.findUnique({
        where: { id },
        select: {
          id: true,
          name: true,
          alternate_name: true,
          description: true,
          email: true,
          url: true,
          tax_status: true,
          tax_id: true,
          year_incorporated: true,
          legal_status: true,
          parent_organization_id: true,
          last_modified: true,
          created: true,
          original_id: true,
          original_translations_id: true,
        }
      });
    case EntityType.SERVICE:
      return prisma.service.findUnique({
        where: { id },
        select: {
          id: true,
          organization_id: true,
          program_id: true,
          name: true,
          alternate_name: true,
          description: true,
          short_description: true,
          url: true,
          email: true,
          status: true,
          interpretation_services: true,
          application_process: true,
          wait_time: true,
          fees_description: true,
          accreditations: true,
          licenses: true,
          minimum_age: true,
          maximum_age: true,
          eligibility_description: true,
          alert: true,
          last_modified: true,
          created: true,
          original_id: true,
          original_translations_id: true,
        }
      });
    // Add cases for other entity types
    case EntityType.LOCATION:
      return prisma.location.findUnique({
        where: { id },
        select: {
          id: true,
          organization_id: true,
          name: true,
          alternate_name: true,
          description: true,
          short_description: true,
          transportation: true,
          latitude: true,
          longitude: true,
          location_type: true,
          last_modified: true,
          created: true,
          original_id: true,
          original_translations_id: true,
        }
      });
    case EntityType.ADDRESS:
      return prisma.address.findUnique({
        where: { id },
        select: {
          id: true,
          location_id: true,
          attention: true,
          address_1: true,
          address_2: true,
          city: true,
          region: true,
          state_province: true,
          postal_code: true,
          country: true,
          address_type: true,
          last_modified: true,
          created: true,
          original_id: true,
        }
      });
    case EntityType.PHONE:
      return prisma.phone.findUnique({
        where: { id },
        select: {
          id: true,
          location_id: true,
          service_id: true,
          organization_id: true,
          contact_id: true,
          service_at_location_id: true,
          number: true,
          extension: true,
          type: true,
          language: true,
          description: true,
          priority: true,
          last_modified: true,
          created: true,
          original_id: true,
          original_translations_id: true,
        }
      });
    case EntityType.SERVICE_AT_LOCATION:
      return prisma.service_at_location.findUnique({
        where: { id },
        select: {
          id: true,
          service_id: true,
          location_id: true,
          description: true,
          last_modified: true,
          created: true,
          original_id: true,
          original_translations_id: true,
        }
      });
    default:
      return null;
  }
}