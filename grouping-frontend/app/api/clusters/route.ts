// app/api/clusters/route.ts
import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/lib/db/prisma';
import { EntityType } from '@/types/entity';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const limit = parseInt(searchParams.get('limit') || '100', 10);
    const offset = parseInt(searchParams.get('offset') || '0', 10);
    
    const clusters = await prisma.match_clusters.findMany({
      take: limit,
      skip: offset,
      orderBy: { created_at: 'desc' },
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

    // Fetch entity data for each cluster
    const clustersWithEntityData = await Promise.all(
      clusters.map(async (cluster) => {
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

        return {
          ...cluster,
          entities: entitiesWithData,
          methods: cluster.methods,
        };
      })
    );

    return NextResponse.json(clustersWithEntityData);
  } catch (error) {
    console.error('Error fetching clusters:', error);
    return NextResponse.json(
      { error: 'Failed to fetch clusters' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { confidence, notes, reasoning, entities, methods } = body;
    
    // Start a transaction to ensure all operations succeed or fail together
    const result = await prisma.$transaction(async (tx) => {
      // Create the cluster
      const cluster = await tx.match_clusters.create({
        data: {
          confidence,
          notes: notes || null,
          reasoning: reasoning || null,
          is_reviewed: false,
        },
        select: {
          id: true,
        },
      });

      // Add entities to the cluster
      for (const entity of entities) {
        await tx.cluster_entities.create({
          data: {
            cluster_id: cluster.id,
            entity_type: entity.entity_type,
            entity_id: entity.entity_id,
          },
        });
      }

      // Add methods to the cluster
      for (const method of methods) {
        await tx.matching_methods.create({
          data: {
            cluster_id: cluster.id,
            method_name: method.method_name,
            confidence: method.confidence,
          },
        });
      }

      return cluster.id;
    });

    // Fetch the complete cluster after creation
    const newCluster = await prisma.match_clusters.findUnique({
      where: { id: result },
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

    if (!newCluster) {
      throw new Error('Failed to retrieve newly created cluster');
    }

    // Fetch entity data for the new cluster
    const entitiesWithData = await Promise.all(
      newCluster.entities.map(async (entity) => {
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
      ...newCluster,
      entities: entitiesWithData,
    }, { status: 201 });
  } catch (error) {
    console.error('Error creating cluster:', error);
    return NextResponse.json(
      { error: 'Failed to create cluster' },
      { status: 500 }
    );
  }
}

// Helper function to get entity data (same as in [id]/route.ts)
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
    // Add cases for other entity types as needed
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