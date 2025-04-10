// app/api/clusters/[id]/split/route.ts
import { NextResponse } from 'next/server';
import prisma from '@/lib/db/prisma';
import { v4 as uuidv4 } from 'uuid';

interface Params {
  params: {
    id: string;
  };
}

export async function POST(request: Request, { params }: Params) {
  try {
    const { id } = params;
    const body = await request.json();
    
    const { new_clusters, unassigned_entities, notes, reviewer } = body;
    
    if (!new_clusters || !Array.isArray(new_clusters) || new_clusters.length < 2) {
      return NextResponse.json(
        { message: 'At least two new clusters are required' },
        { status: 400 }
      );
    }
    
    // Start a transaction to ensure all operations succeed or fail together
    const result = await prisma.$transaction(async (tx) => {
      // 1. Mark the original cluster as reviewed (and denied since we're splitting it)
      const originalCluster = await tx.match_clusters.update({
        where: {
          id
        },
        data: {
          is_reviewed: true,
          review_result: false, // We're rejecting the original cluster
          reviewed_by: reviewer,
          reviewed_at: new Date(),
          notes: notes || null
        },
        include: {
          methods: true
        }
      });
      
      // 2. Create new clusters for each split
      const newClusterIds = [];
      
      for (const cluster of new_clusters) {
        if (!cluster.entities || cluster.entities.length === 0) {
          continue;
        }
        
        // 2a. Create the new cluster
        const newClusterId = uuidv4();
        await tx.match_clusters.create({
          data: {
            id: newClusterId,
            confidence: originalCluster.confidence * 0.9, // Slightly reduce confidence for split clusters
            notes: `Split from cluster ${id}`,
            reasoning: originalCluster.reasoning,
            is_reviewed: false,
            created_at: new Date(),
            updated_at: new Date()
          }
        });
        
        newClusterIds.push(newClusterId);
        
        // 2b. Move entities to the new cluster
        for (const entityId of cluster.entities) {
          // Find the entity in the original cluster
          const entity = await tx.cluster_entities.findUnique({
            where: {
              id: entityId
            }
          });
          
          if (entity) {
            // Create a new entity in the new cluster
            await tx.cluster_entities.create({
              data: {
                id: uuidv4(),
                cluster_id: newClusterId,
                entity_id: entity.entity_id,
                entity_type: entity.entity_type,
                created_at: new Date()
              }
            });
            
            // Delete the entity from the original cluster
            await tx.cluster_entities.delete({
              where: {
                id: entityId
              }
            });
          }
        }
        
        // 2c. Copy the matching methods to the new cluster
        for (const method of originalCluster.methods) {
          await tx.matching_methods.create({
            data: {
              id: uuidv4(),
              cluster_id: newClusterId,
              method_name: method.method_name,
              confidence: method.confidence * 0.9, // Slightly reduce confidence
              created_at: new Date()
            }
          });
        }
      }
      
      // 3. Delete any unassigned entities
      if (unassigned_entities && unassigned_entities.length > 0) {
        await tx.cluster_entities.deleteMany({
          where: {
            id: {
              in: unassigned_entities
            }
          }
        });
      }
      
      return {
        original_cluster: originalCluster.id,
        new_clusters: newClusterIds
      };
    });
    
    return NextResponse.json({
      message: 'Cluster split successfully',
      result
    });
  } catch (error) {
    console.error('Error splitting cluster:', error);
    return NextResponse.json(
      { message: 'Error splitting cluster' },
      { status: 500 }
    );
  }
}