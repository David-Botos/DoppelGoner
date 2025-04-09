// TODO: Types

// app/api/clusters/[id]/edit/route.ts
import { NextResponse } from 'next/server';
import prisma from '@/lib/db/primsa';

interface Params {
  params: {
    id: string;
  };
}

export async function POST(request: Request, { params }: Params) {
  try {
    const { id } = params;
    const body = await request.json();
    
    const { entities_to_remove, notes, reviewer } = body;
    
    if (!entities_to_remove || !Array.isArray(entities_to_remove) || entities_to_remove.length === 0) {
      return NextResponse.json(
        { message: 'No entities to remove specified' },
        { status: 400 }
      );
    }
    
    // Start a transaction to ensure all operations succeed or fail together
    const result = await prisma.$transaction(async (tx) => {
      // 1. Remove the specified entities from the cluster
      await tx.cluster_entities.deleteMany({
        where: {
          id: {
            in: entities_to_remove
          }
        }
      });
      
      // 2. Update the cluster with review information
      const updatedCluster = await tx.match_clusters.update({
        where: {
          id
        },
        data: {
          is_reviewed: true,
          review_result: true, // We're approving the edited cluster
          reviewed_by: reviewer,
          reviewed_at: new Date(),
          notes: notes || null
        }
      });
      
      return updatedCluster;
    });
    
    return NextResponse.json({
      message: 'Cluster edited successfully',
      cluster: result
    });
  } catch (error) {
    console.error('Error editing cluster:', error);
    return NextResponse.json(
      { message: 'Error editing cluster' },
      { status: 500 }
    );
  }
}
