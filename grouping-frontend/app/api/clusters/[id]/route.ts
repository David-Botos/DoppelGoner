// app/api/clusters/[id]/route.ts
import { NextResponse } from 'next/server';
import prisma from '@/lib/db/primsa';

interface Params {
  params: {
    id: string;
  };
}

export async function GET(request: Request, { params }: Params) {
  try {
    const { id } = params;
    
    // Fetch cluster with its entities and methods
    const cluster = await prisma.match_clusters.findUnique({
      where: {
        id
      },
      include: {
        entities: {
          include: {
            entity_data: true // This would need to be implemented in the Prisma schema
          }
        },
        methods: true
      }
    });
    
    if (!cluster) {
      return NextResponse.json(
        { message: 'Cluster not found' },
        { status: 404 }
      );
    }
    
    return NextResponse.json(cluster);
  } catch (error) {
    console.error('Error fetching cluster:', error);
    return NextResponse.json(
      { message: 'Error fetching cluster' },
      { status: 500 }
    );
  }
}