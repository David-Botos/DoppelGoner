// app/api/clusters/route.ts
import { NextResponse } from 'next/server';
import prisma from '@/lib/db/primsa';

export async function GET() {
  try {
    // Fetch all clusters with their entities and methods
    const clusters = await prisma.match_clusters.findMany({
      include: {
        entities: {
          include: {
            entity_data: true // This would need to be implemented in the Prisma schema
          }
        },
        methods: true
      },
      orderBy: {
        created_at: 'desc'
      }
    });
    
    return NextResponse.json(clusters);
  } catch (error) {
    console.error('Error fetching clusters:', error);
    return NextResponse.json(
      { message: 'Error fetching clusters' },
      { status: 500 }
    );
  }
}