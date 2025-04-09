// app/api/clusters/[id]/review/route.ts
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
    
    const { action, notes, reviewer } = body;
    
    if (!action || !reviewer) {
      return NextResponse.json(
        { message: 'Missing required fields' },
        { status: 400 }
      );
    }
    
    // For 'confirm' and 'deny' actions, we update the cluster directly
    if (action === 'confirm' || action === 'deny') {
      const updatedCluster = await prisma.match_clusters.update({
        where: {
          id
        },
        data: {
          is_reviewed: true,
          review_result: action === 'confirm',
          reviewed_by: reviewer,
          reviewed_at: new Date(),
          notes: notes || null
        }
      });
      
      return NextResponse.json({
        message: `Cluster ${action === 'confirm' ? 'confirmed' : 'denied'} successfully`,
        cluster: updatedCluster
      });
    }
    
    // For 'edit' and 'split' actions, we just mark it for review but don't change the result yet
    // This is because the actual editing will happen in the separate edit/split endpoints
    return NextResponse.json({
      message: `Cluster marked for ${action}`
    });
  } catch (error) {
    console.error('Error reviewing cluster:', error);
    return NextResponse.json(
      { message: 'Error reviewing cluster' },
      { status: 500 }
    );
  }
}