// app/api/clusters/route.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Mock data for the MVP - in a real application, this would be fetched from a database
const mockClusters = [
  {
    id: 'cluster1',
    name: 'Healthcare Providers',
    entity_count: 5,
    group_count: 8
  },
  {
    id: 'cluster2',
    name: 'Educational Institutions',
    entity_count: 7,
    group_count: 12
  },
  {
    id: 'cluster3',
    name: 'Community Services',
    entity_count: 10,
    group_count: 15
  }
];

export async function GET(request: NextRequest) {
  // Get pagination parameters
  const searchParams = request.nextUrl.searchParams;
  const page = parseInt(searchParams.get('page') || '1', 10);
  const limit = parseInt(searchParams.get('limit') || '20', 10);
  
  // Calculate offset
  const offset = (page - 1) * limit;
  
  try {
    // In a real app, you would fetch from a database here
    // e.g., const clusters = await db.query('SELECT * FROM public.group_cluster LIMIT $1 OFFSET $2', [limit, offset]);
    
    const paginatedClusters = mockClusters.slice(offset, offset + limit);
    
    return NextResponse.json(paginatedClusters);
  } catch (error) {
    console.error('Error fetching clusters:', error);
    return NextResponse.json(
      { error: 'Failed to fetch clusters' },
      { status: 500 }
    );
  }
}