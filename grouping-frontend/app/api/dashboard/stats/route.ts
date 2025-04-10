// app/api/dashboard/stats/route.ts
import { NextResponse } from 'next/server';
import prisma from '@/lib/db/prisma';

export async function GET() {
  try {
    const [
      totalClusters,
      reviewedClusters,
      avgConfidenceResult,
      totalEntities,
      uniqueMethods,
    ] = await Promise.all([
      prisma.match_clusters.count(),
      prisma.match_clusters.count({ where: { is_reviewed: true } }),
      prisma.match_clusters.aggregate({ _avg: { confidence: true } }),
      prisma.cluster_entities.count(),
      prisma.matching_methods.groupBy({ by: ["method_name"] }),
    ]);

    return NextResponse.json({
      totalClusters,
      reviewedClusters,
      pendingClusters: totalClusters - reviewedClusters,
      averageConfidence: avgConfidenceResult._avg.confidence || 0,
      totalEntities,
      uniqueMatchingMethods: uniqueMethods.length,
    });
  } catch (error) {
    console.error('Error fetching dashboard stats:', error);
    return NextResponse.json(
      { error: 'Failed to fetch dashboard stats' },
      { status: 500 }
    );
  }
}