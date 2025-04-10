// app/api/dashboard/reviews/route.ts
import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/lib/db/prisma';
import { ReviewAction } from '@/types/review';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const limit = parseInt(searchParams.get('limit') || '10', 10);

    const reviews = await prisma.match_clusters.findMany({
      where: {
        is_reviewed: true,
        reviewed_at: { not: null },
      },
      orderBy: {
        reviewed_at: 'desc',
      },
      take: limit,
      select: {
        id: true,
        review_result: true,
        reviewed_by: true,
        reviewed_at: true,
        notes: true,
      },
    });

    const formattedReviews = reviews.map((review) => ({
      id: review.id,
      cluster_id: review.id,
      action: (review.review_result ? 'confirm' : 'deny') as ReviewAction,
      reviewer: review.reviewed_by as string,
      reviewed_at: review.reviewed_at?.toISOString() as string,
      notes: review.notes,
    }));

    return NextResponse.json(formattedReviews);
  } catch (error) {
    console.error('Error fetching recent reviews:', error);
    return NextResponse.json(
      { error: 'Failed to fetch recent reviews' },
      { status: 500 }
    );
  }
}