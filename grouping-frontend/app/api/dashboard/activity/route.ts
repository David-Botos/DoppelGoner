// app/api/dashboard/activity/route.ts
import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/lib/db/prisma';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const days = parseInt(searchParams.get('days') || '30', 10);

    interface DateCount {
      date: string;
      count: string;
    }

    const dates = Array.from({ length: days + 1 }, (_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - (days - i));
      date.setHours(0, 0, 0, 0);
      return date.toISOString().split('T')[0];
    });

    const createdCounts = await prisma.$queryRaw<DateCount[]>`
      SELECT DATE_TRUNC('day', created_at)::date::text as date, COUNT(*)::text as count
      FROM match_clusters
      WHERE created_at >= NOW() - INTERVAL '${days} days'
      GROUP BY DATE_TRUNC('day', created_at)::date
    `;

    const reviewedCounts = await prisma.$queryRaw<DateCount[]>`
      SELECT DATE_TRUNC('day', reviewed_at)::date::text as date, COUNT(*)::text as count
      FROM match_clusters
      WHERE reviewed_at IS NOT NULL AND reviewed_at >= NOW() - INTERVAL '${days} days'
      GROUP BY DATE_TRUNC('day', reviewed_at)::date
    `;

    const activityData = dates.map((date) => {
      const created = createdCounts.find((c) => c.date === date)?.count || '0';
      const reviewed = reviewedCounts.find((r) => r.date === date)?.count || '0';

      return {
        date,
        created: parseInt(created, 10),
        reviewed: parseInt(reviewed, 10),
      };
    });

    return NextResponse.json(activityData);
  } catch (error) {
    console.error('Error fetching cluster activity:', error);
    return NextResponse.json(
      { error: 'Failed to fetch cluster activity' },
      { status: 500 }
    );
  }
}