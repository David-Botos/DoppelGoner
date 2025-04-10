// app/api/dashboard/methods/route.ts
import { NextResponse } from 'next/server';
import prisma from '@/lib/db/prisma';

export async function GET() {
  try {
    // Using raw query to get method breakdown
    const methods = await prisma.$queryRaw<
      { method_name: string; count: string }[]
    >`
      SELECT method_name, COUNT(*)::text as count
      FROM matching_methods
      GROUP BY method_name
      ORDER BY count DESC
    `;

    const methodBreakdown = methods.map((m) => ({
      name: m.method_name,
      value: parseInt(m.count, 10),
    }));

    return NextResponse.json(methodBreakdown);
  } catch (error) {
    console.error('Error fetching method breakdown:', error);
    return NextResponse.json(
      { error: 'Failed to fetch method breakdown' },
      { status: 500 }
    );
  }
}