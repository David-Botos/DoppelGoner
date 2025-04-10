// app/api/dashboard/route.ts
import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const days = parseInt(searchParams.get("days") || "30", 10);

    // Fetch data from each endpoint separately
    const [statsRes, activityRes, methodsRes, reviewsRes] = await Promise.all([
      fetch(new URL("/api/dashboard/stats", request.url)),
      fetch(new URL(`/api/dashboard/activity?days=${days}`, request.url)),
      fetch(new URL("/api/dashboard/methods", request.url)),
      fetch(new URL("/api/dashboard/reviews", request.url)),
    ]);

    // Check if any fetch failed
    if (!statsRes.ok || !activityRes.ok || !methodsRes.ok || !reviewsRes.ok) {
      throw new Error("One or more dashboard APIs failed");
    }

    // Parse the responses
    const [stats, activity, methods, reviews] = await Promise.all([
      statsRes.json(),
      activityRes.json(),
      methodsRes.json(),
      reviewsRes.json(),
    ]);

    return NextResponse.json({
      stats,
      activity,
      methods,
      reviews,
    });
  } catch (error) {
    console.error("Error fetching dashboard data:", error);
    return NextResponse.json(
      { error: "Failed to fetch dashboard data" },
      { status: 500 }
    );
  }
}
