// app/page.tsx
import { Suspense } from 'react';
import DashboardStats from '@/components/dashboard/DashboardStats';
import ClusterActivityChart from '@/components/dashboard/ClusterActivityChart';
import RecentReviews from '@/components/dashboard/RecentReviews';
import MatchingMethodsBreakdown from '@/components/dashboard/MatchingMethodsBreakdown';
import Loading from '@/components/ui/Loading';
import Card, { CardHeader, CardBody } from '@/components/ui/Card';

export default function DashboardPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">Entity Resolution Dashboard</h1>
      
      <Suspense fallback={<Loading />}>
        <DashboardStats />
      </Suspense>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
        <Card>
          <CardHeader>
            <h2 className="text-lg font-semibold text-gray-900">Cluster Activity</h2>
          </CardHeader>
          <CardBody>
            <Suspense fallback={<Loading />}>
              <ClusterActivityChart />
            </Suspense>
          </CardBody>
        </Card>
        
        <Card>
          <CardHeader>
            <h2 className="text-lg font-semibold text-gray-900">Matching Methods Breakdown</h2>
          </CardHeader>
          <CardBody>
            <Suspense fallback={<Loading />}>
              <MatchingMethodsBreakdown />
            </Suspense>
          </CardBody>
        </Card>
      </div>
      
      <div className="mt-8">
        <Card>
          <CardHeader>
            <h2 className="text-lg font-semibold text-gray-900">Recent Reviews</h2>
          </CardHeader>
          <CardBody>
            <Suspense fallback={<Loading />}>
              <RecentReviews />
            </Suspense>
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
