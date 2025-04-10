// app/clusters/page.tsx
import { Suspense } from 'react';
import ClusterList from '@/components/cluster/ClusterList';
import Loading from '@/components/ui/Loading';

export default function ClustersPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">Entity Resolution Clusters</h1>
      <Suspense fallback={<Loading />}>
        <ClusterList />
      </Suspense>
    </div>
  );
}