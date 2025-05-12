// components/ClusterSelector-updated.tsx
import { ClusterSummary } from '@/lib/types';
import React from 'react';

interface ClusterSelectorProps {
  clusters: ClusterSummary[];
  onSelectCluster: (clusterId: string) => void;
  selectedClusterId: string | null;
  loading: boolean;
  error: Error | null;
}

const ClusterSelector: React.FC<ClusterSelectorProps> = ({
  clusters,
  onSelectCluster,
  selectedClusterId,
  loading,
  error
}) => {
  return (
    <div className="mb-6">
      <label htmlFor="cluster-select" className="block text-sm font-medium text-gray-700 mb-1">
        Select Cluster to Review
      </label>
      
      {loading && <div className="text-sm text-gray-500">Loading clusters...</div>}
      
      {error && (
        <div className="p-2 bg-red-100 text-red-800 rounded-md text-sm mb-2">
          Error loading clusters: {error.message}
        </div>
      )}
      
      {!loading && clusters.length === 0 ? (
        <div className="text-sm text-gray-500">No clusters available for review</div>
      ) : (
        <select
          id="cluster-select"
          className="block w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          value={selectedClusterId || ''}
          onChange={(e) => onSelectCluster(e.target.value)}
          disabled={loading}
        >
          <option value="">-- Select a cluster --</option>
          {clusters.map((cluster) => (
            <option key={cluster.id} value={cluster.id}>
              {cluster.name || `Cluster ${cluster.id.substring(0, 8)}...`} ({cluster.entity_count} entities, {cluster.group_count} links)
            </option>
          ))}
        </select>
      )}
    </div>
  );
};

export default ClusterSelector;