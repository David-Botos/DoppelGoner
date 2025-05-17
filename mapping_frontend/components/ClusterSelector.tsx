// components/ClusterSelector.tsx
import React from 'react';
import { ClusterSummary } from '@/lib/types';

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
      
      {loading && (
        <div className="text-sm text-gray-500 flex items-center">
          <svg className="animate-spin h-4 w-4 mr-2 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          Loading clusters...
        </div>
      )}
      
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
              {cluster.average_coherence_score !== undefined && 
                ` - Coherence: ${(cluster.average_coherence_score * 100).toFixed(1)}%`}
            </option>
          ))}
        </select>
      )}
      
      {selectedClusterId && (
        <div className="mt-2 text-xs text-gray-500">
          Showing details for cluster ID: {selectedClusterId}
        </div>
      )}
    </div>
  );
};

export default ClusterSelector;