// components/ClusterTools/ClusterCoherenceChart.tsx
import React, { useMemo } from 'react';
import { ClusterSummary } from '@/lib/types';

interface ClusterCoherenceChartProps {
  clusters: ClusterSummary[];
  selectedClusterId: string | null;
  onSelectCluster: (clusterId: string) => void;
}

const ClusterCoherenceChart: React.FC<ClusterCoherenceChartProps> = ({
  clusters,
  selectedClusterId,
  onSelectCluster
}) => {
  // Sort clusters by coherence score (asc) so low scores (potentially problematic)
  // appear first to draw attention to clusters that need review
  const sortedClusters = useMemo(() => {
    return [...clusters]
      .filter(c => c.average_coherence_score !== undefined)
      .sort((a, b) => {
        const scoreA = a.average_coherence_score || 0;
        const scoreB = b.average_coherence_score || 0;
        return scoreA - scoreB;
      })
      .slice(0, 10); // Limit to top 10 lowest coherence scores
  }, [clusters]);
  
  if (sortedClusters.length === 0) {
    return null;
  }
  
  // Helper function to get color based on coherence score
  const getCoherenceColor = (score: number): string => {
    if (score >= 0.7) return 'bg-green-500';
    if (score >= 0.4) return 'bg-yellow-500';
    return 'bg-red-500';
  };
  
  return (
    <div className="bg-white p-4 rounded-lg shadow-sm border mb-4">
      <h3 className="font-medium text-lg mb-3">Cluster Coherence Overview</h3>
      <p className="text-sm text-gray-600 mb-3">
        Clusters with lower coherence scores may contain questionable matches that need review.
      </p>
      
      <div className="space-y-2">
        {sortedClusters.map(cluster => {
          const coherenceScore = cluster.average_coherence_score || 0;
          const scoreWidth = `${Math.max(5, coherenceScore * 100)}%`; // Ensure at least 5% width for visibility
          const isSelected = cluster.id === selectedClusterId;
          
          return (
            <button
              key={cluster.id}
              className={`w-full text-left block p-2 rounded-md transition-colors ${
                isSelected ? 'bg-blue-50 border border-blue-200' : 'hover:bg-gray-50 border border-gray-100'
              }`}
              onClick={() => onSelectCluster(cluster.id)}
            >
              <div className="flex justify-between items-center mb-1">
                <div className="font-medium truncate pr-2" style={{maxWidth: '80%'}}>
                  {cluster.name || `Cluster ${cluster.id.substring(0, 8)}...`}
                </div>
                <div className="text-sm text-gray-500">
                  {(coherenceScore * 100).toFixed(1)}%
                </div>
              </div>
              
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div 
                  className={`h-2.5 rounded-full ${getCoherenceColor(coherenceScore)}`} 
                  style={{width: scoreWidth}}
                ></div>
              </div>
              
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>{cluster.entity_count} entities</span>
                <span>{cluster.group_count} links</span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default ClusterCoherenceChart;