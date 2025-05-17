// components/ClusterTools/ClusterStatistics.tsx
import React, { useMemo } from 'react';
import { GraphData } from '@/lib/types';
import { calculateAverageConfidence, findCentralNodes, findWeakestEdges } from '@/lib/graph-utils';
import { formatConfidence } from '@/lib/visualization-utils';

interface ClusterStatisticsProps {
  graphData: GraphData | null;
}

const ClusterStatistics: React.FC<ClusterStatisticsProps> = ({ graphData }) => {
  const stats = useMemo(() => {
    if (!graphData || !graphData.nodes || !graphData.links) {
      return {
        nodeCount: 0,
        linkCount: 0,
        averageConfidence: 0,
        centralNodes: [],
        weakestEdges: [],
        // Add defaults for cluster info
        clusterName: '',
        clusterDescription: '',
        databaseEntityCount: 0,
        databaseLinkCount: 0,
        coherenceScore: 0
      };
    }
    
    // Use pre-computed values from database when available
    const databaseEntityCount = graphData.clusterInfo?.entity_count || 0;
    const databaseLinkCount = graphData.clusterInfo?.group_count || 0;
    const coherenceScore = graphData.clusterInfo?.average_coherence_score || 0;
    
    return {
      // Actual counts from the graph data (might be a subset)
      nodeCount: graphData.nodes.length,
      linkCount: graphData.links.length,
      // Use the coherence score from the database if available, otherwise calculate
      averageConfidence: coherenceScore || calculateAverageConfidence(graphData),
      centralNodes: findCentralNodes(graphData, 3),
      weakestEdges: findWeakestEdges(graphData, 3),
      // Add cluster metadata
      clusterName: graphData.clusterInfo?.name || '',
      clusterDescription: graphData.clusterInfo?.description || '',
      databaseEntityCount,
      databaseLinkCount,
      coherenceScore
    };
  }, [graphData]);
  
  if (!graphData) {
    return (
      <div className="bg-white p-4 rounded-lg shadow-sm border text-center text-gray-500">
        <p>Select a cluster to view statistics</p>
      </div>
    );
  }
  
  return (
    <div className="bg-white p-4 rounded-lg shadow-sm border">
      {/* Display cluster name and description if available */}
      {stats.clusterName && (
        <div className="mb-3">
          <h3 className="font-medium text-lg">{stats.clusterName}</h3>
          {stats.clusterDescription && (
            <p className="text-sm text-gray-600 mt-1">{stats.clusterDescription}</p>
          )}
        </div>
      )}
      
      <h3 className="font-medium text-lg mb-3">Cluster Statistics</h3>
      
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-4">
        <div className="border rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-blue-600">
            {stats.databaseEntityCount > 0 ? stats.databaseEntityCount : stats.nodeCount}
          </div>
          <div className="text-xs text-gray-500">Entities</div>
          
          {/* Show both numbers if they differ */}
          {stats.databaseEntityCount > 0 && stats.databaseEntityCount !== stats.nodeCount && (
            <div className="text-xs text-gray-400 mt-1">
              {stats.nodeCount} loaded
            </div>
          )}
        </div>
        
        <div className="border rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-blue-600">
            {stats.databaseLinkCount > 0 ? stats.databaseLinkCount : stats.linkCount}
          </div>
          <div className="text-xs text-gray-500">Connections</div>
          
          {/* Show both numbers if they differ */}
          {stats.databaseLinkCount > 0 && stats.databaseLinkCount !== stats.linkCount && (
            <div className="text-xs text-gray-400 mt-1">
              {stats.linkCount} loaded
            </div>
          )}
        </div>
        
        <div className="border rounded-lg p-3 text-center col-span-2 sm:col-span-1">
          <div className="text-2xl font-bold text-blue-600">{formatConfidence(stats.averageConfidence)}</div>
          <div className="text-xs text-gray-500">
            {stats.coherenceScore > 0 ? 'Coherence Score' : 'Avg. Confidence'}
          </div>
        </div>
      </div>
      
      <div className="mb-3">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Most Connected Entities</h4>
        <ul className="text-sm">
          {stats.centralNodes.length > 0 ? (
            stats.centralNodes.map((node) => (
              <li key={node.id} className="py-1 px-2 hover:bg-gray-50 rounded truncate">
                {node.name}
              </li>
            ))
          ) : (
            <li className="py-1 px-2 text-gray-500 italic">No data available</li>
          )}
        </ul>
      </div>
      
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-2">Lowest Confidence Connections</h4>
        <ul className="text-sm">
          {stats.weakestEdges.length > 0 ? (
            stats.weakestEdges.map((edge) => (
              <li key={edge.id} className="py-1 px-2 hover:bg-gray-50 rounded flex justify-between">
                <div className="truncate flex-1">
                  Connection {edge.id.substring(0, 8)}...
                </div>
                <span className={`text-xs px-1.5 py-0.5 rounded-full ml-2 ${
                  edge.edge_weight < 0.4 ? 'bg-red-100 text-red-800' : 
                  edge.edge_weight < 0.7 ? 'bg-yellow-100 text-yellow-800' : 
                  'bg-green-100 text-green-800'
                }`}>
                  {formatConfidence(edge.edge_weight)}
                </span>
              </li>
            ))
          ) : (
            <li className="py-1 px-2 text-gray-500 italic">No data available</li>
          )}
        </ul>
      </div>
    </div>
  );
};

export default ClusterStatistics;