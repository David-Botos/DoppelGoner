// components/ClusterSplitConfirmation.tsx
import { simulateClusterSplit } from '@/lib/graph-utils';
import { EntityDetails, EntityGroup, GraphData } from '@/lib/types';
import React, { useState } from 'react';


interface ClusterSplitConfirmationProps {
  graphData: GraphData;
  entityGroup: EntityGroup;
  onConfirm: () => void;
  onCancel: () => void;
  entityDetailsCache: Record<string, EntityDetails>;
}

const ClusterSplitConfirmation: React.FC<ClusterSplitConfirmationProps> = ({
  graphData,
  entityGroup,
  onConfirm,
  onCancel,
  entityDetailsCache
}) => {
  const [loading, setLoading] = useState(false);
  
  // Check if cutting this link would split the cluster
  const { canSplit, originalComponent, newComponent } = simulateClusterSplit(graphData, entityGroup.id);
  
  if (!canSplit) {
    return null; // Don't show if no split will occur
  }
  
  // Get entity names from cache or fallback
  const getEntityName = (entityId: string): string => {
    if (entityDetailsCache[entityId]?.organization_name) {
      return entityDetailsCache[entityId].organization_name;
    }
    return `Entity ${entityId.substring(0, 8)}...`;
  };
  
  const originalSize = originalComponent?.nodes.length || 0;
  const newSize = newComponent?.nodes.length || 0;
  
  const handleConfirm = () => {
    setLoading(true);
    onConfirm();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl p-6 max-w-lg w-full">
        <h2 className="text-xl font-bold mb-4">Confirm Cluster Split</h2>
        
        <div className="mb-4">
          <p className="text-red-600 font-medium mb-2">
            Warning: This action will split the cluster into two separate clusters.
          </p>
          <p className="mb-2">
            Cutting the link between <span className="font-medium">{getEntityName(entityGroup.source)}</span> and{' '}
            <span className="font-medium">{getEntityName(entityGroup.target)}</span> will result in:
          </p>
        </div>
        
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="border rounded-md p-3 bg-gray-50">
            <p className="font-medium">Original Cluster</p>
            <p className="text-sm text-gray-600">{originalSize} entities remain</p>
          </div>
          <div className="border rounded-md p-3 bg-gray-50">
            <p className="font-medium">New Cluster</p>
            <p className="text-sm text-gray-600">{newSize} entities will form a new cluster</p>
          </div>
        </div>
        
        <div className="flex justify-end space-x-3">
          <button
            className="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
            onClick={onCancel}
            disabled={loading}
          >
            Cancel
          </button>
          <button
            className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors disabled:opacity-50"
            onClick={handleConfirm}
            disabled={loading}
          >
            {loading ? 'Processing...' : 'Confirm Split'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ClusterSplitConfirmation;