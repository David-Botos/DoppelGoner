// components/TaskDetails/index.tsx
import React from 'react';
import { EntityGroup, EntityDetails } from '../../lib/types';
import ActionButtons from './ActionButtons';

interface TaskDetailsProps {
  task: EntityGroup | null;
  entityDetailsCache: Record<string, EntityDetails>;
  onConfirm: () => void;
  onReject: () => void;
  submitting: boolean;
  error: Error | null;
}

const TaskDetails: React.FC<TaskDetailsProps> = ({
  task,
  entityDetailsCache,
  onConfirm,
  onReject,
  submitting,
  error
}) => {
  if (!task) {
    return (
      <div className="p-4 text-center text-gray-500">
        No task selected. Select a cluster to begin reviewing.
      </div>
    );
  }

  const sourceDetails = entityDetailsCache[task.source];
  const targetDetails = entityDetailsCache[task.target];
  
  // Helper function to safely render values (handle empty objects and nulls)
  const safeRender = (value: unknown): string => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'object' && Object.keys(value).length === 0) return 'N/A';
    return String(value);
  };
  
  // Helper function to render match values based on method type
  const renderMatchValues = () => {
    const { match_values } = task;
    if (!match_values || !match_values.values) {
      return <div>No match evidence available</div>;
    }
    
    switch (task.method_type) {
      case 'email':
        return (
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="font-medium">Organization 1 Email:</p>
                <p className="text-gray-700">{safeRender(match_values.values.original_email1)}</p>
              </div>
              <div>
                <p className="font-medium">Organization 2 Email:</p>
                <p className="text-gray-700">{safeRender(match_values.values.original_email2)}</p>
              </div>
            </div>
            <div>
              <p className="font-medium">Normalized Shared Email:</p>
              <p className="text-gray-700">{safeRender(match_values.values.normalized_shared_email)}</p>
            </div>
          </div>
        );
        
      case 'name':
        return (
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="font-medium">Organization 1 Name:</p>
                <p className="text-gray-700">{safeRender(match_values.values.original_name1)}</p>
              </div>
              <div>
                <p className="font-medium">Organization 2 Name:</p>
                <p className="text-gray-700">{safeRender(match_values.values.original_name2)}</p>
              </div>
            </div>
            <div>
              <p className="font-medium">Name Similarity:</p>
              <p className="text-gray-700">{((match_values.values.pre_rl_similarity_score as number) || 0).toFixed(2)}</p>
            </div>
          </div>
        );
        
      case 'url':
        return (
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="font-medium">Organization 1 URL:</p>
                <p className="text-gray-700 break-all">{safeRender(match_values.values.original_url1)}</p>
              </div>
              <div>
                <p className="font-medium">Organization 2 URL:</p>
                <p className="text-gray-700 break-all">{safeRender(match_values.values.original_url2)}</p>
              </div>
            </div>
            <div>
              <p className="font-medium">Normalized Shared URL:</p>
              <p className="text-gray-700 break-all">{safeRender(match_values.values.normalized_shared_url)}</p>
            </div>
          </div>
        );
        
      case 'phone':
        return (
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="font-medium">Organization 1 Phone:</p>
                <p className="text-gray-700">{safeRender(match_values.values.original_phone1)}</p>
              </div>
              <div>
                <p className="font-medium">Organization 2 Phone:</p>
                <p className="text-gray-700">{safeRender(match_values.values.original_phone2)}</p>
              </div>
            </div>
            <div>
              <p className="font-medium">Normalized Shared Phone:</p>
              <p className="text-gray-700">{safeRender(match_values.values.normalized_shared_phone)}</p>
            </div>
          </div>
        );
        
      default:
        return (
          <div>
            <p className="font-medium">Match Evidence:</p>
            <pre className="p-2 mt-1 text-xs bg-gray-100 rounded overflow-auto">
              {JSON.stringify(match_values.values, null, 2)}
            </pre>
          </div>
        );
    }
  };

  return (
    <div className="p-4 h-full overflow-auto">
      <h2 className="text-xl font-bold mb-4">Entity Link Review</h2>
      
      <div className="p-4 border rounded-md bg-gray-50 mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="font-medium">Match Method:</span>
          <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
            {task.method_type}
          </span>
        </div>
        
        <div className="mb-2">
          <span className="font-medium">Confidence Score:</span>
          <span className="ml-2">{task.confidence_score.toFixed(2)}</span>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-6 mb-6">
        <div className="border p-4 rounded-md">
          <h3 className="font-bold mb-2">Entity 1</h3>
          <p className="font-medium">{sourceDetails?.organization_name || 'Loading...'}</p>
          {sourceDetails?.organization_url && (
            <p className="text-sm text-gray-700 truncate">
              <span className="font-medium">URL:</span> {sourceDetails.organization_url}
            </p>
          )}
        </div>
        
        <div className="border p-4 rounded-md">
          <h3 className="font-bold mb-2">Entity 2</h3>
          <p className="font-medium">{targetDetails?.organization_name || 'Loading...'}</p>
          {targetDetails?.organization_url && (
            <p className="text-sm text-gray-700 truncate">
              <span className="font-medium">URL:</span> {targetDetails.organization_url}
            </p>
          )}
        </div>
      </div>
      
      <div className="mb-6">
        <h3 className="font-bold mb-2">Match Evidence</h3>
        <div className="border p-4 rounded-md">
          {renderMatchValues()}
        </div>
      </div>
      
      <div className="mb-4">
        <p className="font-medium">Decision:</p>
        <p className="text-sm text-gray-700 mb-4">
          Do these two entities represent the same real-world organization?
        </p>
        
        <ActionButtons
          onConfirm={onConfirm}
          onReject={onReject}
          submitting={submitting}
        />
      </div>
      
      {error && (
        <div className="p-3 bg-red-100 text-red-800 rounded-md mt-4">
          Error: {error.message}
        </div>
      )}
    </div>
  );
};

export default TaskDetails;