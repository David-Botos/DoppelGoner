// app/page-with-context.tsx
'use client';

import TaskQueue from '../components/TaskQueue';
import GraphVisualization from '../components/GraphVisualization';
import TaskDetails from '../components/TaskDetails';
import { useClusterContext } from '../context/ClusterContext';
import ClusterSelector from '@/components/ClusterTools/ClusterSelector';

export default function Home() {
  const {
    // Clusters state
    clusters,
    selectedClusterId,
    setSelectedClusterId,
    loadingClusters,
    clusterError,
    
    // Graph data state
    graphData,
    loadingGraph,
    graphError,
    
    // Task queue state
    taskQueue,
    currentTaskIndex,
    currentTask,
    setTaskIndex,
    
    // Entity details state
    entityDetailsCache,
    loadingEntityDetails,
    fetchEntityDetails,
    
    // Feedback state
    submittingFeedback,
    feedbackError,
    submitFeedback
  } = useClusterContext();

  // Handle feedback submission
  const handleFeedback = async (decision: 'confirm' | 'reject') => {
    if (!currentTask) return;
    
    const success = await submitFeedback(decision, currentTask.id);
    
    if (success && decision === 'confirm') {
      // Move to next task if confirmed (reject already removes from queue)
      setTaskIndex(currentTaskIndex + 1);
    }
  };

  return (
    <div className="flex h-screen p-4 bg-gray-50">
      {/* Left Pane: Task Queue */}
      <div className="w-1/3 pr-4 h-full overflow-hidden flex flex-col">
        <div className="mb-4">
          <h1 className="text-2xl font-bold">Entity Cluster Review</h1>
          <p className="text-gray-600">
            Review and validate entity links identified by the backend resolution process.
          </p>
        </div>
        
        <ClusterSelector 
          clusters={clusters}
          onSelectCluster={setSelectedClusterId} 
          selectedClusterId={selectedClusterId}
          loading={loadingClusters}
          error={clusterError}
        />
        
        <div className="flex-1 overflow-hidden border rounded-lg">
          <div className="p-4 bg-white h-full overflow-hidden flex flex-col">
            <h2 className="text-lg font-bold mb-4">Task Queue</h2>
            
            {loadingGraph ? (
              <div className="flex-1 flex items-center justify-center">
                <p>Loading tasks...</p>
              </div>
            ) : graphError ? (
              <div className="flex-1 flex items-center justify-center">
                <p className="text-red-500">Error: {graphError.message}</p>
              </div>
            ) : !graphData ? (
              <div className="flex-1 flex items-center justify-center">
                <p>Select a cluster to view tasks</p>
              </div>
            ) : taskQueue.length === 0 ? (
              <div className="flex-1 flex items-center justify-center">
                <p>No tasks to review in this cluster</p>
              </div>
            ) : (
              <div className="flex-1 overflow-hidden">
                <TaskQueue 
                  tasks={taskQueue}
                  currentTaskIndex={currentTaskIndex}
                  onSelectTask={setTaskIndex}
                  entityDetailsCache={entityDetailsCache}
                />
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Right Container */}
      <div className="w-2/3 h-full flex flex-col">
        {/* Top Right: Graph Visualization */}
        <div className="h-1/2 mb-4 border rounded-lg bg-white overflow-hidden">
          <div className="p-4 h-full flex flex-col">
            <h2 className="text-lg font-bold mb-2">Graph Visualization</h2>
            
            {loadingGraph ? (
              <div className="flex-1 flex items-center justify-center">
                <p>Loading graph...</p>
              </div>
            ) : graphError ? (
              <div className="flex-1 flex items-center justify-center">
                <p className="text-red-500">Error: {graphError.message}</p>
              </div>
            ) : !graphData ? (
              <div className="flex-1 flex items-center justify-center">
                <p>Select a cluster to view the graph</p>
              </div>
            ) : (
              <div className="flex-1">
                <GraphVisualization 
                  graphData={graphData}
                  currentTask={currentTask}
                  entityDetailsCache={entityDetailsCache}
                  onNodeHover={fetchEntityDetails} // Passed as onNodeHover
                  detailsLoading={loadingEntityDetails} // Passed as detailsLoading
                  // Add loadingGraph to inform GraphVisualization about the main graph loading status
                  // This helps GraphVisualization decide if its own internal "settled" state can even begin to be evaluated.
                  isLoadingGraphStructure={loadingGraph} 
                />
              </div>
            )}
          </div>
        </div>
        
        {/* Bottom Right: Task Details */}
        <div className="h-1/2 border rounded-lg bg-white overflow-hidden">
          <div className="h-full">
            <TaskDetails 
              task={currentTask}
              entityDetailsCache={entityDetailsCache}
              onConfirm={() => handleFeedback('confirm')}
              onReject={() => handleFeedback('reject')}
              submitting={submittingFeedback}
              error={feedbackError}
            />
          </div>
        </div>
      </div>
    </div>
  );
}