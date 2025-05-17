// app/page.tsx
'use client';

import React from 'react';
import { useClusterContext } from '../context/ClusterContext';
import AppLayout from '@/components/layout/AppLayout';
import ClusterSelector from '@/components/ClusterSelector';
import GraphVisualization from '@/components/GraphVisualization';
import ConnectionDetails from '@/components/ConnectionDetails';
import ClusterStatistics from '@/components/ClusterTools/ClusterStatistics';
import ClusterCoherenceChart from '@/components/ClusterTools/ClusterCoherenceChart';

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
    
    // Connection state
    selectedConnection,
    setSelectedConnection,
    
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
    if (!selectedConnection) return;
    
    const success = await submitFeedback(decision, selectedConnection);
    
    // If the operation was successful, the context will automatically update the UI
    if (success) {
      // Additional UI feedback could be added here if needed
    }
  };

  return (
    <AppLayout>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="mb-6">
          <h1 className="text-2xl font-bold">Entity Cluster Review</h1>
          <p className="text-gray-600">
            Review and validate entity links identified by the entity resolution pipeline.
          </p>
        </div>
        
        <div className="flex flex-col md:flex-row h-[calc(100vh-12rem)] gap-4">
          {/* Left Pane: Cluster selection and tools */}
          <div className="w-full md:w-1/3 h-full overflow-hidden flex flex-col">
            <ClusterSelector 
              clusters={clusters}
              onSelectCluster={setSelectedClusterId} 
              selectedClusterId={selectedClusterId}
              loading={loadingClusters}
              error={clusterError}
            />
            
            {/* Add the ClusterCoherenceChart component */}
            {!loadingClusters && clusters.length > 0 && (
              <ClusterCoherenceChart
                clusters={clusters}
                selectedClusterId={selectedClusterId}
                onSelectCluster={setSelectedClusterId}
              />
            )}
            
            <div className="mb-4">
              <ClusterStatistics graphData={graphData} />
            </div>
            
            <div className="flex-1 overflow-hidden border rounded-lg bg-white p-4">
              <h2 className="text-lg font-bold mb-4">Review Instructions</h2>
              
              {!selectedClusterId ? (
                <p className="text-gray-600">Select a cluster to begin reviewing entity connections.</p>
              ) : !graphData ? (
                <p className="text-gray-600">Loading cluster data...</p>
              ) : (
                <div>
                  <p className="mb-2">This cluster contains:</p>
                  <ul className="list-disc pl-5 mb-4">
                    <li>{graphData.nodes.length} entities</li>
                    <li>{graphData.links.length} connections to review</li>
                  </ul>
                  
                  <p className="mb-2">To review connections:</p>
                  <ol className="list-decimal pl-5">
                    <li className="mb-1">Click on a connection line in the graph to select it for review</li>
                    <li className="mb-1">Examine the match evidence and entity details</li>
                    <li className="mb-1">Click &apos;Confirm&apos; if the entities are the same or &apos;Reject&apos; if they&apos;re different</li>
                  </ol>
                </div>
              )}
            </div>
          </div>
          
          {/* Right Container */}
          <div className="w-full md:w-2/3 h-full flex flex-col">
            {/* Top Right: Graph Visualization */}
            <div className="h-1/2 mb-4 border rounded-lg bg-white overflow-hidden">
              <div className="p-4 h-full flex flex-col">
                <h2 className="text-lg font-bold mb-2">Graph Visualization</h2>
                
                {loadingGraph ? (
                  <div className="flex-1 flex items-center justify-center">
                    <div className="flex flex-col items-center">
                      <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500 mb-2"></div>
                      <p>Loading graph...</p>
                    </div>
                  </div>
                ) : graphError ? (
                  <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-red-500 text-3xl mb-2">⚠️</div>
                      <p className="text-red-500">Error: {graphError.message}</p>
                    </div>
                  </div>
                ) : !graphData ? (
                  <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                      <svg className="h-12 w-12 text-gray-400 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                      <p>Select a cluster to view the graph</p>
                    </div>
                  </div>
                ) : (
                  <div className="flex-1">
                    <GraphVisualization 
                      graphData={graphData}
                      selectedConnection={selectedConnection}
                      onSelectConnection={setSelectedConnection}
                      entityDetailsCache={entityDetailsCache}
                      onNodeHover={fetchEntityDetails}
                      detailsLoading={loadingEntityDetails}
                      isLoadingGraphStructure={loadingGraph}
                    />
                  </div>
                )}
              </div>
            </div>
            
            {/* Bottom Right: Connection Details */}
            <div className="h-1/2 border rounded-lg bg-white overflow-hidden">
              <div className="h-full">
                <ConnectionDetails 
                  connection={selectedConnection}
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
      </div>
    </AppLayout>
  );
}