// context/ClusterContext.tsx
"use client";

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  ReactNode,
} from "react";
import {
  ClusterSummary,
  GraphData,
  VisualizationEntityEdge,
  EntityDetails,
  FeedbackRequest,
} from "../lib/types";
import {
  fetchClusters,
  fetchClusterGraph,
  fetchEntityDetails,
  confirmLink,
  rejectLink,
} from "../lib/api";

interface ClusterDisplayState {
  graphData: GraphData | null;
  selectedConnection: VisualizationEntityEdge | null;
}

interface ClusterContextValue {
  // Clusters state
  clusters: ClusterSummary[];
  selectedClusterId: string | null;
  setSelectedClusterId: (id: string | null) => void;
  loadingClusters: boolean;
  clusterError: Error | null;

  // Graph display state
  graphData: GraphData | null;
  selectedConnection: VisualizationEntityEdge | null;
  setSelectedConnection: (connection: VisualizationEntityEdge | null) => void;
  loadingGraph: boolean;
  graphError: Error | null;

  // Entity details state
  entityDetailsCache: Record<string, EntityDetails>;
  loadingEntityDetails: Record<string, boolean>;
  entityDetailsError: Record<string, Error | null>;
  fetchEntityDetails: (entityId: string) => Promise<EntityDetails | null>;

  // Feedback state
  submittingFeedback: boolean;
  feedbackError: Error | null;
  submitFeedback: (
    decision: "confirm" | "reject",
    edge: VisualizationEntityEdge
  ) => Promise<boolean>;

  // Graph modification functions
  refreshClusterGraph: () => Promise<void>;
}

const ClusterContext = createContext<ClusterContextValue | undefined>(
  undefined
);

const initialClusterDisplayState: ClusterDisplayState = {
  graphData: null,
  selectedConnection: null,
};

export function ClusterProvider({ children }: { children: ReactNode }) {
  // Clusters state
  const [clusters, setClusters] = useState<ClusterSummary[]>([]);
  const [selectedClusterId, setSelectedClusterId] = useState<string | null>(null);
  const [loadingClusters, setLoadingClusters] = useState(false);
  const [clusterError, setClusterError] = useState<Error | null>(null);

  // Consolidated state for graph display
  const [clusterDisplayState, setClusterDisplayState] = useState<ClusterDisplayState>(initialClusterDisplayState);
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [graphError, setGraphError] = useState<Error | null>(null);

  // Entity details state
  const [entityDetailsCache, setEntityDetailsCache] = useState<Record<string, EntityDetails>>({});
  const [loadingEntityDetails, setLoadingEntityDetails] = useState<Record<string, boolean>>({});
  const [entityDetailsError, setEntityDetailsError] = useState<Record<string, Error | null>>({});

  // Feedback state
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [feedbackError, setFeedbackError] = useState<Error | null>(null);

  // --- Effects ---

  // Fetch list of clusters on mount
  useEffect(() => {
    async function loadClusters() {
      setLoadingClusters(true);
      setClusterError(null);
      try {
        const data = await fetchClusters();
        setClusters(data);
      } catch (err) {
        console.error("Error fetching clusters:", err);
        setClusterError(
          err instanceof Error ? err : new Error("Failed to fetch clusters")
        );
      } finally {
        setLoadingClusters(false);
      }
    }
    loadClusters();
  }, []);

  // Fetch graph data when selected cluster changes
  useEffect(() => {
    async function loadGraphDataForCluster() {
      if (!selectedClusterId) {
        setClusterDisplayState(initialClusterDisplayState);
        setGraphError(null);
        return;
      }

      setLoadingGraph(true);
      setGraphError(null);
      try {
        const data = await fetchClusterGraph(selectedClusterId);
        setClusterDisplayState({
          graphData: data,
          selectedConnection: data.links && data.links.length > 0 ? data.links[0] : null,
        });
      } catch (err) {
        console.error("Error fetching graph data for cluster:", selectedClusterId, err);
        setClusterDisplayState(initialClusterDisplayState);
        setGraphError(
          err instanceof Error ? err : new Error("Failed to fetch graph data")
        );
      } finally {
        setLoadingGraph(false);
      }
    }
    loadGraphDataForCluster();
  }, [selectedClusterId]);

  // Function to fetch entity details (memoized)
  const fetchEntityDetailsById = useCallback(
    async (entityId: string, forceRefetch: boolean = false): Promise<EntityDetails | null> => {
      if (!forceRefetch && entityDetailsCache[entityId]) {
        return entityDetailsCache[entityId];
      }

      if (loadingEntityDetails[entityId] && !forceRefetch) {
        return null; // Already loading
      }

      setLoadingEntityDetails((prev) => ({ ...prev, [entityId]: true }));
      setEntityDetailsError((prev) => ({ ...prev, [entityId]: null }));

      try {
        const details = await fetchEntityDetails(entityId);
        setEntityDetailsCache((prev) => ({ ...prev, [entityId]: details }));
        return details;
      } catch (err) {
        console.error(`Error fetching details for entity ${entityId}:`, err);
        setEntityDetailsError((prev) => ({
          ...prev,
          [entityId]:
            err instanceof Error
              ? err
              : new Error(`Failed to fetch details for entity ${entityId}`),
        }));
        return null;
      } finally {
        setLoadingEntityDetails((prev) => ({ ...prev, [entityId]: false }));
      }
    },
    [entityDetailsCache, loadingEntityDetails]
  );

  // Prefetch entity details for nodes in the graph
  useEffect(() => {
    if (clusterDisplayState.graphData?.nodes) {
      const nodesToPrefetch = clusterDisplayState.graphData.nodes.slice(0, 10); // Limit initial prefetch
      
      nodesToPrefetch.forEach(node => {
        fetchEntityDetailsById(node.id, false);
      });
    }
  }, [clusterDisplayState.graphData, fetchEntityDetailsById]);

  // Update selected connection
  const setSelectedConnection = useCallback((connection: VisualizationEntityEdge | null) => {
    setClusterDisplayState(prev => ({
      ...prev,
      selectedConnection: connection
    }));

    // Prefetch details for the entities in the selected connection
    if (connection) {
      fetchEntityDetailsById(connection.entity_id_1, false);
      fetchEntityDetailsById(connection.entity_id_2, false);
    }
  }, [fetchEntityDetailsById]);

  // Handler to change selected cluster and reset selection
  const handleSelectCluster = useCallback((clusterId: string | null) => {
    setSelectedClusterId(clusterId);
    // The useEffect will reset clusterDisplayState
  }, []);

  // Refresh cluster graph - useful after modifications
  const refreshClusterGraph = useCallback(async () => {
    if (!selectedClusterId) return;

    setLoadingGraph(true);
    setGraphError(null);
    try {
      const data = await fetchClusterGraph(selectedClusterId);
      
      // Try to maintain the selected connection if it still exists
      const currentConnectionId = clusterDisplayState.selectedConnection?.id;
      const newSelectedConnection = currentConnectionId
        ? data.links.find(link => link.id === currentConnectionId) || (data.links.length > 0 ? data.links[0] : null)
        : data.links.length > 0 ? data.links[0] : null;
        
      setClusterDisplayState({
        graphData: data,
        selectedConnection: newSelectedConnection
      });
    } catch (err) {
      console.error("Error refreshing graph data:", err);
      setGraphError(
        err instanceof Error ? err : new Error("Failed to refresh graph data")
      );
    } finally {
      setLoadingGraph(false);
    }
  }, [selectedClusterId, clusterDisplayState.selectedConnection?.id]);

  // Function to submit feedback
  const submitFeedback = useCallback(
    async (
      decision: "confirm" | "reject",
      edge: VisualizationEntityEdge
    ): Promise<boolean> => {
      setSubmittingFeedback(true);
      setFeedbackError(null);

      const feedback: FeedbackRequest = {
        reviewer_id: "user_xyz", // Placeholder - in a real app this would come from auth
        entity_group_id: edge.id,
        method_type: edge.details.methods[0]?.method_type || "unknown", // Use the first method type as the primary one
        entity_id_1: edge.entity_id_1,
        entity_id_2: edge.entity_id_2,
      };

      try {
        if (decision === "confirm") {
          await confirmLink(feedback);
          // Update the edge status in the graph
          setClusterDisplayState(prev => {
            if (!prev.graphData) return prev;
            
            const updatedLinks = prev.graphData.links.map(link => 
              link.id === edge.id ? { ...link, confirmed: true } : link
            );
            
            return {
              ...prev,
              graphData: {
                ...prev.graphData,
                links: updatedLinks
              }
            };
          });
        } else { // Reject
          await rejectLink(feedback);
          // Remove the edge from the graph
          setClusterDisplayState(prev => {
            if (!prev.graphData) return prev;
            
            const updatedLinks = prev.graphData.links.filter(link => link.id !== edge.id);
            const nextSelectedConnection = updatedLinks.length > 0 ? updatedLinks[0] : null;
            
            return {
              graphData: {
                ...prev.graphData,
                links: updatedLinks
              },
              selectedConnection: nextSelectedConnection
            };
          });
        }
        return true;
      } catch (err) {
        console.error("Error submitting feedback:", err);
        setFeedbackError(
          err instanceof Error ? err : new Error("Failed to submit feedback")
        );
        return false;
      } finally {
        setSubmittingFeedback(false);
      }
    },
    []
  );

  const contextValue: ClusterContextValue = {
    clusters,
    selectedClusterId,
    setSelectedClusterId: handleSelectCluster,
    loadingClusters,
    clusterError,

    graphData: clusterDisplayState.graphData,
    selectedConnection: clusterDisplayState.selectedConnection,
    setSelectedConnection,
    loadingGraph,
    graphError,

    entityDetailsCache,
    loadingEntityDetails,
    entityDetailsError,
    fetchEntityDetails: fetchEntityDetailsById,

    submittingFeedback,
    feedbackError,
    submitFeedback,

    refreshClusterGraph,
  };

  return (
    <ClusterContext.Provider value={contextValue}>
      {children}
    </ClusterContext.Provider>
  );
}

export function useClusterContext() {
  const context = useContext(ClusterContext);
  if (context === undefined) {
    throw new Error("useClusterContext must be used within a ClusterProvider");
  }
  return context;
}