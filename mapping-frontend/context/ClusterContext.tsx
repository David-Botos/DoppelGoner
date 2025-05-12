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
  EntityGroup,
  EntityDetails,
  FeedbackRequest,
} from "../lib/types"; // Assuming your types are correctly defined here
import {
  fetchClusters,
  fetchClusterGraph,
  fetchEntityDetails, // Ensure this API function is correctly imported
  confirmLink,
  rejectLink,
} from "../lib/api"; // Assuming your API functions are correctly imported

interface ClusterDisplayState {
  graphData: GraphData | null;
  taskQueue: EntityGroup[];
  currentTaskIndex: number;
}

interface ClusterContextValue {
  // Clusters state
  clusters: ClusterSummary[];
  selectedClusterId: string | null;
  setSelectedClusterId: (id: string | null) => void; // Or a dedicated selectNewCluster function
  loadingClusters: boolean;
  clusterError: Error | null;

  // Graph and Task display state (derived from clusterDisplayState)
  graphData: GraphData | null;
  taskQueue: EntityGroup[];
  currentTaskIndex: number;
  currentTask: EntityGroup | null;
  loadingGraph: boolean; // Manages loading state for graph fetching
  graphError: Error | null; // Manages error state for graph fetching

  // Task navigation
  setTaskIndex: (index: number) => void;
  nextTask: () => boolean;

  // Entity details state
  entityDetailsCache: Record<string, EntityDetails>;
  loadingEntityDetails: Record<string, boolean>; // Tracks loading state for individual entities
  entityDetailsError: Record<string, Error | null>; // Tracks errors for individual entities
  fetchEntityDetails: (entityId: string) => Promise<void>; // Exposed function to fetch details on demand
  // prefetchEntityDetails: (entityIds: string[]) => void; // This will now be handled internally by an effect

  // Feedback state
  submittingFeedback: boolean;
  feedbackError: Error | null;
  submitFeedback: (
    decision: "confirm" | "reject",
    entityGroupId: string
  ) => Promise<boolean>;

  // Graph modification functions
  refreshClusterGraph: () => Promise<void>;
}

const ClusterContext = createContext<ClusterContextValue | undefined>(
  undefined
);

const initialClusterDisplayState: ClusterDisplayState = {
  graphData: null,
  taskQueue: [],
  currentTaskIndex: 0,
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
        setGraphError(null); // Clear previous graph errors
        return;
      }

      setLoadingGraph(true);
      setGraphError(null);
      try {
        const data = await fetchClusterGraph(selectedClusterId);
        setClusterDisplayState({
          graphData: data,
          taskQueue: data.links || [], // Ensure taskQueue is always an array
          currentTaskIndex: 0,
        });
      } catch (err) {
        console.error("Error fetching graph data for cluster:", selectedClusterId, err);
        setClusterDisplayState(initialClusterDisplayState); // Reset on error
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
    async (entityId: string, forceRefetch: boolean = false) => {
      if (!forceRefetch && (entityDetailsCache[entityId] || loadingEntityDetails[entityId])) {
        return;
      }

      setLoadingEntityDetails((prev) => ({ ...prev, [entityId]: true }));
      setEntityDetailsError((prev) => ({ ...prev, [entityId]: null })); // Clear previous error

      try {
        const details = await fetchEntityDetails(entityId);
        setEntityDetailsCache((prev) => ({ ...prev, [entityId]: details }));
      } catch (err) {
        console.error(`Error fetching details for entity ${entityId}:`, err);
        setEntityDetailsError((prev) => ({
          ...prev,
          [entityId]:
            err instanceof Error
              ? err
              : new Error(`Failed to fetch details for entity ${entityId}`),
        }));
      } finally {
        setLoadingEntityDetails((prev) => ({ ...prev, [entityId]: false }));
      }
    },
    [entityDetailsCache, loadingEntityDetails] // Dependencies of the fetch function itself
  );

  // NEW useEffect for prefetching entity details based on taskQueue and currentTaskIndex
  useEffect(() => {
    if (clusterDisplayState.taskQueue.length > 0 && typeof fetchEntityDetailsById === 'function') {
      const entityIdsToPrefetch = new Set<string>();
      let tasksForPrefetching: EntityGroup[];

      const { taskQueue, currentTaskIndex } = clusterDisplayState;

      // Prefetch for a window around the current task, or first few if at the beginning
      const prefetchWindowSize = 5;
      const startIndex = Math.max(0, currentTaskIndex);
      const endIndex = Math.min(taskQueue.length, startIndex + prefetchWindowSize);
      tasksForPrefetching = taskQueue.slice(startIndex, endIndex);
      
      // If no tasks found around current (e.g., currentTaskIndex is at end of a short list),
      // and we are at the beginning of the queue, try prefetching the first few tasks.
      if (tasksForPrefetching.length === 0 && currentTaskIndex === 0 && taskQueue.length > 0) {
        tasksForPrefetching = taskQueue.slice(0, Math.min(prefetchWindowSize, taskQueue.length));
      }

      tasksForPrefetching.forEach((link) => {
        if (link.source) entityIdsToPrefetch.add(link.source);
        if (link.target) entityIdsToPrefetch.add(link.target);
      });

      if (entityIdsToPrefetch.size > 0) {
        Array.from(entityIdsToPrefetch).forEach((id) => {
          // Call without forceRefetch, so it respects cache and loading state
          fetchEntityDetailsById(id, false);
        });
      }
    }
  }, [clusterDisplayState.taskQueue, clusterDisplayState.currentTaskIndex, fetchEntityDetailsById]);


  // --- Callbacks and Derived State ---

  // Handler to change selected cluster and reset task index
  const handleSelectCluster = useCallback((clusterId: string | null) => {
    setSelectedClusterId(clusterId);
    // The useEffect listening to selectedClusterId will reset clusterDisplayState
    // which includes setting currentTaskIndex to 0.
  }, []);


  // Function to set the current task index directly
  const setTaskIndex = useCallback((index: number) => {
    setClusterDisplayState((prev) => {
      if (index >= 0 && index < prev.taskQueue.length) {
        return { ...prev, currentTaskIndex: index };
      }
      // If index is out of bounds but queue is not empty, clamp to last item or 0
      if (prev.taskQueue.length > 0) {
        return { ...prev, currentTaskIndex: Math.max(0, prev.taskQueue.length - 1) };
      }
      return { ...prev, currentTaskIndex: 0 }; // Default to 0 if queue is empty
    });
  }, []);

  // Function to move to the next task
  const nextTask = useCallback(() => {
    let moved = false;
    setClusterDisplayState((prev) => {
      if (prev.currentTaskIndex < prev.taskQueue.length - 1) {
        moved = true;
        return { ...prev, currentTaskIndex: prev.currentTaskIndex + 1 };
      }
      return prev;
    });
    return moved;
  }, []);

  // Derived current task
  const currentTask =
    clusterDisplayState.taskQueue[clusterDisplayState.currentTaskIndex] || null;

  // Refresh cluster graph - useful after cluster modifications
  const refreshClusterGraph = useCallback(async () => {
    if (!selectedClusterId) return;

    setLoadingGraph(true);
    setGraphError(null);
    try {
      const data = await fetchClusterGraph(selectedClusterId);
      setClusterDisplayState((prev) => ({
        ...prev, // Keep currentTaskIndex if possible, or reset if it's out of bounds
        graphData: data,
        taskQueue: data.links || [],
        currentTaskIndex: Math.min(prev.currentTaskIndex, (data.links || []).length > 0 ? (data.links || []).length - 1 : 0),
      }));
    } catch (err) {
      console.error("Error refreshing graph data:", err);
      setGraphError(
        err instanceof Error ? err : new Error("Failed to refresh graph data")
      );
    } finally {
      setLoadingGraph(false);
    }
  }, [selectedClusterId]); // Removed currentTaskIndex dependency to simplify, as it's handled inside

  // Function to submit feedback
  const submitFeedback = useCallback(
    async (
      decision: "confirm" | "reject",
      entityGroupId: string
    ): Promise<boolean> => {
      const taskToUpdate = clusterDisplayState.taskQueue.find(
        (t) => t.id === entityGroupId
      );
      if (!taskToUpdate) return false;

      setSubmittingFeedback(true);
      setFeedbackError(null);

      const feedback: FeedbackRequest = {
        reviewer_id: "user_xyz", // Placeholder
        entity_group_id: taskToUpdate.id,
        method_type: taskToUpdate.method_type,
        entity_id_1: taskToUpdate.source,
        entity_id_2: taskToUpdate.target,
      };

      try {
        if (decision === "confirm") {
          await confirmLink(feedback);
          setClusterDisplayState((prev) => {
            const newGraphData = prev.graphData
              ? {
                  ...prev.graphData,
                  links: prev.graphData.links.map((link) =>
                    link.id === entityGroupId ? { ...link, confirmed: true } : link
                  ),
                }
              : null;
            return {
              ...prev,
              graphData: newGraphData,
              taskQueue: prev.taskQueue.map((t) =>
                t.id === entityGroupId ? { ...t, confirmed: true } : t
              ),
            };
          });
        } else { // Reject
          await rejectLink(feedback);
          setClusterDisplayState((prev) => {
            const newQueue = prev.taskQueue.filter((t) => t.id !== entityGroupId);
            const newGraphData = prev.graphData
              ? {
                  ...prev.graphData,
                  links: prev.graphData.links.filter(
                    (link) => link.id !== entityGroupId
                  ),
                }
              : null;
            
            // Adjust current index if necessary
            let newIndex = prev.currentTaskIndex;
            if (newIndex >= newQueue.length) {
              newIndex = Math.max(0, newQueue.length - 1);
            }
            return {
              ...prev,
              graphData: newGraphData,
              taskQueue: newQueue,
              currentTaskIndex: newIndex,
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
    [clusterDisplayState.taskQueue] // Depends on the current task queue for finding the task
  );

  const contextValue: ClusterContextValue = {
    clusters,
    selectedClusterId,
    setSelectedClusterId: handleSelectCluster, // Use the new handler
    loadingClusters,
    clusterError,

    graphData: clusterDisplayState.graphData,
    taskQueue: clusterDisplayState.taskQueue,
    currentTaskIndex: clusterDisplayState.currentTaskIndex,
    currentTask,
    loadingGraph,
    graphError,

    setTaskIndex,
    nextTask,

    entityDetailsCache,
    loadingEntityDetails,
    entityDetailsError,
    fetchEntityDetails: fetchEntityDetailsById, // Expose the memoized fetcher

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