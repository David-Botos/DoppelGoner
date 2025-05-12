// components/GraphVisualization/index.tsx
"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import * as d3 from "d3";
import { GraphData, EntityGroup, Entity, EntityDetails } from "../../lib/types"; // Adjust path as needed
import NodeTooltip from "./NodeTooltip"; // Adjust path as needed
import LinkTooltip from "./LinkTooltip"; // Adjust path as needed

// Define D3 specific interfaces by extending d3.SimulationNodeDatum and d3.SimulationLinkDatum
interface D3Node extends d3.SimulationNodeDatum {
  id: string;
  name: string;
  organization_id: string;
  // Optional D3 simulation properties; D3 will add these
  x?: number;
  y?: number;
  fx?: number | null; // For fixed positions
  fy?: number | null; // For fixed positions
}

interface D3Link extends d3.SimulationLinkDatum<D3Node> {
  id: string;
  // D3 can handle source/target being string IDs initially, then populates them with D3Node objects
  source: string | D3Node;
  target: string | D3Node;
  confidence_score: number;
  confirmed?: boolean;
  method_type: string;
  match_values: {
    type: string;
    values: Record<string, unknown>;
  };
}

// Define the props for the GraphVisualization component
interface GraphVisualizationProps {
  graphData: GraphData | null; // Graph data can be null if no cluster is selected or it's loading
  currentTask: EntityGroup | null; // The currently active task for highlighting
  entityDetailsCache: Record<string, EntityDetails>; // Cache of fetched entity details
  onNodeHover: (entityId: string) => void; // Callback when a node is hovered
  detailsLoading: Record<string, boolean>; // Tracks loading state for individual entity details
  isLoadingGraphStructure: boolean; // True if the main graph structure is currently being fetched
}

const GraphVisualization: React.FC<GraphVisualizationProps> = ({
  graphData,
  currentTask,
  entityDetailsCache,
  onNodeHover,
  detailsLoading,
  isLoadingGraphStructure,
}) => {
  // Refs for D3 elements and simulation
  const svgRef = useRef<SVGSVGElement>(null); // Ref for the SVG DOM element
  const simulationRef = useRef<d3.Simulation<D3Node, D3Link> | null>(null); // Ref for the D3 force simulation
  const linksRef = useRef<d3.Selection<
    SVGLineElement,
    D3Link,
    SVGGElement,
    unknown
  > | null>(null); // Ref for D3 selection of link lines
  const nodesRef = useRef<d3.Selection<
    SVGCircleElement,
    D3Node,
    SVGGElement,
    unknown
  > | null>(null); // Ref for D3 selection of node circles
  const labelsRef = useRef<d3.Selection<
    SVGTextElement,
    D3Node,
    SVGGElement,
    unknown
  > | null>(null); // Ref for D3 selection of node labels
  const graphContainerRef = useRef<d3.Selection<
    SVGGElement,
    unknown,
    null,
    undefined
  > | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null); // Ref to store the D3 zoom behavior instance

  // State for tooltips
  const [tooltipNode, setTooltipNode] = useState<Entity | null>(null); // Data for the node tooltip
  const [tooltipLink, setTooltipLink] = useState<EntityGroup | null>(null); // Data for the link tooltip
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 }); // Position for tooltips

  // State to manage if all necessary data is loaded and ready for D3 rendering
  const [isDataSettled, setIsDataSettled] = useState(false);
  // Ref to store a signature of the rendered graph data to compare for structural changes
  const graphSignatureRef = useRef<string | null>(null);

  // --- Memoized Styling and Utility Callbacks ---

  const getColorByConfidence = useCallback((score: number): string => {
    if (score >= 0.8) return "#1e88e5"; // High confidence - blue
    if (score >= 0.6) return "#ffc107"; // Medium confidence - amber
    return "#e53935"; // Low confidence - red
  }, []);

  const getThicknessByConfidence = useCallback((score: number): number => {
    if (score >= 0.8) return 3;
    if (score >= 0.6) return 2;
    return 1;
  }, []);

  const getLinkColor = useCallback(
    (link: D3Link): string => {
      if (currentTask && link.id === currentTask.id) {
        return "#000000"; // Current task link is black
      } else if (link.confirmed) {
        return "#4caf50"; // Confirmed links are green
      } else {
        return getColorByConfidence(link.confidence_score);
      }
    },
    [currentTask, getColorByConfidence]
  );

  const getLinkThickness = useCallback(
    (link: D3Link): number => {
      if (currentTask && link.id === currentTask.id) {
        return 4; // Current task link is thicker
      } else if (link.confirmed) {
        return 3; // Confirmed links
      } else {
        return getThicknessByConfidence(link.confidence_score);
      }
    },
    [currentTask, getThicknessByConfidence]
  );

  const getNodeRadius = useCallback(
    (node: D3Node): number => {
      return currentTask &&
        (node.id === (currentTask.source as unknown) ||
          node.id === (currentTask.target as unknown))
        ? 10 // Nodes in current task are larger
        : 8; // Default node size
    },
    [currentTask]
  );

  const getNodeColor = useCallback(
    (node: D3Node): string => {
      return currentTask &&
        (node.id === (currentTask.source as unknown) ||
          node.id === (currentTask.target as unknown))
        ? "#3949ab" // Indigo for highlighted nodes in current task
        : "#69b3a2"; // Default node color
    },
    [currentTask]
  );

  // --- useEffect Hooks ---

  // Effect 1: Determine if all necessary data is "settled" for rendering.
  // This effect runs when the primary graph data, its loading status, or entity details loading status changes.
  useEffect(() => {
    // If the main graph structure is still loading, or if graphData is not yet available,
    // or if there are no nodes, then data is not considered settled.
    if (
      isLoadingGraphStructure ||
      !graphData ||
      !graphData.nodes ||
      graphData.nodes.length === 0
    ) {
      setIsDataSettled(false);
      return;
    }

    // Check if details for any of the nodes currently in graphData are still being loaded.
    // This assumes prefetching logic in the context is attempting to load these.
    let detailsStillLoadingForVisibleNodes = false;
    for (const node of graphData.nodes) {
      if (detailsLoading[node.id]) {
        detailsStillLoadingForVisibleNodes = true;
        break; // Found a node with details loading, no need to check further.
      }
    }

    // Set isDataSettled based on whether critical entity details are still loading.
    setIsDataSettled(!detailsStillLoadingForVisibleNodes);
  }, [graphData, detailsLoading, isLoadingGraphStructure]);

  // Effect 2: Main D3 graph rendering and re-rendering logic.
  // This effect runs when data is settled, or when graphData/currentTask or other visual properties change.
  useEffect(() => {
    const svgElement = svgRef.current;

    // Exit early if data is not settled or the SVG element is not available.
    if (!isDataSettled || !svgElement) {
      // If not settled, ensure any previous D3 elements are removed and simulation is stopped.
      if (svgElement) {
        d3.select(svgElement).selectAll("*").remove(); // Clear the SVG content
      }
      if (simulationRef.current) {
        simulationRef.current.stop();
        simulationRef.current = null;
      }
      graphSignatureRef.current = null; // Reset signature as graph is cleared
      linksRef.current = null;
      nodesRef.current = null;
      labelsRef.current = null;
      graphContainerRef.current = null;
      return;
    }

    // At this point, isDataSettled is true and svgRef.current is available.
    // Get current dimensions of the SVG element.
    const width = svgElement.clientWidth;
    const height = svgElement.clientHeight;

    // If SVG dimensions are not yet resolved by the browser, exit.
    // This is a crucial check to prevent the D3 zoom error.
    if (width === 0 || height === 0) {
      // console.warn("GraphVisualization: SVG dimensions are 0. Deferring D3 setup.");
      // Clear previous D3 elements if any existed and dimensions became zero (e.g. due to parent resize/hide)
      if (simulationRef.current) {
        d3.select(svgElement).selectAll("*").remove();
        simulationRef.current.stop();
        simulationRef.current = null;
        graphSignatureRef.current = null;
        linksRef.current = null;
        nodesRef.current = null;
        labelsRef.current = null;
        graphContainerRef.current = null;
      }
      return;
    }

    // Ensure graphData and its nodes/links are valid (should be due to isDataSettled, but good for safety).
    if (!graphData || !graphData.nodes || !graphData.links) {
      // This state should ideally not be reached if isDataSettled is true and handled graphData correctly.
      // However, as a safeguard:
      d3.select(svgElement).selectAll("*").remove();
      if (simulationRef.current) {
        simulationRef.current.stop();
        simulationRef.current = null;
      }
      graphSignatureRef.current = null;
      return;
    }

    // Generate a signature for the current graph data to detect structural changes.
    const currentGraphSignature = `${graphData.nodes.length}-${
      graphData.links.length
    }-${graphData.links
      .map((l) => l.id)
      .sort()
      .join("_")}`;

    // Determine if a full redraw is needed (e.g., first render, or graph structure changed).
    const shouldRedrawGraph =
      !simulationRef.current ||
      currentGraphSignature !== graphSignatureRef.current;

    if (shouldRedrawGraph) {
      // Store current node positions if a simulation already exists, to try and maintain layout.
      const nodePositions: Record<string, { x: number; y: number }> = {};
      if (simulationRef.current) {
        simulationRef.current.nodes().forEach((node: D3Node) => {
          if (node.x !== undefined && node.y !== undefined) {
            nodePositions[node.id] = { x: node.x, y: node.y };
          }
        });
        simulationRef.current.stop(); // Stop the old simulation before creating a new one.
      }

      graphSignatureRef.current = currentGraphSignature; // Update the signature reference.
      d3.select(svgElement).selectAll("*").remove(); // Clear SVG for a full redraw.

      // Call the function to create and render the D3 graph.
      createGraph(nodePositions, width, height);
    }

    // Cleanup function for this effect: stop the simulation when the component unmounts
    // or before the effect re-runs if dependencies change in a way that requires cleanup.
    return () => {
      if (simulationRef.current) {
        // It's generally good practice to stop the simulation.
        // If createGraph runs again, it will re-initialize or update the simulation.
        // simulationRef.current.stop(); // This might be too aggressive if only styles change later.
        // The stop is handled before redraw if shouldRedrawGraph is true.
      }
    };
  }, [
    isDataSettled, // Key dependency to ensure data readiness.
    graphData, // Primary data source.
    // Callbacks for styling and interaction are dependencies as they might change if currentTask changes.
    onNodeHover,
    getLinkColor,
    getLinkThickness,
    getNodeRadius,
    getNodeColor,
    currentTask,
  ]);

  // Effect 3: Update styles of existing D3 elements if `currentTask` or styling functions change.
  // This runs if the graph is already rendered (`isDataSettled` is true and `simulationRef.current` exists)
  // and avoids a full D3 recreation for mere style updates.
  useEffect(() => {
    // Guard: Only proceed if graph is rendered and references to D3 selections are available.
    if (
      !isDataSettled ||
      !linksRef.current ||
      !nodesRef.current ||
      !labelsRef.current ||
      !simulationRef.current
    ) {
      return;
    }

    // Smoothly transition link styles.
    linksRef.current
      .transition()
      .duration(300)
      .attr("stroke", getLinkColor)
      .attr("stroke-width", getLinkThickness);

    // Smoothly transition node styles.
    nodesRef.current
      .transition()
      .duration(300)
      .attr("r", getNodeRadius)
      .attr("fill", getNodeColor);

    // Potentially update label visibility or style if needed, e.g., for highlighted nodes.
    // For this example, we assume labels don't change style based on currentTask beyond initial creation.
  }, [
    isDataSettled,
    currentTask,
    getLinkColor,
    getLinkThickness,
    getNodeRadius,
    getNodeColor,
  ]);

  // --- D3 Graph Creation Function ---
  function createGraph(
    initialPositions: Record<string, { x: number; y: number }>,
    svgWidth: number,
    svgHeight: number
  ) {
    // Ensure graphData is available (should be guaranteed by the calling useEffect).
    if (!graphData || !graphData.nodes || !graphData.links) return;

    const svg = d3.select(svgRef.current!); // svgRef.current is checked by the caller.

    // Use provided dimensions, with fallbacks if they were somehow zero (though useEffect guard tries to prevent this).
    const resolvedWidth = svgWidth > 0 ? svgWidth : 600;
    const resolvedHeight = svgHeight > 0 ? svgHeight : 400;

    // Prepare nodes and links in D3-compatible format.
    const d3Nodes: D3Node[] = graphData.nodes.map((node) => ({
      ...node, // Spread original node data
      x: initialPositions[node.id]?.x, // Apply previous x position if available
      y: initialPositions[node.id]?.y, // Apply previous y position if available
    }));

    const d3Links: D3Link[] = graphData.links.map((link) => ({
      ...link, // Spread original link data
      source: link.source,
      target: link.target,
    }));

    // console.log("d3Nodes: ", d3Nodes); // Kept for your debugging
    // console.log("d3Links: ", d3Links); // Kept for your debugging

    // Initialize the D3 force simulation.
    const simulation = d3
      .forceSimulation<D3Node, D3Link>(d3Nodes)
      .force(
        "link",
        d3
          .forceLink<D3Node, D3Link>(d3Links)
          .id((d: D3Node) => d.id)
          .distance(100)
      )
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(resolvedWidth / 2, resolvedHeight / 2));
    simulationRef.current = simulation;

    graphContainerRef.current = svg
      .append("g")
      .attr("class", "graph-container");
    const g = graphContainerRef.current;

    // --- Create Links ---
    linksRef.current = g
      .append("g")
      .attr("class", "links")
      .selectAll<SVGLineElement, D3Link>("line")
      .data(d3Links, (d: D3Link) => d.id)
      .join(
        (enter) =>
          enter
            .append("line")
            .attr("stroke", getLinkColor)
            .attr("stroke-width", getLinkThickness)
            .attr("opacity", 0) // Start transparent
            .call((selection) =>
              selection
                .transition("fade-in-opacity") // Named transition
                .duration(300)
                .attr("opacity", 1)
            ), // Fade-in to opacity 1
        (update) =>
          update
            .attr("stroke", getLinkColor)
            .attr("stroke-width", getLinkThickness),
        (exit) =>
          exit.call((selection) =>
            selection.transition().duration(300).attr("opacity", 0).remove()
          )
      )
      .on("mouseover", (event: MouseEvent, d: D3Link) => {
        const sourceNode = d.source as D3Node;
        const targetNode = d.target as D3Node;
        const entityGroupData: EntityGroup = {
          id: d.id,
          source: typeof sourceNode === "string" ? sourceNode : sourceNode.id,
          target: typeof targetNode === "string" ? targetNode : targetNode.id,
          method_type: d.method_type,
          confidence_score: d.confidence_score,
          match_values: d.match_values,
          confirmed: d.confirmed,
        };
        setTooltipLink(entityGroupData);
        setTooltipPosition({ x: event.pageX, y: event.pageY });
      })
      .on("mouseout", () => {
        setTooltipLink(null);
      });

    // --- Create Nodes ---
    nodesRef.current = g
      .append("g")
      .attr("class", "nodes")
      .selectAll<SVGCircleElement, D3Node>("circle")
      .data(d3Nodes, (d: D3Node) => d.id)
      .join(
        (enter) =>
          enter
            .append("circle")
            .attr("r", getNodeRadius)
            .attr("fill", getNodeColor)
            .attr("opacity", 0) // Start transparent
            .call(
              d3
                .drag<SVGCircleElement, D3Node>()
                .on("start", (event, draggedNode) =>
                  dragstarted(event, draggedNode)
                )
                .on("drag", (event, draggedNode) => dragged(event, draggedNode))
                .on("end", (event, draggedNode) =>
                  dragended(event, draggedNode)
                )
            )
            .call((selection) =>
              selection
                .transition("fade-in-opacity") // Named transition
                .duration(300)
                .attr("opacity", 1)
            ), // Fade-in to opacity 1
        (update) => update.attr("r", getNodeRadius).attr("fill", getNodeColor),
        (exit) =>
          exit.call((selection) =>
            selection.transition().duration(300).attr("opacity", 0).remove()
          )
      )
      .on("mouseover", (event: MouseEvent, d: D3Node) => {
        const entityData: Entity = {
          id: d.id,
          name: d.name,
          organization_id: d.organization_id,
        };
        onNodeHover(entityData.id);
        setTooltipNode(entityData);
        setTooltipPosition({ x: event.pageX, y: event.pageY });
      })
      .on("mouseout", () => {
        setTooltipNode(null);
      });

    // --- Create Labels ---
    labelsRef.current = g
      .append("g")
      .attr("class", "labels")
      .selectAll<SVGTextElement, D3Node>("text")
      .data(d3Nodes, (d: D3Node) => d.id)
      .join(
        (enter) =>
          enter
            .append("text")
            .text((d) => d.name)
            .attr("font-size", "10px")
            .attr("dx", 12)
            .attr("dy", 4)
            .attr("opacity", 0) // Start transparent
            .call((selection) =>
              selection
                .transition("fade-in-opacity") // Named transition for consistency
                .duration(300)
                .attr("opacity", 1)
            ), // Fade-in to opacity 1
        (update) => update.text((d) => d.name),
        (exit) =>
          exit.call((selection) =>
            selection.transition().duration(300).attr("opacity", 0).remove()
          )
      );

    // --- Initialize Zoom ---
    if (!zoomRef.current) {
      zoomRef.current = d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 8])
        .on("zoom", (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
          if (graphContainerRef.current) {
            graphContainerRef.current.attr(
              "transform",
              event.transform.toString()
            );
          }
        });
    }
    zoomRef.current.extent([
      [0, 0],
      [resolvedWidth, resolvedHeight],
    ]);
    svg.call(zoomRef.current);

    // --- Simulation Tick Function ---
    simulation.on("tick", () => {
      linksRef.current
        ?.attr("x1", (d) => (d.source as D3Node).x ?? 0)
        .attr("y1", (d) => (d.source as D3Node).y ?? 0)
        .attr("x2", (d) => (d.target as D3Node).x ?? 0)
        .attr("y2", (d) => (d.target as D3Node).y ?? 0);

      nodesRef.current?.attr("cx", (d) => d.x ?? 0).attr("cy", (d) => d.y ?? 0);

      labelsRef.current?.attr("x", (d) => d.x ?? 0).attr("y", (d) => d.y ?? 0);
    });

    // --- Optional: Center on Current Task ---
    if (currentTask && zoomRef.current) {
      const zoomBehavior = zoomRef.current;
      setTimeout(() => {
        const sourceNode = d3Nodes.find((n) => n.id === currentTask.source);
        const targetNode = d3Nodes.find((n) => n.id === currentTask.target);

        if (
          sourceNode?.x != null &&
          sourceNode?.y != null &&
          targetNode?.x != null &&
          targetNode?.y != null
        ) {
          const centerX = (sourceNode.x + targetNode.x) / 2;
          const centerY = (sourceNode.y + targetNode.y) / 2;
          const scale = 1;

          svg
            .transition()
            .duration(750)
            .call(
              zoomBehavior.transform,
              d3.zoomIdentity
                .translate(resolvedWidth / 2, resolvedHeight / 2)
                .scale(scale)
                .translate(-centerX, -centerY)
            );
        }
      }, 500);
    }

    // --- Drag Handler Functions ---
    function dragstarted(
      event: d3.D3DragEvent<SVGCircleElement, D3Node, D3Node>,
      d: D3Node
    ) {
      if (!event.active && simulationRef.current) {
        simulationRef.current.alphaTarget(0.3).restart();
      }
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(
      event: d3.D3DragEvent<SVGCircleElement, D3Node, D3Node>,
      d: D3Node
    ) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(
      event: d3.D3DragEvent<SVGCircleElement, D3Node, D3Node>,
      d: D3Node
    ) {
      if (!event.active && simulationRef.current) {
        simulationRef.current.alphaTarget(0);
      }
      if (event.subject === d) {
        d.fx = null;
        d.fy = null;
      }
    }
  }

  // --- JSX Rendering ---

  // If graphData is available, but we are still waiting for entity details (isDataSettled is false),
  // show a specific loading message.
  // The parent component (page.tsx) handles cases where graphData itself is null or isLoadingGraphStructure is true.
  if (graphData && !isDataSettled && !isLoadingGraphStructure) {
    return (
      <div className="flex w-full h-full items-center justify-center">
        <p>Loading entity details for the graph...</p>
      </div>
    );
  }

  // If data is not settled for other reasons (e.g. isLoadingGraphStructure is true, or graphData is null),
  // page.tsx's loading/empty states will be shown. This component will render its SVG container
  // once isDataSettled is true (which implies graphData is present and details are loaded).
  return (
    <div
      className="relative w-full h-full"
      data-testid="graph-visualization-container"
    >
      {/* SVG element where D3 will render the graph. It's always in the DOM if this return path is reached,
          but D3 content is only added if isDataSettled and dimensions are valid. */}
      <svg ref={svgRef} className="w-full h-full"></svg>

      {/* Render tooltips only if data is settled and a tooltip is active. */}
      {isDataSettled && tooltipNode && (
        <NodeTooltip
          node={tooltipNode}
          position={tooltipPosition}
          entityDetails={entityDetailsCache[tooltipNode.id]}
          loading={detailsLoading[tooltipNode.id]}
        />
      )}
      {isDataSettled && tooltipLink && (
        <LinkTooltip link={tooltipLink} position={tooltipPosition} />
      )}
    </div>
  );
};

export default GraphVisualization;
