// components/GraphVisualization/index.tsx
"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import * as d3 from "d3";
import { GraphData, EntityDetails, Entity, VisualizationEntityEdge } from "../../lib/types";
import NodeTooltip from "./NodeTooltip";
import LinkTooltip from "./LinkTooltip"
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
  entity_id_1: string;
  entity_id_2: string;
  edge_weight: number;
  details: {
    methods: Array<{
      method_type: string;
      pre_rl_confidence: number;
      rl_confidence: number;
      combined_confidence: number;
    }>;
    rl_weight_factor: number;
    method_count: number;
  };
  confirmed?: boolean;
}

// Define the props for the GraphVisualization component
interface GraphVisualizationProps {
  graphData: GraphData | null; // Graph data can be null if no cluster is selected or it's loading
  selectedConnection: VisualizationEntityEdge | null; // The currently selected connection for highlighting
  onSelectConnection: (connection: VisualizationEntityEdge) => void; // Callback when a connection is selected
  entityDetailsCache: Record<string, EntityDetails>; // Cache of fetched entity details
  onNodeHover: (entityId: string) => void; // Callback when a node is hovered
  detailsLoading: Record<string, boolean>; // Tracks loading state for individual entity details
  isLoadingGraphStructure: boolean; // True if the main graph structure is currently being fetched
}

const GraphVisualization: React.FC<GraphVisualizationProps> = ({
  graphData,
  selectedConnection,
  onSelectConnection,
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
  const [hoveredNode, setHoveredNode] = useState<{ 
    entity: Entity; 
    details: EntityDetails | undefined;
  } | null>(null);
  const [hoveredLink, setHoveredLink] = useState<VisualizationEntityEdge | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });

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
      if (selectedConnection && link.id === selectedConnection.id) {
        return "#000000"; // Selected connection is black
      } else if (link.confirmed) {
        return "#4caf50"; // Confirmed links are green
      } else {
        return getColorByConfidence(link.edge_weight);
      }
    },
    [selectedConnection, getColorByConfidence]
  );

  const getLinkThickness = useCallback(
    (link: D3Link): number => {
      if (selectedConnection && link.id === selectedConnection.id) {
        return 4; // Selected connection is thicker
      } else if (link.confirmed) {
        return 3; // Confirmed links
      } else {
        return getThicknessByConfidence(link.edge_weight);
      }
    },
    [selectedConnection, getThicknessByConfidence]
  );

  const getNodeRadius = useCallback(
    (node: D3Node): number => {
      return selectedConnection &&
        (node.id === selectedConnection.entity_id_1 ||
          node.id === selectedConnection.entity_id_2)
        ? 10 // Nodes in selected connection are larger
        : 8; // Default node size
    },
    [selectedConnection]
  );

  const getNodeColor = useCallback(
    (node: D3Node): string => {
      return selectedConnection &&
        (node.id === selectedConnection.entity_id_1 ||
          node.id === selectedConnection.entity_id_2)
        ? "#3949ab" // Indigo for highlighted nodes in selected connection
        : "#69b3a2"; // Default node color
    },
    [selectedConnection]
  );

  // --- useEffect Hooks ---

  // Effect 1: Determine if all necessary data is "settled" for rendering.
  useEffect(() => {
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
    let detailsStillLoadingForVisibleNodes = false;
    for (const node of graphData.nodes) {
      if (detailsLoading[node.id]) {
        detailsStillLoadingForVisibleNodes = true;
        break;
      }
    }

    // Set isDataSettled based on whether critical entity details are still loading.
    setIsDataSettled(!detailsStillLoadingForVisibleNodes);
  }, [graphData, detailsLoading, isLoadingGraphStructure]);

  // Effect 2: Main D3 graph rendering and re-rendering logic.
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
    if (width === 0 || height === 0) {
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

    // Ensure graphData and its nodes/links are valid.
    if (!graphData || !graphData.nodes || !graphData.links) {
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

    // Determine if a full redraw is needed.
    const shouldRedrawGraph =
      !simulationRef.current ||
      currentGraphSignature !== graphSignatureRef.current;

    if (shouldRedrawGraph) {
      // Store current node positions if a simulation already exists.
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

      // Create and render the D3 graph.
      createGraph(nodePositions, width, height);
    }

    // Cleanup function
    return () => {
      if (simulationRef.current) {
        // The stop is handled before redraw if shouldRedrawGraph is true.
      }
    };
  }, [
    isDataSettled,
    graphData,
    onNodeHover,
    getLinkColor,
    getLinkThickness,
    getNodeRadius,
    getNodeColor,
    selectedConnection,
    onSelectConnection,
  ]);

  // Effect 3: Update styles of existing D3 elements if `selectedConnection` or styling functions change.
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

    // If a connection is selected, center the view on it
    if (selectedConnection && zoomRef.current) {
      const zoomBehavior = zoomRef.current;
      const svg = d3.select(svgRef.current!);
      const width = svgRef.current!.clientWidth;
      const height = svgRef.current!.clientHeight;

      // Find the nodes in the simulation
      const source = simulationRef.current.nodes().find(n => n.id === selectedConnection.entity_id_1);
      const target = simulationRef.current.nodes().find(n => n.id === selectedConnection.entity_id_2);

      if (source && target && source.x != null && source.y != null && target.x != null && target.y != null) {
        const centerX = (source.x + target.x) / 2;
        const centerY = (source.y + target.y) / 2;
        const scale = 1.5; // Zoom in a bit more

        svg
          .transition()
          .duration(750)
          .call(
            zoomBehavior.transform,
            d3.zoomIdentity
              .translate(width / 2, height / 2)
              .scale(scale)
              .translate(-centerX, -centerY)
          );
      }
    }
  }, [
    isDataSettled,
    selectedConnection,
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
    // Ensure graphData is available
    if (!graphData || !graphData.nodes || !graphData.links) return;

    const svg = d3.select(svgRef.current!); // svgRef.current is checked by the caller.

    // Use provided dimensions, with fallbacks
    const resolvedWidth = svgWidth > 0 ? svgWidth : 600;
    const resolvedHeight = svgHeight > 0 ? svgHeight : 400;

    // Prepare nodes and links in D3-compatible format.
    const d3Nodes: D3Node[] = graphData.nodes.map((node) => ({
      ...node,
      x: initialPositions[node.id]?.x,
      y: initialPositions[node.id]?.y,
    }));

    const d3Links: D3Link[] = graphData.links.map((link) => ({
      ...link,
      source: link.entity_id_1,
      target: link.entity_id_2,
    }));

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
        const [x, y] = d3.pointer(event, svgRef.current);
        
        // Convert D3Link to VisualizationEntityEdge for the tooltip
        const edgeData: VisualizationEntityEdge = {
          id: d.id,
          cluster_id: '', // This might not be available in the D3Link
          entity_id_1: typeof d.source === 'string' ? d.source : d.source.id,
          entity_id_2: typeof d.target === 'string' ? d.target : d.target.id,
          edge_weight: d.edge_weight,
          details: d.details,
          created_at: new Date() // This might not be available in the D3Link
        };
        
        setHoveredLink(edgeData);
        setTooltipPosition({ x, y });
      })
      .on("mouseout", () => {
        setHoveredLink(null);
      })
      .on("click", (event: MouseEvent, d: D3Link) => {
        // Find the original edge from graphData.links to pass to onSelectConnection
        const originalEdge = graphData.links.find(link => link.id === d.id);
        if (originalEdge) {
          onSelectConnection(originalEdge);
        }
        event.stopPropagation(); // Prevent the click from being captured by the svg
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
        const [x, y] = d3.pointer(event, svgRef.current);
        
        onNodeHover(d.id);
        
        setHoveredNode({
          entity: {
            id: d.id,
            name: d.name,
            organization_id: d.organization_id
          },
          details: entityDetailsCache[d.id]
        });
        setTooltipPosition({ x, y });
      })
      .on("mouseout", () => {
        setHoveredNode(null);
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
        ?.attr("x1", (d) => (typeof d.source === 'string' ? 0 : (d.source as D3Node).x ?? 0))
        .attr("y1", (d) => (typeof d.source === 'string' ? 0 : (d.source as D3Node).y ?? 0))
        .attr("x2", (d) => (typeof d.target === 'string' ? 0 : (d.target as D3Node).x ?? 0))
        .attr("y2", (d) => (typeof d.target === 'string' ? 0 : (d.target as D3Node).y ?? 0));

      nodesRef.current?.attr("cx", (d) => d.x ?? 0).attr("cy", (d) => d.y ?? 0);

      labelsRef.current?.attr("x", (d) => d.x ?? 0).attr("y", (d) => d.y ?? 0);
    });

    // --- Center on Selected Connection if there is one ---
    if (selectedConnection && zoomRef.current) {
      const zoomBehavior = zoomRef.current;
      setTimeout(() => {
        const sourceNode = d3Nodes.find((n) => n.id === selectedConnection.entity_id_1);
        const targetNode = d3Nodes.find((n) => n.id === selectedConnection.entity_id_2);

        if (
          sourceNode?.x != null &&
          sourceNode?.y != null &&
          targetNode?.x != null &&
          targetNode?.y != null
        ) {
          const centerX = (sourceNode.x + targetNode.x) / 2;
          const centerY = (sourceNode.y + targetNode.y) / 2;
          const scale = 1.5;

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

  // If graphData is available, but we are still waiting for entity details, show a loading message
  if (graphData && !isDataSettled && !isLoadingGraphStructure) {
    return (
      <div className="flex w-full h-full items-center justify-center">
        <p>Loading entity details for the graph...</p>
      </div>
    );
  }

  return (
    <div
      className="relative w-full h-full"
      data-testid="graph-visualization-container"
    >
      {/* SVG element where D3 will render the graph */}
      <svg ref={svgRef} className="w-full h-full"></svg>

      {/* Render tooltips only if data is settled and a tooltip is active */}
      {isDataSettled && hoveredNode && (
        <NodeTooltip
          node={hoveredNode.entity}
          details={hoveredNode.details}
          position={tooltipPosition}
          loading={detailsLoading[hoveredNode.entity.id]}
        />
      )}
      
      {isDataSettled && hoveredLink && (
        <LinkTooltip 
          link={hoveredLink} 
          position={tooltipPosition} 
        />
      )}
    </div>
  );
};

export default GraphVisualization;