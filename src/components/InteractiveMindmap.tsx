import {
  forwardRef,
  memo,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent } from "react";
import Tree, { type Orientation, type Point } from "react-d3-tree";
import {
  hierarchy,
  tree,
  type HierarchyPointLink,
  type HierarchyPointNode,
} from "d3-hierarchy";
import { linkRadial } from "d3-shape";
import { usePalette, useThemeMode } from "./mindmap/theme";
import { buildAugmentedTree } from "./mindmap/tree";
import { createNodeRenderer } from "./mindmap/NodeRenderer";
import { exportSvgElement } from "./mindmap/export";
import { smoothStepPath, organicPath } from "./mindmap/path";
import { resetMeasurements } from "./mindmap/measure";
import type {
  HydratedMindmapNode,
  InteractiveMindmapHandle,
  MindmapNode,
  MindmapThemeMode,
  ExportFormat,
  AugmentedRawNodeDatum,
} from "./mindmap/types";

const DEFAULT_HEIGHT = 640;
const DEFAULT_NODE_SIZE = { x: 240, y: 160 };
const DEFAULT_SEPARATION = { siblings: 1.25, nonSiblings: 1.95 };
const DEFAULT_SCALE = { min: 0.5, max: 1.85 };
const DISPLAY_FONT_STACK =
  'var(--display-font, "Space Grotesk", "Futura PT", "Manrope", "Inter", "Helvetica Neue", Arial, sans-serif)';

export interface InteractiveMindmapProps {
  data?: MindmapNode;
  height?: number | string;
  className?: string;
  style?: CSSProperties;
  collapseByDefaultMap?: boolean;
  collapseByDefaultRadial?: boolean;
  orientation?: Orientation;
  initialDepth?: number;
  zoomable?: boolean;
  collapsible?: boolean;
  separation?: { siblings?: number; nonSiblings?: number };
  scaleExtent?: { min?: number; max?: number };
  themeMode?: MindmapThemeMode;
  linkColor?: string;
  leafTint?: { light: string; dark: string };
  branchTint?: { light: string; dark: string };
  onNodeClick?: (node: HydratedMindmapNode) => void;
  linkStyle?: "default" | "organic";
  layout?: "mindmap" | "tidy" | "radial";
  showControls?: boolean;
  letterSpacing?: number;
  exportRef?: React.Ref<InteractiveMindmapHandle>;
  allowLayoutSwitch?: boolean;
}

const BASE_CONTAINER_STYLES: CSSProperties = {
  width: "100%",
  borderRadius: "1.9rem",
  border: "1px solid rgba(148,163,184,0.22)",
  background:
    "linear-gradient(135deg, rgba(15,23,42,0.045) 0%, rgba(37,99,235,0.07) 55%, rgba(15,23,42,0.02) 100%)",
  boxShadow: "0 45px 82px -48px rgba(15,23,42,0.38)",
  position: "relative",
  overflow: "hidden",
};

const calculateTranslate = (width: number, height: number, orientation: Orientation) => {
  if (orientation === "vertical") {
    return {
      x: width / 2,
      y: Math.max(height * 0.16, 120),
    };
  }

  return {
    x: Math.max(width * 0.24, 240),
    y: Math.max(height * 0.5, 240),
  };
};

const useAutoTranslate = (
  ref: React.RefObject<HTMLDivElement | null>,
  orientation: Orientation,
) => {
  const [translate, setTranslate] = useState<Point>(() =>
    calculateTranslate(900, DEFAULT_HEIGHT, orientation),
  );

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.ResizeObserver === "undefined") {
      return;
    }

    const element = ref.current;
    if (!element) return;

    const compute = () => {
      const { width, height } = element.getBoundingClientRect();
      setTranslate(calculateTranslate(width, height, orientation));
    };

    compute();

    const observer = new window.ResizeObserver(() => compute());
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref, orientation]);

  return translate;
};

const resolveHeight = (height?: number | string) => {
  if (typeof height === "number") return `${height}px`;
  if (typeof height === "string") return height;
  return `${DEFAULT_HEIGHT}px`;
};

const useContainerSize = (ref: React.RefObject<HTMLDivElement | null>) => {
  const [size, setSize] = useState({ width: 800, height: 640 });

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.ResizeObserver === "undefined") {
      return;
    }
    const element = ref.current;
    if (!element) return;

    const update = () => {
      const rect = element.getBoundingClientRect();
      setSize({
        width: Math.max(rect.width, 400),
        height: Math.max(rect.height, 400),
      });
    };

    update();

    const observer = new window.ResizeObserver(() => update());
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref]);

  return size;
};

const RADIAL_MARGIN = 180;
const MIN_RADIAL_RADIUS = 440;
const LABEL_CHAR_ESTIMATE = 8;
const LABEL_PADDING_MIN = 120;
const LABEL_PADDING_MAX = 420;
const LABEL_EXTRA_PADDING = 96;

interface RadialTreeProps {
  data: AugmentedRawNodeDatum;
  zoom: number;
  linkColor: string;
  containerRef: React.RefObject<HTMLDivElement | null>;
  textColor: string;
  labelStroke: string;
  onNodeClick?: (node: HydratedMindmapNode) => void;
}

const RadialTree = memo(function RadialTree({
  data,
  zoom,
  linkColor,
  containerRef,
  textColor,
  labelStroke,
  onNodeClick,
}: RadialTreeProps) {
  const size = useContainerSize(containerRef);

  const radius = useMemo(() => {
    const dimension = Math.min(size.width, size.height);
    return Math.max(MIN_RADIAL_RADIUS, (dimension - RADIAL_MARGIN) / 2);
  }, [size.height, size.width]);

  const { nodes, links, labelPadding } = useMemo(() => {
    const root = hierarchy<AugmentedRawNodeDatum>(data, (d) =>
      d.collapsed ? null : d.children,
    );
    const layout = tree<AugmentedRawNodeDatum>()
      .size([2 * Math.PI, radius])
      .separation((a, b) => (a.parent === b.parent ? 1 : 2));
    const layoutRoot = layout(root);
    const radialLink = linkRadial<
      HierarchyPointLink<AugmentedRawNodeDatum>,
      HierarchyPointNode<AugmentedRawNodeDatum>
    >()
      .angle((d) => d.x)
      .radius((d) => d.y);

    const descendants = layoutRoot.descendants();
    const maxLabelLength = descendants.reduce(
      (max, node) => Math.max(max, node.data.name?.length ?? 0),
      0,
    );
    const estimatedPadding = Math.min(
      LABEL_PADDING_MAX,
      Math.max(LABEL_PADDING_MIN, maxLabelLength * LABEL_CHAR_ESTIMATE),
    );

    return {
      nodes: descendants,
      links: layoutRoot.links().map((link) => ({
        key: `${link.source.data.payload.id}-${link.target.data.payload.id}`,
        path: radialLink(link) ?? "",
      })),
      labelPadding: estimatedPadding,
    };
  }, [data, radius]);

  const canvasExtent = useMemo(
    () => radius + labelPadding + RADIAL_MARGIN / 2 + LABEL_EXTRA_PADDING,
    [radius, labelPadding],
  );
  const viewBox = useMemo(
    () => [-canvasExtent, -canvasExtent, canvasExtent * 2, canvasExtent * 2].join(" "),
    [canvasExtent],
  );

  const handleNodeAction = useCallback(
    (node: HierarchyPointNode<AugmentedRawNodeDatum>) => {
      onNodeClick?.(node.data.payload);
    },
    [onNodeClick],
  );

  return (
    <svg
      width="100%"
      height="100%"
      viewBox={viewBox}
      preserveAspectRatio="xMidYMid meet"
      style={{
        background: "var(--canvas-bg, transparent)",
        shapeRendering: "geometricPrecision",
        textRendering: "optimizeLegibility",
      }}
    >
      <g className="mindmap-radial" transform={`translate(0,0) scale(${zoom})`}>
        <g
          fill="none"
          stroke={linkColor}
          strokeOpacity={0.4}
          strokeWidth={1}
          strokeLinecap="round"
        >
          {links.map((link) => (
            <path key={link.key} d={link.path} />
          ))}
        </g>
        {nodes.map((node) => {
          const hasChildren = Boolean(node.children && node.children.length > 0);
          const titleFontWeight = hasChildren ? 360 : 460;
          const rotation = (node.x * 180) / Math.PI - 90;
          const translate = `translate(${node.y},0)`;
          const flipped = node.x >= Math.PI;
          const radialLabelOffset = labelPadding * 0.45 + (hasChildren ? 18 : 22);
          const textOffset = flipped ? -radialLabelOffset : radialLabelOffset;
          const textAnchor = flipped ? "end" : "start";
          const textStrokeWidth = hasChildren ? 1 : 1.2;

          const handleKeyDown =
            onNodeClick !== undefined
              ? (event: ReactKeyboardEvent<SVGGElement>) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    handleNodeAction(node);
                  }
                }
              : undefined;

          return (
            <g
              key={node.data.payload.id}
              transform={`rotate(${rotation}) ${translate}`}
              onClick={onNodeClick ? () => handleNodeAction(node) : undefined}
              tabIndex={onNodeClick ? 0 : undefined}
              onKeyDown={handleKeyDown}
              style={{ cursor: onNodeClick ? "pointer" : "default" }}
            >
              <circle
                r={hasChildren ? 3.2 : 2.4}
                fill={hasChildren ? linkColor : "#f9fbff"}
                stroke={linkColor}
                strokeWidth={hasChildren ? 1 : 0.9}
              />
              <text
                dy="0"
                x={textOffset}
                textAnchor={textAnchor}
                dominantBaseline="middle"
                transform={flipped ? "rotate(180)" : undefined}
                fill={textColor}
                stroke={labelStroke}
                strokeWidth={textStrokeWidth}
                strokeOpacity={0.65}
                paintOrder="stroke fill"
                data-role="title"
                style={{
                  fontFamily: DISPLAY_FONT_STACK,
                  fontSize: "0.68rem",
                  fontWeight: titleFontWeight,
                  letterSpacing: 0.012,
                  WebkitFontSmoothing: "antialiased",
                  textRendering: "optimizeLegibility",
                  fontVariationSettings: `"wght" ${titleFontWeight}`,
                  userSelect: "none",
                }}
              >
                {node.data.name}
              </text>
              {node.data.attributes?.description ? (
                <title>{node.data.attributes.description}</title>
              ) : null}
            </g>
          );
        })}
      </g>
    </svg>
  );
});

const Controls = ({
  layout,
  onLayoutChange,
  isFullscreen,
  onToggleFullscreen,
  onZoomIn,
  onZoomOut,
  onReset,
  onExport,
}: {
  layout: "mindmap" | "radial";
  onLayoutChange?: (layout: "mindmap" | "radial") => void;
  isFullscreen?: boolean;
  onToggleFullscreen?: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
  onExport: (format: ExportFormat) => void;
}) => (
  <div className="mindmap-controls">
    {onLayoutChange ? (
      <div className="mindmap-layout-switch" aria-label="Layout toggle">
        <button
          type="button"
          className={layout === "mindmap" ? "active" : undefined}
          onClick={() => onLayoutChange("mindmap")}
        >
          Map
        </button>
        <button
          type="button"
          className={layout === "radial" ? "active" : undefined}
          onClick={() => onLayoutChange("radial")}
        >
          Radial
        </button>
      </div>
    ) : null}
    {onToggleFullscreen ? (
      <button
        type="button"
        className={`fullscreen-toggle${isFullscreen ? " active" : ""}`}
        onClick={onToggleFullscreen}
        aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
      >
        {isFullscreen ? "Close" : "Full"}
      </button>
    ) : null}
    <button type="button" onClick={onZoomIn} aria-label="Zoom in">
      +
    </button>
    <button type="button" onClick={onZoomOut} aria-label="Zoom out">
      −
    </button>
    <button type="button" onClick={onReset} aria-label="Reset view">
      Reset
    </button>
    <button type="button" onClick={() => onExport("png")} aria-label="Export as PNG">
      PNG
    </button>
    <button type="button" onClick={() => onExport("pdf")} aria-label="Export as PDF">
      PDF
    </button>
    <button type="button" onClick={() => onExport("svg")} aria-label="Export as SVG">
      SVG
    </button>
  </div>
);

// Define the MindmapNode type for context, assuming a structure like:
/*
type MindmapNode = {
  label: string;
  description?: string;
  children?: MindmapNode[];
};
*/

export const DEFAULT_MINDMAP_DATA: MindmapNode = {
    label: "🏆 LeetCode Problem Classification (Interview Prep)", // Added emoji and "Interview Prep"
    description: "Core algorithms and data structures for FAANG-level interviews.",
    children: [
      // --- FOUNDATIONAL DATA STRUCTURES & PATTERNS ---
      {
        label: "📚 Arrays & Strings (Fundamental Patterns)",
        children: [
          {
            label: "Subarrays, Substrings & Windows",
            description: "Problems involving contiguous segments: max/min length, sum, product.",
            children: [
              { label: "Sliding Window" }, // Essential for O(n) solutions
              { label: "Two Pointers (Same Direction)" },
              { label: "Prefix Sums / Difference Array" }, // Crucial for range queries
            ],
          },
          {
            label: "Frequency & Uniqueness",
            description: "Counting, finding duplicates, or mapping relationships.",
            children: [{ label: "Hash Map / Set" }, { label: "Bucket Sort / Counting Sort" }], // Added Bucket Sort
          },
          {
            label: "In-Place & Sorted Array Manipulation",
            description: "Searching or manipulating elements in sorted or nearly sorted arrays.",
            children: [
              { label: "Two Pointers (Opposite Direction)" }, // Great for two-sum variations
              { label: "Cyclic Sort" }, // Excellent for bounded range anomalies (missing/duplicate)
            ],
          },
          {
            label: "Monotonic Sequences",
            description: "Finding the 'next greater' or calculating areas (e.g., histogram).",
            children: [
              { label: "Monotonic Stack" }, // Key for O(n) solutions to next greater/smaller
              { label: "Monotonic Queue / Deque" }, // For sliding window maximum/minimum
            ],
          },
        ],
      },
      // --- RECURSIVE & TREE STRUCTURES ---
      {
        label: "🌲 Trees (Binary & General)",
        children: [
          {
            label: "Traversal & Validation (DFS)",
            description: "Pre-order, In-order, Post-order, path-finding, and property checks.",
            children: [
              { label: "Recursion / Tree DFS" },
              { label: "Iterative Stack-based Traversal" },
              { label: "Lowest Common Ancestor (LCA)" }, // Specific, high-value sub-pattern
            ],
          },
          {
            label: "Level-Based Operations (BFS)",
            description: "Finding depth, shortest path in unweighted graphs, and level order.",
            children: [{ label: "Tree BFS (Level Order Traversal)" }, { label: "Minimum Depth / Height" }],
          },
          {
            label: "Binary Search Trees (BSTs)",
            description: "Utilizing the sorted property for efficient search/insertion.",
            children: [{ label: "BST Properties & Search" }, { label: "In-order Traversal" }],
          },
        ],
      },
      {
        label: "🔗 Linked Lists",
        children: [
          {
            label: "Two-Pointer Techniques",
            description: "Finding midpoints, cycles, or N-th element from the end.",
            children: [{ label: "Fast & Slow Pointers (Floyd's Cycle)" }],
          },
          {
            label: "Structural Manipulation",
            description: "Reversal, merging, grouping, or segment manipulation.",
            children: [{ label: "In-place Reversal" }, { label: "Dummy Head Node" }],
          },
        ],
      },
      // --- SEARCHING & OPTIMIZATION ---
      {
        label: "🔍 Search Algorithms & Divide/Conquer",
        children: [
          {
            label: "Efficient Search in Sorted Data",
            description: "Finding values, thresholds, or first/last occurrences.",
            children: [
              { label: "Binary Search (Classic)" },
              { label: "Modified Binary Search (On Answer)" }, // For optimization problems
            ],
          },
          {
            label: "Greedy Algorithms",
            description: "Local optimal choice leads to global optimal solution.",
            children: [{ label: "Greedy Choice Property" }, { label: "Proof of Optimality" }],
          },
        ],
      },
      {
        label: "🔄 Backtracking & Recursion",
        children: [
          {
            label: "Combinatorics & State Space Search",
            description: "Generating all permutations, combinations, or subsets.",
            children: [
              { label: "Backtracking (DFS with Pruning)" },
              { label: "State Management & Helper Function" },
            ],
          },
          {
            label: "Subsets & Power Set",
            description: "Specialized techniques for generating all sub-elements.",
            children: [{ label: "Recursion / Cascading" }, { label: "Bit Manipulation (for Subsets)" }],
          },
        ],
      },
      // --- ADVANCED TECHNIQUES ---
      {
        label: "📈 Dynamic Programming (DP)",
        children: [
          {
            label: "Sequence & Path Optimization",
            description: "Finding max/min paths, max product, or longest sequences.",
            children: [
              { label: "1D DP (e.g., House Robber, LIS)" },
              { label: "2D DP (e.g., Unique Paths, Edit Distance)" },
            ],
          },
          {
            label: "Decision Making & Counting",
            description: "Knapsack variations, coin change, or counting ways to reach a state.",
            children: [{ label: "Memoization (Top-Down)" }, { label: "Tabulation (Bottom-Up)" }],
          },
          {
            label: "Interval DP & Matrix Chain",
            description: "Problems defined over segments or sub-matrices.",
            children: [{ label: "Interval DP" }, { label: "Space Optimization (Rolling Array)" }],
          },
        ],
      },
      {
        label: "🕸️ Graphs",
        children: [
          {
            label: "Traversal & Connectivity",
            description: "Flood-fill, finding components, or basic path-finding.",
            children: [
              { label: "BFS (Breadth-First Search)" },
              { label: "DFS (Depth-First Search)" },
              { label: "Union-Find (Disjoint Set Union)" }, // Key for component checking
            ],
          },
          {
            label: "Shortest Path & Weighted Graphs",
            description: "Finding the minimum cost path between nodes.",
            children: [
              { label: "Dijkstra's Algorithm" }, // Single-source shortest path (non-negative)
              { label: "Bellman-Ford / Floyd-Warshall" }, // Added for completeness (negative edge/all-pairs)
            ],
          },
          {
            label: "Dependencies & Ordering (DAGs)",
            description: "Solving problems with prerequisites or directed flow.",
            children: [{ label: "Topological Sort (Kahn's or DFS)" }],
          },
        ],
      },
      {
        label: "🗄️ Heaps & Specialized Queues",
        children: [
          {
            label: "K-th Element & Priority Management",
            description: "Finding the smallest/largest K elements, or managing task priorities.",
            children: [
              { label: "Heap / Priority Queue (Min-Heap/Max-Heap)" },
              { label: "Heap-based Top-K" },
            ],
          },
          {
            label: "Median & Stream Data",
            description: "Maintaining central tendency in dynamic data streams.",
            children: [{ label: "Two Heaps Pattern (Min-Heap & Max-Heap)" }],
          },
        ],
      },
      {
        label: "⏱️ Interval & Sweep Line",
        children: [
          {
            label: "Scheduling & Overlap Management",
            description: "Merging, inserting, or scheduling tasks/meetings.",
            children: [
              { label: "Merge Intervals Pattern" }, // Classic sorting and merging
              { label: "Sweep Line / Difference Array" }, // For finding max overlaps
            ],
          },
        ],
      },
      {
        label: "🧠 Advanced Strings & Bitwise",
        children: [
          {
            label: "Prefix & Dictionary Lookups",
            description: "Efficient searching/storage for strings with common prefixes.",
            children: [{ label: "Trie (Prefix Tree)" }],
          },
          {
            label: "Pattern Matching & Automata",
            description: "Searching for complex patterns within a larger text.",
            children: [{ label: "KMP (Knuth-Morris-Pratt)" }, { label: "Rolling Hash (Rabin-Karp)" }],
          },
          {
            label: "Binary Arithmetic & Properties",
            description: "Utilizing bitwise operators for constant-time manipulation.",
            children: [{ label: "Bitwise Operations" }, { label: "XOR Tricks (Finding Uniques)" }],
          },
        ],
      },
      // --- NEWLY ADDED CATEGORIES ---
      {
        label: "🧩 System Design/Math Concepts",
        children: [
          {
            label: "Randomization & Sampling",
            description: "Algorithms for shuffling, random selection, or simulation.",
            children: [{ label: "Reservoir Sampling" }, { label: "Fisher-Yates Shuffle" }],
          },
          {
            label: "Math & Geometry",
            description: "Problems involving number theory, prime numbers, or coordinates.",
            children: [{ label: "Prime Sieve / GCD" }, { label: "Modular Arithmetic" }],
          },
        ],
      },
    ],
  };
// export const DEFAULT_MINDMAP_DATA: MindmapNode = {
//   label: "LeetCode Problem Classification",
//   children: [
//     {
//       label: "Arrays & Strings",
//       children: [
//         {
//           label: "Contiguous subarray / substring",
//           description: "max length, sum, uniqueness",
//           children: [
//             { label: "Sliding Window" },
//             { label: "Two Pointers" },
//             { label: "Prefix Sums" },
//           ],
//         },
//         {
//           label: "Frequency / membership",
//           description: "counts, duplicates, mappings",
//           children: [{ label: "Hash Map / Set" }],
//         },
//         {
//           label: "Pairs / triples / sums in sorted data",
//           children: [{ label: "Two Pointers" }],
//         },
//         {
//           label: "Next greater / smaller / histogram",
//           children: [
//             { label: "Monotonic Stack" },
//             { label: "Monotonic Queue" },
//           ],
//         },
//       ],
//     },
//     {
//       label: "Search & Sort",
//       children: [
//         {
//           label: "Searching value / threshold",
//           children: [{ label: "Binary Search" }, { label: "Modified Binary Search" }],
//         },
//         {
//           label: "Local optima → global answer",
//           children: [{ label: "Greedy" }],
//         },
//         {
//           label: "Bounded range anomalies",
//           description: "missing / duplicate",
//           children: [{ label: "Cyclic Sort" }],
//         },
//       ],
//     },
//     {
//       label: "Dynamic Programming",
//       children: [
//         {
//           label: "Counting combinations / overlapping states",
//           children: [{ label: "Classic DP (tabulation / memoization)" }],
//         },
//         {
//           label: "Optimization over sequences / paths",
//           children: [{ label: "Advanced DP (Knapsack / LIS / interval)" }],
//         },
//       ],
//     },
//     {
//       label: "Graphs",
//       children: [
//         {
//           label: "Connectivity / flood-fill / components",
//           children: [{ label: "BFS" }, { label: "DFS" }, { label: "Union-Find" }],
//         },
//         {
//           label: "Dependencies / prerequisites (DAG)",
//           children: [{ label: "Topological Sort" }],
//         },
//         {
//           label: "Cycles / shortest path / maze",
//           children: [
//             { label: "Graph Traversal (BFS / DFS)" },
//             { label: "Fast & Slow Pointers" },
//           ],
//         },
//       ],
//     },
//     {
//       label: "Trees",
//       children: [
//         {
//           label: "Hierarchy / recursion / validation",
//           children: [
//             { label: "Tree DFS (pre / in / post)" },
//             { label: "Stack-based Traversal" },
//           ],
//         },
//         {
//           label: "Level order / minimum depth",
//           children: [{ label: "Tree BFS" }],
//         },
//       ],
//     },
//     {
//       label: "Heaps & Queues",
//       children: [
//         {
//           label: "Min / max / k-th extreme",
//           children: [
//             { label: "Heap / Priority Queue" },
//             { label: "Two Heaps (median)" },
//             { label: "Monotonic Deque" },
//           ],
//         },
//         {
//           label: "Top-K frequent without full sort",
//           children: [{ label: "Heap-based Top-K" }],
//         },
//       ],
//     },
//     {
//       label: "Backtracking & Combinatorics",
//       children: [
//         {
//           label: "Permutations / combinations with pruning",
//           children: [{ label: "Backtracking" }],
//         },
//         {
//           label: "Subsets / powerset / combination sums",
//           children: [{ label: "Recursion" }, { label: "Bit Manipulation" }],
//         },
//       ],
//     },
//     {
//       label: "Intervals",
//       children: [
//         {
//           label: "Overlaps / merges / scheduling",
//           children: [
//             { label: "Merge Intervals Pattern" },
//             { label: "Sweep Line / Overlapping Intervals" },
//           ],
//         },
//       ],
//     },
//     {
//       label: "Linked Lists",
//       children: [
//         {
//           label: "Cycle detection / midpoints",
//           children: [{ label: "Fast & Slow Pointers" }],
//         },
//         {
//           label: "Reordering / reversing segments",
//           children: [{ label: "In-place Reversal" }],
//         },
//       ],
//     },
//     {
//       label: "Bit Manipulation",
//       children: [
//         {
//           label: "Binary representation / parity / uniques",
//           children: [{ label: "Bitwise Operations" }, { label: "XOR Tricks" }],
//         },
//       ],
//     },
//     {
//       label: "Strings (Advanced)",
//       children: [
//         {
//           label: "Prefix matching / dictionary search",
//           children: [{ label: "Trie (Prefix Tree)" }],
//         },
//         {
//           label: "Pattern matching / automata",
//           children: [{ label: "KMP" }, { label: "Rolling Hash" }],
//         },
//       ],
//     },
//   ],
// };

const InteractiveMindmap = forwardRef<InteractiveMindmapHandle, InteractiveMindmapProps>(
  (
    {
      data = DEFAULT_MINDMAP_DATA,
      height,
      className,
      style,
      collapseByDefaultMap = false,
      collapseByDefaultRadial = false,
      orientation = "vertical",
      initialDepth = 0,
      zoomable = true,
      collapsible = true,
      separation,
      scaleExtent,
      themeMode = "auto",
      linkColor,
      leafTint,
      branchTint,
      onNodeClick,
      showControls = true,
      letterSpacing = 0.008,
      exportRef,
      linkStyle = "default",
      layout = "radial",
      allowLayoutSwitch = true,
    },
    forwardedRef,
  ) => {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const previousBodyOverflow = useRef<string | null>(null);
    const fullscreenPortalRef = useRef<HTMLDivElement | null>(null);
    const [activeLayout, setActiveLayout] = useState<"mindmap" | "tidy" | "radial">(layout);
    const treeOrientation = activeLayout === "mindmap" ? orientation : "vertical";
    const isRadial = activeLayout === "radial";
    const autoTranslate = useAutoTranslate(containerRef, treeOrientation);
    const [translate, setTranslate] = useState<Point>(autoTranslate);
    const [zoom, setZoom] = useState(1);
    const [isFullscreen, setIsFullscreen] = useState(false);

    useEffect(() => {
      setActiveLayout(layout);
    }, [layout]);

    useEffect(() => {
      if (isRadial) {
        setTranslate({ x: 0, y: 0 });
      } else {
        setTranslate(autoTranslate);
      }
    }, [autoTranslate, isRadial]);

    useEffect(() => {
      if (typeof document === "undefined") return;
      const { body } = document;

      if (isFullscreen) {
        if (previousBodyOverflow.current === null) {
          previousBodyOverflow.current = body.style.overflow;
        }
        body.style.overflow = "hidden";
      } else if (previousBodyOverflow.current !== null) {
        body.style.overflow = previousBodyOverflow.current;
        previousBodyOverflow.current = null;
      }

      return () => {
        if (previousBodyOverflow.current !== null) {
          body.style.overflow = previousBodyOverflow.current;
          previousBodyOverflow.current = null;
        }
      };
    }, [isFullscreen]);

    useEffect(() => {
      if (!isFullscreen) return;
      const handleKeyDown = (event: globalThis.KeyboardEvent) => {
        if (event.key === "Escape") {
          setIsFullscreen(false);
        }
      };
      window.addEventListener("keydown", handleKeyDown);
      return () => window.removeEventListener("keydown", handleKeyDown);
    }, [isFullscreen]);

    useEffect(() => {
      if (typeof document === "undefined") return;
      if (isFullscreen) {
        if (!fullscreenPortalRef.current) {
          const el = document.createElement("div");
          el.className = "mindmap-fullscreen-root";
          document.body.appendChild(el);
          fullscreenPortalRef.current = el;
        }
      } else if (fullscreenPortalRef.current) {
        fullscreenPortalRef.current.remove();
        fullscreenPortalRef.current = null;
      }

      return () => {
        if (fullscreenPortalRef.current && !isFullscreen) {
          fullscreenPortalRef.current.remove();
          fullscreenPortalRef.current = null;
        }
      };
    }, [isFullscreen]);

    useEffect(() => {
      setZoom(1);
    }, [activeLayout]);

    useEffect(() => {
      resetMeasurements();
    }, [data]);

    const isDark = useThemeMode(themeMode);
    const palette = usePalette(isDark, branchTint, leafTint);

    const treeData = useMemo(
      () =>
        buildAugmentedTree(data, {
          collapseByDefault:
            activeLayout === "radial" ? collapseByDefaultRadial : collapseByDefaultMap,
        }),
      [data, collapseByDefaultMap, collapseByDefaultRadial, activeLayout],
    );

    const nodeRenderer = useMemo(
      () => createNodeRenderer({ palette, onNodeClick, letterSpacing }),
      [palette, onNodeClick, letterSpacing],
    );

    const separationConfig = useMemo(() => {
      const defaults =
        activeLayout === "mindmap"
          ? DEFAULT_SEPARATION
          : {
              siblings: 1.05,
              nonSiblings: 1.45,
            };
      return { ...defaults, ...separation };
    }, [activeLayout, separation]);
    const scaleConfig = { ...DEFAULT_SCALE, ...scaleExtent };
    const resolvedHeight = resolveHeight(height);

    const containerStyles = useMemo(() => {
      const baseStyles = {
        ...BASE_CONTAINER_STYLES,
        ...style,
      };

      if (isFullscreen) {
        return {
          ...baseStyles,
          position: "fixed" as const,
          inset: 0,
          width: "100vw",
          height: "100vh",
          minHeight: "100vh",
          borderRadius: 0,
          boxShadow: "none",
          margin: 0,
          aspectRatio: "auto",
          overflow: "hidden",
          zIndex: 9999,
          background:
            "linear-gradient(135deg, rgba(10,18,36,0.92) 0%, rgba(37,99,235,0.48) 50%, rgba(10,18,36,0.92) 100%)",
        };
      }

      if (activeLayout === "radial") {
        return {
          ...baseStyles,
          height: "auto",
          minHeight: resolvedHeight,
          aspectRatio: "1 / 1",
          overflow: "visible",
        };
      }

      return {
        ...baseStyles,
        height: resolvedHeight,
      };
    }, [activeLayout, isFullscreen, resolvedHeight, style]);

    const effectiveLinkColor = linkColor ?? palette.link;

    const handleUpdate = useCallback(
      (target: { zoom: number; translate: Point }) => {
        if (typeof target.zoom === "number") setZoom(target.zoom);
        if (target.translate && !isRadial) setTranslate(target.translate);
      },
      [isRadial],
    );

    const adjustZoom = (delta: number) => {
      if (!zoomable) return;
      const next = Math.min(
        scaleConfig.max ?? DEFAULT_SCALE.max,
        Math.max(scaleConfig.min ?? DEFAULT_SCALE.min, zoom + delta),
      );
      setZoom(next);
    };

    const zoomIn = () => adjustZoom(0.12);
    const zoomOut = () => adjustZoom(-0.12);

    const resetView = () => {
      setZoom(1);
      if (isRadial) {
        setTranslate({ x: 0, y: 0 });
      } else {
        setTranslate(autoTranslate);
      }
    };

    const exportAs = async (format: ExportFormat) => {
      if (typeof document === "undefined") return;
      const svg = containerRef.current?.querySelector("svg");
      if (!svg) return;
      await exportSvgElement(svg, format, {
        fileName: `mindmap-${format}`,
        backgroundColor: isDark ? "#0f172a" : "#f8fafc",
        scale: 2,
      });
    };

    const imperativeHandle = useMemo<InteractiveMindmapHandle>(
      () => ({
        exportAs,
        resetView,
      }),
      [exportAs, resetView],
    );

    useImperativeHandle(forwardedRef, () => imperativeHandle, [imperativeHandle]);

    useEffect(() => {
      if (!exportRef) return undefined;
      if (typeof exportRef === "function") {
        exportRef(imperativeHandle);
        return () => {
          exportRef(null);
        };
      }
      if (typeof exportRef === "object" && exportRef !== null) {
        const mutableRef = exportRef as { current: InteractiveMindmapHandle | null };
        mutableRef.current = imperativeHandle;
        return () => {
          mutableRef.current = null;
        };
      }
      return undefined;
    }, [exportRef, imperativeHandle]);

    const graphContent = isRadial ? (
      <RadialTree
        data={treeData}
        zoom={zoom}
        linkColor={effectiveLinkColor}
        containerRef={containerRef}
        textColor={palette.leaf.name}
        labelStroke={isDark ? "#0f172a" : "#ffffff"}
        onNodeClick={onNodeClick}
      />
    ) : (
      <Tree
        data={treeData}
        orientation={treeOrientation}
        collapsible={collapsible}
        zoomable={zoomable}
        translate={translate}
        separation={separationConfig}
        nodeSize={DEFAULT_NODE_SIZE}
        scaleExtent={scaleConfig}
        transitionDuration={300}
        initialDepth={initialDepth}
        renderCustomNodeElement={nodeRenderer}
        pathFunc={linkStyle === "organic" ? organicPath : smoothStepPath}
        zoom={zoom}
        onUpdate={handleUpdate}
      />
    );

    const controlLayout: "mindmap" | "radial" = isRadial ? "radial" : "mindmap";

    const containerClassName = useMemo(() => {
      const classes = [className];
      if (isFullscreen) classes.push("mindmap-fullscreen");
      return classes.filter(Boolean).join(" ");
    }, [className, isFullscreen]);

    const mindmapContent = (
      <div ref={containerRef} className={containerClassName} style={containerStyles}>
        <style>
          {`
            .mindmap-link {
              stroke: ${effectiveLinkColor};
              stroke-width: 1px;
              stroke-linecap: round;
              stroke-opacity: 0.38;
              fill: none;
              pointer-events: none;
              transition: stroke 180ms ease, stroke-width 180ms ease;
            }

            .mindmap-link:hover {
              stroke-opacity: 0.54;
              stroke-width: 1.18px;
            }

            .mindmap-node text,
            .mindmap-radial text {
              font-family: "Inter", "Manrope", -apple-system, BlinkMacSystemFont, sans-serif;
              text-rendering: optimizeLegibility;
              -webkit-font-smoothing: antialiased;
              -moz-osx-font-smoothing: grayscale;
              font-optical-sizing: auto;
              font-feature-settings: "ss02", "tnum";
              letter-spacing: 0.006em;
            }

            .mindmap-node text[data-role="title"] {
              font-weight: 340 !important;
              font-variation-settings: "wght" 340 !important;
            }

            .mindmap-node text[data-role="description"] {
              font-weight: 300 !important;
              font-variation-settings: "wght" 300 !important;
            }

            .mindmap-radial text[data-role="title"] {
              letter-spacing: 0.012em;
              font-weight: 380;
            }

            .mindmap-radial text[data-role="description"] {
              font-weight: 360;
              letter-spacing: 0.008em;
            }

            .mindmap-radial {
              transition: transform 220ms ease;
            }

            .mindmap-radial text {
              transition: fill 160ms ease, stroke-width 160ms ease;
            }

            .mindmap-radial g[role="button"]:hover text,
            .mindmap-radial g[role="button"]:focus-visible text {
              stroke-width: 3.6;
            }

            .mindmap-radial g[role="button"]:hover circle,
            .mindmap-radial g[role="button"]:focus-visible circle {
              stroke-width: 1.4px;
            }

            .mindmap-node rect {
              transition: fill 220ms ease, stroke 220ms ease, transform 220ms ease;
              filter: drop-shadow(0 14px 24px rgba(15,23,42,0.12));
            }

            .mindmap-node:focus rect,
            .mindmap-node:hover rect {
              stroke-width: 1.5px;
              stroke-opacity: 0.9;
              transform: translateY(-1px);
            }

            .mindmap-controls {
              position: absolute;
              top: 1rem;
              right: 1rem;
              display: flex;
              gap: 0.4rem;
              align-items: center;
              background: rgba(255, 255, 255, 0.85);
              border: 1px solid rgba(148,163,184,0.25);
              border-radius: 999px;
              padding: 0.3rem 0.6rem;
              backdrop-filter: blur(16px);
              box-shadow: 0 18px 32px -28px rgba(15,23,42,0.4);
            }

            [data-theme="dark"] .mindmap-controls {
              background: rgba(15,23,42,0.78);
              border-color: rgba(148,163,184,0.35);
            }

            .mindmap-controls button {
              font-family: ${DISPLAY_FONT_STACK};
              font-size: 0.75rem;
              font-weight: 500;
              padding: 0.35rem 0.6rem;
              border-radius: 999px;
              border: none;
              background: rgba(255,255,255,0.9);
              color: rgba(15,23,42,0.8);
              cursor: pointer;
              transition: transform 0.18s ease, background 0.18s ease;
            }

            .mindmap-layout-switch {
              display: inline-flex;
              align-items: center;
              gap: 0.25rem;
              margin-right: 0.45rem;
              padding-right: 0.45rem;
              border-right: 1px solid rgba(148,163,184,0.3);
            }

            .mindmap-layout-switch button {
              padding: 0.28rem 0.6rem;
              font-size: 0.7rem;
              background: rgba(255,255,255,0.7);
              color: rgba(15,23,42,0.75);
            }

            .mindmap-layout-switch button.active {
              background: rgba(96,165,250,0.2);
              color: rgba(37,99,235,0.95);
            }

            .mindmap-controls button.fullscreen-toggle {
              min-width: 2.7rem;
            }

            .mindmap-controls button.fullscreen-toggle.active {
              background: rgba(96,165,250,0.24);
              color: rgba(37,99,235,0.95);
            }

            [data-theme="dark"] .mindmap-controls button {
              background: rgba(30,41,59,0.95);
              color: rgba(226,232,240,0.85);
            }

            .mindmap-controls button:hover,
            .mindmap-controls button:focus-visible {
              transform: translateY(-1px);
              background: rgba(96,165,250,0.18);
              color: rgba(37,99,235,0.9);
              outline: none;
            }

            [data-theme="dark"] .mindmap-controls button:hover,
            [data-theme="dark"] .mindmap-controls button:focus-visible {
              background: rgba(56,189,248,0.22);
              color: rgba(191,219,254,0.92);
            }

            [data-theme="dark"] .mindmap-layout-switch button {
              background: rgba(30,41,59,0.88);
              color: rgba(203,213,225,0.78);
            }

            [data-theme="dark"] .mindmap-layout-switch button.active {
              background: rgba(56,189,248,0.28);
              color: rgba(224,242,254,0.95);
            }

            [data-theme="dark"] .mindmap-controls button.fullscreen-toggle.active {
              background: rgba(56,189,248,0.35);
              color: rgba(224,242,254,0.95);
            }

            .mindmap-fullscreen {
              backdrop-filter: blur(18px);
            }

            .mindmap-fullscreen .mindmap-controls {
              top: clamp(1.25rem, 4vh, 3rem);
              right: clamp(1.25rem, 4vw, 3.25rem);
            }

            .mindmap-fullscreen-overlay {
              position: absolute;
              inset: 0;
              pointer-events: none;
              background: radial-gradient(circle at center, rgba(15,23,42,0.12) 0%, rgba(15,23,42,0.22) 55%, rgba(15,23,42,0.36) 100%);
            }

            .mindmap-fullscreen-hint {
              position: absolute;
              left: 50%;
              bottom: clamp(1.75rem, 5vh, 3.5rem);
              transform: translateX(-50%);
              padding: 0.5rem 0.95rem;
              border-radius: 999px;
              font-size: 0.72rem;
              font-weight: 500;
              letter-spacing: 0.04em;
              text-transform: uppercase;
              background: rgba(15,23,42,0.65);
              color: rgba(241,245,249,0.92);
              display: inline-flex;
              align-items: center;
              gap: 0.6rem;
              pointer-events: none;
              box-shadow: 0 12px 32px -24px rgba(15,23,42,0.9);
            }

            .mindmap-fullscreen-hint span {
              display: inline-flex;
              align-items: center;
              gap: 0.28rem;
              white-space: nowrap;
            }
          `}
        </style>
        {isFullscreen ? <div className="mindmap-fullscreen-overlay" aria-hidden="true" /> : null}
        {graphContent}
        {isFullscreen ? (
          <div className="mindmap-fullscreen-hint">
            <span>Drag to move</span>
            <span>Scroll to zoom</span>
            <span>Press ESC to exit</span>
          </div>
        ) : null}
        {showControls && (
          <Controls
            layout={controlLayout}
            onLayoutChange={
              allowLayoutSwitch
                ? (nextLayout) => {
                    setActiveLayout(nextLayout);
                  }
                : undefined
            }
            isFullscreen={isFullscreen}
            onToggleFullscreen={() => setIsFullscreen((value) => !value)}
            onZoomIn={zoomIn}
            onZoomOut={zoomOut}
            onReset={resetView}
            onExport={exportAs}
          />
        )}
      </div>
    );

    if (isFullscreen && typeof document !== "undefined" && fullscreenPortalRef.current) {
      return createPortal(mindmapContent, fullscreenPortalRef.current);
    }

    return mindmapContent;
  },
);

InteractiveMindmap.displayName = "InteractiveMindmap";

export default InteractiveMindmap;

