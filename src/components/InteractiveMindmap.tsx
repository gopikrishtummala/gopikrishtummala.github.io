import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import Tree, { type RawNodeDatum } from "react-d3-tree";

const BASE_FONT_STACK =
  '"Space Grotesk", "Futura PT", "Manrope", "Inter", "Helvetica Neue", Arial, sans-serif';

type MindmapNode = (RawNodeDatum & { children?: MindmapNode[] }) & {
  collapsed?: boolean;
};

const TREE_DATA: MindmapNode = {
  name: "LeetCode Problem Classification",
  children: [
    {
      name: "Arrays & Strings",
      children: [
        {
          name: "Contiguous subarray / substring",
          attributes: {
            clue: "max length, sum, uniqueness",
          },
          children: [
            { name: "Sliding Window" },
            { name: "Two Pointers" },
            { name: "Prefix Sums" },
          ],
        },
        {
          name: "Frequency / membership",
          attributes: {
            clue: "counts, duplicates, mappings",
          },
          children: [{ name: "Hash Map / Set" }],
        },
        {
          name: "Pairs / triples / sums in sorted data",
          children: [{ name: "Two Pointers" }],
        },
        {
          name: "Next greater / smaller / histogram",
          children: [
            { name: "Monotonic Stack" },
            { name: "Monotonic Queue" },
          ],
        },
      ],
    },
    {
      name: "Search & Sort",
      children: [
        {
          name: "Searching value / threshold",
          children: [
            { name: "Binary Search" },
            { name: "Modified Binary Search" },
          ],
        },
        {
          name: "Local optima → global answer",
          children: [{ name: "Greedy" }],
        },
        {
          name: "Bounded range anomalies",
          attributes: { clue: "missing / duplicate" },
          children: [{ name: "Cyclic Sort" }],
        },
      ],
    },
    {
      name: "Dynamic Programming",
      children: [
        {
          name: "Counting combinations / overlapping states",
          children: [{ name: "Classic DP (tabulation / memoization)" }],
        },
        {
          name: "Optimization over sequences / paths",
          children: [{ name: "Advanced DP (Knapsack / LIS / interval)" }],
        },
      ],
    },
    {
      name: "Graphs",
      children: [
        {
          name: "Connectivity / flood-fill / components",
          children: [
            { name: "BFS" },
            { name: "DFS" },
            { name: "Union-Find" },
          ],
        },
        {
          name: "Dependencies / prerequisites (DAG)",
          children: [{ name: "Topological Sort" }],
        },
        {
          name: "Cycles / shortest path / maze",
          children: [
            { name: "Graph Traversal (BFS / DFS)" },
            { name: "Fast & Slow Pointers" },
          ],
        },
      ],
    },
    {
      name: "Trees",
      children: [
        {
          name: "Hierarchy / recursion / validation",
          children: [
            { name: "Tree DFS (pre / in / post)" },
            { name: "Stack-based Traversal" },
          ],
        },
        {
          name: "Level order / minimum depth",
          children: [{ name: "Tree BFS" }],
        },
      ],
    },
    {
      name: "Heaps & Queues",
      children: [
        {
          name: "Min / max / k-th extreme",
          children: [
            { name: "Heap / Priority Queue" },
            { name: "Two Heaps (median)" },
            { name: "Monotonic Deque" },
          ],
        },
        {
          name: "Top-K frequent without full sort",
          children: [{ name: "Heap-based Top-K" }],
        },
      ],
    },
    {
      name: "Backtracking & Combinatorics",
      children: [
        {
          name: "Permutations / combinations with pruning",
          children: [{ name: "Backtracking" }],
        },
        {
          name: "Subsets / powerset / combination sums",
          children: [{ name: "Recursion" }, { name: "Bit Manipulation" }],
        },
      ],
    },
    {
      name: "Intervals",
      children: [
        {
          name: "Overlaps / merges / scheduling",
          children: [
            { name: "Merge Intervals Pattern" },
            { name: "Sweep Line / Overlapping Intervals" },
          ],
        },
      ],
    },
    {
      name: "Linked Lists",
      children: [
        {
          name: "Cycle detection / midpoints",
          children: [{ name: "Fast & Slow Pointers" }],
        },
        {
          name: "Reordering / reversing segments",
          children: [{ name: "In-place Reversal" }],
        },
      ],
    },
    {
      name: "Bit Manipulation",
      children: [
        {
          name: "Binary representation / parity / uniques",
          children: [{ name: "Bitwise Operations" }, { name: "XOR Tricks" }],
        },
      ],
    },
    {
      name: "Strings (Advanced)",
      children: [
        {
          name: "Prefix matching / dictionary search",
          children: [{ name: "Trie (Prefix Tree)" }],
        },
        {
          name: "Pattern matching / automata",
          children: [{ name: "KMP" }, { name: "Rolling Hash" }],
        },
      ],
    },
  ],
};

const containerStyles: CSSProperties = {
  width: "100%",
  height: "640px",
  borderRadius: "1.9rem",
  border: "1px solid rgba(148,163,184,0.24)",
  background:
    "linear-gradient(135deg, rgba(15,23,42,0.06) 0%, rgba(37,99,235,0.08) 55%, rgba(15,23,42,0.04) 100%)",
  boxShadow: "0 45px 85px -45px rgba(15,23,42,0.42)",
  position: "relative",
  overflow: "hidden",
};

const nodeSize = { x: 260, y: 168 };

const measureLabelWidth = (() => {
  let canvas: HTMLCanvasElement | null = null;
  let context: CanvasRenderingContext2D | null = null;
  return (text: string, fontSize: number, fontWeight = 400) => {
    if (!text) return 0;
    if (typeof document === "undefined") {
      return text.length * fontSize * 0.6;
    }
    if (!canvas) {
      canvas = document.createElement("canvas");
      context = canvas.getContext("2d");
    }
    if (!context) {
      return text.length * fontSize * 0.6;
    }

    context.font = `${fontWeight} ${fontSize}px ${BASE_FONT_STACK}`;
    const metrics = context.measureText(text);
    return metrics.width;
  };
})();

const useTheme = () => {
  const [isDark, setIsDark] = useState<boolean>(() => {
    if (typeof document === "undefined") return false;
    return document.documentElement.getAttribute("data-theme") === "dark";
  });

  useEffect(() => {
    if (typeof document === "undefined") return;
    const doc = document.documentElement;
    const update = () => setIsDark(doc.getAttribute("data-theme") === "dark");
    update();
    const observer = new MutationObserver(update);
    observer.observe(doc, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);

  return isDark;
};

const renderNode = (isDark: boolean) =>
  ({ nodeDatum, toggleNode }: any) => {
    const name = nodeDatum.name as string;
    const clue = nodeDatum.attributes?.clue as string | undefined;
    const paddingX = 26;
    const paddingY = clue ? 30 : 20;
    const fontSize = 14;
    const clueFontSize = 12;
    const lineHeight = 20;
    const nameWeight = 500;
    const clueWeight = 400;

    const nameWidth = measureLabelWidth(name, fontSize, nameWeight);
    const clueWidth = clue ? measureLabelWidth(clue, clueFontSize, clueWeight) : 0;
    const textWidth = Math.max(nameWidth, clueWidth);
    const width = Math.max(180, Math.min(320, textWidth + paddingX * 2));
    const height = clue ? paddingY * 2 + lineHeight * 2 : paddingY * 2 + lineHeight;

    const bg = isDark ? "rgba(15,23,42,0.92)" : "rgba(255,255,255,0.96)";
    const border = isDark ? "rgba(148,163,184,0.42)" : "rgba(15,23,42,0.14)";
    const nameColor = isDark ? "rgba(248,250,252,0.98)" : "rgba(22,30,46,0.9)";
    const clueColor = isDark ? "rgba(198,213,231,0.9)" : "rgba(71,85,105,0.86)";
    const shadow = isDark
      ? "drop-shadow(0 18px 32px rgba(15,23,42,0.55))"
      : "drop-shadow(0 20px 38px rgba(15,23,42,0.18))";

    return (
      <g onClick={toggleNode} style={{ cursor: "pointer", filter: shadow }}>
        <rect
          width={width}
          height={height}
          x={-width / 2}
          y={-height / 2}
          rx={26}
          fill={bg}
          stroke={border}
          strokeWidth={0.8}
        />
        <text
          fill={nameColor}
          textAnchor="middle"
          alignmentBaseline="middle"
          fontSize={fontSize}
          fontFamily={BASE_FONT_STACK}
          fontWeight={nameWeight}
          letterSpacing="0.01em"
          dy={clue ? -10 : 2}
          style={{ textRendering: "geometricPrecision" }}
        >
          {name}
        </text>
        {clue && (
          <text
            fill={clueColor}
            textAnchor="middle"
            alignmentBaseline="middle"
            fontSize={clueFontSize}
            fontFamily={BASE_FONT_STACK}
            fontWeight={clueWeight}
            letterSpacing="0.02em"
            dy={18}
            style={{ textRendering: "geometricPrecision" }}
          >
            {clue}
          </text>
        )}
      </g>
    );
  };

const prepareTree = (node: MindmapNode, isRoot = true): MindmapNode => {
  const children = node.children?.map((child) => prepareTree(child, false));
  return {
    ...node,
    ...(children ? { children } : {}),
    ...(isRoot ? {} : { collapsed: true }),
  };
};

export default function InteractiveMindmap() {
  const data = useMemo(() => prepareTree(TREE_DATA), []);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [translate, setTranslate] = useState({ x: 280, y: 290 });
  const isDark = useTheme();
  const linkStroke = isDark ? "rgba(148,163,184,0.55)" : "rgba(15,23,42,0.25)";

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.ResizeObserver === "undefined") {
      return;
    }

    const ResizeObserverInstance = window.ResizeObserver;
    const element = containerRef.current;
    if (!element) return;

    const setFromRect = () => {
      const { width, height } = element.getBoundingClientRect();
      setTranslate({ x: width * 0.32, y: Math.max(height * 0.5, 260) });
    };

    setFromRect();

    const observer = new ResizeObserverInstance(() => setFromRect());
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={containerRef} style={containerStyles}>
      <style>
        {`
          .mindmap-link {
            stroke: ${linkStroke};
            stroke-width: 1.25px;
            fill: none;
          }
        `}
      </style>
      <Tree
        data={data}
        orientation="horizontal"
        collapsible
        zoomable
        translate={translate}
        separation={{ siblings: 1.35, nonSiblings: 2.05 }}
        nodeSize={nodeSize}
        scaleExtent={{ min: 0.55, max: 2 }}
        transitionDuration={350}
        initialDepth={0}
        renderCustomNodeElement={renderNode(isDark)}
        pathClassFunc={() => "mindmap-link"}
      />
    </div>
  );
}
