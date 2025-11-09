import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import Tree from "react-d3-tree";

const TREE_DATA = {
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
  height: "580px",
  borderRadius: "1.75rem",
  border: "1px solid rgba(148,163,184,0.25)",
  background: "var(--code-bg)",
  boxShadow: "0 35px 70px -35px rgba(15,23,42,0.35)",
  position: "relative",
};

const nodeSize = { x: 210, y: 120 };

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
    const paddingX = 16;
    const paddingY = clue ? 20 : 12;
    const fontSize = 12;
    const clueFontSize = 10;
    const lineHeight = 14;

    const textWidth = Math.max(
      name.length * fontSize * 0.55,
      clue ? clue.length * clueFontSize * 0.52 : 0,
    );
    const width = Math.max(150, textWidth + paddingX * 2);
    const height = clue ? paddingY * 2 + lineHeight * 2 : paddingY * 2 + lineHeight;

    const bg = isDark ? "rgba(15,23,42,0.92)" : "rgba(255,255,255,0.98)";
    const border = isDark ? "rgba(148,163,184,0.5)" : "rgba(15,23,42,0.12)";
    const nameColor = isDark ? "rgba(248,250,252,0.96)" : "rgba(15,23,42,0.92)";
    const clueColor = isDark ? "rgba(203,213,225,0.88)" : "rgba(71,85,105,0.82)";

    return (
      <g onClick={toggleNode} style={{ cursor: "pointer" }}>
        <rect
          width={width}
          height={height}
          x={-width / 2}
          y={-height / 2}
          rx={22}
          fill={bg}
          stroke={border}
          strokeWidth={1}
        />
        <text
          fill={nameColor}
          textAnchor="middle"
          alignmentBaseline="middle"
          fontSize={fontSize}
          fontFamily="var(--code-font)"
          dy={clue ? -6 : 0}
        >
          {name}
        </text>
        {clue && (
          <text
            fill={clueColor}
            textAnchor="middle"
            alignmentBaseline="middle"
            fontSize={clueFontSize}
            fontFamily="var(--code-font)"
            dy={16}
          >
            {clue}
          </text>
        )}
      </g>
    );
  };

export default function InteractiveMindmap() {
  const data = useMemo(() => TREE_DATA, []);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [translate, setTranslate] = useState({ x: 280, y: 290 });
  const isDark = useTheme();

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;

    const setFromRect = () => {
      const { width, height } = element.getBoundingClientRect();
      setTranslate({ x: width * 0.25, y: height / 2 });
    };

    setFromRect();

    const observer = new ResizeObserver(() => setFromRect());
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={containerRef} style={containerStyles}>
      <Tree
        data={data}
        orientation="horizontal"
        collapsible
        zoomable
        translate={translate}
        separation={{ siblings: 1, nonSiblings: 1.7 }}
        nodeSize={nodeSize}
        scaleExtent={{ min: 0.5, max: 1.8 }}
        transitionDuration={350}
        renderCustomNodeElement={renderNode(isDark)}
        linkSvgProps={{
          stroke: isDark ? "rgba(148,163,184,0.55)" : "rgba(15,23,42,0.25)",
          strokeWidth: 1.25,
        }}
      />
    </div>
  );
}
