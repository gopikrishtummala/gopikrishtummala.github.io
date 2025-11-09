import { useMemo, useState } from "react";
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

const containerStyles: React.CSSProperties = {
  width: "100%",
  height: "520px",
  borderRadius: "1.5rem",
  border: "1px solid rgba(148,163,184,0.25)",
  background: "var(--code-bg)",
  boxShadow: "0 30px 60px -32px rgba(15,23,42,0.35)",
};

const nodeSize = { x: 180, y: 110 };

export default function InteractiveMindmap() {
  const data = useMemo(() => TREE_DATA, []);
  const [translate] = useState({ x: 220, y: 260 });

  return (
    <div style={containerStyles}>
      <Tree
        data={data}
        orientation="horizontal"
        collapsible
        zoomable
        translate={translate}
        separation={{ siblings: 1, nonSiblings: 1.5 }}
        nodeSize={nodeSize}
        shouldCollapseNeighborNodes={true}
        renderCustomNodeElement={({ nodeDatum, toggleNode }) => (
          <g onClick={toggleNode} style={{ cursor: "pointer" }}>
            <rect
              width={160}
              height={52}
              x={-80}
              y={-26}
              rx={18}
              fill="rgb(30, 41, 59)"
              stroke="rgba(148,163,184,0.35)"
              strokeWidth={1}
            />
            <text
              fill="rgb(226,232,240)"
              textAnchor="middle"
              alignmentBaseline="middle"
              fontSize={11}
              fontFamily="var(--code-font)"
            >
              {nodeDatum.name}
            </text>
            {nodeDatum.attributes?.clue && (
              <text
                fill="rgba(148,163,184,0.85)"
                textAnchor="middle"
                alignmentBaseline="middle"
                fontSize={9}
                fontFamily="var(--code-font)"
                y={18}
              >
                {nodeDatum.attributes.clue}
              </text>
            )}
          </g>
        )}
      />
    </div>
  );
}
