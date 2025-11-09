import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, RefObject } from "react";
import Tree, {
  type Orientation,
  type RawNodeDatum,
  type TreeNodeDatum,
} from "react-d3-tree";

const DISPLAY_FONT_STACK =
  'var(--display-font, "Space Grotesk", "Futura PT", "Manrope", "Inter", "Helvetica Neue", Arial, sans-serif)';
const CANVAS_FONT_STACK =
  '"Space Grotesk", "Futura PT", "Manrope", "Inter", "Helvetica Neue", Arial, sans-serif';
const DEFAULT_HEIGHT = 640;
const DEFAULT_NODE_SIZE = { x: 260, y: 168 };
const DEFAULT_SEPARATION = { siblings: 1.35, nonSiblings: 2.05 };
const DEFAULT_SCALE = { min: 0.55, max: 2 };

export type MindmapThemeMode = "auto" | "light" | "dark";

export interface MindmapNode {
  id?: string;
  label: string;
  description?: string;
  collapsed?: boolean;
  children?: MindmapNode[];
  meta?: Record<string, unknown>;
}

export type HydratedMindmapNode = Omit<MindmapNode, "id"> & { id: string };

interface AugmentedMetadata {
  payload: HydratedMindmapNode;
  ui: { hasChildren: boolean };
}

type AugmentedRawNodeDatum = RawNodeDatum &
  AugmentedMetadata & {
    children?: AugmentedRawNodeDatum[];
    collapsed?: boolean;
  };

type AugmentedTreeNodeDatum = TreeNodeDatum &
  AugmentedMetadata & {
    children?: AugmentedTreeNodeDatum[];
  };

interface ColorSet {
  bg: string;
  border: string;
  name: string;
  description: string;
  shadow: string;
}

interface Palette {
  link: string;
  leaf: ColorSet;
  branch: ColorSet;
}

export interface InteractiveMindmapProps {
  data?: MindmapNode;
  height?: number | string;
  className?: string;
  style?: CSSProperties;
  collapseByDefault?: boolean;
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
}

const BASE_CONTAINER_STYLES: CSSProperties = {
  width: "100%",
  borderRadius: "1.9rem",
  border: "1px solid rgba(148,163,184,0.24)",
  background:
    "linear-gradient(135deg, rgba(15,23,42,0.06) 0%, rgba(37,99,235,0.08) 55%, rgba(15,23,42,0.04) 100%)",
  boxShadow: "0 45px 85px -45px rgba(15,23,42,0.42)",
  position: "relative",
  overflow: "hidden",
};

const DEFAULT_COLORS = {
  leaf: {
    lightBg: "rgba(37,99,235,0.08)",
    lightBorder: "rgba(37,99,235,0.24)",
    darkBg: "rgba(56,189,248,0.18)",
    darkBorder: "rgba(56,189,248,0.32)",
  },
  branch: {
    lightBg: "rgba(255,255,255,0.95)",
    lightBorder: "rgba(15,23,42,0.12)",
    darkBg: "rgba(15,23,42,0.88)",
    darkBorder: "rgba(148,163,184,0.38)",
  },
  text: {
    lightName: "rgba(22,30,46,0.76)",
    lightDescription: "rgba(71,85,105,0.7)",
    darkName: "rgba(248,250,252,0.92)",
    darkDescription: "rgba(198,213,231,0.78)",
  },
  shadow: {
    light: "drop-shadow(0 20px 38px rgba(15,23,42,0.18))",
    dark: "drop-shadow(0 18px 32px rgba(15,23,42,0.55))",
  },
  link: {
    light: "rgba(15,23,42,0.25)",
    dark: "rgba(148,163,184,0.55)",
  },
};

const slugify = (value: string) =>
  value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") ||
  "node";

interface BuildOptions {
  collapseByDefault: boolean;
}

const ensureId = (node: MindmapNode, path: string): HydratedMindmapNode => {
  const generatedId = node.id ?? `${path}-${slugify(node.label)}`;
  return {
    ...node,
    id: generatedId,
  };
};

const buildAugmentedTree = (
  node: MindmapNode,
  options: BuildOptions,
  path = "root",
): AugmentedRawNodeDatum => {
  const hydrated = ensureId(node, path);
  const children = hydrated.children?.map((child, index) =>
    buildAugmentedTree(child, options, `${path}.${index}`),
  );
  const hasChildren = Boolean(children && children.length > 0);

  return {
    name: hydrated.label,
    attributes: hydrated.description ? { description: hydrated.description } : undefined,
    payload: { ...hydrated },
    ui: { hasChildren },
    children,
    collapsed:
      hydrated.collapsed ?? (options.collapseByDefault && path !== "root"),
  };
};

const measureLabelWidth = (() => {
  let canvas: HTMLCanvasElement | null = null;
  let context: CanvasRenderingContext2D | null = null;
  return (text: string, fontSize: number, fontWeight = 300) => {
    if (!text) return 0;
    if (typeof document === "undefined") {
      return text.length * fontSize * 0.58;
    }
    if (!canvas) {
      canvas = document.createElement("canvas");
      context = canvas.getContext("2d");
    }
    if (!context) {
      return text.length * fontSize * 0.58;
    }

    context.font = `${fontWeight} ${fontSize}px ${CANVAS_FONT_STACK}`;
    const metrics = context.measureText(text);
    return metrics.width;
  };
})();

const useTheme = (mode: MindmapThemeMode = "auto") => {
  const [systemDark, setSystemDark] = useState<boolean>(() => {
    if (mode === "dark") return true;
    if (mode === "light") return false;
    if (typeof document === "undefined") return false;
    return document.documentElement.getAttribute("data-theme") === "dark";
  });

  useEffect(() => {
    if (mode !== "auto" || typeof document === "undefined") return;
    const doc = document.documentElement;
    const update = () => setSystemDark(doc.getAttribute("data-theme") === "dark");
    update();
    const observer = new MutationObserver(update);
    observer.observe(doc, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, [mode]);

  if (mode === "dark") return true;
  if (mode === "light") return false;
  return systemDark;
};

const calculateTranslate = (width: number, height: number, orientation: Orientation) => {
  if (orientation === "vertical") {
    return {
      x: width / 2,
      y: Math.max(height * 0.18, 120),
    };
  }

  return {
    x: width * 0.32,
    y: Math.max(height * 0.5, 260),
  };
};

const useAutoTranslate = (
  ref: RefObject<HTMLDivElement | null>,
  orientation: Orientation,
) => {
  const [translate, setTranslate] = useState(() =>
    calculateTranslate(800, DEFAULT_HEIGHT, orientation),
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

const getPalette = (
  isDark: boolean,
  leafTint?: { light: string; dark: string },
  branchTint?: { light: string; dark: string },
): Palette => {
  const leafBg = isDark
    ? leafTint?.dark ?? DEFAULT_COLORS.leaf.darkBg
    : leafTint?.light ?? DEFAULT_COLORS.leaf.lightBg;
  const branchBg = isDark
    ? branchTint?.dark ?? DEFAULT_COLORS.branch.darkBg
    : branchTint?.light ?? DEFAULT_COLORS.branch.lightBg;

  return {
    link: isDark ? DEFAULT_COLORS.link.dark : DEFAULT_COLORS.link.light,
    leaf: {
      bg: leafBg,
      border: isDark ? DEFAULT_COLORS.leaf.darkBorder : DEFAULT_COLORS.leaf.lightBorder,
      name: isDark ? DEFAULT_COLORS.text.darkName : DEFAULT_COLORS.text.lightName,
      description: isDark
        ? DEFAULT_COLORS.text.darkDescription
        : DEFAULT_COLORS.text.lightDescription,
      shadow: isDark ? DEFAULT_COLORS.shadow.dark : DEFAULT_COLORS.shadow.light,
    },
    branch: {
      bg: branchBg,
      border: isDark
        ? DEFAULT_COLORS.branch.darkBorder
        : DEFAULT_COLORS.branch.lightBorder,
      name: isDark ? DEFAULT_COLORS.text.darkName : DEFAULT_COLORS.text.lightName,
      description: isDark
        ? DEFAULT_COLORS.text.darkDescription
        : DEFAULT_COLORS.text.lightDescription,
      shadow: isDark ? DEFAULT_COLORS.shadow.dark : DEFAULT_COLORS.shadow.light,
    },
  };
};

interface NodeRendererOptions {
  palette: Palette;
  onNodeClick?: (node: HydratedMindmapNode) => void;
}

const createNodeRenderer =
  ({ palette, onNodeClick }: NodeRendererOptions) =>
  ({ nodeDatum, toggleNode }: { nodeDatum: TreeNodeDatum; toggleNode: () => void }) => {
    const node = nodeDatum as AugmentedTreeNodeDatum;
    const {
      payload: { label, description },
      ui: { hasChildren },
    } = node;

    const isLeaf = !hasChildren;
    const paddingX = 26;
    const paddingY = description ? 30 : 20;
    const fontSize = 14;
    const descriptionFontSize = 12;
    const lineHeight = 20;
    const nameWeight = 320;
    const descriptionWeight = 300;

    const nameWidth = measureLabelWidth(label, fontSize, nameWeight);
    const descriptionWidth = description
      ? measureLabelWidth(description, descriptionFontSize, descriptionWeight)
      : 0;
    const textWidth = Math.max(nameWidth, descriptionWidth);
    const width = Math.max(180, Math.min(320, textWidth + paddingX * 2));
    const height = description
      ? paddingY * 2 + lineHeight * 2
      : paddingY * 2 + lineHeight;

    const colors = isLeaf ? palette.leaf : palette.branch;

    const handleClick = () => {
      toggleNode();
      onNodeClick?.(node.payload);
    };

    return (
      <g onClick={handleClick} style={{ cursor: "pointer", filter: colors.shadow }}>
        <rect
          width={width}
          height={height}
          x={-width / 2}
          y={-height / 2}
          rx={26}
          fill={colors.bg}
          stroke={colors.border}
          strokeWidth={0.9}
        />
        <text
          fill={colors.name}
          textAnchor="middle"
          alignmentBaseline="middle"
          fontSize={fontSize}
          fontFamily={DISPLAY_FONT_STACK}
          fontWeight={nameWeight}
          letterSpacing="0.03em"
          dy={description ? -10 : 2}
          style={{ textRendering: "geometricPrecision" }}
        >
          {label}
        </text>
        {description && (
          <text
            fill={colors.description}
            textAnchor="middle"
            alignmentBaseline="middle"
            fontSize={descriptionFontSize}
            fontFamily={DISPLAY_FONT_STACK}
            fontWeight={descriptionWeight}
            letterSpacing="0.035em"
            dy={18}
            style={{ textRendering: "geometricPrecision" }}
          >
            {description}
          </text>
        )}
      </g>
    );
  };

const resolveHeight = (height?: number | string) => {
  if (typeof height === "number") return `${height}px`;
  if (typeof height === "string") return height;
  return `${DEFAULT_HEIGHT}px`;
};

export const DEFAULT_MINDMAP_DATA: MindmapNode = {
  label: "LeetCode Problem Classification",
  children: [
    {
      label: "Arrays & Strings",
      children: [
        {
          label: "Contiguous subarray / substring",
          description: "max length, sum, uniqueness",
          children: [
            { label: "Sliding Window" },
            { label: "Two Pointers" },
            { label: "Prefix Sums" },
          ],
        },
        {
          label: "Frequency / membership",
          description: "counts, duplicates, mappings",
          children: [{ label: "Hash Map / Set" }],
        },
        {
          label: "Pairs / triples / sums in sorted data",
          children: [{ label: "Two Pointers" }],
        },
        {
          label: "Next greater / smaller / histogram",
          children: [
            { label: "Monotonic Stack" },
            { label: "Monotonic Queue" },
          ],
        },
      ],
    },
    {
      label: "Search & Sort",
      children: [
        {
          label: "Searching value / threshold",
          children: [{ label: "Binary Search" }, { label: "Modified Binary Search" }],
        },
        {
          label: "Local optima → global answer",
          children: [{ label: "Greedy" }],
        },
        {
          label: "Bounded range anomalies",
          description: "missing / duplicate",
          children: [{ label: "Cyclic Sort" }],
        },
      ],
    },
    {
      label: "Dynamic Programming",
      children: [
        {
          label: "Counting combinations / overlapping states",
          children: [{ label: "Classic DP (tabulation / memoization)" }],
        },
        {
          label: "Optimization over sequences / paths",
          children: [{ label: "Advanced DP (Knapsack / LIS / interval)" }],
        },
      ],
    },
    {
      label: "Graphs",
      children: [
        {
          label: "Connectivity / flood-fill / components",
          children: [{ label: "BFS" }, { label: "DFS" }, { label: "Union-Find" }],
        },
        {
          label: "Dependencies / prerequisites (DAG)",
          children: [{ label: "Topological Sort" }],
        },
        {
          label: "Cycles / shortest path / maze",
          children: [
            { label: "Graph Traversal (BFS / DFS)" },
            { label: "Fast & Slow Pointers" },
          ],
        },
      ],
    },
    {
      label: "Trees",
      children: [
        {
          label: "Hierarchy / recursion / validation",
          children: [
            { label: "Tree DFS (pre / in / post)" },
            { label: "Stack-based Traversal" },
          ],
        },
        {
          label: "Level order / minimum depth",
          children: [{ label: "Tree BFS" }],
        },
      ],
    },
    {
      label: "Heaps & Queues",
      children: [
        {
          label: "Min / max / k-th extreme",
          children: [
            { label: "Heap / Priority Queue" },
            { label: "Two Heaps (median)" },
            { label: "Monotonic Deque" },
          ],
        },
        {
          label: "Top-K frequent without full sort",
          children: [{ label: "Heap-based Top-K" }],
        },
      ],
    },
    {
      label: "Backtracking & Combinatorics",
      children: [
        {
          label: "Permutations / combinations with pruning",
          children: [{ label: "Backtracking" }],
        },
        {
          label: "Subsets / powerset / combination sums",
          children: [{ label: "Recursion" }, { label: "Bit Manipulation" }],
        },
      ],
    },
    {
      label: "Intervals",
      children: [
        {
          label: "Overlaps / merges / scheduling",
          children: [
            { label: "Merge Intervals Pattern" },
            { label: "Sweep Line / Overlapping Intervals" },
          ],
        },
      ],
    },
    {
      label: "Linked Lists",
      children: [
        {
          label: "Cycle detection / midpoints",
          children: [{ label: "Fast & Slow Pointers" }],
        },
        {
          label: "Reordering / reversing segments",
          children: [{ label: "In-place Reversal" }],
        },
      ],
    },
    {
      label: "Bit Manipulation",
      children: [
        {
          label: "Binary representation / parity / uniques",
          children: [{ label: "Bitwise Operations" }, { label: "XOR Tricks" }],
        },
      ],
    },
    {
      label: "Strings (Advanced)",
      children: [
        {
          label: "Prefix matching / dictionary search",
          children: [{ label: "Trie (Prefix Tree)" }],
        },
        {
          label: "Pattern matching / automata",
          children: [{ label: "KMP" }, { label: "Rolling Hash" }],
        },
      ],
    },
  ],
};

export default function InteractiveMindmap({
  data = DEFAULT_MINDMAP_DATA,
  height,
  className,
  style,
  collapseByDefault = true,
  orientation = "horizontal",
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
}: InteractiveMindmapProps) {
  const isDark = useTheme(themeMode);
  const palette = useMemo(
    () => getPalette(isDark, leafTint, branchTint),
    [isDark, leafTint, branchTint],
  );

  const treeData = useMemo(
    () => buildAugmentedTree(data, { collapseByDefault }),
    [data, collapseByDefault],
  );

  const containerRef = useRef<HTMLDivElement | null>(null);
  const translate = useAutoTranslate(containerRef, orientation);

  const nodeRenderer = useMemo(
    () => createNodeRenderer({ palette, onNodeClick }),
    [palette, onNodeClick],
  );

  const separationConfig = { ...DEFAULT_SEPARATION, ...separation };
  const scaleConfig = { ...DEFAULT_SCALE, ...scaleExtent };
  const resolvedHeight = resolveHeight(height);
  const containerStyles = useMemo(
    () => ({
      ...BASE_CONTAINER_STYLES,
      height: resolvedHeight,
      ...style,
    }),
    [resolvedHeight, style],
  );
  const effectiveLinkColor = linkColor ?? palette.link;

  return (
    <div ref={containerRef} className={className} style={containerStyles}>
      <style>
        {`
          .mindmap-link {
            stroke: ${effectiveLinkColor};
            stroke-width: 1.25px;
            fill: none;
          }
        `}
      </style>
      <Tree
        data={treeData}
        orientation={orientation}
        collapsible={collapsible}
        zoomable={zoomable}
        translate={translate}
        separation={separationConfig}
        nodeSize={DEFAULT_NODE_SIZE}
        scaleExtent={scaleConfig}
        transitionDuration={320}
        initialDepth={initialDepth}
        renderCustomNodeElement={nodeRenderer}
        pathClassFunc={() => "mindmap-link"}
      />
    </div>
  );
}

