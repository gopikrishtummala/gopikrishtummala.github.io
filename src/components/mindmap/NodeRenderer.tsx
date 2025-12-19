import { memo, useCallback } from "react";
import type { KeyboardEvent } from "react";
import type { CustomNodeElementProps } from "react-d3-tree";
import type { AugmentedTreeNodeDatum, HydratedMindmapNode, Palette } from "./types";
import { measureLabelWidth } from "./measure";

const DISPLAY_FONT_STACK =
  'var(--display-font, "Space Grotesk", "Futura PT", "Manrope", "Inter", "Helvetica Neue", Arial, sans-serif)';
const CANVAS_FONT_STACK =
  '"Space Grotesk", "Futura PT", "Manrope", "Inter", "Helvetica Neue", Arial, sans-serif';

interface NodeRendererProps {
  palette: Palette;
  onNodeClick?: (node: HydratedMindmapNode) => void;
  letterSpacing?: number;
  onCenter?: (info: { x: number; y: number; depth: number }) => void;
}

const NAME_FONT_SIZE = 14;
const DESCRIPTION_FONT_SIZE = 12.4;
const NAME_WEIGHT_LEAF = 380;
const NAME_WEIGHT_PARENT = 320;
const DESCRIPTION_WEIGHT = 300;
const PADDING_X = 30;
const PADDING_Y = 30;
const NAME_LINE_HEIGHT = 24;
const DESCRIPTION_LINE_HEIGHT = 20;
const DESCRIPTION_GAP = 8;
const CORNER_RADIUS = 28;
const MIN_NODE_WIDTH = 190;
const MAX_CONTENT_WIDTH = 200;

const wrapText = (
  text: string,
  fontSize: number,
  fontWeight: number,
  maxWidth: number,
) => {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  const widths: number[] = [];
  let current = "";

  words.forEach((word) => {
    const candidate = current ? `${current} ${word}` : word;
    const width = measureLabelWidth(candidate, fontSize, fontWeight, CANVAS_FONT_STACK);
    if (width <= maxWidth || !current) {
      current = candidate;
    } else {
      lines.push(current);
      widths.push(
        measureLabelWidth(current, fontSize, fontWeight, CANVAS_FONT_STACK),
      );
      current = word;
    }
  });

  if (current) {
    lines.push(current);
    widths.push(
      measureLabelWidth(current, fontSize, fontWeight, CANVAS_FONT_STACK),
    );
  }

  return { lines, widths };
};

const renderFactory = ({
  palette,
  onNodeClick,
  letterSpacing = 0.0075,
  onCenter,
}: NodeRendererProps) => {
  const Renderer = memo(
    ({ nodeDatum, toggleNode, hierarchyPointNode }: CustomNodeElementProps) => {
      const augmented = nodeDatum as AugmentedTreeNodeDatum;
      const {
        payload,
        ui: { hasChildren, depth },
      } = augmented;
      const { label, description } = payload;

      const nameFontWeight = hasChildren ? NAME_WEIGHT_PARENT : NAME_WEIGHT_LEAF;
      const nameWrap = wrapText(label, NAME_FONT_SIZE, nameFontWeight, MAX_CONTENT_WIDTH);
      const descriptionWrap = description
        ? wrapText(description, DESCRIPTION_FONT_SIZE, DESCRIPTION_WEIGHT, MAX_CONTENT_WIDTH)
        : { lines: [] as string[], widths: [] as number[] };

      const contentWidth = Math.max(
        nameWrap.widths.length ? Math.max(...nameWrap.widths) : 0,
        descriptionWrap.widths.length ? Math.max(...descriptionWrap.widths) : 0,
      );

      const width = Math.max(
        MIN_NODE_WIDTH,
        Math.min(MAX_CONTENT_WIDTH + PADDING_X * 2, contentWidth + PADDING_X * 2),
      );

      const nameHeight = Math.max(1, nameWrap.lines.length) * NAME_LINE_HEIGHT;
      const descriptionHeight = descriptionWrap.lines.length
        ? DESCRIPTION_GAP + descriptionWrap.lines.length * DESCRIPTION_LINE_HEIGHT
        : 0;
      const height = PADDING_Y * 2 + nameHeight + descriptionHeight;

      const colors = hasChildren
        ? palette.branches[Math.min(depth, palette.branches.length - 1)]
        : palette.leaf;

      const accentWidth = 5;
      const textX = -width / 2 + PADDING_X;
      const nameStartY = -height / 2 + PADDING_Y;
      const descriptionStartY =
        nameStartY +
        nameWrap.lines.length * NAME_LINE_HEIGHT +
        (descriptionWrap.lines.length ? DESCRIPTION_GAP : 0);

      const handleClick = useCallback(() => {
        toggleNode();
        if (hierarchyPointNode && onCenter) {
          const { x = 0, y = 0, depth = 0 } = hierarchyPointNode;
          onCenter({ x, y, depth });
        }
        onNodeClick?.(payload);
      }, [toggleNode, onNodeClick, payload, hierarchyPointNode, onCenter]);

      const handleKeyDown = (event: KeyboardEvent<SVGGElement>) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          handleClick();
        }
      };

      return (
        <g
          onClick={handleClick}
          onKeyDown={handleKeyDown}
          role="treeitem"
          aria-expanded={hasChildren ? !augmented.__rd3t.collapsed : undefined}
          tabIndex={0}
          className="mindmap-node"
          style={{ cursor: "pointer" }}
        >
          <rect
            width={accentWidth}
            height={height - PADDING_Y * 0.6}
            x={-width / 2 + (PADDING_Y * 0.6) / 2}
            y={-height / 2 + (PADDING_Y * 0.6) / 2}
            rx={accentWidth / 2}
            fill={hasChildren ? colors.border : palette.link}
            opacity={0.65}
          />
          <rect
            width={width}
            height={height}
            x={-width / 2}
            y={-height / 2}
            rx={CORNER_RADIUS}
            fill={colors.bg}
            stroke={colors.border}
            strokeWidth={0.9}
            filter={colors.shadow}
          />
          {nameWrap.lines.map((line, index) => (
            <text
              key={`title-${line}-${index}`}
              fill={colors.name}
              textAnchor="middle"
              alignmentBaseline="middle"
              fontSize={NAME_FONT_SIZE}
              fontFamily={DISPLAY_FONT_STACK}
              fontWeight={nameFontWeight}
              letterSpacing={letterSpacing}
              style={{
                textRendering: "optimizeLegibility",
                WebkitFontSmoothing: "antialiased",
                fontVariationSettings: `"wght" ${nameFontWeight}`,
              }}
              y={nameStartY + index * NAME_LINE_HEIGHT}
              data-role="title"
            >
              {line}
            </text>
          ))}
          {descriptionWrap.lines.map((line, index) => (
            <text
              key={`desc-${line}-${index}`}
              fill={colors.description}
              textAnchor="middle"
              alignmentBaseline="middle"
              fontSize={DESCRIPTION_FONT_SIZE}
              fontFamily={DISPLAY_FONT_STACK}
              fontWeight={DESCRIPTION_WEIGHT}
              letterSpacing={letterSpacing * 1.2}
              style={{
                textRendering: "optimizeLegibility",
                WebkitFontSmoothing: "antialiased",
                fontVariationSettings: `"wght" ${DESCRIPTION_WEIGHT}`,
              }}
              y={descriptionStartY + index * DESCRIPTION_LINE_HEIGHT}
              data-role="description"
            >
              {line}
            </text>
          ))}
        </g>
      );
    },
  );

  Renderer.displayName = "MindmapNodeRenderer";
  return Renderer;
};

export const createNodeRenderer = (props: NodeRendererProps) => {
  const Renderer = renderFactory(props);
  return (nodeProps: CustomNodeElementProps) => <Renderer {...nodeProps} />;
};

