import type { Orientation, TreeLinkDatum } from "react-d3-tree";

const MIN_HORIZONTAL_OFFSET = 140;
const MIN_VERTICAL_OFFSET = 90;
const SEGMENT_RATIO = 0.52;
const MIN_SEGMENT = 32;

const horizontalCurve = (source: { x: number; y: number }, target: { x: number; y: number }) => {
  const deltaX = target.x - source.x;
  const abs = Math.abs(deltaX);
  const base = Math.max(abs * SEGMENT_RATIO, MIN_HORIZONTAL_OFFSET);
  const offset = Math.max(MIN_SEGMENT, Math.min(base, abs / 2 - 12));
  const direction = Math.sign(deltaX || 1);
  const control = direction * offset;

  return `M${source.x},${source.y}
      C${source.x + control},${source.y}
       ${target.x - control},${target.y}
       ${target.x},${target.y}`;
};

const verticalCurve = (source: { x: number; y: number }, target: { x: number; y: number }) => {
  const deltaY = target.y - source.y;
  const abs = Math.abs(deltaY);
  const base = Math.max(abs * SEGMENT_RATIO, MIN_VERTICAL_OFFSET);
  const offset = Math.max(MIN_SEGMENT, Math.min(base, abs / 2 - 10));
  const direction = Math.sign(deltaY || 1);
  const control = direction * offset;

  return `M${source.x},${source.y}
      C${source.x},${source.y + control}
       ${target.x},${target.y - control}
       ${target.x},${target.y}`;
};

export const smoothStepPath = (
  { source, target }: TreeLinkDatum,
  orientation: Orientation,
) => {
  if (orientation === "vertical") {
    return horizontalCurve(source, target);
  }
  return verticalCurve(source, target);
};

export const organicPath = ({ source, target }: TreeLinkDatum) => {
  const midY = (source.y + target.y) / 2;
  return `M${source.x},${source.y}
          C${source.x},${midY}
           ${target.x},${midY}
           ${target.x},${target.y}`;
};

