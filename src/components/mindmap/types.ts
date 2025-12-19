import type { RawNodeDatum, TreeNodeDatum } from "react-d3-tree";

export interface MindmapNode {
  id?: string;
  label: string;
  description?: string;
  collapsed?: boolean;
  children?: MindmapNode[];
  meta?: Record<string, unknown>;
}

export type HydratedMindmapNode = Omit<MindmapNode, "id"> & { id: string };

export interface AugmentedMetadata {
  payload: HydratedMindmapNode;
  ui: {
    hasChildren: boolean;
    depth: number;
  };
}

export type AugmentedRawNodeDatum = RawNodeDatum &
  AugmentedMetadata & {
    children?: AugmentedRawNodeDatum[];
    collapsed?: boolean;
  };

export type AugmentedTreeNodeDatum = TreeNodeDatum &
  AugmentedMetadata & {
    children?: AugmentedTreeNodeDatum[];
  };

export interface ColorSet {
  bg: string;
  border: string;
  name: string;
  description: string;
  shadow: string;
}

export interface Palette {
  link: string;
  leaf: ColorSet;
  branches: ColorSet[];
}

export type MindmapThemeMode = "auto" | "light" | "dark";

export interface ExportOptions {
  fileName?: string;
  backgroundColor?: string;
  scale?: number;
}

export type ExportFormat = "svg" | "png" | "pdf";

export interface InteractiveMindmapHandle {
  exportAs: (format: ExportFormat, options?: ExportOptions) => Promise<void>;
  resetView: () => void;
}

