import type { MindmapNode, HydratedMindmapNode, AugmentedRawNodeDatum } from "./types";

interface BuildOptions {
  collapseByDefault: boolean;
}

const slugify = (value: string) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "") || "node";

const ensureId = (
  node: MindmapNode,
  path: string,
  autoId: number,
): HydratedMindmapNode => ({
  ...node,
  id: node.id ?? `${path}-${slugify(node.label || `node-${autoId}`)}`,
});

export const buildAugmentedTree = (
  node: MindmapNode,
  options: BuildOptions,
  path = "root",
  depth = 0,
  autoId = 0,
): AugmentedRawNodeDatum => {
  const hydrated = ensureId(node, path, autoId);
  const children = hydrated.children?.map((child, index) =>
    buildAugmentedTree(child, options, `${path}.${index}`, depth + 1, autoId + index + 1),
  );
  const hasChildren = Boolean(children && children.length > 0);

  return {
    name: hydrated.label,
    attributes: hydrated.description
      ? { description: hydrated.description }
      : undefined,
    payload: hydrated,
    ui: { hasChildren, depth },
    children: children?.map((child) => ({
      ...child,
      collapsed: options.collapseByDefault ? true : child.collapsed,
    })),
    collapsed:
      hydrated.collapsed ?? (options.collapseByDefault && path !== "root"),
  };
};

