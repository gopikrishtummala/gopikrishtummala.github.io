import type { CollectionEntry } from "astro:content";
import getSortedPosts from "./getSortedPosts";

export type Track = "Fundamentals" | "GenAI Systems" | "MLOps & Production" | "Robotics" | "Agentic AI";

/**
 * Extract part number from slug (e.g., "agentic-ai-design-patterns-part-1" -> 1)
 * or module number (e.g., "autonomous-stack-module-1" -> 1 or "satellite-photogrammetry-module-1-core-principles" -> 1)
 */
function extractPartNumber(slug: string): number | null {
  // Match -part-N or -part-N- pattern (at end or followed by more text)
  const partMatch = slug.match(/-part-(\d+)(?:$|-)/);
  if (partMatch) {
    return parseInt(partMatch[1], 10);
  }
  
  // Match -module-N or -module-N- pattern (at end or followed by more text)
  const moduleMatch = slug.match(/-module-(\d+)(?:$|-)/);
  if (moduleMatch) {
    return parseInt(moduleMatch[1], 10);
  }
  
  return null;
}

/**
 * Extract series base name (e.g., "agentic-ai-design-patterns-part-1" -> "agentic-ai-design-patterns")
 * or "satellite-photogrammetry-module-1-core-principles" -> "satellite-photogrammetry")
 */
function getSeriesBase(slug: string): string {
  // Remove -part-N or -part-N- and everything after
  let base = slug.replace(/-part-\d+(-.*)?$/, '');
  // Remove -module-N or -module-N- and everything after
  base = base.replace(/-module-\d+(-.*)?$/, '');
  return base;
}

/**
 * Get identifier from post (slug or filename from id)
 */
function getPostIdentifier(post: CollectionEntry<"blog">): string {
  if (post.data.slug) {
    return post.data.slug;
  }
  // Extract filename from post.id (handle paths like "subdir/file" or just "file")
  const idParts = post.id.split('/');
  const filename = idParts[idParts.length - 1];
  // Remove .md or .mdx extension if present
  return filename.replace(/\.(md|mdx)$/, '');
}

/**
 * Sort posts: series posts first (sorted by part number), then non-series posts (sorted by date)
 */
function sortPostsWithSeries(posts: CollectionEntry<"blog">[]): CollectionEntry<"blog">[] {
  const seriesPosts: CollectionEntry<"blog">[] = [];
  const nonSeriesPosts: CollectionEntry<"blog">[] = [];
  const seriesMap = new Map<string, CollectionEntry<"blog">[]>();
  
  // Separate series and non-series posts
  for (const post of posts) {
    const identifier = getPostIdentifier(post);
    const partNumber = extractPartNumber(identifier);
    if (partNumber !== null) {
      const base = getSeriesBase(identifier);
      if (!seriesMap.has(base)) {
        seriesMap.set(base, []);
      }
      seriesMap.get(base)!.push(post);
    } else {
      nonSeriesPosts.push(post);
    }
  }
  
  // Sort each series by part number
  for (const [, seriesPostsList] of seriesMap.entries()) {
    seriesPostsList.sort((a, b) => {
      const identifierA = getPostIdentifier(a);
      const identifierB = getPostIdentifier(b);
      const partA = extractPartNumber(identifierA) ?? 0;
      const partB = extractPartNumber(identifierB) ?? 0;
      return partA - partB;
    });
    seriesPosts.push(...seriesPostsList);
  }
  
  // Sort series posts by part number across all series (in case of multiple series)
  seriesPosts.sort((a, b) => {
    const identifierA = getPostIdentifier(a);
    const identifierB = getPostIdentifier(b);
    const partA = extractPartNumber(identifierA) ?? 0;
    const partB = extractPartNumber(identifierB) ?? 0;
    // If same part number, maintain original order (by series base)
    if (partA === partB) {
      const baseA = getSeriesBase(identifierA);
      const baseB = getSeriesBase(identifierB);
      return baseA.localeCompare(baseB);
    }
    return partA - partB;
  });
  
  // Return series posts first, then non-series posts (already sorted by date from getSortedPosts)
  return [...seriesPosts, ...nonSeriesPosts];
}

export const getPostsByTrack = (
  posts: CollectionEntry<"blog">[],
  track: Track
) => {
  const filtered = posts.filter((post) => post.data.track === track);
  const sorted = getSortedPosts(filtered);
  return sortPostsWithSeries(sorted);
};

export default getPostsByTrack;

