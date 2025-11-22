import type { CollectionEntry } from "astro:content";
import getSortedPosts from "./getSortedPosts";

export type Track = "Fundamentals" | "GenAI Systems" | "MLOps & Production" | "Robotics" | "Agentic AI";

/**
 * Extract part number from slug (e.g., "agentic-ai-design-patterns-part-1" -> 1)
 * or module number (e.g., "autonomous-stack-module-1" -> 1)
 */
function extractPartNumber(slug: string): number | null {
  // Match -part-N or -module-N pattern
  const partMatch = slug.match(/-part-(\d+)$/);
  if (partMatch) {
    return parseInt(partMatch[1], 10);
  }
  
  const moduleMatch = slug.match(/-module-(\d+)$/);
  if (moduleMatch) {
    return parseInt(moduleMatch[1], 10);
  }
  
  return null;
}

/**
 * Extract series base name (e.g., "agentic-ai-design-patterns-part-1" -> "agentic-ai-design-patterns")
 */
function getSeriesBase(slug: string): string {
  return slug.replace(/-part-\d+$/, '').replace(/-module-\d+$/, '');
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
    const partNumber = extractPartNumber(post.id);
    if (partNumber !== null) {
      const base = getSeriesBase(post.id);
      if (!seriesMap.has(base)) {
        seriesMap.set(base, []);
      }
      seriesMap.get(base)!.push(post);
    } else {
      nonSeriesPosts.push(post);
    }
  }
  
  // Sort each series by part number
  for (const [base, seriesPostsList] of seriesMap.entries()) {
    seriesPostsList.sort((a, b) => {
      const partA = extractPartNumber(a.id) ?? 0;
      const partB = extractPartNumber(b.id) ?? 0;
      return partA - partB;
    });
    seriesPosts.push(...seriesPostsList);
  }
  
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

