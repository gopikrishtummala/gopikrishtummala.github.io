import type { CollectionEntry } from "astro:content";
import getSortedPosts from "./getSortedPosts";

export type Track = "Fundamentals" | "GenAI Systems" | "Robotics" | "Agentic AI";

export const getPostsByTrack = (
  posts: CollectionEntry<"blog">[],
  track: Track
) => {
  const sorted = getSortedPosts(posts);
  return sorted.filter((post) => post.data.track === track);
};

export default getPostsByTrack;

