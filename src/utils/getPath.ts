import { BLOG_PATH } from "@/content.config";
import { slugifyStr } from "./slugify";
import type { CollectionEntry } from "astro:content";

/**
 * Get full path of a blog post
 * @param id - id of the blog post (aka slug)
 * @param filePath - the blog post full file location
 * @param includeBase - whether to include `/posts` in return value
 * @returns blog post path
 */
export function getPath(
  id: string,
  filePath: string | undefined,
  includeBase = true
) {
  const pathSegments = filePath
    ?.replace(BLOG_PATH, "")
    .split("/")
    .filter(path => path !== "") // remove empty string in the segments ["", "other-path"] <- empty string will be removed
    .filter(path => !path.startsWith("_")) // exclude directories start with underscore "_"
    .slice(0, -1) // remove the last segment_ file name_ since it's unnecessary
    .map(segment => slugifyStr(segment)); // slugify each segment path

  const basePath = includeBase ? "/posts" : "";

  // Making sure `id` does not contain the directory
  const blogId = id.split("/");
  const slug = blogId.length > 0 ? blogId.slice(-1) : blogId;

  // If not inside the sub-dir, simply return the file path
  if (!pathSegments || pathSegments.length < 1) {
    return [basePath, slug].join("/");
  }

  return [basePath, ...pathSegments, slug].join("/");
}

/**
 * Get the URL path for a blog post, using frontmatter slug if available
 * @param post - the blog post entry
 * @param includeBase - whether to include `/posts` in return value
 * @returns blog post URL path
 */
export function getPostPath(
  post: CollectionEntry<"blog"> | null | undefined,
  includeBase = true
) {
  // Safety check for post
  if (!post) {
    throw new Error("Invalid post entry: post is null or undefined");
  }
  
  // Check if post.data exists, if not try to use id and filePath
  if (!post.data) {
    if (post.id && post.filePath) {
      return getPath(post.id, post.filePath, includeBase);
    }
    throw new Error("Invalid post entry: post.data is undefined and cannot generate path");
  }
  
  // Use frontmatter slug if available, otherwise use path-based slug
  // Use optional chaining for extra safety
  const slug = post.data?.slug;
  if (slug) {
    const basePath = includeBase ? "/posts" : "";
    return [basePath, slug].filter(Boolean).join("/");
  }
  return getPath(post.id, post.filePath, includeBase);
}
