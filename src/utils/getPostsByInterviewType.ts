import type { CollectionEntry } from "astro:content";
import getSortedPosts from "./getSortedPosts";

export type InterviewType = "Theory" | "System Design" | "Coding" | "Behavioral" | "ML-Infra";

export const getPostsByInterviewType = (
  posts: CollectionEntry<"blog">[],
  interviewType: InterviewType
) => {
  const sorted = getSortedPosts(posts);
  return sorted.filter(
    (post) =>
      post.data.interview_relevance?.includes(interviewType) ?? false
  );
};

export default getPostsByInterviewType;

