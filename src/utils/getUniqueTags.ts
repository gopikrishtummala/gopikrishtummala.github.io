import type { CollectionEntry } from "astro:content";
import { slugifyStr } from "./slugify";
import postFilter from "./postFilter";

export interface TagWithCount {
  tag: string;
  tagName: string;
  count: number;
}

const getUniqueTags = (posts: CollectionEntry<"blog">[]): TagWithCount[] => {
  const tagMap = new Map<string, TagWithCount>();

  posts
    .filter(postFilter)
    .forEach(post => {
      post.data.tags.forEach(originalTag => {
        const tag = slugifyStr(originalTag);
        const current = tagMap.get(tag);
        if (current) {
          current.count += 1;
        } else {
          tagMap.set(tag, {
            tag,
            tagName: originalTag,
            count: 1,
          });
        }
      });
    });

  return Array.from(tagMap.values()).sort((a, b) => {
    if (b.count === a.count) {
      return a.tag.localeCompare(b.tag);
    }
    return b.count - a.count;
  });
};

export default getUniqueTags;
