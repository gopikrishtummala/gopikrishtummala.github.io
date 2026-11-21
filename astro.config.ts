import { defineConfig, envField } from "astro/config";
import tailwindcss from "@tailwindcss/vite";
import sitemap from "@astrojs/sitemap";
import remarkToc from "remark-toc";
import remarkCollapse from "remark-collapse";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import {
  transformerNotationDiff,
  transformerNotationHighlight,
  transformerNotationWordHighlight,
} from "@shikijs/transformers";
import { transformerFileName } from "./src/utils/transformers/fileName";
import { SITE } from "./src/config";
import mdx from "@astrojs/mdx";
import react from "@astrojs/react";

// https://astro.build/config
export default defineConfig({
  site: SITE.website,
  redirects: {
    "/posts/computer-vision/computer-vision-module-2-convolutional-networks": "/posts/computer-vision/computer-vision-module-4-convolutional-networks",
    "/posts/computer-vision/computer-vision-module-3-vision-transformers": "/posts/computer-vision/computer-vision-module-5-vision-transformers",
    "/posts/computer-vision/computer-vision-module-4-object-detection": "/posts/computer-vision/computer-vision-module-6-object-detection",
    "/posts/computer-vision/computer-vision-module-5-segmentation": "/posts/computer-vision/computer-vision-module-7-segmentation",
    "/posts/computer-vision/computer-vision-module-6-video-and-tracking": "/posts/computer-vision/computer-vision-module-8-video-and-tracking",
    "/posts/computer-vision/computer-vision-module-7-3d-vision": "/posts/computer-vision/computer-vision-module-9-3d-vision",
    "/posts/computer-vision/computer-vision-module-8-vision-foundation-models": "/posts/computer-vision/computer-vision-module-10-vision-foundation-models",
  },
  integrations: [
    mdx(),
    react(),
    sitemap({
      filter: page => SITE.showArchives || !page.endsWith("/archives"),
    }),
  ],
  markdown: {
    remarkPlugins: [remarkToc, [remarkCollapse, { test: "Table of contents" }], remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      // For more themes, visit https://shiki.style/themes
      // Modern themes that look great with Fira Code:
      // - catppuccin-latte / catppuccin-mocha (warm, modern)
      // - rose-pine-dawn / rose-pine-moon (elegant)
      // - one-light / one-dark-pro (VS Code style)
      themes: { light: "catppuccin-latte", dark: "catppuccin-mocha" },
      defaultColor: false,
      wrap: false,
      transformers: [
        transformerFileName({ style: "v2", hideDot: false }),
        transformerNotationHighlight(),
        transformerNotationWordHighlight(),
        transformerNotationDiff({ matchAlgorithm: "v3" }),
      ],
    },
  },
  vite: {
    // eslint-disable-next-line
    // @ts-ignore
    // This will be fixed in Astro 6 with Vite 7 support
    // See: https://github.com/withastro/astro/issues/14030
    plugins: [tailwindcss()],
    optimizeDeps: {
      exclude: ["@resvg/resvg-js"],
    },
  },
  image: {
    responsiveStyles: true,
    layout: "constrained",
  },
  env: {
    schema: {
      PUBLIC_GOOGLE_SITE_VERIFICATION: envField.string({
        access: "public",
        context: "client",
        optional: true,
      }),
    },
  },
  experimental: {
    preserveScriptOrder: true,
    svgo: true, // Automatically optimize SVG files during build
  },
});
