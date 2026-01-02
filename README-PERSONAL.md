# Personal Website - Gopi Krishna Tummala

This is my personal academic and professional website built with [AstroPaper](https://github.com/satnaing/astro-paper), a minimal and responsive blog theme.

## About

Senior Machine Learning Engineer at Adobe with 10+ years of experience in designing ML algorithms, training frameworks, and building large-scale systems for autonomous vehicles and generative AI.

## Website Structure

The site includes:

- **Home**: Welcome page with overview
- **About**: Detailed background and experience
- **Posts**: Research publications, blog posts about work
- **Tags**: Organized by topics (research, ML, autonomous vehicles, etc.)

## Local Development

```bash
# Install dependencies
cd astro-paper
pnpm install

# Run development server
pnpm dev

# Build for production
pnpm build

# Preview production build
pnpm preview
```

## Deployment

The site is deployed automatically via GitHub Actions when changes are pushed to the `master` branch.

The deployment workflow:
1. Builds the Astro site
2. Generates static search index
3. Deploys to GitHub Pages

## Content Management

Blog posts are located in `src/data/blog/` as markdown files with frontmatter.

### Adding New Posts

Create a new `.md` file in `src/data/blog/` with the following structure:

```markdown
---
author: Gopi Krishna Tummala
pubDatetime: 2024-01-01T00:00:00Z
title: Your Post Title
slug: your-post-slug
featured: false
draft: false
tags:
  - machine-learning
  - research
description: Brief description for SEO and previews
---

Your content here...
```

## Customization

- Site configuration: `src/config.ts`
- Theme colors: Edit Tailwind config
- Social links: `src/constants.ts`
- Navigation: `src/components/Header.astro`

## License

Personal website content. Theme based on MIT licensed AstroPaper.

