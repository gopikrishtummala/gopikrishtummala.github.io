# Deployment Instructions

## Current State
- Old site: `index.html`, `w3.css` (backup of original portfolio)
- New site: All files in `astro-paper/` directory
- GitHub Actions workflow: `.github/workflows/deploy.yml`

## Recommended: Deploy AstroPaper Site

### Option 1: Move AstroPaper to Root (Clean approach)

```bash
# 1. Backup old site
mkdir old-site-backup
mv index.html w3.css README.md old-site-backup/

# 2. Move AstroPaper contents to root
mv astro-paper/* .
mv astro-paper/.* . 2>/dev/null || true
rmdir astro-paper

# 3. Update GitHub workflow path
# (Already points to root, no changes needed)

# 4. Commit and push
git add .
git commit -m "Migrate to AstroPaper site"
git push origin master
```

### Option 2: Keep in Subdirectory (Current structure)

The workflow is already set up to build from `astro-paper/`. Just push:

```bash
git add .
git commit -m "Add personal website with AstroPaper"
git push origin master
```

## Enable GitHub Pages

1. Go to: https://github.com/gopikrishtummalaold/gopikrishtummalaold.github.io/settings/pages
2. Under "Build and deployment":
   - Source: **GitHub Actions**
3. Wait 2-3 minutes for deployment
4. Visit: https://gopikrishtummalaold.github.io/

## Local Development

```bash
cd astro-paper  # or root if you moved files
pnpm install
pnpm run dev
```

Visit http://localhost:4321

## Build Locally

```bash
pnpm run build
pnpm run preview
```

## Troubleshooting

- **Build fails**: Check Node.js version (need v18+)
- **404 errors**: Ensure GitHub Pages source is "GitHub Actions"
- **Workflow fails**: Check `.github/workflows/deploy.yml` paths

## Current Blog Posts

1. AutoCalib - Automatic Camera Calibration
2. PhD Research - From Wireless to Vision
3. Behavior Prediction in Autonomous Driving

All posts are in: `src/data/blog/`

