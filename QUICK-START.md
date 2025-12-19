# Quick Start - Deploy Your Site

## Step 1: Commit Everything

```bash
cd /Users/gopi/Downloads/repos/gopikrishtummalaold.github.io

# Check what files are ready
git status

# Add all changes
git add .

# Commit
git commit -m "Personal website with AstroPaper - AutoCalib, PhD research, and prediction posts"
```

## Step 2: Push to GitHub

```bash
git push origin master
```

## Step 3: Enable GitHub Pages

1. Go to: https://github.com/gopikrishtummalaold/gopikrishtummalaold.github.io/settings/pages
2. Under "Build and deployment" → Source: Select **"GitHub Actions"**
3. Save

## Step 4: Wait for Deployment

- Go to: https://github.com/gopikrishtummalaold/gopikrishtummalaold.github.io/actions
- Watch the "Deploy to GitHub Pages" workflow run (takes 2-3 minutes)
- Once complete (green checkmark), visit: https://gopikrishtummalaold.github.io/

## Your Site Content

### Pages
- **Home**: Personalized welcome with professional summary
- **About**: Your background, experience, education, research
- **Posts**: 3 blog posts
- **Tags**: Organized by topics

### Blog Posts
1. **AutoCalib** - Best Paper Award winning camera calibration research
2. **PhD Research** - Journey from wireless to autonomous vision
3. **Behavior Prediction** - Closed-loop reasoning in autonomous driving

### Contact Info
- Email: tummalag.cseosu@gmail.com
- LinkedIn: https://www.linkedin.com/in/gopi-krishna-tummala-00294415/
- GitHub: https://github.com/gopikrishtummalaold

## Local Testing (Optional)

Before pushing, you can test locally:

```bash
cd astro-paper
pnpm install
pnpm run build
pnpm run preview
```

Visit: http://localhost:4321

## After Deployment

Your site will be live at:
**https://gopikrishtummalaold.github.io/**

Updates are automatic - just push to master branch and GitHub Actions will rebuild.

