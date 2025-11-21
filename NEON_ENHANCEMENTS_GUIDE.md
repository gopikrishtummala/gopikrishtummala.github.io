# Neon/Cyberpunk Enhancements Guide

This guide shows how to use the new neon/cyberpunk design elements inspired by [Neodev](https://astro.build/themes/details/neodev-portfolio-template-for-web-developer/) and [DevPro](https://astro.build/themes/details/devpro-portfolio-template-for-developer/) themes.

## 🎨 What's Been Added

### 1. **Neon Color Palette**
- Cyan (`--neon-cyan: #00f0ff`)
- Purple (`--neon-purple: #b026ff`)
- Glow effects for both colors

### 2. **Visual Effects**
- Cyber grid backgrounds
- Neon glow effects (box-shadow and text-shadow)
- Pulse animations
- Reveal animations (scroll-triggered)
- Enhanced hover effects

### 3. **Components**
- Scroll progress indicator
- Enhanced series navigation
- Neon buttons
- Neon cards
- Gradient text

## 🚀 How to Use

### Scroll Progress Indicator

Add to your layout (e.g., `PostDetails.astro`):

```astro
import ScrollProgress from "@/components/ScrollProgress.astro";

<ScrollProgress />
```

### Enhanced Series Navigation

Your existing series nav will automatically get hover effects. To add more glow:

```html
<div class="series-nav neon-glow-soft">
  <!-- existing content -->
</div>
```

### Neon Buttons

```html
<button class="neon-button">Click Me</button>
```

### Neon Cards

```html
<div class="neon-card">
  <h3>Card Title</h3>
  <p>Card content...</p>
</div>
```

### Gradient Text

```html
<h1 class="gradient-text-cyan-purple">Futuristic Heading</h1>
```

### Neon Tags

```html
<span class="tag-neon">Tag Name</span>
```

### Cyber Grid Background

```html
<section class="cyber-grid">
  <!-- content with grid background -->
</section>
```

### Reveal Animations

```html
<div class="reveal-on-scroll">
  Content that fades in on scroll
</div>

<div class="reveal-on-scroll delay-200">
  Content with delay
</div>
```

### Enhanced Code Blocks

Code blocks automatically get:
- Neon border on hover
- Top gradient line
- Enhanced glow effects

### Enhanced Headings

H2 and H3 headings automatically get a cyan gradient underline.

## 🎯 Recommended Enhancements

### 1. Add Scroll Progress to PostDetails

In `src/layouts/PostDetails.astro`, add:

```astro
import ScrollProgress from "@/components/ScrollProgress.astro";

<!-- At the top of the body -->
<ScrollProgress />
```

### 2. Enhance Series Navigation

Update series nav inline styles to use classes:

```html
<div class="series-nav neon-glow-soft cyber-grid">
  <!-- existing content -->
</div>
```

### 3. Add Reveal Animations to Sections

```html
<section class="reveal-on-scroll">
  <h2>Section Title</h2>
  <p>Content...</p>
</section>
```

### 4. Enhance Code Blocks

Code blocks already have enhanced styling. You can add a custom class:

```html
<pre class="neon-glow-soft">
  <!-- code -->
</pre>
```

### 5. Add Neon Accents to Important Text

```html
<span class="neon-text-cyan">Important text</span>
```

## 🎨 Customization

### Adjust Glow Intensity

In `src/styles/neon-enhancements.css`, modify:

```css
--neon-glow-cyan: 0 0 20px rgba(0, 240, 255, 0.5),
                  0 0 40px rgba(0, 240, 255, 0.3),
                  0 0 60px rgba(0, 240, 255, 0.2);
```

### Change Colors

```css
:root {
  --neon-cyan: #00f0ff;  /* Change to your preferred cyan */
  --neon-purple: #b026ff; /* Change to your preferred purple */
}
```

### Adjust Animation Speed

```css
.pulse-neon {
  animation: pulse-neon 2s ease-in-out infinite; /* Change 2s to your preference */
}
```

## 📱 Responsive Behavior

All effects are automatically adjusted for mobile:
- Reduced glow intensity
- Smaller grid patterns
- Shorter underlines

## ♿ Accessibility

- Respects `prefers-reduced-motion`
- All effects are visual enhancements (no functionality changes)
- Maintains contrast ratios

## 🔥 Examples from Neodev/DevPro

### Neodev Features Replicated:
- ✅ Cyber-grid backgrounds
- ✅ Neon glow effects
- ✅ Pulse animations
- ✅ Smooth transitions
- ✅ Dark theme support

### DevPro Features Replicated:
- ✅ Scroll progress indicator
- ✅ Enhanced hover effects
- ✅ Gradient text
- ✅ Smooth scroll transitions
- ✅ Programmer-friendly aesthetics

## 🎯 Next Steps

1. Add `ScrollProgress` component to layouts
2. Apply `neon-glow-soft` to series nav
3. Add `reveal-on-scroll` to key sections
4. Use `gradient-text-cyan-purple` for headings
5. Experiment with `neon-card` for special content blocks

Enjoy your futuristic blog! 🚀

