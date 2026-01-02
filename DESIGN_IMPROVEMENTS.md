# Design Improvements Guide

## ✅ What's Been Updated

### 1. **Color Scheme** (Less Blue/Pink, More Modern)
- **Light Mode**: Clean grays with indigo/purple accents
- **Dark Mode**: Deep blacks with indigo/purple accents
- **Gradients**: Changed from sky-blue to indigo-purple-fuchsia
- **Accent Color**: Updated to indigo-600 (#6366f1) for a more professional look

### 2. **Organization Icons**
- Created `OrgIcon.astro` component
- Added icon support for: Adobe, Microsoft, Qualcomm, OSU, IIT Madras, Zoox, Standard Chartered, Tata Elxsi
- Timeline now displays organization logos

### 3. **Visual Enhancements**
- Updated hero section gradients
- Enhanced timeline with icons
- Improved button styling with gradients
- Better shadow effects

## 📥 Downloading Icons

### Option 1: Use the Script
```bash
bash scripts/download-icons.sh
```

### Option 2: Manual Download

**Recommended Sources:**

1. **Adobe**
   - SVG: https://www.adobe.com/content/dam/shared/images/product-icons/svg/adobe.svg
   - Save as: `public/icons/organizations/adobe.svg`

2. **Microsoft**
   - Search: "Microsoft logo SVG download"
   - Or: https://www.microsoft.com/en-us/store/b/microsoft
   - Save as: `public/icons/organizations/microsoft.svg`

3. **Qualcomm**
   - Search: "Qualcomm logo SVG"
   - Or company website
   - Save as: `public/icons/organizations/qualcomm.svg`

4. **Ohio State University**
   - Official: https://www.osu.edu/brand-guide/visual-identity/logos.html
   - Save as: `public/icons/organizations/osu.svg`

5. **IIT Madras**
   - Official: https://www.iitm.ac.in/
   - Save as: `public/icons/organizations/iit-madras.svg`

6. **Zoox** (if needed)
   - Search: "Zoox logo SVG"
   - Save as: `public/icons/organizations/zoox.svg`

### Icon Requirements
- **Format**: SVG preferred (scalable, crisp at any size)
- **Size**: Will be auto-sized to 20px in timeline
- **Fallback**: If icon not found, it will be hidden (no broken image)

## 🎨 Color Palette Changes

### Before (Blue/Pink)
- Accent: `#2563eb` (blue)
- Gradients: sky-500, indigo-500, purple-500

### After (Modern Indigo/Purple)
- Accent: `#6366f1` (indigo-600)
- Gradients: indigo-600, purple-600, fuchsia-600
- More professional, less "toy-like"

## 🚀 Next Steps

1. **Download Icons**: Run the script or download manually
2. **Test**: Check timeline on homepage - icons should appear
3. **Customize**: Adjust colors further if needed in `global.css`

## 📱 Mobile

Mobile design is already good - no changes needed. All enhancements are responsive.

