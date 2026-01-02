#!/bin/bash

# Script to download organization icons (SVG preferred)
# Run this from the project root: bash scripts/download-icons.sh

ICON_DIR="public/icons/organizations"
mkdir -p "$ICON_DIR"

echo "Downloading organization icons..."

# --- Adobe ---
ADOBE_URL="https://www.adobe.com/content/dam/shared/images/product-icons/svg/adobe.svg"
if curl -L "$ADOBE_URL" -o "$ICON_DIR/adobe.svg" 2>/dev/null; then
  echo "Adobe icon downloaded"
else
  echo "Adobe SVG download failed — please download manually: $ADOBE_URL"
fi

# --- Microsoft ---
MS_URL="https://img.icons8.com/fluency-systems-filled/48/microsoft.svg"
if curl -L "$MS_URL" -o "$ICON_DIR/microsoft.svg" 2>/dev/null; then
  echo "Microsoft icon downloaded"
else
  echo "Microsoft SVG failed — search 'Microsoft logo SVG' manually"
fi

# --- Qualcomm ---
QUALCOMM_URL="https://upload.wikimedia.org/wikipedia/commons/6/6e/Qualcomm-logo.svg"
if curl -L "$QUALCOMM_URL" -o "$ICON_DIR/qualcomm.svg" 2>/dev/null; then
  echo "Qualcomm icon downloaded"
else
  echo "Qualcomm SVG failed — search 'Qualcomm logo SVG' manually"
fi

# --- Ohio State University ---
OSU_URL="https://upload.wikimedia.org/wikipedia/commons/1/1d/Ohio_State_University_seal.svg"
if curl -L "$OSU_URL" -o "$ICON_DIR/osu.svg" 2>/dev/null; then
  echo "OSU icon downloaded"
else
  echo "OSU SVG failed — see: https://www.osu.edu/brand-guide/visual-identity/logos.html"
fi

# --- IIT Madras ---
IITM_URL="https://upload.wikimedia.org/wikipedia/en/6/6e/IIT_Madras_Logo.svg"
if curl -L "$IITM_URL" -o "$ICON_DIR/iit-madras.svg" 2>/dev/null; then
  echo "IIT Madras icon downloaded"
else
  echo "IIT Madras SVG failed — visit: https://www.iitm.ac.in/"
fi

echo ""
echo "Icons downloaded to $ICON_DIR"
