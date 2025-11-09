import type { ExportFormat, ExportOptions } from "./types";

const createFileName = (format: ExportFormat, options?: ExportOptions) => {
  const base = options?.fileName ?? "mindmap";
  return base.endsWith(`.${format}`) ? base : `${base}.${format}`;
};

const downloadBlob = (blob: Blob, fileName: string) => {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(url);
};

const serializeSvg = (svg: SVGSVGElement) => {
  const serializer = new XMLSerializer();
  return serializer.serializeToString(svg);
};

const svgElementToBlob = (svg: SVGSVGElement) =>
  new Blob([serializeSvg(svg)], { type: "image/svg+xml;charset=utf-8" });

const isTransparent = (color: string) =>
  color === "rgba(0, 0, 0, 0)" || color === "transparent";

const resolveBackgroundColor = (svg: SVGSVGElement, override?: string) => {
  if (override) return override;
  const svgStyle = window.getComputedStyle(svg);
  if (!isTransparent(svgStyle.backgroundColor)) return svgStyle.backgroundColor;
  const parent = svg.parentElement ? window.getComputedStyle(svg.parentElement) : null;
  if (parent && !isTransparent(parent.backgroundColor)) return parent.backgroundColor;
  return "#ffffff";
};

const getSvgDimensions = (svg: SVGSVGElement) => {
  const viewBox = svg.getAttribute("viewBox");
  if (viewBox) {
    const [, , widthStr, heightStr] = viewBox.trim().split(/\s+/);
    return {
      width: parseFloat(widthStr),
      height: parseFloat(heightStr),
    };
  }

  return {
    width: parseFloat(svg.getAttribute("width") ?? "0") || svg.clientWidth || 1,
    height: parseFloat(svg.getAttribute("height") ?? "0") || svg.clientHeight || 1,
  };
};

const copyComputedStyles = (source: Element, target: Element, properties: string[]) => {
  const computed = window.getComputedStyle(source as Element);
  properties.forEach((prop) => {
    const value = computed.getPropertyValue(prop);
    if (value) {
      target.setAttribute(prop.replace(/[A-Z]/g, (m) => `-${m.toLowerCase()}`), value.trim());
    }
  });
};

const inlineStyles = (original: SVGSVGElement, clone: SVGSVGElement) => {
  const originalLinks = original.querySelectorAll<SVGPathElement>('.rd3t-link');
  const clonedLinks = clone.querySelectorAll<SVGPathElement>('.rd3t-link');
  originalLinks.forEach((link, index) => {
    const counterpart = clonedLinks[index];
    if (!counterpart) return;
    copyComputedStyles(link, counterpart, [
      'stroke',
      'strokeWidth',
      'strokeOpacity',
      'strokeLinecap',
    ]);
    counterpart.setAttribute('fill', 'none');
  });

  const originalRects = original.querySelectorAll<SVGRectElement>('.mindmap-node rect');
  const clonedRects = clone.querySelectorAll<SVGRectElement>('.mindmap-node rect');
  originalRects.forEach((rect, index) => {
    const counterpart = clonedRects[index];
    if (!counterpart) return;
    copyComputedStyles(rect, counterpart, ['fill', 'stroke', 'strokeWidth', 'strokeOpacity']);
  });

  const originalTexts = original.querySelectorAll<SVGTextElement>('.mindmap-node text');
  const clonedTexts = clone.querySelectorAll<SVGTextElement>('.mindmap-node text');
  originalTexts.forEach((text, index) => {
    const counterpart = clonedTexts[index];
    if (!counterpart) return;
    copyComputedStyles(text, counterpart, [
      'fill',
      'stroke',
      'strokeWidth',
      'strokeOpacity',
      'fontFamily',
      'fontSize',
      'fontWeight',
      'letterSpacing',
      'fontVariationSettings',
      'fontStretch',
      'fontStyle',
    ]);
  });

  const originalRadialPaths = original.querySelectorAll<SVGPathElement>('.mindmap-radial path');
  const clonedRadialPaths = clone.querySelectorAll<SVGPathElement>('.mindmap-radial path');
  originalRadialPaths.forEach((path, index) => {
    const counterpart = clonedRadialPaths[index];
    if (!counterpart) return;
    copyComputedStyles(path, counterpart, ['stroke', 'strokeWidth', 'strokeOpacity', 'fill']);
  });

  const originalRadialCircles = original.querySelectorAll<SVGCircleElement>('.mindmap-radial circle');
  const clonedRadialCircles = clone.querySelectorAll<SVGCircleElement>('.mindmap-radial circle');
  originalRadialCircles.forEach((circle, index) => {
    const counterpart = clonedRadialCircles[index];
    if (!counterpart) return;
    copyComputedStyles(circle, counterpart, ['fill', 'stroke', 'strokeWidth', 'strokeOpacity']);
  });

  const originalRadialTexts = original.querySelectorAll<SVGTextElement>('.mindmap-radial text');
  const clonedRadialTexts = clone.querySelectorAll<SVGTextElement>('.mindmap-radial text');
  originalRadialTexts.forEach((text, index) => {
    const counterpart = clonedRadialTexts[index];
    if (!counterpart) return;
    copyComputedStyles(text, counterpart, [
      'fill',
      'stroke',
      'strokeWidth',
      'strokeOpacity',
      'fontFamily',
      'fontSize',
      'fontWeight',
      'letterSpacing',
      'fontVariationSettings',
      'fontStretch',
      'fontStyle',
    ]);
  });
};

const cloneSvgForExport = (svg: SVGSVGElement) => {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  inlineStyles(svg, clone);
  const sourceViewBox = svg.getAttribute('viewBox');
  if (sourceViewBox) {
    clone.setAttribute('viewBox', sourceViewBox);
    const [, , w, h] = sourceViewBox.trim().split(/\s+/);
    if (w && h) {
      clone.setAttribute('width', w);
      clone.setAttribute('height', h);
    }
  } else {
    const rect = svg.getBoundingClientRect();
    clone.setAttribute('viewBox', `0 0 ${rect.width} ${rect.height}`);
    clone.setAttribute('width', `${rect.width}`);
    clone.setAttribute('height', `${rect.height}`);
  }
  return clone;
};

const svgToPngBlob = async (
  svg: SVGSVGElement,
  backgroundColor?: string,
  scale = window.devicePixelRatio || 2,
) =>
  new Promise<Blob>((resolve, reject) => {
    const prepared = cloneSvgForExport(svg);
    const serialized = serializeSvg(prepared);
    const svgBlob = new Blob([serialized], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svgBlob);
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () => {
      URL.revokeObjectURL(url);
      const { width, height } = getSvgDimensions(prepared);
      const canvas = document.createElement('canvas');
      canvas.width = Math.max(width * scale, 1);
      canvas.height = Math.max(height * scale, 1);
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('Unable to create canvas context'));
        return;
      }
      ctx.save();
      ctx.scale(scale, scale);
      ctx.fillStyle = resolveBackgroundColor(svg, backgroundColor);
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(image, 0, 0, width, height);
      ctx.restore();
      canvas.toBlob(
        (blob) => {
          if (!blob) {
            reject(new Error('Unable to export canvas'));
            return;
          }
          resolve(blob);
        },
        'image/png',
        0.95,
      );
    };
    image.onerror = reject;
    image.src = url;
  });

const blobToDataUrl = (blob: Blob) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });

export const exportSvgElement = async (
  svg: SVGSVGElement,
  format: ExportFormat,
  options?: ExportOptions,
) => {
  if (format === "svg") {
    const prepared = cloneSvgForExport(svg);
    const blob = svgElementToBlob(prepared);
    downloadBlob(blob, createFileName(format, options));
    return;
  }

  if (format === "png") {
    const blob = await svgToPngBlob(svg, options?.backgroundColor, options?.scale);
    downloadBlob(blob, createFileName(format, options));
    return;
  }

  if (format === "pdf") {
    const pngBlob = await svgToPngBlob(svg, options?.backgroundColor, options?.scale);
    const dataUrl = await blobToDataUrl(pngBlob);
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.src = dataUrl;
    await image.decode();
    const { jsPDF } = await import("jspdf");
    const pdf = new jsPDF({
      orientation: image.width >= image.height ? "landscape" : "portrait",
      unit: "px",
      format: [image.width, image.height],
    });
    pdf.addImage(dataUrl, "PNG", 0, 0, image.width, image.height);
    const blob = pdf.output("blob");
    downloadBlob(blob, createFileName(format, options));
    return;
  }

  throw new Error(`Unsupported export format: ${format}`);
};

