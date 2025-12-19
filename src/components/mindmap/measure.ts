const labelCache = new Map<string, number>();

let canvas: HTMLCanvasElement | null = null;
let context: CanvasRenderingContext2D | null = null;

const ensureContext = () => {
  if (!canvas) {
    canvas = document.createElement("canvas");
    context = canvas.getContext("2d");
  }
  return context;
};

export const measureLabelWidth = (
  text: string,
  fontSize: number,
  fontWeight: number,
  fontFamily: string,
) => {
  if (!text) return 0;
  const key = `${fontSize}-${fontWeight}-${fontFamily}-${text}`;
  const cached = labelCache.get(key);
  if (typeof cached === "number") return cached;

  if (typeof document === "undefined") {
    const fallback = text.length * fontSize * 0.58;
    labelCache.set(key, fallback);
    return fallback;
  }

  const ctx = ensureContext();
  if (!ctx) {
    const fallback = text.length * fontSize * 0.58;
    labelCache.set(key, fallback);
    return fallback;
  }

  ctx.font = `${fontWeight} ${fontSize}px ${fontFamily}`;
  const width = ctx.measureText(text).width;
  labelCache.set(key, width);
  return width;
};

export const resetMeasurements = () => {
  labelCache.clear();
};

