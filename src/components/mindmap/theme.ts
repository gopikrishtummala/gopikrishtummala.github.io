import { useEffect, useMemo, useState } from "react";
import { DEFAULT_COLORS } from "./tokens";
import type { MindmapThemeMode, Palette } from "./types";

const BRANCH_LIGHT_BG = [0.95, 0.92, 0.89, 0.86, 0.83];
const BRANCH_DARK_BG = [0.88, 0.84, 0.8, 0.76, 0.72];

const clampDepth = (depth: number) => Math.min(depth, BRANCH_LIGHT_BG.length - 1);

export const resolveBranchTint = (depth: number, isDark: boolean) => {
  const index = clampDepth(depth);
  const alpha = isDark ? BRANCH_DARK_BG[index] : BRANCH_LIGHT_BG[index];
  const base = isDark ? "15,23,42" : "255,255,255";
  return `rgba(${base},${alpha.toFixed(2)})`;
};

export const resolveLeafTint = (depth: number, isDark: boolean) => {
  const alpha = Math.max(0.12, 0.18 - depth * 0.02);
  return isDark
    ? `rgba(56,189,248,${alpha.toFixed(2)})`
    : `rgba(37,99,235,${alpha.toFixed(2)})`;
};

const resolveShadow = (isDark: boolean) =>
  isDark ? DEFAULT_COLORS.shadow.dark : DEFAULT_COLORS.shadow.light;

export const useThemeMode = (mode: MindmapThemeMode = "auto") => {
  const [isDark, setIsDark] = useState<boolean>(() => {
    if (mode === "dark") return true;
    if (mode === "light") return false;
    if (typeof document !== "undefined") {
      const dataTheme = document.documentElement.getAttribute("data-theme");
      if (dataTheme === "dark") return true;
      if (dataTheme === "light") return false;
    }
    if (typeof window !== "undefined" && window.matchMedia) {
      return window.matchMedia("(prefers-color-scheme: dark)").matches;
    }
    return false;
  });

  useEffect(() => {
    if (mode === "dark") {
      setIsDark(true);
      return;
    }
    if (mode === "light") {
      setIsDark(false);
      return;
    }
    if (typeof window === "undefined") return;
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const updateFromMedia = () => setIsDark(mql.matches);
    updateFromMedia();
    mql.addEventListener("change", updateFromMedia);

    const doc = document.documentElement;
    const observer = new MutationObserver(() => {
      const theme = doc.getAttribute("data-theme");
      if (theme === "dark") setIsDark(true);
      else if (theme === "light") setIsDark(false);
    });
    observer.observe(doc, { attributes: true, attributeFilter: ["data-theme"] });

    return () => {
      mql.removeEventListener("change", updateFromMedia);
      observer.disconnect();
    };
  }, [mode]);

  return isDark;
};

export const usePalette = (
  isDark: boolean,
  branchTint?: { light: string; dark: string },
  leafTint?: { light: string; dark: string },
): Palette =>
  useMemo(() => {
    const leafBg = isDark
      ? leafTint?.dark ?? DEFAULT_COLORS.leaf.darkBg
      : leafTint?.light ?? DEFAULT_COLORS.leaf.lightBg;
    const leafBorder = isDark
      ? DEFAULT_COLORS.leaf.darkBorder
      : DEFAULT_COLORS.leaf.lightBorder;
    const branchBorder = isDark
      ? DEFAULT_COLORS.branch.darkBorder
      : DEFAULT_COLORS.branch.lightBorder;
    const branchText = isDark ? DEFAULT_COLORS.text.darkName : DEFAULT_COLORS.text.lightName;
    const branchDescription = isDark
      ? DEFAULT_COLORS.text.darkDescription
      : DEFAULT_COLORS.text.lightDescription;

    const branchShadows = resolveShadow(isDark);

    return {
      link: isDark ? DEFAULT_COLORS.link.dark : DEFAULT_COLORS.link.light,
      leaf: {
        bg: leafBg,
        border: leafBorder,
        name: isDark ? DEFAULT_COLORS.text.darkName : DEFAULT_COLORS.text.lightName,
        description: isDark
          ? DEFAULT_COLORS.text.darkDescription
          : DEFAULT_COLORS.text.lightDescription,
        shadow: resolveShadow(isDark),
      },
      branches: Array.from({ length: BRANCH_LIGHT_BG.length }).map((_, depth) => ({
        bg:
          depth === 0
            ? branchTint?.[isDark ? "dark" : "light"] ??
              (isDark
                ? DEFAULT_COLORS.branch.darkBg
                : DEFAULT_COLORS.branch.lightBg)
            : resolveBranchTint(depth, isDark),
        border: branchBorder,
        name: branchText,
        description: branchDescription,
        shadow: branchShadows,
      })),
    };
  }, [isDark, branchTint, leafTint]);

