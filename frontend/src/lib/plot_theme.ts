export interface PlotTheme {
  surface: string;
  grid: string;
  text: string;
}

function cssVariable(name: string, fallback: string): string {
  if (typeof document === "undefined") {
    return fallback;
  }
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

export function plotTheme(): PlotTheme {
  return {
    surface: cssVariable("--plot-surface", "#ffffff"),
    grid: cssVariable("--plot-grid", "#d9dde3"),
    text: cssVariable("--plot-ink", "#17191d"),
  };
}
