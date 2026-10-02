export interface PlotTheme {
  surface: string;
  grid: string;
  text: string;
  raw: string;
  fit: string;
  peak: string;
  peakStart: string;
  peakEnd: string;
  derivative: string;
  found: string;
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
    raw: cssVariable("--plot-raw", "#0b63ce"),
    fit: cssVariable("--plot-fit", "#c62828"),
    peak: cssVariable("--plot-peak", "#b45309"),
    peakStart: cssVariable("--plot-peak-start", "#00796b"),
    peakEnd: cssVariable("--plot-peak-end", "#6d28d9"),
    derivative: cssVariable("--plot-derivative", "#2563eb"),
    found: cssVariable("--plot-found", "#047857"),
  };
}
