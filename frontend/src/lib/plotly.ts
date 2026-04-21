import type Plotly from "plotly.js-dist-min";

let plotlyPromise: Promise<typeof Plotly> | undefined;

export async function loadPlotly(): Promise<typeof Plotly> {
  if (!plotlyPromise) {
    plotlyPromise = import("plotly.js-dist-min").then(
      (module) => module.default,
    );
  }
  return plotlyPromise;
}
