import type Plotly from "plotly.js/lib/core";

let plotlyPromise: Promise<typeof Plotly> | undefined;

export async function loadPlotly(): Promise<typeof Plotly> {
  if (!plotlyPromise) {
    plotlyPromise = import("plotly.js/lib/core").then(async (module) => {
      const PlotlyCore = module.default;
      const [
        bar,
        heatmap,
        scatter,
        scattergl,
        scatter3d,
      ] = await Promise.all([
        import("plotly.js/lib/bar"),
        import("plotly.js/lib/heatmap"),
        import("plotly.js/lib/scatter"),
        import("plotly.js/lib/scattergl"),
        import("plotly.js/lib/scatter3d"),
      ]);
      PlotlyCore.register([
        bar.default,
        heatmap.default,
        scatter.default,
        scattergl.default,
        scatter3d.default,
      ]);
      return PlotlyCore;
    });
  }
  return plotlyPromise;
}
