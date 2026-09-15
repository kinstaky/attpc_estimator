import type Plotly from "plotly.js/lib/core";

import { plotTheme } from "./plot_theme";

let plotlyPromise: Promise<typeof Plotly> | undefined;

function applyThemeToLayout(layout: any): any {
  const theme = plotTheme();
  const themedLayout = {
    ...layout,
    paper_bgcolor: theme.surface,
    plot_bgcolor: theme.surface,
    font: { ...layout?.font, color: theme.text },
  };
  for (const [key, axis] of Object.entries(layout || {})) {
    if (key.startsWith("xaxis") || key.startsWith("yaxis")) {
      themedLayout[key] = { ...(axis as object), gridcolor: theme.grid, color: theme.text };
    }
  }
  if (layout?.scene) {
    const themeSceneAxis = (axis: any) => ({
      ...axis,
      backgroundcolor: theme.surface,
      gridcolor: theme.grid,
      color: theme.text,
    });
    themedLayout.scene = {
      ...layout.scene,
      xaxis: themeSceneAxis(layout.scene.xaxis),
      yaxis: themeSceneAxis(layout.scene.yaxis),
      zaxis: themeSceneAxis(layout.scene.zaxis),
    };
  }
  return themedLayout;
}

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
      const originalReact = PlotlyCore.react.bind(PlotlyCore);
      (PlotlyCore as any).react = (root: any, data: any, layout: any, config: any) => (
        originalReact(root, data, applyThemeToLayout(layout), config)
      );
      return PlotlyCore;
    });
  }
  return plotlyPromise;
}
