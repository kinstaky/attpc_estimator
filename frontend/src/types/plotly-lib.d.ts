declare module "plotly.js/lib/core" {
  import type Plotly from "plotly.js";

  const PlotlyCore: typeof Plotly & {
    register(modules: unknown[]): void;
  };

  export default PlotlyCore;
}

declare module "plotly.js/lib/bar" {
  const bar: unknown;
  export default bar;
}

declare module "plotly.js/lib/heatmap" {
  const heatmap: unknown;
  export default heatmap;
}

declare module "plotly.js/lib/scatter" {
  const scatter: unknown;
  export default scatter;
}

declare module "plotly.js/lib/scattergl" {
  const scattergl: unknown;
  export default scattergl;
}

declare module "plotly.js/lib/scatter3d" {
  const scatter3d: unknown;
  export default scatter3d;
}
