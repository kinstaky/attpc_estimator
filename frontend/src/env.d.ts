/// <reference types="vite/client" />

declare module "*.vue";

declare module "plotly.js-dist-min" {
  const Plotly: {
    react: (...args: unknown[]) => unknown;
    purge: (...args: unknown[]) => unknown;
    relayout: (...args: unknown[]) => unknown;
  };
  export default Plotly;
}
