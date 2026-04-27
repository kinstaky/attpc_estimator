import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      buffer: "buffer/",
    },
  },
  optimizeDeps: {
    esbuildOptions: {
      define: {
        global: "globalThis",
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    reportCompressedSize: false,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) {
            return undefined;
          }
          if (id.includes("plotly.js")) {
            return "vendor-plotly";
          }
          if (id.includes("vuetify")) {
            return "vendor-vuetify";
          }
          if (id.includes("vue-router")) {
            return "vendor-router";
          }
          return "vendor";
        },
      },
    },
  },
});
