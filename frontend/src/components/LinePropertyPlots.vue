<template>
  <v-row dense>
    <v-col
      v-for="plot in plots"
      :key="plot.key"
      cols="12"
      lg="6"
    >
      <v-card class="result-card-vuetify" rounded="xl">
        <v-card-title class="result-card-title">
          <div>
            <p class="page-kicker">Phase 2</p>
            <h2>{{ plot.title }}</h2>
          </div>
        </v-card-title>
        <v-card-text>
          <div :ref="(element) => setPlotRoot(plot.key, element)" class="trace-plot result-plot"></div>
        </v-card-text>
      </v-card>
    </v-col>
  </v-row>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, watch } from "vue";

import { loadPlotly } from "../lib/plotly";
import type { HistogramPlot } from "../types";

const props = defineProps<{
  plots: HistogramPlot[];
  scaleMode: "linear" | "log";
}>();

const plotRoots = new Map<string, HTMLElement>();

function setPlotRoot(key: string, element: unknown): void {
  if (element instanceof HTMLElement) {
    plotRoots.set(key, element);
    return;
  }
  plotRoots.delete(key);
}

function baseLayout() {
  return {
    margin: { t: 48, r: 24, b: 56, l: 68 },
    paper_bgcolor: "#fffdf8",
    plot_bgcolor: "#fffdf8",
    font: {
      family: "ui-sans-serif, system-ui, sans-serif",
      color: "#222",
    },
  };
}

function transformHeatmapValues(values: number[][], scaleMode: "linear" | "log"): Array<Array<number | null>> {
  return values.map((row) => row.map((value) => {
    const count = Number(value || 0);
    if (count <= 0) {
      return null;
    }
    return scaleMode === "log" ? Math.log10(count + 1) : count;
  }));
}

async function renderPlot(
  Plotly: Awaited<ReturnType<typeof loadPlotly>>,
  plot: HistogramPlot,
): Promise<void> {
  const root = plotRoots.get(plot.key);
  if (!root) {
    return;
  }

  if (plot.render === "bar") {
    const histogram = (plot.histogram || []) as number[];
    await Plotly.react(
      root,
      [
        {
          type: "bar",
          x: plot.binCenters || [],
          y: histogram.map((value) => {
            const count = Number(value || 0);
            if (props.scaleMode === "log" && count <= 0) {
              return null;
            }
            return count;
          }),
          customdata: histogram,
          marker: { color: "#174f40" },
          hovertemplate: `${plot.binLabel || "Value"} %{x}<br>${plot.countLabel || "Count"} %{customdata}<extra></extra>`,
        },
      ],
      {
        ...baseLayout(),
        title: { text: plot.title, x: 0.02, xanchor: "left" },
        uirevision: plot.key,
        xaxis: {
          title: plot.binLabel || "Value",
          zeroline: false,
          gridcolor: "#e7dfcf",
        },
        yaxis: {
          title: plot.countLabel || "Count",
          type: props.scaleMode,
          zeroline: false,
          gridcolor: "#e7dfcf",
        },
        bargap: 0,
      },
      { displayModeBar: false, responsive: true },
    );
    return;
  }

  if (plot.render === "grouped_bar") {
    await Plotly.react(
      root,
      (plot.series || []).map((series, index) => ({
        type: "bar",
        name: series.title,
        x: plot.binCenters || [],
        y: series.histogram.map((value) => {
          const count = Number(value || 0);
          if (props.scaleMode === "log" && count <= 0) {
            return null;
          }
          return count;
        }),
        customdata: series.histogram,
        hovertemplate: `${plot.binLabel || "Value"} %{x}<br>${plot.countLabel || "Count"} %{customdata}<extra>${series.title}</extra>`,
        marker: { color: ["#174f40", "#b35900", "#4f5d75"][index % 3] },
      })),
      {
        ...baseLayout(),
        title: { text: plot.title, x: 0.02, xanchor: "left" },
        uirevision: plot.key,
        xaxis: {
          title: plot.binLabel || "Value",
          zeroline: false,
          gridcolor: "#e7dfcf",
        },
        yaxis: {
          title: plot.countLabel || "Count",
          type: props.scaleMode,
          zeroline: false,
          gridcolor: "#e7dfcf",
        },
        barmode: "group",
        bargap: 0.08,
      },
      { displayModeBar: false, responsive: true },
    );
    return;
  }

  const histogram = (plot.histogram || []) as number[][];
  await Plotly.react(
    root,
    [
      {
        type: "heatmap",
        z: transformHeatmapValues(histogram, props.scaleMode),
        customdata: histogram,
        x: plot.xBinCenters || [],
        y: plot.yBinCenters || [],
        colorscale: "YlGnBu",
        hovertemplate: `${plot.xLabel || "X"} %{x}<br>${plot.yLabel || "Y"} %{y}<br>${plot.countLabel || "Count"} %{customdata}<extra></extra>`,
        colorbar: {
          title: props.scaleMode === "log" ? "log10(count + 1)" : (plot.countLabel || "Count"),
        },
      },
    ],
    {
      ...baseLayout(),
      title: { text: plot.title, x: 0.02, xanchor: "left" },
      uirevision: plot.key,
      xaxis: {
        title: plot.xLabel || "X",
        zeroline: false,
        gridcolor: "#e7dfcf",
      },
      yaxis: {
        title: plot.yLabel || "Y",
        zeroline: false,
        gridcolor: "#e7dfcf",
      },
    },
    { displayModeBar: false, responsive: true },
  );
}

async function renderPlots(): Promise<void> {
  const Plotly = await loadPlotly();
  for (const plot of props.plots) {
    await renderPlot(Plotly, plot);
  }
}

onMounted(() => {
  void renderPlots();
});

watch(
  () => props.plots,
  () => {
    void renderPlots();
  },
  { deep: true, immediate: true },
);

watch(
  () => props.scaleMode,
  () => {
    void renderPlots();
  },
);

onBeforeUnmount(() => {
  void loadPlotly().then((Plotly) => {
    for (const root of plotRoots.values()) {
      Plotly.purge(root);
    }
  });
});
</script>
