<template>
  <v-row dense>
    <v-col cols="12" lg="6">
      <v-card v-if="lineCountPlot" class="result-card-vuetify" rounded="xl">
        <v-card-title class="result-card-title">
          <div>
            <p class="page-kicker">Phase 2</p>
            <h2>{{ lineCountPlot.title }}</h2>
          </div>
        </v-card-title>
        <v-card-text>
          <div :ref="(element) => setPlotRoot('line_count', element)" class="trace-plot result-plot"></div>
        </v-card-text>
      </v-card>
    </v-col>

    <v-col cols="12" lg="6">
      <v-card v-if="labeledRatioPlot" class="result-card-vuetify" rounded="xl">
        <v-card-title class="result-card-title">
          <div>
            <p class="page-kicker">Phase 2</p>
            <h2>{{ labeledRatioPlot.title }}</h2>
          </div>
        </v-card-title>
        <v-card-text>
          <div :ref="(element) => setPlotRoot('labeled_ratio', element)" class="trace-plot result-plot"></div>
        </v-card-text>
      </v-card>
    </v-col>

    <v-col cols="12" lg="6">
      <v-card v-if="projectedDistancePlot" class="result-card-vuetify" rounded="xl">
        <v-card-title class="result-card-title">
          <div>
            <p class="page-kicker">Phase 2</p>
            <h2>{{ projectedDistancePlot.title }}</h2>
          </div>
        </v-card-title>
        <v-card-text>
          <v-row dense class="mb-2">
            <v-col cols="12" md="6">
              <v-text-field
                v-model.number="angleMin"
                min="0"
                max="180"
                label="Angle min (deg)"
                type="number"
                variant="outlined"
              />
            </v-col>
            <v-col cols="12" md="6">
              <v-text-field
                v-model.number="angleMax"
                min="0"
                max="180"
                label="Angle max (deg)"
                type="number"
                variant="outlined"
              />
            </v-col>
          </v-row>
          <div :ref="(element) => setPlotRoot('distances1', element)" class="trace-plot result-plot"></div>
        </v-card-text>
      </v-card>
    </v-col>

    <v-col cols="12" lg="6">
      <v-card v-if="projectedDotPlot" class="result-card-vuetify" rounded="xl">
        <v-card-title class="result-card-title">
          <div>
            <p class="page-kicker">Phase 2</p>
            <h2>{{ projectedDotPlot.title }}</h2>
          </div>
        </v-card-title>
        <v-card-text>
          <v-row dense class="mb-2">
            <v-col cols="12" md="6">
              <v-text-field
                v-model.number="distanceMin"
                :min="distanceAxisBounds.min"
                :max="distanceAxisBounds.max"
                label="Pair distance min (mm)"
                type="number"
                variant="outlined"
              />
            </v-col>
            <v-col cols="12" md="6">
              <v-text-field
                v-model.number="distanceMax"
                :min="distanceAxisBounds.min"
                :max="distanceAxisBounds.max"
                label="Pair distance max (mm)"
                type="number"
                variant="outlined"
              />
            </v-col>
          </v-row>
          <div :ref="(element) => setPlotRoot('distances2', element)" class="trace-plot result-plot"></div>
        </v-card-text>
      </v-card>
    </v-col>

    <v-col cols="12" lg="6">
      <v-card v-if="jointPlot" class="result-card-vuetify" rounded="xl">
        <v-card-title class="result-card-title">
          <div>
            <p class="page-kicker">Phase 2</p>
            <h2>{{ jointPlot.title }}</h2>
          </div>
        </v-card-title>
        <v-card-text>
          <div :ref="(element) => setPlotRoot('joint', element)" class="trace-plot result-plot"></div>
        </v-card-text>
      </v-card>
    </v-col>

  </v-row>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";

import { loadPlotly } from "../lib/plotly";
import type { HistogramPlot } from "../types";

const props = defineProps<{
  plots: HistogramPlot[];
  scaleMode: "linear" | "log";
}>();

const plotRoots = new Map<string, HTMLElement>();
const angleMin = ref(0);
const angleMax = ref(180);
const distanceMin = ref(0);
const distanceMax = ref(600);

const plotMap = computed(() => new Map(props.plots.map((plot) => [plot.key, plot])));

const lineCountPlot = computed(() => plotMap.value.get("line_count"));
const labeledRatioPlot = computed(() => plotMap.value.get("labeled_ratio"));
const jointPlot = computed(() => plotMap.value.get("joint"));
const distanceTemplatePlot = computed(() => plotMap.value.get("distances1"));
const angleTemplatePlot = computed(() => plotMap.value.get("distances2"));

const distanceAxisBounds = computed(() => {
  const centers = jointPlot.value?.xBinCenters || [];
  if (!centers.length) {
    return { min: 0, max: 600 };
  }
  return {
    min: Math.min(...centers),
    max: Math.max(...centers),
  };
});

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function finiteNumber(value: number, fallback: number): number {
  return Number.isFinite(value) ? value : fallback;
}

function normalizeRange(first: number, second: number): { min: number; max: number } {
  return first <= second ? { min: first, max: second } : { min: second, max: first };
}

function projectToX(
  histogram: number[][],
  yCenters: number[],
  xSize: number[],
  yMin: number,
  yMax: number,
): number[] {
  return histogram.reduce(
    (projection, row, index) => {
      const center = yCenters[index];
      if (center < yMin || center > yMax) {
        return projection;
      }
      row.forEach((value, columnIndex) => {
        projection[columnIndex] += Number(value || 0);
      });
      return projection;
    },
    new Array(xSize.length).fill(0),
  );
}

function projectToY(
  histogram: number[][],
  xCenters: number[],
  ySize: number[],
  xMin: number,
  xMax: number,
): number[] {
  return histogram.map((row) => row.reduce((sum, value, columnIndex) => {
    const center = xCenters[columnIndex];
    if (center < xMin || center > xMax) {
      return sum;
    }
    return sum + Number(value || 0);
  }, 0)).slice(0, ySize.length);
}

const projectedDistancePlot = computed<HistogramPlot | null>(() => {
  const template = distanceTemplatePlot.value;
  const joint = jointPlot.value;
  if (!template || !joint || !joint.histogram || !joint.xBinCenters || !joint.yBinCenters) {
    return template || null;
  }
  const histogram = joint.histogram as number[][];
  const angleRange = normalizeRange(
    clamp(finiteNumber(angleMin.value, 0), 0, 180),
    clamp(finiteNumber(angleMax.value, 180), 0, 180),
  );
  return {
    ...template,
    histogram: projectToX(
      histogram,
      joint.yBinCenters,
      joint.xBinCenters,
      angleRange.min,
      angleRange.max,
    ),
    binCenters: joint.xBinCenters,
  };
});

const projectedDotPlot = computed<HistogramPlot | null>(() => {
  const template = angleTemplatePlot.value;
  const joint = jointPlot.value;
  if (!template || !joint || !joint.histogram || !joint.xBinCenters || !joint.yBinCenters) {
    return template || null;
  }
  const histogram = joint.histogram as number[][];
  const fallbackBounds = distanceAxisBounds.value;
  const clampedRange = normalizeRange(
    clamp(finiteNumber(distanceMin.value, fallbackBounds.min), fallbackBounds.min, fallbackBounds.max),
    clamp(finiteNumber(distanceMax.value, fallbackBounds.max), fallbackBounds.min, fallbackBounds.max),
  );
  return {
    ...template,
    histogram: projectToY(
      histogram,
      joint.xBinCenters,
      joint.yBinCenters,
      clampedRange.min,
      clampedRange.max,
    ),
    binCenters: joint.yBinCenters,
  };
});

const renderedPlots = computed(() => [
  lineCountPlot.value,
  labeledRatioPlot.value,
  projectedDistancePlot.value,
  projectedDotPlot.value,
  jointPlot.value,
].filter((plot): plot is HistogramPlot => Boolean(plot)));

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

async function renderPlot(Plotly: Awaited<ReturnType<typeof loadPlotly>>, plot: HistogramPlot): Promise<void> {
  const root = plotRoots.get(plot.key);
  if (!root) {
    return;
  }
  if (plot.render === "bar") {
    const histogram = (plot.histogram || []) as number[];
    const yValues = histogram.map((value) => {
      const count = Number(value || 0);
      if (props.scaleMode === "log" && count <= 0) {
        return null;
      }
      return count;
    });
    await Plotly.react(
      root,
      [
        {
          type: "bar",
          x: plot.binCenters || [],
          y: yValues,
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

async function renderProjectedDistancePlot(): Promise<void> {
  const plot = projectedDistancePlot.value;
  if (!plot) {
    return;
  }
  const Plotly = await loadPlotly();
  await renderPlot(Plotly, plot);
}

async function renderProjectedAnglePlot(): Promise<void> {
  const plot = projectedDotPlot.value;
  if (!plot) {
    return;
  }
  const Plotly = await loadPlotly();
  await renderPlot(Plotly, plot);
}

async function renderPlots(): Promise<void> {
  const Plotly = await loadPlotly();
  for (const plot of renderedPlots.value) {
    const root = plotRoots.get(plot.key);
    if (!root) {
      continue;
    }
    await renderPlot(Plotly, plot);
  }
}

onMounted(() => {
  void renderPlots();
});

watch(
  () => props.plots,
  () => {
    const bounds = distanceAxisBounds.value;
    angleMin.value = 0;
    angleMax.value = 180;
    distanceMin.value = Math.max(0, bounds.min);
    distanceMax.value = Math.min(600, bounds.max);
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

watch(
  () => projectedDistancePlot.value,
  () => {
    void renderProjectedDistancePlot();
  },
  { deep: true },
);

watch(
  () => projectedDotPlot.value,
  () => {
    void renderProjectedAnglePlot();
  },
  { deep: true },
);

onBeforeUnmount(() => {
  void loadPlotly().then((Plotly) => {
    for (const root of plotRoots.values()) {
      Plotly.purge(root);
    }
  });
});
</script>
