<template>
  <div ref="root" class="trace-plot result-plot"></div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from "vue";

import { loadPlotly } from "../lib/plotly";

const props = defineProps({
  detector: { type: String, default: "ATTPC" },
  metric: { type: String, required: true },
  series: { type: Object, required: true },
  variant: { type: String, default: "" },
  thresholds: { type: Array, default: () => [] },
  valueBinCount: { type: Number, default: 0 },
  binCount: { type: Number, default: 0 },
  binCenters: { type: Array, default: () => [] },
  binLabel: { type: String, default: "" },
  countLabel: { type: String, default: "Count" },
  scaleMode: { type: String, default: "linear" },
  cdfRenderMode: { type: String, default: "2d" },
  cdfProjectionBin: { type: Number, default: 60 },
});

const root = ref(null);

function histogramIndices(count) {
  return Array.from({ length: count }, (_, index) => index);
}

function cdfValueCenters(count) {
  return Array.from(
    { length: count },
    (_, index) => (index + 0.5) / count,
  );
}

function transposeMatrix(matrix) {
  if (!matrix.length) {
    return [];
  }
  return matrix[0].map((_, columnIndex) => matrix.map((row) => row[columnIndex]));
}

function transformHeatmapValues(matrix, scaleMode) {
  return matrix.map((row) => row.map((value) => {
    const count = Number(value || 0);
    if (count <= 0) {
      return null;
    }
    if (scaleMode === "log") {
      return Math.log10(count + 1);
    }
    return count;
  }));
}

function resolveProjectionThreshold(thresholds, requestedBin) {
  if (!thresholds.length) {
    return { threshold: requestedBin, index: 0 };
  }
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  thresholds.forEach((threshold, index) => {
    const distance = Math.abs(Number(threshold) - requestedBin);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return {
    threshold: Number(thresholds[bestIndex]),
    index: bestIndex,
  };
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

async function renderCdfHeatmap() {
  const Plotly = await loadPlotly();
  const thresholds = props.thresholds.map(Number);
  const values = cdfValueCenters(props.valueBinCount);
  const original = transposeMatrix(props.series.histogram);
  const transformed = transformHeatmapValues(original, props.scaleMode);

  Plotly.react(
    root.value,
    [
      {
        type: "heatmap",
        z: transformed,
        customdata: original,
        x: thresholds,
        y: values,
        colorscale: "YlGnBu",
        hovertemplate: "CDF bin %{x}<br>CDF value %{y:.3f}<br>Count %{customdata}<extra></extra>",
        colorbar: {
          title: props.scaleMode === "log" ? "log10(count + 1)" : "Count",
        },
      },
    ],
    {
      ...baseLayout(),
      title: { text: props.series.title, x: 0.02, xanchor: "left" },
      xaxis: {
        title: "CDF bin",
        range: [0, 150],
        zeroline: false,
        gridcolor: "#e7dfcf",
      },
      yaxis: {
        title: "CDF value",
        range: [0, 1],
        zeroline: false,
        gridcolor: "#e7dfcf",
      },
    },
    {
      displayModeBar: false,
      responsive: true,
    },
  );
}

async function renderCdfProjection() {
  const Plotly = await loadPlotly();
  const thresholds = props.thresholds.map(Number);
  const values = cdfValueCenters(props.valueBinCount);
  const { index, threshold } = resolveProjectionThreshold(
    thresholds,
    props.cdfProjectionBin,
  );
  const histogram = props.series.histogram[index] || [];
  const yValues = histogram.map((value) => {
    const count = Number(value || 0);
    if (props.scaleMode === "log" && count <= 0) {
      return null;
    }
    return count;
  });

  Plotly.react(
    root.value,
    [
      {
        type: "bar",
        x: values,
        y: yValues,
        customdata: histogram,
        marker: {
          color: "#174f40",
        },
        hovertemplate: "CDF value %{x:.3f}<br>Count %{customdata}<extra></extra>",
      },
    ],
    {
      ...baseLayout(),
      title: {
        text: `${props.series.title} · CDF bin ${threshold}`,
        x: 0.02,
        xanchor: "left",
      },
      xaxis: {
        title: "CDF value",
        range: [0, 1],
        zeroline: false,
        gridcolor: "#e7dfcf",
      },
      yaxis: {
        title: "Count",
        type: props.scaleMode === "log" ? "log" : "linear",
        zeroline: false,
        gridcolor: "#e7dfcf",
      },
      bargap: 0,
    },
    {
      displayModeBar: false,
      responsive: true,
    },
  );
}

function hoverLabel() {
  if (props.metric === "amplitude") {
    return "Amplitude";
  }
  return props.binLabel || "Value";
}

function isIcAmplitude() {
  return props.detector === "IC" && props.metric === "amplitude";
}

function rebinIcAmplitude(histogram) {
  const targetBinCount = 400;
  const sourceMin = 0;
  const sourceMax = 2000;
  const sourceWidth = 5;
  const rebinned = Array.from({ length: targetBinCount }, () => 0);
  for (let index = sourceMin; index < sourceMax && index < histogram.length; index += 1) {
    const targetIndex = Math.floor(index / sourceWidth);
    rebinned[targetIndex] += Number(histogram[index] || 0);
  }
  const centers = Array.from(
    { length: targetBinCount },
    (_, index) => index * sourceWidth + sourceWidth / 2,
  );
  return { histogram: rebinned, centers, binCount: targetBinCount };
}

function isBaseline() {
  return props.metric === "baseline";
}

function rebinBaseline(histogram) {
  const targetBinCount = 200;
  const sourceMin = -100;
  const sourceMax = 100;
  const sourceWidth = (sourceMax - sourceMin) / targetBinCount;
  const rebinned = Array.from({ length: targetBinCount }, () => 0);
  const sourceCenters = props.binCenters?.length
    ? props.binCenters.map(Number)
    : histogramIndices(histogram.length);
  for (let sourceIndex = 0; sourceIndex < histogram.length; sourceIndex += 1) {
    const center = Number(sourceCenters[sourceIndex]);
    if (Number.isNaN(center) || center < sourceMin || center > sourceMax) {
      continue;
    }
    const targetIndex = Math.min(
      targetBinCount - 1,
      Math.floor((center - sourceMin) / sourceWidth),
    );
    rebinned[targetIndex] += Number(histogram[sourceIndex] || 0);
  }
  const centers = Array.from(
    { length: targetBinCount },
    (_, index) => sourceMin + index * sourceWidth + sourceWidth / 2,
  );
  return { histogram: rebinned, centers, binCount: targetBinCount };
}

async function renderOneDimensional() {
  const Plotly = await loadPlotly();
  const display = isIcAmplitude()
    ? rebinIcAmplitude(props.series.histogram)
    : isBaseline()
      ? rebinBaseline(props.series.histogram)
    : {
      histogram: props.series.histogram,
      centers: props.binCenters?.length
        ? props.binCenters
        : histogramIndices(props.binCount || props.series.histogram.length),
      binCount: props.binCount || props.series.histogram.length,
    };
  const xValues = display.centers;
  const yValues = display.histogram.map((value) => {
    const total = Number(value || 0);
    if (props.scaleMode === "log" && total <= 0) {
      return null;
    }
    return total;
  });

  Plotly.react(
    root.value,
    [
      {
        type: "bar",
        x: xValues,
        y: yValues,
        customdata: display.histogram,
        marker: {
          color: "#174f40",
        },
        hovertemplate: `${hoverLabel()} %{x}<br>${props.countLabel || "Count"} %{customdata}<extra></extra>`,
      },
    ],
    {
      ...baseLayout(),
      title: { text: props.series.title, x: 0.02, xanchor: "left" },
      xaxis: {
        title: props.binLabel || hoverLabel(),
        range: isIcAmplitude() ? [0, 2000] : isBaseline() ? [-100, 100] : undefined,
        zeroline: false,
        gridcolor: "#e7dfcf",
      },
      yaxis: {
        title: props.countLabel || "Count",
        type: props.scaleMode === "log" ? "log" : "linear",
        zeroline: false,
        gridcolor: "#e7dfcf",
      },
      bargap: 0,
    },
    {
      displayModeBar: false,
      responsive: true,
    },
  );
}

async function renderPlot() {
  if (!root.value) {
    return;
  }
  if (props.metric === "cdf") {
    if (props.cdfRenderMode === "projection") {
      await renderCdfProjection();
      return;
    }
    await renderCdfHeatmap();
    return;
  }
  await renderOneDimensional();
}

onMounted(() => {
  void renderPlot();
});

watch(
  () => [
    props.metric,
    props.detector,
    props.series,
    props.variant,
    props.thresholds,
    props.valueBinCount,
    props.binCount,
    props.binCenters,
    props.binLabel,
    props.countLabel,
    props.scaleMode,
    props.cdfRenderMode,
    props.cdfProjectionBin,
  ],
  () => {
    void renderPlot();
  },
  { deep: true },
);

onBeforeUnmount(() => {
  void loadPlotly().then((Plotly) => {
    if (root.value) {
      Plotly.purge(root.value);
    }
  });
});
</script>
