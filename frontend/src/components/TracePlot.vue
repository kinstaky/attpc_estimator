<template>
  <div class="trace-plot-stack">
    <div ref="primaryRoot" class="trace-plot"></div>
    <div
      v-if="visualMode !== 'raw'"
      ref="secondaryRoot"
      class="trace-plot trace-plot--secondary"
    ></div>
    <div
      v-if="visualMode === 'curvature'"
      ref="tertiaryRoot"
      class="trace-plot trace-plot--secondary"
    ></div>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

import { loadPlotly } from "../lib/plotly";

const props = defineProps({
  trace: { type: Object, default: null },
  visualMode: { type: String, default: "raw" },
});

const primaryRoot = ref(null);
const secondaryRoot = ref(null);
const tertiaryRoot = ref(null);
let syncingCurvatureRange = false;
let curvatureRange = null;

function shouldClipCompatTrace(trace) {
  if (!trace || trace.detector !== "GAGG" || trace.traceId >= 16) {
    return false;
  }
  if (!Array.isArray(trace.raw) || trace.raw.length < 256) {
    return false;
  }
  return trace.raw[128] === 0 && trace.raw[192] === 0 && trace.raw[255] === 0;
}

function clipPeakAnalysis(analysis, sampleCount) {
  if (!analysis) {
    return null;
  }
  const kept = analysis.peakIndices
    .map((peakIndex, index) => ({ peakIndex, index }))
    .filter(({ peakIndex }) => peakIndex < sampleCount)
    .map(({ index }) => index);
  return {
    peakIndices: kept.map((index) => analysis.peakIndices[index]),
    leftIndices: kept.map((index) => analysis.leftIndices[index]),
    rightIndices: kept.map((index) => analysis.rightIndices[index]),
    peakCount: kept.length,
  };
}

const displayTrace = computed(() => {
  const trace = props.trace;
  if (!shouldClipCompatTrace(trace)) {
    return trace;
  }
  const sampleCount = 128;
  return {
    ...trace,
    raw: trace.raw.slice(0, sampleCount),
    trace: trace.trace.slice(0, sampleCount),
    bitflipAnalysis: trace.bitflipAnalysis
      ? {
          ...trace.bitflipAnalysis,
          xIndices: trace.bitflipAnalysis.xIndices.slice(0, sampleCount),
          firstDerivative: trace.bitflipAnalysis.firstDerivative.slice(0, sampleCount),
          secondDerivative: trace.bitflipAnalysis.secondDerivative.slice(0, sampleCount),
          structures: (trace.bitflipAnalysis.structures || []).filter(
            (structure) => structure.startBaselineIndex < sampleCount,
          ),
        }
      : trace.bitflipAnalysis,
    peakAnalysis: clipPeakAnalysis(trace.peakAnalysis, sampleCount),
  };
});

function sampleIndices(values) {
  return values.map((_, index) => index);
}

function sampleAxis(sampleCount, rangeOverride = null) {
  return {
    title: "Sample",
    range: rangeOverride ?? [0, Math.max(0, sampleCount - 1)],
    autorange: false,
    zeroline: false,
    gridcolor: "#e7dfcf",
  };
}

function baseLayout() {
  return {
    margin: { t: 48, r: 24, b: 48, l: 56 },
    paper_bgcolor: "#fffdf8",
    plot_bgcolor: "#fffdf8",
    font: {
      family: "ui-sans-serif, system-ui, sans-serif",
      color: "#222",
    },
  };
}

function defaultConfig() {
  return {
    displayModeBar: false,
    responsive: true,
  };
}

async function renderRawPlot(root) {
  if (!root || !props.trace) {
    return;
  }
  const Plotly = await loadPlotly();
  const trace = displayTrace.value;
  Plotly.react(
    root,
    [
      {
        type: "scatter",
        mode: "lines",
        x: sampleIndices(trace.raw),
        y: trace.raw,
        line: {
          color: "#174f40",
          width: 2,
        },
        name: "Raw trace",
      },
    ],
    {
      ...baseLayout(),
      title: { text: "Raw Trace", x: 0.02, xanchor: "left" },
      xaxis: sampleAxis(trace.raw.length),
      yaxis: {
        title: "Amplitude",
        zeroline: false,
        gridcolor: "#e7dfcf",
      },
      showlegend: false,
    },
    defaultConfig(),
  );
}

async function renderTimeDomainPlot(root) {
  if (!root || !props.trace) {
    return;
  }
  const Plotly = await loadPlotly();
  const trace = displayTrace.value;

  Plotly.react(
    root,
    [
      {
        type: "scatter",
        mode: "lines",
        x: sampleIndices(trace.raw),
        y: trace.raw,
        line: {
          color: "#b46b2f",
          width: 1.8,
        },
        name: "Raw trace",
      },
      {
        type: "scatter",
        mode: "lines",
        x: sampleIndices(trace.trace),
        y: trace.trace,
        line: {
          color: "#174f40",
          width: 2.2,
        },
        name: "Baseline removed",
      },
    ],
    {
      ...baseLayout(),
      title: { text: "Time Domain", x: 0.02, xanchor: "left" },
      legend: {
        orientation: "h",
        x: 1,
        xanchor: "right",
        y: 1.16,
      },
      xaxis: sampleAxis(
        trace.trace.length,
        props.visualMode === "curvature" ? curvatureRange : null,
      ),
      yaxis: {
        title: "Amplitude",
        zeroline: false,
        gridcolor: "#e7dfcf",
      },
    },
    defaultConfig(),
  );
}

async function renderFrequencyPlot(root) {
  if (!root || !props.trace) {
    return;
  }
  const Plotly = await loadPlotly();
  const trace = displayTrace.value;
  Plotly.react(
    root,
    [
      {
        type: "scatter",
        mode: "lines",
        x: sampleIndices(trace.transformed),
        y: trace.transformed,
        line: {
          color: "#3b5ba9",
          width: 2,
        },
        fill: "tozeroy",
        fillcolor: "rgba(59, 91, 169, 0.16)",
        name: "Frequency distribution",
      },
    ],
    {
      ...baseLayout(),
      title: { text: "Frequency Distribution", x: 0.02, xanchor: "left" },
      xaxis: {
        title: "Frequency bin",
        zeroline: false,
        gridcolor: "#e7dfcf",
      },
      yaxis: {
        title: "Magnitude",
        zeroline: false,
        gridcolor: "#e7dfcf",
      },
      showlegend: false,
    },
    defaultConfig(),
  );
}

async function renderPeakPlot(root) {
  if (!root || !props.trace) {
    return;
  }
  const Plotly = await loadPlotly();
  const trace = displayTrace.value;
  const analysis = trace.peakAnalysis || {
    peakIndices: [],
    leftIndices: [],
    rightIndices: [],
    peakCount: 0,
  };
  const traces = [
    {
      type: "scatter",
      mode: "lines",
      x: sampleIndices(trace.trace),
      y: trace.trace,
      line: {
        color: "#174f40",
        width: 2.2,
      },
      name: "Baseline removed",
    },
  ];
  if (analysis.peakIndices.length > 0) {
    traces.push({
      type: "scatter",
      mode: "markers",
      x: analysis.peakIndices,
      y: analysis.peakIndices.map((index) => trace.trace[index]),
      marker: {
        color: "#a84a24",
        size: 9,
      },
      name: "Peaks",
    });
    traces.push({
      type: "scatter",
      mode: "markers",
      x: analysis.leftIndices,
      y: analysis.leftIndices.map((index) => trace.trace[index]),
      marker: {
        color: "#3b5ba9",
        size: 8,
        symbol: "diamond",
      },
      name: "Peak start",
    });
    traces.push({
      type: "scatter",
      mode: "markers",
      x: analysis.rightIndices,
      y: analysis.rightIndices.map((index) => trace.trace[index]),
      marker: {
        color: "#6a4c93",
        size: 8,
        symbol: "diamond-open",
      },
      name: "Peak end",
    });
  }
  Plotly.react(
    root,
    traces,
    {
      ...baseLayout(),
      title: {
        text: `Peak Analysis<br><sup>${analysis.peakCount} peak${analysis.peakCount === 1 ? "" : "s"}</sup>`,
        x: 0.02,
        xanchor: "left",
      },
      legend: {
        orientation: "h",
        x: 1,
        xanchor: "right",
        y: 1.18,
      },
      xaxis: sampleAxis(trace.trace.length),
      yaxis: {
        title: "Amplitude",
        zeroline: false,
        gridcolor: "#e7dfcf",
      },
    },
    defaultConfig(),
  );
}

function clearRelayoutSyncHandlers() {
  primaryRoot.value?.removeAllListeners?.("plotly_relayout");
  secondaryRoot.value?.removeAllListeners?.("plotly_relayout");
  tertiaryRoot.value?.removeAllListeners?.("plotly_relayout");
}

async function bindCurvatureRangeSync() {
  const roots = [primaryRoot.value, secondaryRoot.value, tertiaryRoot.value].filter(Boolean);
  if (roots.length < 2) {
    return;
  }
  const Plotly = await loadPlotly();
  const trace = displayTrace.value;
  clearRelayoutSyncHandlers();

  const syncTarget = async (target, eventData) => {
    if (syncingCurvatureRange) {
      return;
    }
    if (eventData["xaxis.autorange"]) {
      curvatureRange = null;
      syncingCurvatureRange = true;
      await Plotly.relayout(target, {
        "xaxis.range[0]": 0,
        "xaxis.range[1]": Math.max(0, trace.trace.length - 1),
      });
      syncingCurvatureRange = false;
      return;
    }
    const rangeStart = eventData["xaxis.range[0]"];
    const rangeEnd = eventData["xaxis.range[1]"];
    if (rangeStart === undefined || rangeEnd === undefined) {
      return;
    }
    curvatureRange = [rangeStart, rangeEnd];
    syncingCurvatureRange = true;
    await Plotly.relayout(target, {
      "xaxis.range[0]": rangeStart,
      "xaxis.range[1]": rangeEnd,
    });
    syncingCurvatureRange = false;
  };

  for (const sourceRoot of roots) {
    sourceRoot.on("plotly_relayout", (eventData) => {
      for (const targetRoot of roots) {
        if (targetRoot === sourceRoot) {
          continue;
        }
        void syncTarget(targetRoot, eventData);
      }
    });
  }
}

async function renderFirstDerivativePlot(root) {
  if (!root || !props.trace) {
    return;
  }
  const Plotly = await loadPlotly();
  const trace = displayTrace.value;
  const analysis = trace.bitflipAnalysis;

  Plotly.react(
    root,
    [
      {
        type: "scatter",
        mode: "lines",
        x: analysis.xIndices,
        y: analysis.firstDerivative,
        line: {
          color: "#3b5ba9",
          width: 1.8,
        },
        name: "First derivative",
      },
    ],
    {
      ...baseLayout(),
      title: { text: "First Derivative", x: 0.02, xanchor: "left" },
      xaxis: sampleAxis(analysis.xIndices.length, curvatureRange),
      yaxis: {
        title: "First derivative",
        zeroline: false,
        gridcolor: "#e7dfcf",
      },
      showlegend: false,
    },
    defaultConfig(),
  );
}

async function renderSecondDerivativePlot(root) {
  if (!root || !props.trace) {
    return;
  }
  const Plotly = await loadPlotly();
  const trace = displayTrace.value;
  const analysis = trace.bitflipAnalysis;
  const structureWaveformXs = [];
  const structureWaveformYs = [];
  const structureMarkerXs = [];
  const structureMarkerYs = [];
  for (const structure of analysis.structures || []) {
    for (
      let index = structure.startBaselineIndex;
      index <= structure.endBaselineIndex;
      index += 1
    ) {
      structureWaveformXs.push(index);
      structureWaveformYs.push(analysis.secondDerivative[index]);
    }
    structureWaveformXs.push(null);
    structureWaveformYs.push(null);
    structureMarkerXs.push(
      structure.startBaselineIndex,
      structure.endBaselineIndex,
    );
    structureMarkerYs.push(
      analysis.secondDerivative[structure.startBaselineIndex],
      analysis.secondDerivative[structure.endBaselineIndex],
    );
  }
  const traces = [
    {
      type: "scatter",
      mode: "lines",
      x: analysis.xIndices,
      y: analysis.secondDerivative,
      line: {
        color: "#a84a24",
        width: 2,
      },
      name: "Second derivative",
    },
  ];
  if (structureWaveformXs.length > 0) {
    traces.push({
      type: "scatter",
      mode: "lines",
      x: structureWaveformXs,
      y: structureWaveformYs,
      line: {
        color: "#1a7f5a",
        width: 3,
      },
      name: "Found bitflip",
    });
    traces.push({
      type: "scatter",
      mode: "markers",
      x: structureMarkerXs,
      y: structureMarkerYs,
      marker: {
        color: "#1a7f5a",
        size: 8,
      },
      name: "Bitflip boundaries",
    });
  }

  Plotly.react(
    root,
    traces,
    {
      ...baseLayout(),
      title: { text: "Second Derivative", x: 0.02, xanchor: "left" },
      legend: {
        orientation: "h",
        x: 1,
        xanchor: "right",
        y: 1.16,
      },
      xaxis: sampleAxis(analysis.xIndices.length, curvatureRange),
      yaxis: {
        title: "Second derivative",
        zeroline: false,
        gridcolor: "#e7dfcf",
      },
    },
    defaultConfig(),
  );
}

async function renderPlots() {
  if (!props.trace || !primaryRoot.value) {
    return;
  }
  await nextTick();

  const Plotly = await loadPlotly();
  if (props.visualMode === "raw") {
    curvatureRange = null;
    await renderRawPlot(primaryRoot.value);
    clearRelayoutSyncHandlers();
    if (secondaryRoot.value) {
      Plotly.purge(secondaryRoot.value);
    }
    if (tertiaryRoot.value) {
      Plotly.purge(tertiaryRoot.value);
    }
    return;
  }

  if (props.visualMode === "peak") {
    curvatureRange = null;
    await renderRawPlot(primaryRoot.value);
    if (!secondaryRoot.value) {
      return;
    }
    clearRelayoutSyncHandlers();
    if (tertiaryRoot.value) {
      Plotly.purge(tertiaryRoot.value);
    }
    await renderPeakPlot(secondaryRoot.value);
    return;
  }

  await renderTimeDomainPlot(primaryRoot.value);
  if (!secondaryRoot.value) {
    return;
  }
  if (props.visualMode === "curvature") {
    await renderFirstDerivativePlot(secondaryRoot.value);
    if (!tertiaryRoot.value) {
      return;
    }
    await renderSecondDerivativePlot(tertiaryRoot.value);
    await bindCurvatureRangeSync();
    return;
  }
  clearRelayoutSyncHandlers();
  if (tertiaryRoot.value) {
    Plotly.purge(tertiaryRoot.value);
  }
  await renderFrequencyPlot(secondaryRoot.value);
}

onMounted(() => {
  void renderPlots();
});
watch(() => props.trace, () => {
  void renderPlots();
}, { deep: true });
watch(() => props.visualMode, () => {
  void renderPlots();
});

onBeforeUnmount(() => {
  clearRelayoutSyncHandlers();
  void loadPlotly().then((Plotly) => {
    if (primaryRoot.value) {
      Plotly.purge(primaryRoot.value);
    }
    if (secondaryRoot.value) {
      Plotly.purge(secondaryRoot.value);
    }
    if (tertiaryRoot.value) {
      Plotly.purge(tertiaryRoot.value);
    }
  });
});
</script>
